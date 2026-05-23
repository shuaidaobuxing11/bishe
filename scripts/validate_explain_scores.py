#!/usr/bin/env python3
"""校验可解释性报告与评分是否合理。"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from explain.feature_groups import get_feature_groups
from explain.feature_names import get_feature_names

FINE_GRAINED_SCORE_COLS = [
    "target_absolute_score",
    "uav_absolute_score",
    "target_relative_score",
    "inter_uav_coordination_score",
    "safety_score",
    "velocity_score",
    "action_stability_score",
    "task_efficiency_score",
    "scenario_alignment_score",
]

TARGET_RELATIVE_FEATURES = {
    "distance_uav1_target",
    "distance_uav2_target",
    "mean_distance_to_target",
    "rel_uav1_target_x",
    "rel_uav1_target_y",
    "rel_uav2_target_x",
    "rel_uav2_target_y",
    "uav1_dist_to_target",
    "uav2_dist_to_target",
}


def _check_scores_in_range(scores: dict, policy: str) -> list[str]:
    issues = []
    for k in FINE_GRAINED_SCORE_COLS:
        v = scores.get(k)
        if v is None:
            issues.append(f"{policy}: 缺少 {k}")
            continue
        if isinstance(v, (int, float)) and (np.isnan(v) or v < 0 or v > 1):
            issues.append(f"{policy}: {k}={v} 不在 [0,1]")
    return issues


def _check_expert_source(policy: str, scores: dict) -> tuple[list[str], list[str]]:
    """返回 (errors, warnings)。"""
    src = scores.get("explanation_source", "")
    errors: list[str] = []
    warnings: list[str] = []
    if policy == "expert_v2" and src != "rule_and_trajectory":
        errors.append(f"expert_v2 应标注 rule_and_trajectory，当前为 {src}")
    elif policy != "expert_v2" and src == "rule_and_trajectory":
        warnings.append(f"{policy}: 使用 rule_and_trajectory（Captum 不可用或未加载模型时属正常降级）")
    return errors, warnings


def _check_feature_alignment(report_dir: str, policy: str) -> list[str]:
    imp_csv = os.path.join(report_dir, "feature_importance.csv")
    if not os.path.isfile(imp_csv):
        return [f"{policy}: 缺少 feature_importance.csv"]
    df = pd.read_csv(imp_csv)
    fnames = get_feature_names()
    if len(df) != len(fnames):
        return [f"{policy}: importance 行数 {len(df)} != feature_names {len(fnames)}"]
    return []


def _check_target_relative_grouping(report_dir: str, policy: str) -> list[str]:
    gcsv = os.path.join(report_dir, "feature_group_importance.csv")
    if not os.path.isfile(gcsv):
        return []
    gdf = pd.read_csv(gcsv)
    groups, _ = get_feature_groups()
    tr_feats = set(groups.get("target_relative", []))
    coord_row = gdf[gdf["group"] == "inter_uav_coordination"]
    if coord_row.empty:
        return []
    used = str(coord_row.iloc[0].get("features_used", ""))
    wrongly = [f for f in TARGET_RELATIVE_FEATURES if f in used]
    if wrongly:
        return [f"{policy}: 相对目标特征 {wrongly} 被归入 inter_uav_coordination"]
    return []


def _check_default_alignment(policy: str, scores: dict, scenario: str) -> list[str]:
    if scenario != "default":
        return []
    align = scores.get("scenario_alignment_score", 0)
    tr = scores.get("target_relative_score", 0)
    ta = scores.get("target_absolute_score", 0)
    if align < 0.08 and (tr >= 0.1 or ta >= 0.1):
        return [f"{policy}: default 场景 alignment={align:.3f} 可能偏低（target 相关分={tr:.3f}/{ta:.3f}）"]
    return []


def _check_streamlit_files(scenario: str, seed: int) -> list[str]:
    issues = []
    comp = os.path.join(ROOT, "results", "explain_comparisons", f"policy_compare_{scenario}_seed{seed}")
    for suffix in ("_fine_grained_scores.png", "_heatmap.png", "_scores.png", ".csv", ".md"):
        p = comp + suffix if suffix.startswith("_") else comp + suffix
        if not os.path.isfile(p):
            issues.append(f"Streamlit 对比文件缺失: {os.path.basename(p)}")
    return issues


def validate(scenario: str, seed: int, report_dir: str) -> int:
    report_root = os.path.join(ROOT, report_dir)
    if not os.path.isdir(report_root):
        print(f"[FAIL] 报告目录不存在: {report_root}")
        return 1

    all_issues: list[str] = []
    all_warnings: list[str] = []
    checked = 0

    for name in sorted(os.listdir(report_root)):
        if not name.endswith(f"_{scenario}_seed{seed}"):
            continue
        d = os.path.join(report_root, name)
        if not os.path.isdir(d):
            continue
        prefix = name.rsplit("_seed", 1)[0]
        if not prefix.endswith(f"_{scenario}"):
            continue
        policy = prefix[: -(len(scenario) + 1)]
        scores_path = os.path.join(d, "explain_scores.json")
        if not os.path.isfile(scores_path):
            all_issues.append(f"{policy}: 缺少 explain_scores.json")
            continue
        with open(scores_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
        checked += 1
        all_issues.extend(_check_scores_in_range(scores, policy))
        errs, warns = _check_expert_source(policy, scores)
        all_issues.extend(errs)
        all_warnings.extend(warns)
        all_issues.extend(_check_feature_alignment(d, policy))
        all_issues.extend(_check_target_relative_grouping(d, policy))
        all_issues.extend(_check_default_alignment(policy, scores, scenario))

    all_issues.extend(_check_streamlit_files(scenario, seed))

    print(f"校验场景={scenario} seed={seed}，检查策略数={checked}")
    if all_warnings:
        print(f"[INFO] {len(all_warnings)} 项提示：")
        for w in all_warnings:
            print(f"  - {w}")
    if all_issues:
        print(f"[WARN] 发现 {len(all_issues)} 项问题：")
        for i in all_issues:
            print(f"  - {i}")
        return 1
    print("[OK] 所有检查通过")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="校验可解释性评分")
    ap.add_argument("--scenario", type=str, default="default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_dir", type=str, default="results/explain_reports")
    args = ap.parse_args()
    sys.exit(validate(args.scenario, args.seed, args.report_dir))


if __name__ == "__main__":
    main()
