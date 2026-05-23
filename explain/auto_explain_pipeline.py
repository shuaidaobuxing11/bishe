"""自动解释流水线：轨迹 → 归因 → 分组 → 评分 → 报告。"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import yaml

from explain.counterfactual_explainer import run_counterfactual_tests, save_counterfactual
from explain.explain_scores import compute_explain_scores, save_explain_scores
from explain.explain_utils import plot_importance_bar, save_importance_csv
from explain.feature_groups import (
    aggregate_importance_by_group,
    get_feature_groups,
    importance_df_from_array,
    plot_group_importance,
    save_group_importance,
)
from explain.feature_names import get_feature_names, validate_obs_dim
from explain.keyframe_explainer import build_keyframes_markdown, select_keyframes, steps_to_df
from explain.scenario_explain_rules import evaluate_scenario_alignment
from explain.episode_explain_report import build_report_markdown
from visualization.trajectory_metrics import compute_trajectory_metrics
from visualization.trajectory_recorder import load_trajectory


def _load_config(path: str | None = None) -> dict:
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "configs", "explain_report_config.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _trajectory_derived_weights(steps: list[dict]) -> dict[str, float]:
    if not steps:
        return {}
    d12 = np.array([s.get("distance_uav1_uav2", 0) for s in steps], dtype=float)
    md = np.array([s.get("mean_distance_to_target", 0) for s in steps], dtype=float)
    d1 = np.array([s.get("distance_uav1_target", 0) for s in steps], dtype=float)
    d2 = np.array([s.get("distance_uav2_target", 0) for s in steps], dtype=float)
    return {
        "distance_uav1_uav2": float(np.std(d12) + 1.0 / (np.min(d12) + 1e-3)),
        "mean_distance_to_target": float(np.std(md)),
        "distance_uav1_target": float(np.std(d1)),
        "distance_uav2_target": float(np.std(d2)),
    }


def _expert_pseudo_importance(steps: list[dict], feature_names: list[str]) -> np.ndarray:
    obs_list = [s.get("obs") for s in steps if s.get("obs")]
    if not obs_list:
        return np.ones(len(feature_names)) / len(feature_names)
    arr = np.array(obs_list, dtype=float)
    var = np.var(arr, axis=0)
    if var.sum() < 1e-8:
        var = np.ones_like(var)
    return var / var.sum()


def _compute_episode_importance(
    policy_name: str,
    model_path: str | None,
    steps: list[dict],
    method: str,
    device: str,
) -> tuple[np.ndarray, list[str], dict[int, np.ndarray], bool]:
    """返回 (mean_abs, feature_names, attrs_by_step, used_captum)。"""
    feature_names = get_feature_names()
    attrs_by_step: dict[int, np.ndarray] = {}

    if policy_name == "expert_v2" or not model_path or not os.path.isfile(model_path):
        mean_abs = _expert_pseudo_importance(steps, feature_names)
        return mean_abs, feature_names, attrs_by_step, False

    try:
        from explain.captum_explainer import _compute_attribution, _load_wrapper, _predict_action
    except Exception as exc:
        print(f"[warn] Captum/Torch 不可用，使用行为伪归因: {exc}")
        mean_abs = _expert_pseudo_importance(steps, feature_names)
        return mean_abs, feature_names, attrs_by_step, False

    try:
        wrapper, _ = _load_wrapper(model_path, device=device)
    except Exception as exc:
        print(f"[warn] 模型加载失败，使用行为伪归因: {exc}")
        mean_abs = _expert_pseudo_importance(steps, feature_names)
        return mean_abs, feature_names, attrs_by_step, False

    attrs_list = []
    for s in steps:
        obs = np.asarray(s.get("obs"), dtype=np.float32)
        if obs.size == 0:
            continue
        validate_obs_dim(len(obs))
        act = int(s.get("action", _predict_action(wrapper, obs, device)))
        try:
            attr = _compute_attribution(wrapper, obs, act, method, device)
        except Exception:
            attr = np.abs(obs) / (np.abs(obs).sum() + 1e-8)
        attrs_list.append(np.abs(attr))
        attrs_by_step[int(s["step"])] = attr
    if not attrs_list:
        mean_abs = _expert_pseudo_importance(steps, feature_names)
        return mean_abs, feature_names, attrs_by_step, False
    mean_abs = np.mean(np.stack(attrs_list, axis=0), axis=0)
    return mean_abs, feature_names, attrs_by_step, True


def run_auto_explain_pipeline(
    policy_name: str,
    scenario_name: str,
    seed: int,
    trajectory_path: str,
    model_path: str | None = None,
    save_dir: str = "results/explain_reports",
    config: dict | None = None,
) -> dict[str, Any]:
    cfg = _load_config() if config is None else config
    explain_cfg = cfg.get("explain", {})
    thresholds = cfg.get("thresholds", {})
    method = explain_cfg.get("method", "integrated_gradients")
    device = explain_cfg.get("device", "cpu")

    out_dir = os.path.join(save_dir, f"{policy_name}_{scenario_name}_seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    data = load_trajectory(trajectory_path)
    steps = data.get("steps", [])
    metrics = compute_trajectory_metrics(trajectory_path)

    mean_abs, feature_names, attrs_by_step, used_captum = _compute_episode_importance(
        policy_name, model_path, steps, method, device
    )
    if policy_name == "expert_v2":
        explanation_source = "rule_and_trajectory"
    elif used_captum:
        explanation_source = "captum_and_trajectory"
    else:
        explanation_source = "rule_and_trajectory"

    imp_df = importance_df_from_array(feature_names, mean_abs).sort_values("abs_attribution", ascending=False)
    feat_csv = os.path.join(out_dir, "feature_importance.csv")
    feat_png = os.path.join(out_dir, "fig_feature_importance.png")
    save_importance_csv(feat_csv, feature_names, mean_abs)
    plot_importance_bar(feature_names, mean_abs, feat_png, title=f"{policy_name} @ {scenario_name}")

    tderived = _trajectory_derived_weights(steps)
    groups, group_warnings = get_feature_groups(feature_names)
    group_df = aggregate_importance_by_group(imp_df, groups, allow_overlap=True, trajectory_derived=tderived)
    group_csv = os.path.join(out_dir, "feature_group_importance.csv")
    group_png = os.path.join(out_dir, "fig_group_importance.png")
    save_group_importance(group_df, group_csv)
    footnote = ""
    if explanation_source == "rule_and_trajectory":
        footnote = "Note: rule/trajectory-based scores (not Captum IG)."
    elif group_df.get("allow_overlap", pd.Series([True])).iloc[0]:
        footnote = "Semantic groups may overlap (e.g. distance_uav1_uav2 in coordination & safety)."
    plot_group_importance(group_df, group_png, title=f"Fine-grained groups — {policy_name}", footnote=footnote)

    scores = compute_explain_scores(
        group_df,
        metrics,
        scenario_name,
        importance_df=imp_df,
        explanation_source=explanation_source,
    )
    scores_path = os.path.join(out_dir, "explain_scores.json")
    save_explain_scores(scores, policy_name, scenario_name, seed, save_dir=os.path.join(out_dir, "..", "explain_scores"))
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in scores.items() if k != "scenario_alignment_detail"}, f, indent=2, ensure_ascii=False)

    keyframes_md = ""
    if explain_cfg.get("enable_keyframe_explanation", True) and steps:
        kf = select_keyframes(steps_to_df(steps))
        keyframes_md = build_keyframes_markdown(kf, steps, attrs_by_step or None, feature_names)
        with open(os.path.join(out_dir, "keyframes.md"), "w", encoding="utf-8") as f:
            f.write(keyframes_md)

    cf_df = pd.DataFrame()
    if explain_cfg.get("enable_counterfactual", True) and steps and steps[0].get("obs"):
        obs0 = np.asarray(steps[len(steps) // 2]["obs"], dtype=np.float32)

        def _predict(o: np.ndarray) -> int:
            if policy_name == "expert_v2":
                from visualization.policy_loaders import ExpertPolicyWrapper
                from visualization.trajectory_recorder import make_env_for_scenario

                env = make_env_for_scenario(scenario_name, seed=seed)
                pol = ExpertPolicyWrapper()
                pol.bind_env(env)
                a = pol.predict(o)
                env.close()
                return int(a)
            if model_path and os.path.isfile(model_path):
                try:
                    from visualization.policy_loaders import load_policy

                    pol = load_policy(model_path=model_path)
                    return int(pol.predict(o))
                except Exception:
                    pass
            return int(steps[len(steps) // 2].get("action", 0))

        try:
            cf_df = run_counterfactual_tests(_predict, obs0, feature_names)
            save_counterfactual(
                cf_df,
                os.path.join(out_dir, "counterfactual.csv"),
                os.path.join(out_dir, "counterfactual.md"),
            )
        except Exception as exc:
            with open(os.path.join(out_dir, "counterfactual.md"), "w", encoding="utf-8") as f:
                f.write(f"# 反事实解释\n\n跳过：{exc}\n")

    align = scores.get("scenario_alignment_detail") or evaluate_scenario_alignment(scores, scenario_name)
    bundle = {
        "policy_name": policy_name,
        "scenario_name": scenario_name,
        "seed": seed,
        "metrics": metrics,
        "scores": scores,
        "alignment": align,
        "importance_df": imp_df,
        "group_df": group_df,
        "thresholds": thresholds,
        "out_dir": out_dir,
        "explanation_source": explanation_source,
        "used_captum": used_captum,
        "group_warnings": group_warnings,
    }
    report_md = build_report_markdown(bundle)
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    bundle["report_path"] = report_path
    bundle["paths"] = {
        "report": report_path,
        "feature_importance": feat_csv,
        "group_importance": group_csv,
        "scores": scores_path,
    }
    return bundle
