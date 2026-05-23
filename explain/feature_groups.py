"""细粒度特征分组、语义评分与组级聚合。"""
from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd

from explain.feature_names import get_feature_names

# Captum 可解释组（不含 action_stability / task_efficiency）
CAPTUM_GROUP_TEMPLATES: dict[str, tuple[str, ...]] = {
    "target_absolute": (
        "target_x_norm",
        "target_y_norm",
        "target_x",
        "target_y",
    ),
    "uav_absolute": (
        "uav1_x_norm",
        "uav1_y_norm",
        "uav2_x_norm",
        "uav2_y_norm",
        "uav1_x",
        "uav1_y",
        "uav2_x",
        "uav2_y",
    ),
    "target_relative": (
        "distance_uav1_target",
        "distance_uav2_target",
        "mean_distance_to_target",
        "rel_uav1_target_x",
        "rel_uav1_target_y",
        "rel_uav2_target_x",
        "rel_uav2_target_y",
        "uav1_dist_to_target",
        "uav2_dist_to_target",
    ),
    "inter_uav_coordination": (
        "distance_uav1_uav2",
        "dist_uav1_uav2",
        "rel_uav1_uav2_x",
        "rel_uav1_uav2_y",
        "uav_pair_distance",
    ),
    "safety": (
        "distance_uav1_uav2",
        "dist_uav1_uav2",
        "min_uav_distance",
        "boundary_distance",
        "uav1_boundary_distance",
        "uav2_boundary_distance",
        "collision_risk",
        "out_of_bounds_risk",
    ),
    "velocity": (
        "uav1_vx_norm",
        "uav1_vy_norm",
        "uav2_vx_norm",
        "uav2_vy_norm",
        "target_vx_norm",
        "target_vy_norm",
        "uav1_vx",
        "uav1_vy",
        "uav2_vx",
        "uav2_vy",
        "target_vx",
        "target_vy",
        "target_speed",
        "target_heading",
    ),
}

# 互斥分配优先级（allow_overlap=False）
EXCLUSIVE_PRIORITY = [
    "target_relative",
    "inter_uav_coordination",
    "safety",
    "target_absolute",
    "uav_absolute",
    "velocity",
]

# 轨迹字段 → 语义组辅助权重
TRAJECTORY_SEMANTIC_WEIGHTS: dict[str, str] = {
    "distance_uav1_target": "target_relative",
    "distance_uav2_target": "target_relative",
    "mean_distance_to_target": "target_relative",
    "distance_uav1_uav2": "inter_uav_coordination",
    "min_uav_distance": "safety",
    "action_switch_rate": "action_stability",
    "episode_length": "task_efficiency",
    "episode_return": "task_efficiency",
    "success": "task_efficiency",
}


def get_feature_groups(
    feature_names: list[str] | None = None,
) -> tuple[dict[str, list[str]], list[str]]:
    """返回 (groups, missing_features_warning)。"""
    names = feature_names or get_feature_names()
    name_set = set(names)
    groups: dict[str, list[str]] = {}
    all_template_feats = set()

    for gname, tpl in CAPTUM_GROUP_TEMPLATES.items():
        matched = [f for f in tpl if f in name_set]
        if matched:
            groups[gname] = matched
        all_template_feats.update(tpl)

    missing = sorted(all_template_feats - name_set)
    # 10 维 obs 下缺少 distance/rel 属正常，仅提示相对关系类需轨迹补充
    rel_missing = [m for m in missing if "dist" in m or "rel_" in m or "distance" in m]
    warnings_list: list[str] = []
    if rel_missing:
        warnings_list.append(
            "观测中无相对距离特征，target_relative / inter_uav 将结合 trajectory 距离字段补充。"
        )
    return groups, warnings_list


def importance_df_from_array(feature_names: list[str], attributions: np.ndarray) -> pd.DataFrame:
    attr = np.asarray(attributions, dtype=float).reshape(-1)
    if len(attr) != len(feature_names):
        raise ValueError(
            f"importance 维度 ({len(attr)}) 与 feature_names ({len(feature_names)}) 不一致"
        )
    return pd.DataFrame(
        {
            "feature": feature_names,
            "attribution": attr,
            "abs_attribution": np.abs(attr),
        }
    )


def _assign_exclusive(feature_groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """同一特征只归入优先级最高组。"""
    feat_to_group: dict[str, str] = {}
    for g in EXCLUSIVE_PRIORITY:
        for f in feature_groups.get(g, []):
            if f not in feat_to_group:
                feat_to_group[f] = g
    out: dict[str, list[str]] = {g: [] for g in feature_groups}
    for f, g in feat_to_group.items():
        out.setdefault(g, []).append(f)
    return {k: v for k, v in out.items() if v}


def aggregate_importance_by_group(
    importance_df: pd.DataFrame,
    feature_groups: dict[str, list[str]] | None = None,
    allow_overlap: bool = True,
    trajectory_derived: dict[str, float] | None = None,
) -> pd.DataFrame:
    if feature_groups is None:
        feature_groups, _ = get_feature_groups(list(importance_df["feature"]))

    groups_use = feature_groups if allow_overlap else _assign_exclusive(feature_groups)
    rows: list[dict[str, Any]] = []

    for group, feats in groups_use.items():
        sub = importance_df[importance_df["feature"].isin(feats)]
        raw = float(sub["abs_attribution"].sum()) if len(sub) else 0.0
        rows.append(
            {
                "group": group,
                "raw_importance": raw,
                "feature_count": len(feats),
                "features_used": ",".join(feats),
            }
        )

    # 轨迹语义补充（不重复计入 exclusive 的 obs 特征）
    if trajectory_derived:
        traj_group_boost: dict[str, float] = {}
        for tkey, val in trajectory_derived.items():
            g = TRAJECTORY_SEMANTIC_WEIGHTS.get(tkey)
            if g:
                traj_group_boost[g] = traj_group_boost.get(g, 0.0) + float(val)
        for gname, boost in traj_group_boost.items():
            found = False
            for r in rows:
                if r["group"] == gname:
                    r["raw_importance"] += boost
                    r["features_used"] += f";trajectory:{gname}"
                    found = True
                    break
            if not found:
                rows.append(
                    {
                        "group": gname,
                        "raw_importance": boost,
                        "feature_count": 0,
                        "features_used": f"trajectory:{gname}",
                    }
                )

    total = sum(r["raw_importance"] for r in rows) or 1.0
    for r in rows:
        r["normalized_importance"] = r["raw_importance"] / total
    df = pd.DataFrame(rows)
    df["allow_overlap"] = allow_overlap
    return df


def _trajectory_semantic_boost(metrics: dict[str, Any], steps: list[dict] | None = None) -> dict[str, float]:
    """从轨迹指标推导各语义组 0~1 辅助分。"""
    out: dict[str, float] = {}

    sw = float(metrics.get("action_switch_rate", 0.5))
    out["action_stability"] = float(np.clip(1.0 - sw, 0.0, 1.0))

    success = bool(metrics.get("success", False))
    ep_len = float(metrics.get("episode_length", 200))
    ep_ret = float(metrics.get("episode_return", -100))
    mean_d = float(metrics.get("mean_distance_to_target", 5.0))
    min_d = float(metrics.get("min_mean_distance", mean_d))

    eff = 0.0
    if success:
        eff += 0.5
    eff += 0.25 * float(np.clip(1.0 - ep_len / 200.0, 0.0, 1.0))
    eff += 0.25 * float(np.clip(1.0 - mean_d / 3.0, 0.0, 1.0))
    out["task_efficiency"] = float(np.clip(eff, 0.0, 1.0))

    min_uav = float(metrics.get("min_uav_distance", 1.0))
    out["safety"] = float(np.clip(min_uav / 1.5, 0.0, 1.0))

    # target_relative：距离越小、变化越有序 → 分越高
    out["target_relative"] = float(np.clip(1.0 - min_d / 2.5, 0.0, 1.0))

    out["inter_uav_coordination"] = float(np.clip(min_uav / 2.0, 0.0, 1.0))

    if steps:
        d1 = np.array([s.get("distance_uav1_target", 0) for s in steps], dtype=float)
        d2 = np.array([s.get("distance_uav2_target", 0) for s in steps], dtype=float)
        if len(d1):
            trend = float(np.mean(d1 + d2))
            out["target_relative"] = max(out["target_relative"], float(np.clip(1.0 - trend / 4.0, 0, 1)))

    return out


def compute_semantic_group_scores(
    importance_df: pd.DataFrame,
    trajectory_metrics: dict[str, Any] | None = None,
    group_df: pd.DataFrame | None = None,
    blend_trajectory: float = 0.35,
) -> dict[str, float]:
    """
    融合 Captum 组归一化 + 轨迹语义辅助。
    blend_trajectory: 轨迹辅助在最终分中的权重（expert 可提高到 0.7）。
    """
    if group_df is None:
        group_df = aggregate_importance_by_group(importance_df, allow_overlap=True)

    captum_scores: dict[str, float] = {}
    for _, r in group_df.iterrows():
        captum_scores[r["group"]] = float(r["normalized_importance"])

    metrics = trajectory_metrics or {}
    traj_scores = _trajectory_semantic_boost(metrics)

    all_groups = [
        "target_absolute",
        "uav_absolute",
        "target_relative",
        "inter_uav_coordination",
        "safety",
        "velocity",
        "action_stability",
        "task_efficiency",
    ]

    result: dict[str, float] = {}
    for g in all_groups:
        c = captum_scores.get(g, 0.0)
        t = traj_scores.get(g, 0.0)
        if g in ("action_stability", "task_efficiency"):
            result[g] = t if t > 0 else c
        else:
            result[g] = float(np.clip((1.0 - blend_trajectory) * c + blend_trajectory * t, 0.0, 1.0))

    return {
        "target_absolute_score": round(result.get("target_absolute", 0.0), 3),
        "uav_absolute_score": round(result.get("uav_absolute", 0.0), 3),
        "target_relative_score": round(result.get("target_relative", 0.0), 3),
        "inter_uav_coordination_score": round(result.get("inter_uav_coordination", 0.0), 3),
        "safety_score": round(result.get("safety", 0.0), 3),
        "velocity_score": round(result.get("velocity", 0.0), 3),
        "action_stability_score": round(result.get("action_stability", 0.0), 3),
        "task_efficiency_score": round(result.get("task_efficiency", 0.0), 3),
    }


def save_group_importance(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_group_importance(df: pd.DataFrame, save_path: str, title: str = "", footnote: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.45)))
    df = df.sort_values("normalized_importance", ascending=True)
    ax.barh(df["group"], df["normalized_importance"], color="#2166ac", edgecolor="#333")
    ax.set_xlabel("Normalized importance")
    ax.set_title(title or "Fine-grained feature groups")
    if footnote:
        fig.text(0.01, 0.01, footnote, fontsize=7, color="#555")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
