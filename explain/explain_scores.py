"""细粒度可解释性评分。"""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from explain.feature_groups import compute_semantic_group_scores
from explain.scenario_explain_rules import evaluate_scenario_alignment


def compute_explain_scores(
    group_importance: pd.DataFrame,
    trajectory_metrics: dict[str, Any] | None,
    scenario_name: str,
    importance_df: pd.DataFrame | None = None,
    explanation_source: str = "captum_and_trajectory",
) -> dict[str, Any]:
    blend = 0.65 if explanation_source == "rule_and_trajectory" else 0.35
    semantic = compute_semantic_group_scores(
        importance_df if importance_df is not None else pd.DataFrame(),
        trajectory_metrics=trajectory_metrics,
        group_df=group_importance,
        blend_trajectory=blend,
    )
    align = evaluate_scenario_alignment(semantic, scenario_name)
    semantic["scenario_alignment_score"] = float(align["alignment_score"])
    semantic["explanation_source"] = explanation_source
    semantic["scenario_alignment_detail"] = align
    return semantic


def save_explain_scores(
    scores: dict[str, Any],
    policy_name: str,
    scenario_name: str,
    seed: int,
    save_dir: str = "results/explain_scores",
) -> tuple[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    flat = {k: v for k, v in scores.items() if k != "scenario_alignment_detail"}
    base = f"{policy_name}_{scenario_name}_seed{seed}_scores"
    csv_path = os.path.join(save_dir, base + ".csv")
    json_path = os.path.join(save_dir, base + ".json")
    pd.DataFrame([flat]).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(flat, f, indent=2, ensure_ascii=False)
    return csv_path, json_path
