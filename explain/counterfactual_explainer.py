"""简单反事实扰动测试。"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pandas as pd

from explain.feature_names import get_feature_names
from visualization.trajectory_recorder import ACTION_NAMES, decode_joint_action


def _action_label(action: int) -> str:
    a1, a2 = decode_joint_action(action)
    return f"{ACTION_NAMES.get(a1,a1)}/{ACTION_NAMES.get(a2,a2)}"


def run_counterfactual_tests(
    predict_fn: Callable[[np.ndarray], int],
    obs: np.ndarray,
    feature_names: list[str] | None = None,
    perturb_config: dict[str, float] | None = None,
) -> pd.DataFrame:
    feature_names = feature_names or get_feature_names()
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    original = int(predict_fn(obs.copy()))

    default_delta = {
        "target_x_norm": 0.1,
        "target_y_norm": 0.1,
        "uav1_x_norm": 0.1,
        "uav2_x_norm": 0.1,
        "uav1_vx_norm": 0.1,
        "uav2_vx_norm": 0.1,
    }
    deltas = {**default_delta, **(perturb_config or {})}

    rows: list[dict[str, Any]] = []
    for feat, delta in deltas.items():
        if feat not in feature_names:
            continue
        idx = feature_names.index(feat)
        perturbed = obs.copy()
        perturbed[idx] += float(delta)
        new_a = int(predict_fn(perturbed))
        changed = new_a != original
        expl = (
            f"当 {feat} 增加 {delta} 后，动作由 {_action_label(original)} 变为 {_action_label(new_a)}，"
            + ("说明策略对该特征较敏感。" if changed else "动作未变，说明对该扰动不敏感。")
        )
        rows.append(
            {
                "feature_perturbed": feat,
                "delta": delta,
                "original_action": original,
                "new_action": new_a,
                "changed": changed,
                "explanation": expl,
            }
        )
    return pd.DataFrame(rows)


def save_counterfactual(df: pd.DataFrame, csv_path: str, md_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    lines = ["# 反事实解释\n"]
    for _, r in df.iterrows():
        lines.append(f"- **{r['feature_perturbed']}** (+{r['delta']}): {r['explanation']}\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
