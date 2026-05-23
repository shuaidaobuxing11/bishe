"""关键帧选择与动作解释。"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from visualization.trajectory_recorder import ACTION_NAMES, decode_joint_action


def steps_to_df(steps: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(steps)


def select_keyframes(trajectory_df: pd.DataFrame) -> list[dict[str, Any]]:
    if trajectory_df.empty:
        return []
    n = len(trajectory_df)
    frames: list[dict[str, Any]] = []

    frames.append({"name": "initial", "step": int(trajectory_df.iloc[0]["step"])})

    mid_idx = n // 2
    frames.append({"name": "mid", "step": int(trajectory_df.iloc[mid_idx]["step"])})

    if "reward" in trajectory_df.columns:
        idx = int(trajectory_df["reward"].idxmin())
        frames.append({"name": "critical_min_reward", "step": int(trajectory_df.loc[idx, "step"])})

    if "distance_uav1_uav2" in trajectory_df.columns:
        idx = int(trajectory_df["distance_uav1_uav2"].idxmin())
        frames.append({"name": "critical_min_uav_distance", "step": int(trajectory_df.loc[idx, "step"])})

    if "mean_distance_to_target" in trajectory_df.columns:
        idx = int(trajectory_df["mean_distance_to_target"].idxmin())
        frames.append({"name": "critical_min_distance", "step": int(trajectory_df.loc[idx, "step"])})

    coll = trajectory_df[trajectory_df.get("collision", False) == True]  # noqa: E712
    if len(coll):
        first_coll = int(coll.index[0])
        prev = max(0, first_coll - 1)
        frames.append({"name": "pre_collision", "step": int(trajectory_df.iloc[prev]["step"])})

    succ = trajectory_df[trajectory_df.get("success", False) == True]  # noqa: E712
    if len(succ):
        idx = int(succ.index[0])
        prev = max(0, idx - 1)
        frames.append({"name": "pre_success", "step": int(trajectory_df.iloc[prev]["step"])})

    if "action" in trajectory_df.columns and n > 2:
        acts = trajectory_df["action"].astype(int).values
        jumps = np.where(np.abs(np.diff(acts)) > 0)[0]
        if len(jumps):
            j = int(jumps[len(jumps) // 2])
            frames.append({"name": "action_switch", "step": int(trajectory_df.iloc[j + 1]["step"])})

    frames.append({"name": "final", "step": int(trajectory_df.iloc[-1]["step"])})

    # 去重 step
    seen: set[int] = set()
    uniq = []
    for f in frames:
        if f["step"] not in seen:
            seen.add(f["step"])
            uniq.append(f)
    return uniq


def explain_keyframe_action(
    step_row: dict,
    attribution: np.ndarray | None,
    feature_names: list[str],
    top_k: int = 5,
) -> str:
    step = step_row.get("step", "?")
    action = int(step_row.get("action", 0))
    a1, a2 = decode_joint_action(action)
    u1n = ACTION_NAMES.get(a1, str(a1))
    u2n = ACTION_NAMES.get(a2, str(a2))

    d1 = step_row.get("distance_uav1_target", "?")
    d2 = step_row.get("distance_uav2_target", "?")
    d12 = step_row.get("distance_uav1_uav2", "?")

    text = (
        f"在第 {step} 步，UAV1 距目标 {d1:.3f}，UAV2 距目标 {d2:.3f}，双机间距 {d12:.3f}。"
        f"策略选择 action={action}（UAV1: {u1n}，UAV2: {u2n}）。"
    )

    if attribution is not None and len(attribution) == len(feature_names):
        idx = np.argsort(np.abs(attribution))[::-1][:top_k]
        tops = [f"{feature_names[i]}({attribution[i]:+.4f})" for i in idx]
        text += f" 特征归因显示 {', '.join(tops)} 对该动作贡献较高。"
    return text


def build_keyframes_markdown(
    keyframes: list[dict],
    steps: list[dict],
    attributions_by_step: dict[int, np.ndarray] | None,
    feature_names: list[str],
) -> str:
    step_map = {int(s["step"]): s for s in steps}
    lines = ["# 关键帧动作解释\n"]
    for kf in keyframes:
        st = kf["step"]
        row = step_map.get(st)
        if not row:
            continue
        attr = attributions_by_step.get(st) if attributions_by_step else None
        lines.append(f"## {kf['name']} (step={st})\n")
        lines.append(explain_keyframe_action(row, attr, feature_names) + "\n")
    return "\n".join(lines)
