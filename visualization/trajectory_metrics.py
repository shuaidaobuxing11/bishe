"""轨迹逐步指标汇总。"""
from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np

from visualization.trajectory_recorder import load_trajectory, trajectory_basename


def compute_trajectory_metrics(trajectory_path: str) -> dict[str, Any]:
    data = load_trajectory(trajectory_path)
    meta = data.get("meta", {})
    steps = data.get("steps", [])
    if not steps:
        return {
            "policy_name": meta.get("policy_name", ""),
            "scenario_name": meta.get("scenario_name", ""),
            "seed": meta.get("seed"),
            "success": False,
            "collision": False,
            "episode_return": 0.0,
            "episode_length": 0,
        }

    d1 = np.array([s.get("distance_uav1_target", np.nan) for s in steps], dtype=float)
    d2 = np.array([s.get("distance_uav2_target", np.nan) for s in steps], dtype=float)
    d12 = np.array([s.get("distance_uav1_uav2", np.nan) for s in steps], dtype=float)
    mean_d = (d1 + d2) / 2.0

    actions = [int(s.get("action", 0)) for s in steps]
    switch_count = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i - 1])

    return {
        "policy_name": meta.get("policy_name", ""),
        "scenario_name": meta.get("scenario_name", ""),
        "seed": meta.get("seed"),
        "success": bool(meta.get("final_success", steps[-1].get("success", False))),
        "collision": bool(meta.get("had_collision", any(s.get("collision") for s in steps))),
        "episode_return": float(meta.get("total_reward", sum(s.get("reward", 0) for s in steps))),
        "episode_length": int(meta.get("episode_length", len(steps))),
        "final_mean_distance": float(mean_d[-1]) if len(mean_d) else float("nan"),
        "min_mean_distance": float(np.nanmin(mean_d)) if len(mean_d) else float("nan"),
        "mean_distance_to_target": float(np.nanmean(mean_d)) if len(mean_d) else float("nan"),
        "min_uav_distance": float(np.nanmin(d12)) if len(d12) else float("nan"),
        "action_switch_count": int(switch_count),
        "action_switch_rate": float(switch_count / max(len(steps) - 1, 1)),
    }


def summarize_all_policy_metrics(
    trajectory_paths: dict[str, str] | list[str],
    save_path: str,
) -> list[dict[str, Any]]:
    if isinstance(trajectory_paths, dict):
        items = list(trajectory_paths.items())
    else:
        items = [(os.path.basename(p).split("_")[0], p) for p in trajectory_paths]

    rows = []
    for _name, path in items:
        if not os.path.isfile(path):
            continue
        rows.append(compute_trajectory_metrics(path))

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    if rows:
        keys = list(rows[0].keys())
        with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return rows
