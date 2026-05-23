#!/usr/bin/env python3
"""读取多策略轨迹，生成同步动画、静态对比、step_table、metrics_summary。"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from visualization.compare_trajectories import compare_all_policies_outputs
from visualization.trajectory_animation import make_stepwise_animation
from visualization.trajectory_metrics import summarize_all_policy_metrics
from visualization.trajectory_recorder import discover_trajectories_for_scenario_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="多策略对比动画与汇总表")
    ap.add_argument("--scenario", type=str, default="default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trajectory_dir", type=str, default="results/trajectories")
    ap.add_argument("--save_dir", type=str, default="results/animations")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--skip_stepwise", action="store_true")
    args = ap.parse_args()

    paths = discover_trajectories_for_scenario_seed(args.trajectory_dir, args.scenario, args.seed)
    if not paths:
        print(
            f"未找到轨迹 *_{args.scenario}_seed{args.seed}.json，"
            f"请先运行 scripts/record_all_policy_trajectories.py"
        )
        return

    print(f"发现 {len(paths)} 条轨迹: {list(paths.keys())}")
    os.makedirs(args.save_dir, exist_ok=True)

    if not args.skip_stepwise:
        for name, tpath in paths.items():
            out = os.path.join(
                args.save_dir,
                f"{name}_{args.scenario}_seed{args.seed}_stepwise.gif",
            )
            try:
                saved = make_stepwise_animation(tpath, out, fps=args.fps)
                print(f"stepwise: {saved}")
            except Exception as exc:
                print(f"stepwise 跳过 {name}: {exc}")

    outs = compare_all_policies_outputs(
        paths, args.scenario, args.seed, args.save_dir, fps=args.fps
    )
    for k, v in outs.items():
        print(f"{k}: {v}")

    metrics_path = os.path.join(args.save_dir, f"metrics_summary_{args.scenario}_seed{args.seed}.csv")
    summarize_all_policy_metrics(paths, metrics_path)
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
