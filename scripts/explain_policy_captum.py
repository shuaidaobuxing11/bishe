#!/usr/bin/env python3
"""Captum 策略可解释性：单状态 / 整 episode / 对比图。"""
from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from explain.captum_explainer import (
    compare_policy_importance,
    compare_success_failure_importance,
    explain_episode,
    explain_single_state,
)
from visualization.trajectory_recorder import load_trajectory


def main() -> None:
    ap = argparse.ArgumentParser(description="Captum 策略解释")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--policy_name", type=str, default="policy")
    ap.add_argument("--scenario", type=str, default="default")
    ap.add_argument("--trajectory_path", type=str, default="")
    ap.add_argument("--method", type=str, default="integrated_gradients", choices=["integrated_gradients", "saliency"])
    ap.add_argument("--save_dir", type=str, default="results/explain")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--mode", type=str, default="episode", choices=["single", "episode", "success_vs_fail", "compare_policies"])
    ap.add_argument("--success_csv", type=str, default="")
    ap.add_argument("--failure_csv", type=str, default="")
    ap.add_argument("--importance_glob", type=str, default="", help="compare_policies 时 episode importance CSV 通配符")
    ap.add_argument("--policy_names", type=str, nargs="*", default=[])
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "single":
        if not args.trajectory_path:
            raise SystemExit("single 模式需要 --trajectory_path")
        data = load_trajectory(args.trajectory_path)
        obs = data["steps"][0]["obs"]
        explain_single_state(
            args.model_path,
            obs,
            method=args.method,
            save_path=args.save_dir,
            policy_name=args.policy_name,
            scenario_name=args.scenario,
            device=args.device,
        )
        print("单状态解释完成。")
        return

    if args.mode == "episode":
        if not args.trajectory_path:
            raise SystemExit("episode 模式需要 --trajectory_path")
        r = explain_episode(
            args.model_path,
            args.trajectory_path,
            method=args.method,
            save_dir=args.save_dir,
            device=args.device,
        )
        print(f"Episode 解释: {r['png_path']}")
        return

    if args.mode == "success_vs_fail":
        succ = args.success_csv or os.path.join(args.save_dir, f"{args.policy_name}_{args.scenario}_success_episode_importance.csv")
        fail = args.failure_csv or os.path.join(args.save_dir, f"{args.policy_name}_{args.scenario}_failure_episode_importance.csv")
        out = os.path.join(args.save_dir, "success_vs_failure_importance.png")
        compare_success_failure_importance(succ, fail, out)
        print(f"对比图: {out}")
        return

    if args.mode == "compare_policies":
        pattern = args.importance_glob or os.path.join(args.save_dir, f"*_{args.scenario}_episode_importance.csv")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise SystemExit(f"未找到 CSV: {pattern}")
        names = args.policy_names if args.policy_names else [os.path.basename(p).split("_")[0] for p in paths]
        out = os.path.join(args.save_dir, f"policy_importance_compare_{args.scenario}.png")
        compare_policy_importance(paths, names, out)
        print(f"策略对比图: {out}")


if __name__ == "__main__":
    main()
