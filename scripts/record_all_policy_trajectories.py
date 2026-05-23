#!/usr/bin/env python3
"""同一 seed 下录制所有默认策略轨迹。"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from visualization.policy_loaders import load_default_policies
from visualization.trajectory_recorder import record_all_policies_same_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="同 seed 录制 expert/BC/PPO 全部轨迹")
    ap.add_argument("--scenario", type=str, default="default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="results/trajectories")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--config", type=str, default="configs/visualization_config.yaml")
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    import yaml

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    policies = load_default_policies(cfg)
    if not policies:
        print("未加载到任何策略，请检查 configs/visualization_config.yaml 与 models/ 路径。")
        return

    print(f"已加载策略: {list(policies.keys())}")
    paths = record_all_policies_same_seed(
        policies,
        scenario_name=args.scenario,
        seed=args.seed,
        save_dir=args.save_dir,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
    )
    print(f"完成，共 {len(paths)} 条轨迹 -> {args.save_dir}")


if __name__ == "__main__":
    main()
