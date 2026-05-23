#!/usr/bin/env python3
"""同一场景下多策略轨迹录制与静态对比（可含 expert / BC / PPO）。"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from visualization.compare_trajectories import plot_trajectory_comparison
from visualization.policy_loaders import load_default_policies, load_policy
from visualization.trajectory_recorder import (
    discover_trajectories_for_scenario_seed,
    record_all_policies_same_seed,
)


def _str2bool(v: str) -> bool:
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def main() -> None:
    ap = argparse.ArgumentParser(description="多策略轨迹对比")
    ap.add_argument("--scenario", type=str, default="default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="results/animations")
    ap.add_argument("--trajectory_dir", type=str, default="results/trajectories")
    ap.add_argument("--baseline_path", type=str, default="models/ppo_online_baseline.zip")
    ap.add_argument("--finetune_path", type=str, default="models/ppo_finetune_from_bc.zip")
    ap.add_argument("--kl_path", type=str, default="models/ppo_finetune_kl.zip")
    ap.add_argument("--bc_path", type=str, default="models/bc_pretrain_best.pt")
    ap.add_argument("--include_expert", type=_str2bool, default=True)
    ap.add_argument("--include_bc", type=_str2bool, default=True)
    ap.add_argument("--include_ppo", type=_str2bool, default=True)
    ap.add_argument("--skip_record", action="store_true")
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--config", type=str, default="configs/visualization_config.yaml")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    import yaml

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if args.skip_record:
        paths = discover_trajectories_for_scenario_seed(args.trajectory_dir, args.scenario, args.seed)
    else:
        policies = {}
        if args.include_expert:
            policies["expert_v2"] = load_policy(policy_name="expert_v2", config=cfg)
        if args.include_bc:
            for bc in (args.bc_path, "models/bc_pretrain_best.pt", "models/bc_pretrain.pt"):
                p = bc if os.path.isabs(bc) else os.path.join(ROOT, bc)
                if os.path.isfile(p):
                    policies["bc"] = load_policy(model_path=p, policy_type="bc")
                    break
        if args.include_ppo:
            for name, mpath in [
                ("ppo_baseline", args.baseline_path),
                ("ppo_finetune", args.finetune_path),
                ("ppo_finetune_kl", args.kl_path),
            ]:
                p = mpath if os.path.isabs(mpath) else os.path.join(ROOT, mpath)
                if os.path.isfile(p):
                    policies[name] = load_policy(model_path=p, policy_type="ppo")
        if not policies:
            policies = load_default_policies(cfg)
        paths = record_all_policies_same_seed(
            policies,
            args.scenario,
            args.seed,
            args.trajectory_dir,
            max_steps=args.max_steps,
        )

    if not paths:
        print("无轨迹可对比。")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    out_png = os.path.join(args.save_dir, f"compare_{args.scenario}_seed{args.seed}.png")
    plot_trajectory_comparison(
        paths,
        out_png,
        title=f"Policy comparison — {args.scenario} seed={args.seed}",
    )
    print(f"对比图: {out_png}")


if __name__ == "__main__":
    main()
