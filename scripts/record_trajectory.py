#!/usr/bin/env python3
"""录制策略轨迹到 results/trajectories/。"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from visualization.policy_loaders import load_policy
from visualization.trajectory_recorder import make_env_for_scenario, record_episode


def main() -> None:
    ap = argparse.ArgumentParser(description="录制单策略轨迹")
    ap.add_argument("--model_path", type=str, default="", help="PPO/BC 模型路径；expert 可留空")
    ap.add_argument("--policy_name", type=str, required=True)
    ap.add_argument("--policy_type", type=str, default="auto", choices=["auto", "ppo", "bc", "expert"])
    ap.add_argument("--scenario", type=str, default="default")
    ap.add_argument("--save_dir", type=str, default="results/trajectories")
    ap.add_argument("--n_cases", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--stochastic", action="store_true", help="随机采样动作")
    args = ap.parse_args()

    if args.policy_type == "expert" or args.policy_name == "expert_v2":
        policy = load_policy(policy_name="expert_v2")
    else:
        policy = load_policy(model_path=args.model_path, policy_type=args.policy_type)
    os.makedirs(args.save_dir, exist_ok=True)

    for case_id in range(args.n_cases):
        ep_seed = args.seed + case_id * 9973
        env = make_env_for_scenario(args.scenario, seed=ep_seed, max_steps=args.max_steps)
        out = record_episode(
            env,
            policy,
            policy_name=args.policy_name,
            scenario_name=args.scenario,
            save_path=args.save_dir,
            max_steps=args.max_steps,
            deterministic=not args.stochastic,
            seed=ep_seed,
            case_id=case_id,
        )
        env.close()
        ok = out["meta"]["final_success"]
        print(f"[{case_id}] success={ok} len={out['meta']['episode_length']} -> {args.save_dir}")


if __name__ == "__main__":
    main()
