#!/usr/bin/env python3
"""批量为所有策略生成解释报告。"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml

from explain.auto_explain_pipeline import run_auto_explain_pipeline
from visualization.trajectory_recorder import discover_trajectories_for_scenario_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="批量生成解释报告")
    ap.add_argument("--scenario", type=str, default="default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trajectory_dir", type=str, default="results/trajectories")
    ap.add_argument("--save_dir", type=str, default="results/explain_reports")
    ap.add_argument("--config", type=str, default="configs/explain_report_config.yaml")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    paths = discover_trajectories_for_scenario_seed(args.trajectory_dir, args.scenario, args.seed)
    if not paths:
        print(f"未找到轨迹 *_{args.scenario}_seed{args.seed}.json")
        return

    policies_cfg = cfg.get("policies", {})
    for policy_name, traj_path in paths.items():
        pc = policies_cfg.get(policy_name, {})
        if pc.get("enabled") is False:
            continue
        model_path = pc.get("model_path")
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(ROOT, model_path)
        if policy_name == "expert_v2":
            model_path = None
        try:
            bundle = run_auto_explain_pipeline(
                policy_name,
                args.scenario,
                args.seed,
                traj_path,
                model_path=model_path,
                save_dir=args.save_dir,
                config=cfg,
            )
            print(f"OK {policy_name} -> {bundle['report_path']}")
        except Exception as exc:
            print(f"FAIL {policy_name}: {exc}")


if __name__ == "__main__":
    main()
