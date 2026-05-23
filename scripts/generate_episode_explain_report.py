#!/usr/bin/env python3
"""单 episode 自动解释报告。"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from explain.auto_explain_pipeline import run_auto_explain_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="生成单策略 episode 解释报告")
    ap.add_argument("--policy_name", type=str, required=True)
    ap.add_argument("--scenario", type=str, default="default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trajectory_path", type=str, required=True)
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--save_dir", type=str, default="results/explain_reports")
    ap.add_argument("--config", type=str, default="configs/explain_report_config.yaml")
    args = ap.parse_args()

    import yaml

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    cfg = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    model_path = args.model_path or None
    if not model_path and args.policy_name in cfg.get("policies", {}):
        mp = cfg["policies"][args.policy_name].get("model_path")
        if mp:
            model_path = mp if os.path.isabs(mp) else os.path.join(ROOT, mp)

    bundle = run_auto_explain_pipeline(
        args.policy_name,
        args.scenario,
        args.seed,
        args.trajectory_path,
        model_path=model_path,
        save_dir=args.save_dir,
        config=cfg,
    )
    print(f"报告已生成: {bundle['report_path']}")


if __name__ == "__main__":
    main()
