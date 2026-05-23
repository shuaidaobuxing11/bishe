#!/usr/bin/env python3
"""同策略成功 vs 失败轨迹解释对比。"""
from __future__ import annotations

import argparse
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from explain.auto_explain_pipeline import run_auto_explain_pipeline
from explain.compare_explain_reports import compare_success_failure_explanations
from visualization.trajectory_recorder import load_trajectory


def _find_success_failure(traj_dir: str, policy: str, scenario: str) -> tuple[str | None, str | None]:
    pattern = os.path.join(traj_dir, f"{policy}_{scenario}_seed*.json")
    succ, fail = None, None
    for p in glob.glob(pattern):
        data = load_trajectory(p)
        if data.get("meta", {}).get("final_success"):
            succ = p
        else:
            fail = p
    return succ, fail


def main() -> None:
    ap = argparse.ArgumentParser(description="成功/失败解释对比")
    ap.add_argument("--policy_name", type=str, required=True)
    ap.add_argument("--scenario", type=str, default="noisy_target")
    ap.add_argument("--report_dir", type=str, default="results/explain_reports")
    ap.add_argument("--trajectory_dir", type=str, default="results/trajectories")
    ap.add_argument("--save_dir", type=str, default="results/explain_comparisons")
    ap.add_argument("--model_path", type=str, default="")
    args = ap.parse_args()

    succ_path, fail_path = _find_success_failure(args.trajectory_dir, args.policy_name, args.scenario)
    if not succ_path or not fail_path:
        print(f"需要同策略成功与失败轨迹各一条（当前 succ={succ_path}, fail={fail_path}）")
        print("提示：可对同场景用不同 seed 录制，或扩展 trajectory 采集。")
        return

    def _seed_from_path(p: str) -> int:
        base = os.path.basename(p).replace(".json", "")
        return int(base.rsplit("_seed", 1)[-1])

    bs = run_auto_explain_pipeline(
        args.policy_name, args.scenario, _seed_from_path(succ_path), succ_path,
        model_path=args.model_path or None, save_dir=args.report_dir,
    )
    bf = run_auto_explain_pipeline(
        args.policy_name, args.scenario, _seed_from_path(fail_path), fail_path,
        model_path=args.model_path or None, save_dir=args.report_dir,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, f"{args.policy_name}_{args.scenario}_success_vs_failure.md")
    compare_success_failure_explanations(bs, bf, out)
    print(f"对比报告: {out}")


if __name__ == "__main__":
    main()
