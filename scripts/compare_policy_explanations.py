#!/usr/bin/env python3
"""对比多策略解释报告。"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from explain.compare_explain_reports import compare_policy_explanations
from visualization.trajectory_recorder import discover_trajectories_for_scenario_seed


def main() -> None:
    ap = argparse.ArgumentParser(description="策略解释对比")
    ap.add_argument("--scenario", type=str, default="default")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report_dir", type=str, default="results/explain_reports")
    ap.add_argument("--save_dir", type=str, default="results/explain_comparisons")
    args = ap.parse_args()

    traj_paths = discover_trajectories_for_scenario_seed(
        os.path.join(ROOT, "results/trajectories"), args.scenario, args.seed
    )
    report_map = {}
    for pname in traj_paths:
        d = os.path.join(args.report_dir, f"{pname}_{args.scenario}_seed{args.seed}")
        if os.path.isdir(d):
            report_map[pname] = d

    if len(report_map) < 2:
        print("需要至少 2 份报告目录，请先 batch_generate_explain_reports.py")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    out = os.path.join(args.save_dir, f"policy_compare_{args.scenario}_seed{args.seed}.md")
    paths = compare_policy_explanations(report_map, out, args.scenario, args.seed)
    print(f"对比报告 MD:  {paths.get('md')}")
    print(f"对比表格 CSV: {paths.get('csv')}")
    if paths.get("groups_png"):
        print(f"特征组热力图: {paths['groups_png']}")
    if paths.get("fine_grained_scores_png"):
        print(f"细粒度柱状图: {paths['fine_grained_scores_png']}")
    if paths.get("heatmap_png"):
        print(f"评分热力图:   {paths['heatmap_png']}")
    if paths.get("scores_png"):
        print(f"评分柱状图:   {paths['scores_png']}")
    if paths.get("perf_joint_png"):
        print(f"性能-解释图:  {paths['perf_joint_png']}")


if __name__ == "__main__":
    main()
