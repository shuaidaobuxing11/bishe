#!/usr/bin/env python3
"""SHAP 策略解释（可选）。"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description="SHAP 策略解释（可选）")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--trajectory_path", type=str, required=True)
    ap.add_argument("--policy_name", type=str, default="policy")
    ap.add_argument("--scenario", type=str, default="default")
    ap.add_argument("--save_dir", type=str, default="results/explain")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    try:
        from explain.shap_explainer import explain_with_shap
        from visualization.trajectory_recorder import load_trajectory
    except ImportError as e:
        print("SHAP explainer is optional. Please use Captum explainer first.")
        print(e)
        return

    data = load_trajectory(args.trajectory_path)
    steps = data["steps"]
    if not steps or "obs" not in steps[0]:
        print("轨迹缺少 obs，请用 scripts/record_trajectory.py 重新录制。")
        return

    import numpy as np

    obs_arr = np.array([s["obs"] for s in steps], dtype=np.float32)
    r = explain_with_shap(
        args.model_path,
        background_obs=obs_arr,
        explain_obs=obs_arr[0],
        save_dir=args.save_dir,
        policy_name=args.policy_name,
        scenario_name=args.scenario,
        device=args.device,
    )
    print(f"SHAP 完成: {r['png_path']}")


if __name__ == "__main__":
    main()
