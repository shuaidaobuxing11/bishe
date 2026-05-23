#!/usr/bin/env python3
"""
双机协同：按「与 run_unified_eval 一致」的若干场景，依次串行训练纯在线 PPO 基线。
每场景保存独立 zip，并在 results/training_curves.csv 中写入不同 method 名。

用法（项目根目录）:
  python scripts/train_coop_ppo_all_scenarios.py
  python scripts/train_coop_ppo_all_scenarios.py --total_timesteps 200000 --seed 42

场景说明:
  default       — spawn_mode=default, noise_sigma=0
  near_uavs     — spawn_mode=near_uavs
  near_border   — spawn_mode=near_border
  noisy_target  — spawn_mode=default, noise_sigma=0.3（与 scripts/run_unified_eval.py 中 noisy_target 一致）
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 每项: (model_tag, spawn_mode, noise_sigma)
SCENARIOS: list[tuple[str, str, float]] = [
    ("default", "default", 0.0),
    ("near_uavs", "near_uavs", 0.0),
    ("near_border", "near_border", 0.0),
    ("noisy_target", "default", 0.3),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="多场景串行训练 PPO baseline（双机）")
    ap.add_argument("--total_timesteps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=42, help="各场景 seed 在此基础上 +1000*i")
    ap.add_argument("--save_dir", type=str, default="models")
    ap.add_argument("--curve_out_csv", type=str, default="results/training_curves.csv")
    ap.add_argument("--dry_run", action="store_true", help="只打印命令不执行")
    args = ap.parse_args()

    train_py = ROOT / "online_rl" / "train_online_baseline.py"
    for i, (tag, spawn, noise) in enumerate(SCENARIOS):
        ep_seed = int(args.seed) + i * 1000
        cmd = [
            sys.executable,
            str(train_py),
            "--total_timesteps",
            str(args.total_timesteps),
            "--seed",
            str(ep_seed),
            "--save_dir",
            args.save_dir,
            "--spawn_mode",
            spawn,
            "--noise_sigma",
            str(noise),
            "--model_tag",
            tag,
            "--curve_out_csv",
            args.curve_out_csv,
        ]
        print("\n>>>", " ".join(cmd))
        if args.dry_run:
            continue
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"场景 {tag} 训练失败，退出码 {r.returncode}")
            sys.exit(r.returncode)

    print("\n全部场景训练完成。模型位于:", Path(args.save_dir).resolve())


if __name__ == "__main__":
    main()
