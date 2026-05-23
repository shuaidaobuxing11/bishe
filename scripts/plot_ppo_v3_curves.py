#!/usr/bin/env python3
"""
从 `results/v3/training_curves.csv`（由 train_ppo_v3 的 curve_eval 生成）绘制训练曲线。

用法（项目根）:
  python scripts/plot_ppo_v3_curves.py
  python scripts/plot_ppo_v3_curves.py --method ppo_v3 --csv results/v3/training_curves.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_method_rows(csv_path: Path, method: str):
    steps, succ, ret, coll = [], [], [], []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("method", "").strip() != method.strip():
                continue
            ts = int(float(row["training_steps"]))
            steps.append(ts)
            succ.append(float(row["success_rate"]))
            ret.append(float(row["mean_return"]))
            if "collision_rate" in row and row["collision_rate"] != "":
                coll.append(float(row["collision_rate"]))
            else:
                coll.append(0.0)
    idx = np.argsort(steps)
    return (
        np.asarray(steps)[idx],
        np.asarray(succ)[idx],
        np.asarray(ret)[idx],
        np.asarray(coll)[idx],
    )


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(root / "results" / "v3" / "training_curves.csv"))
    ap.add_argument("--method", type=str, default="ppo_v3")
    ap.add_argument(
        "--out",
        type=str,
        default=str(root / "results" / "v3" / "ppo_v3_training_curves.png"),
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到 CSV: {csv_path}（请先运行 train_ppo_v3 且勿加 --no-curve-logging）")

    steps, succ, ret, coll = read_method_rows(csv_path, args.method)
    if len(steps) == 0:
        raise ValueError(f"CSV 中无 method={args.method!r} 的行")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(steps, succ * 100.0, "o-", color="C0", lw=1.8)
    axes[0].set_title("Success rate vs training steps")
    axes[0].set_xlabel("Training steps")
    axes[0].set_ylabel("Success rate (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, ret, "o-", color="C1", lw=1.8)
    axes[1].set_title("Mean return vs training steps")
    axes[1].set_xlabel("Training steps")
    axes[1].set_ylabel("Mean return")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, coll * 100.0, "o-", color="C2", lw=1.8)
    axes[2].set_title("Collision rate vs training steps")
    axes[2].set_xlabel("Training steps")
    axes[2].set_ylabel("Collision rate (%)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"CoopTrackingEnvV3 · {args.method}", fontsize=12)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
