import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_training_curves(csv_path: str):
    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            method = row.get("method", "").strip()
            if not method:
                continue
            ts = int(float(row.get("training_steps", "0")))
            succ = float(row.get("success_rate", "0"))
            ret = float(row.get("mean_return", "0"))
            data.setdefault(method, []).append((ts, succ, ret))
    # sort by steps
    for method in list(data.keys()):
        data[method].sort(key=lambda x: x[0])
    return data


def main():
    ap = argparse.ArgumentParser(description="训练曲线对比：success_rate 与 mean_return")
    ap.add_argument("--csv", type=str, default="results/training_curves.csv", help="训练曲线 CSV（由 training_curve_callback 写入）")
    ap.add_argument("--out", type=str, default="results/fig8_training_curves.png", help="输出图片路径")
    ap.add_argument("--baseline_method", type=str, default="ppo_baseline")
    ap.add_argument("--finetune_method", type=str, default="ppo_finetune")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"未找到训练曲线 CSV：{args.csv}")

    data = _read_training_curves(args.csv)
    if args.baseline_method not in data:
        raise ValueError(f"CSV 中未找到 baseline 方法：{args.baseline_method}")
    if args.finetune_method not in data:
        raise ValueError(f"CSV 中未找到 finetune 方法：{args.finetune_method}")

    base = np.asarray(data[args.baseline_method], dtype=np.float64)  # [ts, succ, ret]
    fin = np.asarray(data[args.finetune_method], dtype=np.float64)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(base[:, 0], base[:, 1] * 100.0, marker="o", label="PPO baseline")
    ax1.plot(fin[:, 0], fin[:, 1] * 100.0, marker="o", label="PPO finetune")
    ax1.set_title("Success Rate vs Training Steps")
    ax1.set_xlabel("Training steps")
    ax1.set_ylabel("Success rate (%)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2.plot(base[:, 0], base[:, 2], marker="o", label="PPO baseline")
    ax2.plot(fin[:, 0], fin[:, 2], marker="o", label="PPO finetune")
    ax2.set_title("Return vs Training Steps")
    ax2.set_xlabel("Training steps")
    ax2.set_ylabel("Mean return")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    os.makedirs(os.path.dirname(args.out) or "results", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"图表已保存: {args.out}")


if __name__ == "__main__":
    main()

