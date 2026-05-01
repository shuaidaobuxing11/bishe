"""
绘制 BC 混合回放 + 保守约束（phase2）训练曲线。

输入：results/bc_mixed_metrics.npz（由 run_phased_training.py / train_bc_mixed 保存）
输出：results/fig7_bc_mixed_curves.png
"""
import os
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=str, default="results/bc_mixed_metrics.npz")
    p.add_argument("--save", type=str, default="results/fig7_bc_mixed_curves.png")
    args = p.parse_args()

    if not os.path.isfile(args.metrics):
        print(f"未找到 {args.metrics}，请先运行 phase2 训练生成。")
        return

    data = np.load(args.metrics, allow_pickle=False)
    steps = data["steps"]
    loss = data["loss"]
    acc = data["acc"]

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(steps, loss, color="#1f77b4")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("mixed BC loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("BC loss (offline/online mixed + conservative)")

    ax2.plot(steps, acc, color="#ff7f0e")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("action accuracy")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("BC action accuracy")

    plt.tight_layout()
    fig.savefig(args.save, dpi=150)
    print(f"BC mixed 曲线已保存: {args.save}")


if __name__ == "__main__":
    main()

