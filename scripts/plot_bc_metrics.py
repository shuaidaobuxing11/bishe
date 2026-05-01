"""
绘制 BC 训练曲线：loss 与 action accuracy（train/val）。
输入：results/bc_metrics.npz（由 scripts/generate_offline_data.py 训练时生成）
输出：results/bc_curves.png
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=str, default="results/bc_metrics.npz")
    p.add_argument("--save", type=str, default="results/bc_curves.png")
    args = p.parse_args()

    data = np.load(args.metrics, allow_pickle=False)
    train_loss = data["train_loss"]
    train_acc = data["train_acc"]
    val_loss = data["val_loss"]
    val_acc = data["val_acc"]

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    x = np.arange(1, len(train_loss) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(x, train_loss, label="train_loss")
    if np.isfinite(val_loss).any():
        ax1.plot(x, val_loss, label="val_loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(x, train_acc, label="train_acc")
    if np.isfinite(val_acc).any():
        ax2.plot(x, val_acc, label="val_acc")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("action accuracy")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.save, dpi=150)
    print(f"BC 曲线已保存: {args.save}")


if __name__ == "__main__":
    main()

