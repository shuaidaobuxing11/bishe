#!/usr/bin/env python3
"""
行为克隆训练曲线：从 train_bc 保存的 npz 绘制
  train_loss, val_loss, train_acc, val_acc（按 epoch）。

默认输入: results/bc_metrics.npz（与 configs 中 behavior_cloning.metrics_path 一致）

用法（项目根）:
  python scripts/plot_bc_training_curves.py
  python scripts/plot_bc_training_curves.py --metrics results/bc_metrics.npz --out results/bc_training_curves.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _flat_float(arr: np.ndarray) -> np.ndarray:
    """处理 savez 读出的一维 float 数组。"""
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description="绘制 BC train/val loss 与 accuracy 曲线")
    ap.add_argument(
        "--metrics",
        type=str,
        default=str(ROOT / "results" / "bc_metrics.npz"),
        help="train_bc 保存的 metrics npz",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "bc_training_curves.png"),
        help="输出 PNG 路径",
    )
    args = ap.parse_args()

    path = Path(args.metrics)
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到 {path}。\n请先训练 BC：`python scripts/train_bc.py` 或 "
            "`python scripts/generate_offline_data.py`（会写 metrics_path）。"
        )

    data = np.load(path, allow_pickle=False)
    for key in ("train_loss", "val_loss", "train_acc", "val_acc"):
        if key not in data.files:
            raise KeyError(f"npz 中缺少键 `{key}`，当前包含: {data.files}")

    train_loss = _flat_float(data["train_loss"])
    val_loss = _flat_float(data["val_loss"])
    train_acc = _flat_float(data["train_acc"])
    val_acc = _flat_float(data["val_acc"])

    n = min(len(train_loss), len(val_loss), len(train_acc), len(val_acc))
    train_loss = train_loss[:n]
    val_loss = val_loss[:n]
    train_acc = train_acc[:n]
    val_acc = val_acc[:n]
    epochs = np.arange(1, n + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)

    ax_loss.plot(epochs, train_loss, "o-", color="C0", lw=2.0, ms=4, label="train_loss")
    ax_loss.plot(epochs, val_loss, "s-", color="C1", lw=2.0, ms=4, label="val_loss")
    ax_loss.set_ylabel("Loss (CE)")
    ax_loss.set_title("行为克隆 · 交叉熵损失")
    ax_loss.legend(loc="upper right", fontsize=10)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, train_acc * 100.0, "o-", color="C2", lw=2.0, ms=4, label="train_acc")
    ax_acc.plot(epochs, val_acc * 100.0, "s-", color="C3", lw=2.0, ms=4, label="val_acc")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title("行为克隆 · 动作分类准确率（argmax vs 标签）")
    ax_acc.legend(loc="lower right", fontsize=10)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim(bottom=0, top=min(103, max(ax_acc.get_ylim()[1], 100)))

    split_note = ""
    if "split_episode_level" in data.files:
        lev = int(np.asarray(data["split_episode_level"]).reshape(-1)[0])
        split_note = " · 按整条轨迹划分" if lev else " · 按 transition 随机划分"

    fig.suptitle("Behavior Cloning 训练曲线" + split_note, fontsize=13, y=1.02)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"已保存: {out_path}")


if __name__ == "__main__":
    main()
