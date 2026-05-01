"""
读取训练日志或评估结果，绘制学习曲线或对比柱状图（奖励、胜率）。
若使用 SB3 的 tensorboard 或 CSV 回调，可在此解析并绘图；此处提供简单示例框架。
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def plot_comparison(metrics_path=None, save_path="results/comparison.png"):
    """
    若存在 metrics_path（如 JSON/CSV）则读取并画图；
    否则可手动传入 baseline_return, baseline_win_rate, finetune_return, finetune_win_rate。
    """
    os.makedirs(os.path.dirname(save_path) or "results", exist_ok=True)
    # 示例：假设已有一组评估结果
    labels = ["纯在线基线", "离线+在线"]
    returns = [0.0, 0.0]  # 占位，实际从 eval 输出或文件读取
    win_rates = [0.0, 0.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(labels))
    ax1.bar(x, returns, color=["#1f77b4", "#ff7f0e"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("平均回合奖励")
    ax2.bar(x, win_rates, color=["#1f77b4", "#ff7f0e"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("胜率")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"图表已保存: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="results/comparison.png")
    args = parser.parse_args()
    plot_comparison(save_path=args.save)
