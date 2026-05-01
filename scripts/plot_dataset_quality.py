"""
离线数据质量展示图（专家数据集）：胜率、回报分布、长度分布。

输入：data/offline/expert_episodes.npz（由 scripts/collect_expert_episodes.py 生成）
输出：results/fig2_dataset_quality.png
"""
import os
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--expert_npz", type=str, default="data/offline/expert_episodes.npz")
    p.add_argument("--save", type=str, default="results/fig2_dataset_quality.png")
    args = p.parse_args()

    if not os.path.isfile(args.expert_npz):
        print(f"未找到 {args.expert_npz}，请先运行 scripts/collect_expert_episodes.py 生成。")
        return

    data = np.load(args.expert_npz, allow_pickle=False)
    returns = data["episode_returns"].astype(np.float32)
    lengths = data["episode_lengths"].astype(np.float32)
    wins = data["episode_wins"].astype(np.int32)

    win_rate = float(np.mean(wins)) if len(wins) else 0.0

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    # win rate bar
    axes[0].bar([0], [win_rate], color="#2ca02c")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks([0])
    axes[0].set_xticklabels(["Expert v2"])
    axes[0].set_ylabel("Success rate")
    axes[0].set_title(f"Success rate = {win_rate*100:.1f}%")
    axes[0].grid(True, axis="y", alpha=0.3)

    # return hist
    axes[1].hist(returns, bins=30, color="#1f77b4", alpha=0.85)
    axes[1].set_title(f"Episode return (mean={returns.mean():.1f}, std={returns.std():.1f})")
    axes[1].set_xlabel("Return")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.3)

    # length hist
    axes[2].hist(lengths, bins=30, color="#ff7f0e", alpha=0.85)
    axes[2].set_title(f"Episode length (mean={lengths.mean():.1f}, std={lengths.std():.1f})")
    axes[2].set_xlabel("Length")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Offline dataset quality (expert demonstrations)", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(args.save, dpi=150)
    print(f"离线数据质量图已保存: {args.save}")


if __name__ == "__main__":
    main()

