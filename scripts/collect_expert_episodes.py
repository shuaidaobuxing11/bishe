"""
循环采样规则专家策略，保存 episode 数据并打印统计分布。

用法示例：
python scripts/collect_expert_episodes.py --n_episodes 200 --max_steps 200 --save_dir data/offline
"""
import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from offline_rl.dataset_builder import collect_episodes_with_stats


def _summary(x: np.ndarray):
    if x.size == 0:
        return "empty"
    q = np.percentile(x, [0, 25, 50, 75, 100]).tolist()
    return f"min/25/50/75/max = {q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}, {q[4]:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="data/offline")
    p.add_argument("--save_name", type=str, default="expert_episodes.npz")
    p.add_argument("--expert", type=str, default="v2", choices=("v1", "v2"), help="v1=原规则, v2=带转向+避碰")
    args = p.parse_args()

    data, stats = collect_episodes_with_stats(
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        save_name=args.save_name,
        seed=args.seed,
        expert=args.expert,
    )

    rets = data["episode_returns"]
    lens = data["episode_lengths"].astype(np.float32)
    wins = data["episode_wins"]

    print("=" * 60)
    print("规则专家策略 episode 统计 (expert={})".format(args.expert))
    print("=" * 60)
    print(f"success_rate = {stats['success_rate']:.2%}")
    print(f"return_mean ± std = {stats['return_mean']:.2f} ± {stats['return_std']:.2f}")
    print(f"len_mean ± std    = {stats['len_mean']:.2f} ± {stats['len_std']:.2f}")
    print("-" * 60)
    print(f"回报分布: { _summary(rets) }")
    print(f"长度分布: { _summary(lens) }")
    print(f"wins(0/1) 计数: {int((wins==0).sum())} / {int((wins==1).sum())}")
    print("=" * 60)


if __name__ == "__main__":
    main()

