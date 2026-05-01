"""
从 BC 评估结果中提取 10 条成功轨迹与 10 条失败轨迹，单独保存。

输入：eval_bc_policy.py 使用 --save_episodes 保存的 npz 文件，例如：
  python scripts/eval_bc_policy.py --bc_path models/bc_pretrain.pt --n_episodes 500 --save_episodes results/bc_eval_episodes.npz

输出：
  results/bc_success_episodes_10.npz
  results/bc_failure_episodes_10.npz
"""
import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_npz", type=str, default="results/bc_eval_episodes.npz")
    p.add_argument("--n_success", type=int, default=10)
    p.add_argument("--n_failure", type=int, default=10)
    p.add_argument("--out_dir", type=str, default="results")
    args = p.parse_args()

    data = np.load(args.eval_npz, allow_pickle=True)
    episodes = data["episodes"]
    win = data["episode_win"]

    success_idx = np.where(win == 1)[0]
    failure_idx = np.where(win == 0)[0]

    if len(success_idx) == 0 or len(failure_idx) == 0:
        print("评估结果中成功或失败 episode 数量不足，无法各取 10 条。")
    os.makedirs(args.out_dir or ".", exist_ok=True)

    n_s = min(args.n_success, len(success_idx))
    n_f = min(args.n_failure, len(failure_idx))
    rng = np.random.default_rng(42)
    sel_s = rng.choice(success_idx, size=n_s, replace=False) if n_s > 0 else np.array([], dtype=int)
    sel_f = rng.choice(failure_idx, size=n_f, replace=False) if n_f > 0 else np.array([], dtype=int)

    success_eps = episodes[sel_s] if n_s > 0 else []
    failure_eps = episodes[sel_f] if n_f > 0 else []

    succ_path = os.path.join(args.out_dir, "bc_success_episodes_10.npz")
    fail_path = os.path.join(args.out_dir, "bc_failure_episodes_10.npz")
    np.savez_compressed(succ_path, episodes=success_eps)
    np.savez_compressed(fail_path, episodes=failure_eps)
    print(f"成功轨迹 {n_s} 条已保存到 {succ_path}")
    print(f"失败轨迹 {n_f} 条已保存到 {fail_path}")


if __name__ == "__main__":
    main()

