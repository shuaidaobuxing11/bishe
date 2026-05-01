"""
从提取好的 BC 成功/失败 episode 中画出 2D 轨迹图，用于论文/中期报告展示。

前置步骤（已完成）：
1) 评估并保存全部 BC episode：
   python scripts/eval_bc_policy.py --bc_path models/bc_pretrain_best.pt --n_episodes 500 --seed 2024 --save_episodes results/bc_eval_episodes.npz
2) 抽取 10 条成功 + 10 条失败：
   python scripts/extract_bc_episodes.py --eval_npz results/bc_eval_episodes.npz --out_dir results

当前脚本用法示例：
   python scripts/plot_bc_trajectories.py --success_npz results/bc_success_episodes_10.npz --failure_npz results/bc_failure_episodes_10.npz --save results/bc_trajs.png
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

from envs import load_env_config


def _extract_xy_from_episode(ep, arena_size: float):
    """
    ep: 单个 episode 字典，包含 obs (T,10)
    返回：u1_xy, u2_xy, tgt_xy，均为 (T,2) 的真实坐标（未归一化）。
    """
    obs = ep["obs"]  # (T,10), 归一化过
    # 还原到实际坐标
    u1 = obs[:, 0:2] * arena_size
    u2 = obs[:, 4:6] * arena_size
    tgt = obs[:, 8:10] * arena_size
    return u1, u2, tgt


def plot_group(episodes, arena_size: float, title: str, max_plots: int = 4):
    """
    episodes: list/array of episode dicts
    """
    n = min(max_plots, len(episodes))
    if n == 0:
        return None
    cols = n
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4), squeeze=False)
    axes = axes[0]
    for i in range(n):
        ep = episodes[i]
        u1, u2, tgt = _extract_xy_from_episode(ep, arena_size)
        ax = axes[i]
        ax.plot(tgt[:, 0], tgt[:, 1], "k--", label="Target")
        ax.plot(u1[:, 0], u1[:, 1], "b-", label="UAV1")
        ax.plot(u2[:, 0], u2[:, 1], "r-", label="UAV2")
        # 标记起点/终点
        ax.scatter(tgt[0, 0], tgt[0, 1], c="k", marker="x", s=40)
        ax.scatter(u1[0, 0], u1[0, 1], c="b", marker="o", s=20)
        ax.scatter(u2[0, 0], u2[0, 1], c="r", marker="o", s=20)
        ax.scatter(tgt[-1, 0], tgt[-1, 1], c="k", marker="*", s=60)
        ax.set_title(f"{title} #{i}, win={ep.get('win', False)}")
        ax.set_xlim(-arena_size, arena_size)
        ax.set_ylim(-arena_size, arena_size)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--success_npz", type=str, default="results/bc_success_episodes_10.npz")
    p.add_argument("--failure_npz", type=str, default="results/bc_failure_episodes_10.npz")
    p.add_argument("--save", type=str, default="results/bc_trajs.png")
    p.add_argument("--max_plots", type=int, default=3, help="每组最多画多少条轨迹")
    args = p.parse_args()

    cfg = load_env_config()
    arena_size = float(cfg.get("arena_size", 10.0))

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    # 载入成功/失败 episode
    succ_data = np.load(args.success_npz, allow_pickle=True)
    fail_data = np.load(args.failure_npz, allow_pickle=True)
    succ_eps = succ_data["episodes"]
    fail_eps = fail_data["episodes"]

    # 画成功组
    fig1 = plot_group(succ_eps, arena_size, "BC Success", max_plots=args.max_plots)
    # 画失败组
    fig2 = plot_group(fail_eps, arena_size, "BC Failure", max_plots=args.max_plots)

    # 拼成一张总图：上行成功，下行失败
    if fig1 is None and fig2 is None:
        print("无可用 episode 进行绘制。")
        return

    # 将两个 figure 的内容合并到一个 figure 里（简单做法：重新绘制）
    cols = args.max_plots
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 8), squeeze=False)

    def _draw_to_axes(eps, row, title_prefix):
        n = min(cols, len(eps))
        for i in range(n):
            ep = eps[i]
            u1, u2, tgt = _extract_xy_from_episode(ep, arena_size)
            ax = axes[row][i]
            ax.plot(tgt[:, 0], tgt[:, 1], "k--", label="Target")
            ax.plot(u1[:, 0], u1[:, 1], "b-", label="UAV1")
            ax.plot(u2[:, 0], u2[:, 1], "r-", label="UAV2")
            ax.scatter(tgt[0, 0], tgt[0, 1], c="k", marker="x", s=40)
            ax.scatter(u1[0, 0], u1[0, 1], c="b", marker="o", s=20)
            ax.scatter(u2[0, 0], u2[0, 1], c="r", marker="o", s=20)
            ax.scatter(tgt[-1, 0], tgt[-1, 1], c="k", marker="*", s=60)
            ax.set_title(f"{title_prefix} #{i}, win={ep.get('win', False)}")
            ax.set_xlim(-arena_size, arena_size)
            ax.set_ylim(-arena_size, arena_size)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()

    _draw_to_axes(succ_eps, 0, "Success")
    _draw_to_axes(fail_eps, 1, "Failure")

    fig.suptitle("BC Policy Trajectories (Success vs Failure)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(args.save, dpi=150)
    print(f"BC 轨迹示意图已保存到 {args.save}")


if __name__ == "__main__":
    main()

