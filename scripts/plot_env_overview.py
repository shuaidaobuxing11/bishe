"""
环境示意图：绘制竞技场边界 + 一条代表性轨迹（UAV1/UAV2/Target）。

优先使用 results/bc_success_episodes_10.npz 中第 0 条成功轨迹；
若不存在，则使用 data/offline/expert_episodes.npz 中第 0 条 episode（按 episode_id=0 聚合）。
输出：results/fig1_env_overview.png
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


def _extract_from_episode_obs(obs: np.ndarray, arena_size: float):
    u1 = obs[:, 0:2] * arena_size
    u2 = obs[:, 4:6] * arena_size
    tgt = obs[:, 8:10] * arena_size
    return u1, u2, tgt


def _load_bc_success(success_npz: str):
    if not os.path.isfile(success_npz):
        return None
    data = np.load(success_npz, allow_pickle=True)
    eps = data["episodes"]
    if len(eps) == 0:
        return None
    return eps[0]


def _load_expert_episode(expert_npz: str, episode_index: int = 0):
    if not os.path.isfile(expert_npz):
        return None
    data = np.load(expert_npz, allow_pickle=False)
    ep_id = data["episode_id"]
    mask = ep_id == int(episode_index)
    if not np.any(mask):
        return None
    obs = data["obs"][mask]
    return {"obs": obs, "win": bool(data["episode_wins"][episode_index])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", type=str, default="results/fig1_env_overview.png")
    p.add_argument("--bc_success_npz", type=str, default="results/bc_success_episodes_10.npz")
    p.add_argument("--expert_npz", type=str, default="data/offline/expert_episodes.npz")
    args = p.parse_args()

    cfg = load_env_config()
    arena_size = float(cfg.get("arena_size", 10.0))

    ep = _load_bc_success(args.bc_success_npz)
    source = "BC success"
    if ep is None:
        ep = _load_expert_episode(args.expert_npz, episode_index=0)
        source = "Expert episode0"
    if ep is None:
        print("未找到可用轨迹数据。请先生成 bc_success_episodes_10.npz 或 expert_episodes.npz。")
        return

    obs = ep["obs"]
    u1, u2, tgt = _extract_from_episode_obs(obs, arena_size)

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # arena boundary
    ax.plot(
        [-arena_size, arena_size, arena_size, -arena_size, -arena_size],
        [-arena_size, -arena_size, arena_size, arena_size, -arena_size],
        "k-",
        lw=1.5,
    )
    ax.plot(tgt[:, 0], tgt[:, 1], "k--", label="Target")
    ax.plot(u1[:, 0], u1[:, 1], "b-", label="UAV1")
    ax.plot(u2[:, 0], u2[:, 1], "r-", label="UAV2")
    ax.scatter(tgt[0, 0], tgt[0, 1], c="k", marker="x", s=50)
    ax.scatter(u1[0, 0], u1[0, 1], c="b", marker="o", s=30)
    ax.scatter(u2[0, 0], u2[0, 1], c="r", marker="o", s=30)
    ax.scatter(tgt[-1, 0], tgt[-1, 1], c="k", marker="*", s=80)
    ax.set_title(f"CoopTrackingEnv overview ({source})")
    ax.set_xlim(-arena_size * 1.05, arena_size * 1.05)
    ax.set_ylim(-arena_size * 1.05, arena_size * 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.save, dpi=150)
    print(f"环境示意图已保存: {args.save}")


if __name__ == "__main__":
    main()

