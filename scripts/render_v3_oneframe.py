#!/usr/bin/env python3
"""
一次性保存三机环境轨迹图（无窗口），用于 README / 答辩插图。
示例:
  python scripts/render_v3_oneframe.py --out results/v3/v3_trajectory_demo.png --steps 150
  python scripts/render_v3_oneframe.py --model models/ppo_v3/ppo_coop_v3.zip --out results/v3/v3_trajectory_ppo.png
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from envs.coop_tracking_env_v3 import CoopTrackingEnvV3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default=os.path.join(ROOT, "results", "v3", "v3_trajectory_demo.png"))
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--model",
        type=str,
        default="",
        help="可选：SB3 PPO .zip；确定性 rollout；留空则随机动作",
    )
    args = p.parse_args()

    env = CoopTrackingEnvV3(render_mode=None, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)

    model = None
    model_path = args.model.strip()
    if model_path:
        from pathlib import Path

        if Path(model_path).is_file():
            from stable_baselines3 import PPO

            model = PPO.load(model_path)
        else:
            print(f"[警告] 未找到模型，改用随机动作: {model_path}")

    for _ in range(args.steps):
        if model is not None:
            a, _ = model.predict(obs, deterministic=True)
            a = np.asarray(a, dtype=np.int64).reshape(-1)
            obs, _, term, trunc, _ = env.step(a)
        else:
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            break

    title = f"三机协同追踪 (V3) · 步数={env._step_count}"
    title += " · PPO deterministic" if model is not None else " · 随机策略"

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    lim = float(env.arena_size) * 1.08
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)

    colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    for i, c in enumerate(colors):
        tr = np.array(env._trail_uavs[i])
        if len(tr) > 1:
            ax.plot(tr[:, 0], tr[:, 1], "-", color=c, lw=1.8, label=f"UAV{i+1}")

    tg = np.array(env._trail_tgt)
    if len(tg) > 1:
        ax.plot(tg[:, 0], tg[:, 1], "-", color="#d62728", lw=1.6, label="目标")

    for i, c in enumerate(colors):
        if len(env._trail_uavs[i]):
            xy = env._trail_uavs[i][-1]
            ax.scatter([xy[0]], [xy[1]], c=c, s=55, zorder=6, edgecolors="k", linewidths=0.4)
    if len(env._trail_tgt):
        txy = env._trail_tgt[-1]
        ax.scatter([txy[0]], [txy[1]], c="#d62728", s=70, zorder=6, marker="*", edgecolors="k")

    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"已保存: {args.out}")

    env.close()


if __name__ == "__main__":
    main()
