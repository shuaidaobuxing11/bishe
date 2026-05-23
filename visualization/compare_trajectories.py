"""多策略静态/同步动画轨迹对比与逐步表导出。"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from visualization.trajectory_animation import _save_anim
from visualization.trajectory_recorder import load_trajectory, trajectory_basename

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

POLICY_STYLE = {
    "expert_v2": {"color": "#31a354", "ls": "-"},
    "bc": {"color": "#fdb462", "ls": "-."},
    "ppo_baseline": {"color": "#6baed6", "ls": "-"},
    "ppo_finetune": {"color": "#08519c", "ls": "--"},
    "ppo_finetune_kl": {"color": "#756bb1", "ls": ":"},
}

DEFAULT_STYLE_CYCLE = [
    {"color": "#1f77b4", "ls": "-", "lw": 2.0},
    {"color": "#ff7f0e", "ls": "-.", "lw": 2.0},
    {"color": "#2ca02c", "ls": "--", "lw": 2.0},
    {"color": "#9467bd", "ls": ":", "lw": 2.0},
    {"color": "#d62728", "ls": "-", "lw": 1.8},
]


def _style_for(name: str, idx: int) -> dict:
    if name in POLICY_STYLE:
        s = POLICY_STYLE[name]
        return {"color": s["color"], "ls": s["ls"], "lw": 2.0}
    return DEFAULT_STYLE_CYCLE[idx % len(DEFAULT_STYLE_CYCLE)]


def plot_trajectory_comparison(
    trajectory_paths: list[str] | dict[str, str],
    save_path: str,
    title: str = "",
    policy_labels: list[str] | None = None,
) -> None:
    if isinstance(trajectory_paths, dict):
        paths = list(trajectory_paths.values())
        policy_labels = policy_labels or list(trajectory_paths.keys())
    else:
        paths = trajectory_paths

    fig, ax = plt.subplots(figsize=(9, 9))
    arena = 10.0

    for idx, path in enumerate(paths):
        data = load_trajectory(path)
        meta = data.get("meta", {})
        steps = data.get("steps", [])
        if not steps:
            continue
        arena = float(meta.get("arena_size", arena))
        name = (
            policy_labels[idx]
            if policy_labels and idx < len(policy_labels)
            else meta.get("policy_name", f"policy_{idx}")
        )
        success = bool(meta.get("final_success", steps[-1].get("success", False)))
        style = _style_for(name, idx)
        ls = style["ls"] if success else ":"
        alpha = 0.95 if success else 0.6

        u1x = np.array([s["uav1_x"] for s in steps])
        u1y = np.array([s["uav1_y"] for s in steps])
        u2x = np.array([s["uav2_x"] for s in steps])
        u2y = np.array([s["uav2_y"] for s in steps])
        tx = np.array([s["target_x"] for s in steps])
        ty = np.array([s["target_y"] for s in steps])

        tag = f"{name} ({'OK' if success else 'FAIL'})"
        ax.plot(u1x, u1y, color=style["color"], ls=ls, lw=style["lw"], alpha=alpha, label=f"{tag} UAV1")
        ax.plot(u2x, u2y, color=style["color"], ls=ls, lw=style["lw"] * 0.85, alpha=alpha * 0.9)
        if idx == 0:
            ax.plot(tx, ty, color="#d62728", ls="--", lw=1.2, alpha=0.7, label="Target")
        ax.scatter(u1x[0], u1y[0], c=style["color"], s=40, marker="o", zorder=4)
        ax.scatter(u2x[-1], u2y[-1], c=style["color"], s=70, marker="*", zorder=5)

    lim = arena * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "多策略轨迹对比")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    if save_path.lower().endswith(".png"):
        fig.savefig(save_path[:-4] + ".pdf")
    elif not save_path.lower().endswith(".pdf"):
        fig.savefig(save_path + ".pdf")
    plt.close(fig)


def export_step_table(trajectory_paths: dict[str, str], save_path: str) -> None:
    """合并多策略逐步数据为长表 CSV。"""
    fields = [
        "policy_name",
        "step",
        "action",
        "uav1_action_name",
        "uav2_action_name",
        "reward",
        "cum_reward",
        "distance_uav1_target",
        "distance_uav2_target",
        "distance_uav1_uav2",
        "success",
        "collision",
        "done",
    ]
    rows: list[dict] = []
    for policy_name, path in trajectory_paths.items():
        data = load_trajectory(path)
        pname = data.get("meta", {}).get("policy_name", policy_name)
        for s in data.get("steps", []):
            rows.append(
                {
                    "policy_name": pname,
                    "step": s.get("step"),
                    "action": s.get("action"),
                    "uav1_action_name": s.get("uav1_action_name"),
                    "uav2_action_name": s.get("uav2_action_name"),
                    "reward": s.get("reward"),
                    "cum_reward": s.get("cum_reward", s.get("reward")),
                    "distance_uav1_target": s.get("distance_uav1_target"),
                    "distance_uav2_target": s.get("distance_uav2_target"),
                    "distance_uav1_uav2": s.get("distance_uav1_uav2"),
                    "success": s.get("success"),
                    "collision": s.get("collision"),
                    "done": s.get("done"),
                }
            )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_multi_policy_comparison_animation(
    trajectory_paths: dict[str, str],
    save_path: str,
    fps: int = 10,
    max_steps: int | None = None,
) -> str:
    """多策略同步对比动画（2×2 或 2×3 子图）。"""
    loaded: dict[str, dict] = {}
    arena = 10.0
    max_len = 0
    for name, path in trajectory_paths.items():
        if not os.path.isfile(path):
            continue
        data = load_trajectory(path)
        steps = data.get("steps", [])
        if not steps:
            continue
        loaded[name] = data
        arena = float(data.get("meta", {}).get("arena_size", arena))
        max_len = max(max_len, len(steps))

    if not loaded:
        raise ValueError("无有效轨迹用于多策略动画")

    if max_steps is not None:
        max_len = min(max_len, max_steps)

    names = list(loaded.keys())
    n = len(names)
    ncols = 3 if n > 4 else 2
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
    gs = GridSpec(nrows, ncols, figure=fig)
    lim = arena * 1.05

    axes_info = []
    for idx, name in enumerate(names):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        steps = loaded[name]["steps"]
        meta = loaded[name]["meta"]
        success = bool(meta.get("final_success", steps[-1].get("success", False)))
        ax.set_title(f"{name} ({'Success' if success else 'Fail'})", fontsize=10)

        u1x = np.array([s["uav1_x"] for s in steps])
        u1y = np.array([s["uav1_y"] for s in steps])
        u2x = np.array([s["uav2_x"] for s in steps])
        u2y = np.array([s["uav2_y"] for s in steps])
        tx = np.array([s["target_x"] for s in steps])
        ty = np.array([s["target_y"] for s in steps])

        (l1,) = ax.plot([], [], "-", color="#1f77b4", lw=1.5)
        (l2,) = ax.plot([], [], "-", color="#2ca02c", lw=1.5)
        (lt,) = ax.plot([], [], "--", color="#d62728", lw=1.0)
        p1 = ax.scatter([], [], c="#1f77b4", s=50, zorder=5)
        p2 = ax.scatter([], [], c="#2ca02c", s=50, zorder=5)
        pt = ax.scatter([], [], c="#d62728", marker="*", s=80, zorder=6)
        info_txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, va="bottom", fontsize=7)

        axes_info.append(
            {
                "name": name,
                "steps": steps,
                "ep_len": len(steps),
                "u1x": u1x,
                "u1y": u1y,
                "u2x": u2x,
                "u2y": u2y,
                "tx": tx,
                "ty": ty,
                "l1": l1,
                "l2": l2,
                "lt": lt,
                "p1": p1,
                "p2": p2,
                "pt": pt,
                "info_txt": info_txt,
            }
        )

    fig.suptitle("Multi-policy synchronized comparison", fontsize=12, y=1.02)

    def _update(frame: int):
        artists = []
        for info in axes_info:
            i = min(frame, info["ep_len"] - 1)
            done_early = frame >= info["ep_len"] - 1 and info["steps"][-1].get("done")
            l1, l2, lt = info["l1"], info["l2"], info["lt"]
            l1.set_data(info["u1x"][: i + 1], info["u1y"][: i + 1])
            l2.set_data(info["u2x"][: i + 1], info["u2y"][: i + 1])
            lt.set_data(info["tx"][: i + 1], info["ty"][: i + 1])
            info["p1"].set_offsets([[info["u1x"][i], info["u1y"][i]]])
            info["p2"].set_offsets([[info["u2x"][i], info["u2y"][i]]])
            info["pt"].set_offsets([[info["tx"][i], info["ty"][i]]])
            s = info["steps"][i]
            tag = "Done" if done_early and s.get("done") else f"step={s['step']}"
            info["info_txt"].set_text(f"a={s.get('action')} r={s.get('reward', 0):.2f}\n{tag}")
            artists.extend([l1, l2, lt, info["p1"], info["p2"], info["pt"], info["info_txt"]])
        return artists

    anim = FuncAnimation(fig, _update, frames=max_len, interval=1000 // max(fps, 1), blit=False)
    fig.tight_layout()
    saved = _save_anim(anim, save_path, fps)
    plt.close(fig)
    return saved


def compare_all_policies_outputs(
    trajectory_paths: dict[str, str],
    scenario_name: str,
    seed: int,
    save_dir: str,
    fps: int = 10,
) -> dict[str, str]:
    """一键：同步动画 + 静态图 + step_table + 返回路径 dict。"""
    os.makedirs(save_dir, exist_ok=True)
    base = f"compare_all_policies_{scenario_name}_seed{seed}"
    out: dict[str, str] = {}

    gif_path = os.path.join(save_dir, base + ".gif")
    out["compare_gif"] = make_multi_policy_comparison_animation(trajectory_paths, gif_path, fps=fps)

    png_path = os.path.join(save_dir, base + ".png")
    plot_trajectory_comparison(
        trajectory_paths,
        png_path,
        title=f"All policies @ {scenario_name} seed={seed}",
    )
    out["compare_png"] = png_path
    out["compare_pdf"] = png_path[:-4] + ".pdf"

    step_csv = os.path.join(save_dir, f"step_table_{scenario_name}_seed{seed}.csv")
    export_step_table(trajectory_paths, step_csv)
    out["step_table"] = step_csv

    return out
