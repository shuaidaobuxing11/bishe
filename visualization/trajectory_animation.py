"""Matplotlib 轨迹动画：基础 / 逐步（stepwise）。"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec

from visualization.trajectory_recorder import load_trajectory

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save_anim(anim: FuncAnimation, save_path: str, fps: int) -> str:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    out = save_path
    if not out.lower().endswith((".gif", ".mp4")):
        out = out + ".gif"
    saved = out
    if out.lower().endswith(".mp4"):
        try:
            anim.save(out, writer="ffmpeg", fps=fps)
            return saved
        except Exception:
            saved = out[:-4] + ".gif"
    anim.save(saved, writer=PillowWriter(fps=fps))
    return saved


def make_animation(
    trajectory_path: str,
    save_path: str,
    title: str = "",
    fps: int = 10,
) -> str:
    data = load_trajectory(trajectory_path)
    meta = data.get("meta", {})
    steps = data.get("steps", [])
    if not steps:
        raise ValueError(f"轨迹为空: {trajectory_path}")

    policy_name = meta.get("policy_name", "policy")
    scenario_name = meta.get("scenario_name", "scenario")
    arena = float(meta.get("arena_size", 10.0))
    final_success = bool(meta.get("final_success", steps[-1].get("success", False)))

    u1x = np.array([s["uav1_x"] for s in steps])
    u1y = np.array([s["uav1_y"] for s in steps])
    u2x = np.array([s["uav2_x"] for s in steps])
    u2y = np.array([s["uav2_y"] for s in steps])
    tx = np.array([s["target_x"] for s in steps])
    ty = np.array([s["target_y"] for s in steps])

    fig, ax = plt.subplots(figsize=(7, 7))
    lim = arena * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    (line1,) = ax.plot([], [], "-", color="#1f77b4", lw=2, label="UAV1")
    (line2,) = ax.plot([], [], "-", color="#2ca02c", lw=2, label="UAV2")
    (line_t,) = ax.plot([], [], "--", color="#d62728", lw=1.5, label="Target")
    pt1 = ax.scatter([], [], c="#1f77b4", s=80, zorder=5)
    pt2 = ax.scatter([], [], c="#2ca02c", s=80, zorder=5)
    pt_t = ax.scatter([], [], c="#d62728", marker="*", s=120, zorder=6)
    step_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=10)
    status_text = ax.text(0.02, 0.02, "", transform=ax.transAxes, va="bottom", fontsize=11, fontweight="bold")

    ax.set_title(title or f"{policy_name} @ {scenario_name}")
    ax.legend(loc="upper right", fontsize=9)

    def _update(i: int):
        i = min(i, len(steps) - 1)
        line1.set_data(u1x[: i + 1], u1y[: i + 1])
        line2.set_data(u2x[: i + 1], u2y[: i + 1])
        line_t.set_data(tx[: i + 1], ty[: i + 1])
        pt1.set_offsets([[u1x[i], u1y[i]]])
        pt2.set_offsets([[u2x[i], u2y[i]]])
        pt_t.set_offsets([[tx[i], ty[i]]])
        step_text.set_text(f"step = {steps[i]['step']}")
        if i == len(steps) - 1:
            status_text.set_text("Success" if final_success else "Fail")
            status_text.set_color("#1a9850" if final_success else "#d73027")
        else:
            coll = steps[i].get("collision", False)
            status_text.set_text("COLLISION" if coll else "")
            status_text.set_color("#d73027")
        return line1, line2, line_t, pt1, pt2, pt_t, step_text, status_text

    anim = FuncAnimation(fig, _update, frames=len(steps), interval=1000 // max(fps, 1), blit=False)
    saved = _save_anim(anim, save_path, fps)
    plt.close(fig)
    return saved


def make_stepwise_animation(
    trajectory_path: str,
    save_path: str,
    fps: int = 10,
    show_metrics: bool = True,
) -> str:
    """左：2D 轨迹；右上：动作/奖励/状态文本；右下：距离曲线。"""
    data = load_trajectory(trajectory_path)
    meta = data.get("meta", {})
    steps = data.get("steps", [])
    if not steps:
        raise ValueError(f"轨迹为空: {trajectory_path}")

    policy_name = meta.get("policy_name", "policy")
    scenario_name = meta.get("scenario_name", "scenario")
    seed = meta.get("seed", "")
    arena = float(meta.get("arena_size", 10.0))
    final_success = bool(meta.get("final_success", steps[-1].get("success", False)))

    u1x = np.array([s["uav1_x"] for s in steps])
    u1y = np.array([s["uav1_y"] for s in steps])
    u2x = np.array([s["uav2_x"] for s in steps])
    u2y = np.array([s["uav2_y"] for s in steps])
    tx = np.array([s["target_x"] for s in steps])
    ty = np.array([s["target_y"] for s in steps])
    d1 = np.array([s.get("distance_uav1_target", 0) for s in steps])
    d2 = np.array([s.get("distance_uav2_target", 0) for s in steps])
    d12 = np.array([s.get("distance_uav1_uav2", 0) for s in steps])
    step_ids = np.array([s["step"] for s in steps])

    fig = plt.figure(figsize=(11, 6))
    gs = GridSpec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_txt = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, 1]) if show_metrics else None

    lim = arena * 1.05
    ax_map.set_xlim(-lim, lim)
    ax_map.set_ylim(-lim, lim)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(True, alpha=0.3)

    (line1,) = ax_map.plot([], [], "-", color="#1f77b4", lw=2, label="UAV1")
    (line2,) = ax_map.plot([], [], "-", color="#2ca02c", lw=2, label="UAV2")
    (line_t,) = ax_map.plot([], [], "--", color="#d62728", lw=1.5, label="Target")
    pt1 = ax_map.scatter([], [], c="#1f77b4", s=80, zorder=5)
    pt2 = ax_map.scatter([], [], c="#2ca02c", s=80, zorder=5)
    pt_t = ax_map.scatter([], [], c="#d62728", marker="*", s=120, zorder=6)
    ax_map.legend(loc="upper right", fontsize=8)
    ax_map.set_title(f"{policy_name} @ {scenario_name} (seed={seed})")

    ax_txt.axis("off")
    txt_box = ax_txt.text(0.02, 0.98, "", va="top", ha="left", fontsize=10, family="monospace")

    ln_d1 = ln_d2 = ln_d12 = vline = None
    if ax_dist is not None:
        (ln_d1,) = ax_dist.plot([], [], "-", color="#1f77b4", label="d(UAV1,T)")
        (ln_d2,) = ax_dist.plot([], [], "-", color="#2ca02c", label="d(UAV2,T)")
        (ln_d12,) = ax_dist.plot([], [], "-", color="#9467bd", label="d(UAV1,UAV2)")
        vline = ax_dist.axvline(0, color="#333333", ls="--", lw=0.8)
        ax_dist.set_xlabel("step")
        ax_dist.set_ylabel("distance")
        ax_dist.legend(fontsize=8)
        ax_dist.grid(True, alpha=0.3)

    def _update(i: int):
        i = min(i, len(steps) - 1)
        s = steps[i]
        line1.set_data(u1x[: i + 1], u1y[: i + 1])
        line2.set_data(u2x[: i + 1], u2y[: i + 1])
        line_t.set_data(tx[: i + 1], ty[: i + 1])
        pt1.set_offsets([[u1x[i], u1y[i]]])
        pt2.set_offsets([[u2x[i], u2y[i]]])
        pt_t.set_offsets([[tx[i], ty[i]]])

        status = "SUCCESS" if s.get("success") else ("COLLISION" if s.get("collision") else ("DONE" if s.get("done") else "RUN"))
        txt_box.set_text(
            f"step: {s['step']}\n"
            f"action: {s.get('action')} ({s.get('uav1_action_name')}, {s.get('uav2_action_name')})\n"
            f"reward: {s.get('reward', 0):.3f}  cum: {s.get('cum_reward', 0):.3f}\n"
            f"status: {status}\n"
            f"d1={s.get('distance_uav1_target', 0):.3f}  d2={s.get('distance_uav2_target', 0):.3f}\n"
            f"d12={s.get('distance_uav1_uav2', 0):.3f}"
        )
        if ax_dist is not None and ln_d1 is not None:
            ln_d1.set_data(step_ids[: i + 1], d1[: i + 1])
            ln_d2.set_data(step_ids[: i + 1], d2[: i + 1])
            ln_d12.set_data(step_ids[: i + 1], d12[: i + 1])
            if vline is not None:
                vline.set_xdata([step_ids[i], step_ids[i]])
            ax_dist.relim()
            ax_dist.autoscale_view()
        return line1, line2, line_t, pt1, pt2, pt_t, txt_box

    anim = FuncAnimation(fig, _update, frames=len(steps), interval=1000 // max(fps, 1), blit=False)
    fig.tight_layout()
    if not save_path.endswith("_stepwise.gif") and "_stepwise" not in os.path.basename(save_path):
        base, ext = os.path.splitext(save_path)
        if not ext:
            save_path = save_path + "_stepwise.gif"
    saved = _save_anim(anim, save_path, fps)
    plt.close(fig)
    return saved
