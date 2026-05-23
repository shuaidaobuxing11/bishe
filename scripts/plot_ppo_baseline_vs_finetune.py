#!/usr/bin/env python3
"""
PPO baseline 与 PPO finetune（或 PPO+KL）训练曲线四图对比：
  - 平均回报、成功率、平均步长、碰撞率（均来自周期性评估）。
依赖 results/training_curves.csv（需新版 TrainingCurveEvalCallback 写入 mean_length、collision_rate）。

用法（项目根）:
  python scripts/plot_ppo_baseline_vs_finetune.py

视觉：线型与原习惯对调（baseline 方点、finetune 圆点）；finetune 加粗实线主色、baseline 灰色虚线弱化，便于突出微调曲线。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from online_rl.training_curve_csv import read_curves_full  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _pick_baseline(data: dict, explicit: str) -> str:
    if explicit.strip() and explicit in data:
        return explicit.strip()
    for k in ("ppo_baseline", "ppo_baseline_default"):
        if k in data:
            return k
    raise KeyError(
        "未找到 baseline 曲线键。请先训练并写入 CSV，或通过 --baseline_method 指定。"
        f" 已有键：{sorted(data.keys())}"
    )


def _pick_finetune(data: dict, explicit: str | None, prefer_kl: bool) -> tuple[str, str]:
    labels = {"ppo_finetune": "PPO finetune", "ppo_kl": "PPO finetune（KL+BC）"}
    if explicit and explicit.strip() and explicit in data:
        k = explicit.strip()
        return k, labels.get(k, k)
    if prefer_kl and "ppo_kl" in data:
        return "ppo_kl", labels["ppo_kl"]
    if "ppo_finetune" in data:
        return "ppo_finetune", labels["ppo_finetune"]
    if "ppo_kl" in data:
        return "ppo_kl", labels["ppo_kl"]
    raise KeyError(
        "未找到 finetune 曲线。请先运行 train_online_finetune.py，或通过 --finetune_method 指定。"
        f" 已有键：{sorted(data.keys())}"
    )


def _plot_pair(
    ax,
    arr_b: np.ndarray,
    arr_f: np.ndarray,
    col: int,
    ylabel: str,
    title: str,
    scale_pct: bool,
    fin_display: str,
):
    """col: 1=succ, 2=ret, 3=len, 4=collision

    线与数据一一对应；图标互换：baseline→方形，finetune→圆点。
    finetune 用粗实线主色、后绘 zorder 更高；baseline 灰虚线弱化。
    图例顺序：finetune 在上（论文主方法）。
    """
    handles = []
    legends = []

    # 先画 baseline（底层）
    if arr_b.size and arr_b.shape[1] > col:
        xb = arr_b[:, 0]
        yb = arr_b[:, col]
        if scale_pct:
            yb = yb * 100.0
        m = np.isfinite(yb)
        (ln,) = ax.plot(
            xb[m],
            yb[m],
            "s--",
            lw=1.65,
            ms=4.5,
            label="PPO baseline",
            color="#9e9e9e",
            alpha=0.85,
            zorder=2,
        )
        handles.append(ln)
        legends.append("PPO baseline")

    # 后画 finetune（更醒目、压住交点）
    if arr_f.size and arr_f.shape[1] > col:
        xf = arr_f[:, 0]
        yf = arr_f[:, col]
        if scale_pct:
            yf = yf * 100.0
        m = np.isfinite(yf)
        (ln,) = ax.plot(
            xf[m],
            yf[m],
            "o-",
            lw=2.85,
            ms=6.5,
            label=fin_display,
            color="#08519c",
            markerfacecolor="#2171b5",
            markeredgecolor="#08306b",
            markeredgewidth=0.8,
            zorder=5,
            clip_on=False,
        )
        handles.append(ln)
        legends.append(fin_display)

    ax.set_title(title)
    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if handles:
        ax.legend(handles[::-1], legends[::-1], fontsize=9)



def main() -> None:
    ap = argparse.ArgumentParser(description="PPO baseline vs finetune 四指标曲线")
    ap.add_argument("--csv", type=str, default=str(ROOT / "results" / "training_curves.csv"))
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "ppo_baseline_vs_finetune_curves.png"),
    )
    ap.add_argument("--baseline_method", type=str, default="ppo_baseline")
    ap.add_argument("--finetune_method", type=str, default="", help="空则优先 ppo_kl 其次 ppo_finetune")
    ap.add_argument("--no_prefer_kl", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"未找到 CSV: {args.csv}")

    data = read_curves_full(args.csv)
    bk = _pick_baseline(data, args.baseline_method)
    fk, fin_display = _pick_finetune(data, args.finetune_method or None, prefer_kl=not args.no_prefer_kl)

    arr_b = data[bk]
    arr_f = data[fk]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.8))

    _plot_pair(axes[0, 0], arr_b, arr_f, 2, "Mean return", "平均回报（评估回合）", False, fin_display)
    _plot_pair(axes[0, 1], arr_b, arr_f, 1, "Success rate (%)", "成功率", True, fin_display)
    _plot_pair(axes[1, 0], arr_b, arr_f, 3, "Mean episode length", "平均回合步长", False, fin_display)
    _plot_pair(axes[1, 1], arr_b, arr_f, 4, "Collision rate (%)", "碰撞率（回合级）", True, fin_display)

    fig.suptitle(f"PPO baseline vs finetune · {bk} / {fk}", fontsize=13, y=1.02)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"已保存: {out_path}")
    if arr_b.shape[0] and not np.any(np.isfinite(arr_b[:, 3])):
        print(
            "[提示] mean_length/collision_rate 为空：请用当前代码重新跑 PPO baseline 与 finetune 写入完整评估指标。"
        )


if __name__ == "__main__":
    main()
