#!/usr/bin/env python3
"""
PPO finetune + KL/BC 与 纯 PPO finetune（无 KL 辅助）四指标对比：
  平均回报、成功率、平均步长、碰撞率。

CSV 中方法名默认：
  - ppo_kl      — train_online_finetune 默认（含 KL + BC 辅助）
  - ppo_finetune — train_online_finetune --no_bc_regularizers

「只有 KL 有条长/碰撞、finetune 没有」的常见原因：**ppo_finetune 行是老版 CSV 写入的**，当时仅记 4 列；
迁移占位后仍为空白。需在升级 Callback 之后 **重新跑一次** `--no_bc_regularizers`，
同 method + 同 training_steps 在 CSV 后部的新行会覆盖读入时的旧行。

用法（项目根）:
  python scripts/plot_ppo_finetune_vs_kl.py
  python online_rl/train_online_finetune.py --no_bc_regularizers --total_timesteps 200000
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

LBL_KL = "PPO finetune + KL/BC"
LBL_PLAIN = "PPO finetune（无 KL）"


def _resolve_key(data: dict, explicit: str, default_key: str, role: str) -> str:
    exp = (explicit or "").strip()
    if exp:
        if exp not in data:
            raise KeyError(
                f"{role}：未找到键 `{exp}`。CSV 中的 method：{sorted(data.keys())}"
            )
        return exp
    if default_key in data:
        return default_key
    raise KeyError(
        f"未找到默认 {role} 键 `{default_key}`，请用 --kl_method / --fin_method 指定。"
        f" CSV 中的 method：{sorted(data.keys())}"
    )


def _finetune_plain_missing_length_collision(arr_plain: np.ndarray) -> bool:
    """纯 finetune 行为旧 CSV：无有效 mean_length / collision_rate。"""
    if arr_plain.size == 0 or arr_plain.shape[1] < 5:
        return True
    ok_len = np.any(np.isfinite(arr_plain[:, 3]))
    ok_col = np.any(np.isfinite(arr_plain[:, 4]))
    return not (ok_len and ok_col)


def _plot_two(
    ax,
    arr_kl: np.ndarray,
    arr_plain: np.ndarray,
    col: int,
    ylabel: str,
    title: str,
    scale_pct: bool,
    skip_plain: bool,
):
    if arr_kl.size and arr_kl.shape[1] > col:
        x = arr_kl[:, 0]
        y = arr_kl[:, col]
        if scale_pct:
            y = y * 100.0
        m = np.isfinite(y)
        ax.plot(x[m], y[m], "o-", lw=2, ms=5, label=LBL_KL, color="C0")
    if (
        not skip_plain
        and arr_plain.size
        and arr_plain.shape[1] > col
    ):
        x = arr_plain[:, 0]
        y = arr_plain[:, col]
        if scale_pct:
            y = y * 100.0
        m = np.isfinite(y)
        if np.any(m):
            ax.plot(x[m], y[m], "s-", lw=2, ms=5, label=LBL_PLAIN, color="C1")

    ax.set_title(title)
    ax.set_xlabel("Training steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=9)


def main() -> None:
    ap = argparse.ArgumentParser(description="PPO finetune+KL vs 纯 finetune 四指标曲线")
    ap.add_argument("--csv", type=str, default=str(ROOT / "results" / "training_curves.csv"))
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "results" / "ppo_finetune_vs_kl_curves.png"),
    )
    ap.add_argument("--kl_method", type=str, default="ppo_kl", help="含 KL/BC 的 CSV method 键")
    ap.add_argument("--fin_method", type=str, default="ppo_finetune", help="无 KL 的 CSV method 键")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"未找到 CSV: {args.csv}")

    data = read_curves_full(args.csv)
    kk = _resolve_key(data, args.kl_method, "ppo_kl", "PPO finetune+KL")
    fk = _resolve_key(data, args.fin_method, "ppo_finetune", "PPO finetune（无 KL）")

    arr_kl = data[kk]
    arr_f = data[fk]

    skip_len_coll = _finetune_plain_missing_length_collision(arr_f)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.8))

    _plot_two(axes[0, 0], arr_kl, arr_f, 2, "Mean return", "平均回报（评估回合）", False, False)
    _plot_two(axes[0, 1], arr_kl, arr_f, 1, "Success rate (%)", "成功率", True, False)

    _plot_two(
        axes[1, 0],
        arr_kl,
        arr_f,
        3,
        "Mean episode length",
        "平均回合步长",
        False,
        skip_plain=skip_len_coll,
    )
    _plot_two(
        axes[1, 1],
        arr_kl,
        arr_f,
        4,
        "Collision rate (%)",
        "碰撞率（回合级）",
        True,
        skip_plain=skip_len_coll,
    )

    foot = ""
    if skip_len_coll:
        foot = (
            "纯 finetune（橙色）CSV 缺少 mean_length / collision_rate → 下两图仅 KL。"
            "\n补救：在项目根运行  python online_rl/train_online_finetune.py --no_bc_regularizers --total_timesteps 200000"
            "\n（新日志写满 6 列；同 method 且同一 training_steps 时，以 CSV 文件后部行为准）"
        )
        for axmiss, cap in (
            (axes[1, 0], "纯 finetune 缺步长\n（旧 CSV，见下方）"),
            (axes[1, 1], "纯 finetune 缺碰撞率\n（旧 CSV，见下方）"),
        ):
            axmiss.text(
                0.5,
                0.14,
                cap,
                transform=axmiss.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color="C1",
                zorder=20,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.92),
            )

    title = f"PPO finetune + KL/BC vs 纯 finetune · {kk} / {fk}"
    if skip_len_coll:
        title += "（纯 finetune 缺少步长/碰撞列，须重训）"
    fig.suptitle(title, fontsize=12, y=1.03)
    if foot:
        fig.text(
            0.5,
            0.01,
            foot,
            ha="center",
            fontsize=9,
            color="0.2",
            transform=fig.transFigure,
        )
    plt.subplots_adjust(bottom=0.14, top=0.92)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"已保存: {out_path}")
    if skip_len_coll:
        print(
            f"\n[说明] `{fk}` 在 CSV 里没有有效的 mean_length / collision_rate。\n"
            "这是旧日志格式导致的；请重新跑一次无 KL 微调以写入完整 6 列：\n"
            "  python online_rl/train_online_finetune.py --no_bc_regularizers "
            "--total_timesteps <与 KL 相同的步数或其它预算>\n"
        )
    kl_miss = arr_kl.size and (
        not np.any(np.isfinite(arr_kl[:, 3])) or not np.any(np.isfinite(arr_kl[:, 4]))
    )
    if kl_miss:
        print(
            f"[说明] `{kk}` 也缺少步长/碰撞列，请在当前代码版本下重新跑带 KL 的 train_online_finetune。"
        )


if __name__ == "__main__":
    main()
