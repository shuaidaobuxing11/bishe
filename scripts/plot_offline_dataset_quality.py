#!/usr/bin/env python3
"""
离线数据集质量可视化（混合 pickle：mixed_offline_dataset.pkl）。

生成：
  1) 各子场景回合成功率柱状图（与生成数据时的 variant 对齐）
  2) 轨迹（回合）总回报分布
  3) 轨迹（回合）长度分布

用法（项目根）:
  python scripts/plot_offline_dataset_quality.py
  python scripts/plot_offline_dataset_quality.py --dataset data/offline/mixed_offline_dataset.pkl
  python scripts/plot_offline_dataset_quality.py --filter_expert_mode 仅统计 data_mode=expert 的回合
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_rl.mixed_dataset import load_mixed_bundle

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_MODE_EXPERT_INT = 0


def _unique_episode_rows(merged: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, np.ndarray]:
    """每个 episode 一行：win, return, length, scenario_id, data_mode_id。"""
    ep_ids = merged["episode_id"].reshape(-1)
    uniq = np.unique(ep_ids)
    wins: list[float] = []
    rets: list[float] = []
    lens: list[float] = []
    scen: list[int] = []
    dms: list[int] = []

    sid_to_name = meta.get("scenario_id_to_name", {})

    for eid in uniq:
        m = ep_ids == eid
        idx = np.flatnonzero(m)[0]
        wins.append(float(merged["episode_success"][idx]))
        rets.append(float(merged["episode_return"][idx]))
        lens.append(float(merged["episode_length"][idx]))
        scen.append(int(merged["scenario_id"][idx]))
        dms.append(int(merged["data_mode_id"][idx]))

    return {
        "win": np.asarray(wins),
        "return": np.asarray(rets),
        "length": np.asarray(lens),
        "scenario_id": np.asarray(scen),
        "data_mode_id": np.asarray(dms, dtype=np.int32),
        "scenario_name": np.asarray([sid_to_name.get(int(s), str(int(s))) for s in scen]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="离线数据集质量图（成功率 / 回报 / 长度）")
    ap.add_argument(
        "--dataset",
        type=str,
        default=str(ROOT / "data" / "offline" / "mixed_offline_dataset.pkl"),
        help="mixed_offline_dataset.pkl 路径",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT / "results"),
        help="PNG 输出目录",
    )
    ap.add_argument(
        "--combined_name",
        type=str,
        default="offline_dataset_quality.png",
        help="三联图文件名",
    )
    ap.add_argument(
        "--filter_expert_mode",
        action="store_true",
        help="只保留 data_mode=expert 的回合（剔除 recovery/suboptimal 段）",
    )
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.is_file():
        raise FileNotFoundError(
            f"未找到数据集: {ds_path}\n请先运行: python scripts/generate_offline_data.py --config configs/offline_config.yaml"
        )

    merged, meta = load_mixed_bundle(str(ds_path))
    if "episode_id" not in merged:
        raise ValueError("数据集中缺少 episode_id，是否为旧版 pickle？")

    ep = _unique_episode_rows(merged, meta)

    mask = np.ones(len(ep["win"]), dtype=bool)
    if args.filter_expert_mode:
        mask = ep["data_mode_id"] == DATA_MODE_EXPERT_INT

    wins = ep["win"][mask]
    rets = ep["return"][mask]
    lens = ep["length"][mask]
    names = ep["scenario_name"][mask]

    if len(wins) == 0:
        raise ValueError("过滤后无可用回合，请去掉 --filter_expert_mode 或检查数据集。")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filt_tag = "_expert_mode_only" if args.filter_expert_mode else ""

    # -------- 柱状图：各场景成功率 --------
    uniq_names: list[str] = []
    for n in names:
        ns = str(n)
        if ns not in uniq_names:
            uniq_names.append(ns)

    rates: list[float] = []
    counts: list[int] = []
    for lbl in uniq_names:
        m = names == lbl
        rates.append(float(np.mean(wins[m])) if m.any() else 0.0)
        counts.append(int(m.sum()))

    order = sorted(range(len(uniq_names)), key=lambda i: -rates[i])
    labels_ord = [uniq_names[i] for i in order]
    rates_ord = [rates[i] for i in order]
    counts_ord = [counts[i] for i in order]

    x = np.arange(len(labels_ord))
    fig_comb, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))

    ax0 = axes[0]
    bars = ax0.bar(x, [r * 100.0 for r in rates_ord], color="steelblue", edgecolor="0.35", linewidth=0.6)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels_ord, rotation=28, ha="right", fontsize=9)
    ax0.set_ylabel("成功率 (%)")
    ax0.set_title("离线数据 · 各子场景回合成功率" + ("（仅 expert 模式）" if args.filter_expert_mode else ""))
    ax0.set_ylim(0, max(105, max([r * 100 for r in rates_ord]) * 1.12))
    ax0.grid(True, axis="y", alpha=0.28)
    for i, (b, c, r) in enumerate(zip(bars, counts_ord, rates_ord)):
        ax0.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.2,
            f"n={c}\n{r*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax1 = axes[1]
    ax1.hist(rets, bins=min(50, max(15, len(rets) // 8)), color="coral", edgecolor="white", alpha=0.9)
    ax1.axvline(float(np.mean(rets)), color="darkred", ls="--", lw=1.5, label=f"均值 {np.mean(rets):.2f}")
    ax1.axvline(float(np.median(rets)), color="navy", ls=":", lw=1.5, label=f"中位 {np.median(rets):.2f}")
    ax1.set_xlabel("回合总回报")
    ax1.set_ylabel("回合数")
    ax1.set_title("轨迹回报分布（按回合）")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = axes[2]
    lens_i = lens.astype(int)
    ax2.hist(lens_i, bins=np.arange(lens_i.min(), lens_i.max() + 2) - 0.5, color="seagreen", edgecolor="white", alpha=0.9)
    ax2.axvline(float(np.mean(lens_i)), color="darkred", ls="--", lw=1.5, label=f"均值 {np.mean(lens_i):.1f}")
    ax2.axvline(float(np.median(lens_i)), color="navy", ls=":", lw=1.5, label=f"中位 {np.median(lens_i):.1f}")
    ax2.set_xlabel("轨迹长度（步数）")
    ax2.set_ylabel("回合数")
    ax2.set_title("轨迹长度分布（按回合）")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, axis="y", alpha=0.25)

    fig_comb.suptitle(f"离线数据集质量 · {ds_path.name}{filt_tag}", fontsize=12, y=1.02)
    fig_comb.tight_layout()
    comb_path = out_dir / f"{Path(args.combined_name).stem}{filt_tag}{Path(args.combined_name).suffix}"
    fig_comb.savefig(comb_path, dpi=160, bbox_inches="tight")
    plt.close(fig_comb)

    # 另存三张独立图（便于插入 PPT）
    def _save_single(fig, name: str) -> None:
        p = out_dir / f"{Path(name).stem}{filt_tag}{Path(name).suffix}"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return p

    f1, a1 = plt.subplots(figsize=(7.2, 4.8))
    bars = a1.bar(np.arange(len(labels_ord)), [r * 100.0 for r in rates_ord], color="steelblue", edgecolor="0.35")
    a1.set_xticks(np.arange(len(labels_ord)))
    a1.set_xticklabels(labels_ord, rotation=28, ha="right", fontsize=9)
    a1.set_ylabel("成功率 (%)")
    a1.set_title("各子场景专家/混合 Rollout 回合成功率")
    a1.set_ylim(0, max(105, max([r * 100 for r in rates_ord]) * 1.15))
    a1.grid(True, axis="y", alpha=0.28)
    for i, (b, c, r) in enumerate(zip(bars, counts_ord, rates_ord)):
        a1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"n={c}", ha="center", va="bottom", fontsize=8)
    p1 = _save_single(f1, "offline_dataset_success_by_scenario.png")

    f2, a2 = plt.subplots(figsize=(6.8, 4.6))
    a2.hist(rets, bins=min(50, max(15, len(rets) // 8)), color="coral", edgecolor="white", alpha=0.9)
    a2.axvline(float(np.mean(rets)), color="darkred", ls="--", lw=1.5, label=f"均值 {np.mean(rets):.2f}")
    a2.axvline(float(np.median(rets)), color="navy", ls=":", lw=1.5, label=f"中位 {np.median(rets):.2f}")
    a2.set_xlabel("回合总回报")
    a2.set_ylabel("回合数")
    a2.set_title("轨迹回报分布")
    a2.legend()
    a2.grid(True, axis="y", alpha=0.25)
    p2 = _save_single(f2, "offline_dataset_return_distribution.png")

    f3, a3 = plt.subplots(figsize=(6.8, 4.6))
    a3.hist(lens_i, bins=np.arange(lens_i.min(), lens_i.max() + 2) - 0.5, color="seagreen", edgecolor="white", alpha=0.9)
    a3.axvline(float(np.mean(lens_i)), color="darkred", ls="--", lw=1.5, label=f"均值 {np.mean(lens_i):.1f}")
    a3.axvline(float(np.median(lens_i)), color="navy", ls=":", lw=1.5, label=f"中位 {np.median(lens_i):.1f}")
    a3.set_xlabel("轨迹长度（步数）")
    a3.set_ylabel("回合数")
    a3.set_title("轨迹长度分布")
    a3.legend()
    a3.grid(True, axis="y", alpha=0.25)
    p3 = _save_single(f3, "offline_dataset_length_distribution.png")

    stats = {
        "dataset": str(ds_path),
        "n_episodes_used": int(len(wins)),
        "filter_expert_mode_only": bool(args.filter_expert_mode),
        "overall_success_rate": float(np.mean(wins)),
        "mean_return": float(np.mean(rets)),
        "median_return": float(np.median(rets)),
        "mean_episode_length": float(np.mean(lens_i)),
        "median_episode_length": float(np.median(lens_i)),
        "success_rate_by_scenario": {lbl: rates[i] for i, lbl in enumerate(uniq_names)},
        "figures": [str(comb_path), str(p1), str(p2), str(p3)],
    }
    stats_path = out_dir / f"offline_dataset_quality_stats{filt_tag}.json"
    with open(stats_path, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2, ensure_ascii=False)

    print("已保存:")
    print(" ", comb_path)
    print(" ", p1)
    print(" ", p2)
    print(" ", p3)
    print(" ", stats_path)


if __name__ == "__main__":
    main()
