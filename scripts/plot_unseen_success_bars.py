#!/usr/bin/env python3
"""
未见条件（near_uavs / near_border / noisy_target）成功率分组柱状图。

横轴：三个场景各为一组；
每组：random, expert_v2, bc, ppo_baseline, ppo_finetune 五根柱子；
纵轴：成功率（%）。

数据：scripts/run_unified_eval.py → results/eval_latest.json 或 eval_log.jsonl

用法（项目根）:
  python scripts/plot_unseen_success_bars.py
  python scripts/plot_unseen_success_bars.py --input results/eval_log.jsonl --run_id 20260317_214751
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCENARIO_ORDER = ["near_uavs", "near_border", "noisy_target"]
METHOD_ORDER = ["random", "expert_v2", "bc", "ppo_baseline", "ppo_finetune"]
METHOD_DISPLAY = {
    "random": "Random",
    "expert_v2": "Expert v2",
    "bc": "BC",
    "ppo_baseline": "PPO baseline",
    "ppo_finetune": "PPO finetune",
}
BAR_COLORS = {
    "random": "#bdbdbd",
    "expert_v2": "#31a354",
    "bc": "#fdb462",
    "ppo_baseline": "#6baed6",
    "ppo_finetune": "#08519c",
}


def _load_json_list(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _filter_unseen(rows: list[dict]) -> list[dict]:
    s = set(SCENARIO_ORDER)
    return [r for r in rows if r.get("scenario") in s]


def _filter_run(rows: list[dict], run_id: str | None) -> list[dict]:
    if not run_id:
        return rows
    return [r for r in rows if r.get("run_id") == run_id]


def _latest_run_id(rows: list[dict]) -> str | None:
    ids = [r.get("run_id") for r in rows if r.get("run_id")]
    return max(ids) if ids else None


def _method_map(rows: list[dict]) -> dict[str, dict]:
    m: dict[str, dict] = {}
    for r in rows:
        k = r.get("method")
        if isinstance(k, str):
            m[k] = r
    return m


def _success_matrix(rows: list[dict]) -> np.ndarray:
    """shape (n_scen, n_method)，缺失为 nan。"""
    mat = np.full((len(SCENARIO_ORDER), len(METHOD_ORDER)), np.nan, dtype=np.float64)
    for si, scen in enumerate(SCENARIO_ORDER):
        scen_rows = [r for r in rows if r.get("scenario") == scen]
        mmap = _method_map(scen_rows)
        for mi, method in enumerate(METHOD_ORDER):
            row = mmap.get(method)
            if row is not None and "success_rate" in row:
                mat[si, mi] = float(row["success_rate"]) * 100.0
    return mat


def plot_grouped_success(
    matrix: np.ndarray,
    title_suffix: str,
    out_path: str,
    dpi: int = 150,
) -> None:
    n_scen = len(SCENARIO_ORDER)
    n_m = len(METHOD_ORDER)
    x = np.arange(n_scen, dtype=np.float64)
    group_span = 0.72
    each_w = group_span / n_m
    offsets = (np.arange(n_m) - (n_m - 1) / 2.0) * each_w

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for mi, method in enumerate(METHOD_ORDER):
        heights = matrix[:, mi]
        pos = x + offsets[mi]
        color = BAR_COLORS.get(method, "#777777")
        ax.bar(
            pos,
            heights,
            width=each_w * 0.92,
            label=METHOD_DISPLAY[method],
            color=color,
            edgecolor="#333333",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_ORDER)
    ax.set_xlabel("场景")
    ax.set_ylabel("成功率 / %")
    ax.set_title(f"未见条件成功率对比 — {title_suffix}")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="未见条件成功率分组柱状图")
    ap.add_argument("--input", type=str, default=str(ROOT / "results" / "eval_latest.json"))
    ap.add_argument("--run_id", type=str, default="", help="jsonl 时选用；空则取未见场景行中的最新 run_id")
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "unseen_conditions_success_bars.png"))
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    path = args.input
    if not os.path.isfile(path):
        print(f"未找到 {path}，请先运行 scripts/run_unified_eval.py。")
        return

    if path.endswith(".jsonl"):
        rows = _load_jsonl(path)
        rows = _filter_unseen(rows)
        rid = args.run_id.strip() or _latest_run_id(rows)
        if not rid:
            print("未见场景在 jsonl 中无记录或缺少 run_id。")
            return
        rows = _filter_run(rows, rid)
    else:
        rows = _load_json_list(path)
        rows = _filter_unseen(rows)
        if args.run_id.strip():
            rows = _filter_run(rows, args.run_id.strip())

    mat = _success_matrix(rows)
    if not np.any(np.isfinite(mat)):
        print("未见场景无有效成功率数据，请检查评估日志是否包含这三个场景及五种方法。")
        return

    missing_methods: set[str] = set()
    for si in range(len(SCENARIO_ORDER)):
        for mi, method in enumerate(METHOD_ORDER):
            if not np.isfinite(mat[si, mi]):
                missing_methods.add(method)
    if missing_methods:
        print("警告: 部分 (场景, 方法) 缺失，对应柱高为空:", ", ".join(sorted(missing_methods)))

    meta = next((r for r in rows if r.get("scenario") in SCENARIO_ORDER), None)
    if meta:
        title_suffix = (
            f"n_episodes={meta.get('n_episodes', '?')}, seed={meta.get('seed', '?')}"
            + (f", run_id={meta.get('run_id')}" if meta.get("run_id") else "")
        )
    else:
        title_suffix = "（无元数据）"

    plot_grouped_success(mat, title_suffix, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
