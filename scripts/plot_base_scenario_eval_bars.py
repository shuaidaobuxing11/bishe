#!/usr/bin/env python3
"""
基础场景（默认 default）下各方法评估结果柱状图。

方法（固定顺序）: random, expert_v2, bc, ppo_baseline, ppo_finetune
指标: 成功率、平均回报、碰撞率（按回合 collision_rate_episodes，与 eval 表一致）

数据来源: scripts/run_unified_eval.py 生成的 results/eval_latest.json
         或 results/eval_log.jsonl（可指定 run_id 取某次完整评估）

用法（项目根）:
  python scripts/plot_base_scenario_eval_bars.py
  python scripts/plot_base_scenario_eval_bars.py --scenario default --out results/base_scenario_eval_bars.png
  python scripts/plot_base_scenario_eval_bars.py --input results/eval_log.jsonl --run_id 20260317_214751
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

METHOD_ORDER = ["random", "expert_v2", "bc", "ppo_baseline", "ppo_finetune"]
METHOD_DISPLAY = {
    "random": "Random",
    "expert_v2": "Expert v2",
    "bc": "BC",
    "ppo_baseline": "PPO baseline",
    "ppo_finetune": "PPO finetune",
}

# 与表格/论文区分度兼顾的配色
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


def _filter_scenario(rows: list[dict], scenario: str) -> list[dict]:
    return [r for r in rows if r.get("scenario") == scenario]


def _filter_run(rows: list[dict], run_id: str | None) -> list[dict]:
    if not run_id:
        return rows
    return [r for r in rows if r.get("run_id") == run_id]


def _latest_run_id(rows: list[dict]) -> str | None:
    ids = [r.get("run_id") for r in rows if r.get("run_id")]
    return max(ids) if ids else None


def _method_map(rows: list[dict]) -> dict[str, dict]:
    """同 method 多条时保留最后一次出现。"""
    m: dict[str, dict] = {}
    for r in rows:
        k = r.get("method")
        if isinstance(k, str):
            m[k] = r
    return m


def _collect_vectors(method_map: dict[str, dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    labels: list[str] = []
    succ: list[float] = []
    ret: list[float] = []
    coll_ep: list[float] = []
    for key in METHOD_ORDER:
        labels.append(METHOD_DISPLAY[key])
        row = method_map.get(key)
        if row is None:
            succ.append(np.nan)
            ret.append(np.nan)
            coll_ep.append(np.nan)
            continue
        succ.append(float(row["success_rate"]) * 100.0)
        ret.append(float(row["mean_return"]))
        coll_ep.append(float(row["collision_rate_episodes"]) * 100.0)
    return labels, np.asarray(succ), np.asarray(ret), np.asarray(coll_ep)


def plot_bars(
    labels: list[str],
    succ_pct: np.ndarray,
    mean_ret: np.ndarray,
    coll_ep_pct: np.ndarray,
    title_suffix: str,
    out_path: str,
    dpi: int = 150,
) -> None:
    x = np.arange(len(labels))
    w = 0.62

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    colors = [BAR_COLORS.get(METHOD_ORDER[i], "#777777") for i in range(len(METHOD_ORDER))]

    ax0 = axes[0]
    for i in range(len(x)):
        if np.isfinite(succ_pct[i]):
            ax0.bar(x[i], succ_pct[i], width=w, color=colors[i], edgecolor="#333333", linewidth=0.6)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=22, ha="right")
    ax0.set_ylabel("成功率 (%)")
    ax0.set_title("成功率")
    if np.any(np.isfinite(succ_pct)):
        ax0.set_ylim(0, max(100.0, float(np.nanmax(succ_pct)) * 1.12))
    else:
        ax0.set_ylim(0, 100)
    ax0.grid(True, axis="y", alpha=0.3)

    ax1 = axes[1]
    for i in range(len(x)):
        if np.isfinite(mean_ret[i]):
            ax1.bar(x[i], mean_ret[i], width=w, color=colors[i], edgecolor="#333333", linewidth=0.6)
    ax1.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--", zorder=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=22, ha="right")
    ax1.set_ylabel("平均回报")
    ax1.set_title("平均回报")
    rmin, rmax = np.nanmin(mean_ret), np.nanmax(mean_ret)
    if np.all(~np.isfinite(mean_ret)):
        ax1.set_ylim(-1.0, 1.0)
    else:
        pad = max(5.0, (rmax - rmin) * 0.12)
        ax1.set_ylim(rmin - pad, rmax + pad)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = axes[2]
    for i in range(len(x)):
        if np.isfinite(coll_ep_pct[i]):
            ax2.bar(x[i], coll_ep_pct[i], width=w, color=colors[i], edgecolor="#333333", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=22, ha="right")
    ax2.set_ylabel("碰撞率 (%)")
    ax2.set_title("碰撞率（按回合）")
    ymax = float(np.nanmax(coll_ep_pct)) if np.any(np.isfinite(coll_ep_pct)) else 80.0
    ax2.set_ylim(0, min(100.0, ymax * 1.15))
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"基础场景评估对比 — {title_suffix}", fontsize=13, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="基础场景各方法评估柱状图")
    ap.add_argument("--input", type=str, default=str(ROOT / "results" / "eval_latest.json"), help="JSON 列表或 .jsonl")
    ap.add_argument("--scenario", type=str, default="default", help="场景名，通常为 default")
    ap.add_argument("--run_id", type=str, default="", help="从 eval_log.jsonl 截取某次 run_id（空则取最新）")
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "base_scenario_eval_bars.png"))
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    path = args.input
    if not os.path.isfile(path):
        print(f"未找到 {path}，请先运行 scripts/run_unified_eval.py（含 random 与 include_ppo_baseline 按需打开）。")
        return

    if path.endswith(".jsonl"):
        rows = _load_jsonl(path)
        rows = _filter_scenario(rows, args.scenario)
        rid = args.run_id.strip() or _latest_run_id(rows)
        if not rid:
            print("jsonl 中无 run_id，无法汇总。")
            return
        rows = _filter_run(rows, rid)
    else:
        rows = _load_json_list(path)
        rows = _filter_scenario(rows, args.scenario)
        if args.run_id.strip():
            rows = _filter_run(rows, args.run_id.strip())

    mmap = _method_map(rows)
    missing = [k for k in METHOD_ORDER if k not in mmap]
    if missing:
        print("警告: 下列方法在本次结果中缺失，柱状图为空缺口:", ", ".join(missing))

    labels, succ, ret, coll = _collect_vectors(mmap)
    meta = next((mmap[k] for k in METHOD_ORDER if k in mmap), None)
    if meta:
        title_suffix = (
            f"scenario={args.scenario}, n_episodes={meta.get('n_episodes', '?')}, "
            f"seed={meta.get('seed', '?')}"
            + (f", run_id={meta.get('run_id')}" if meta.get("run_id") else "")
        )
    else:
        title_suffix = f"scenario={args.scenario}（无可用行）"

    plot_bars(labels, succ, ret, coll, title_suffix, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
