"""
未见条件场景（near_uavs / near_border / noisy_target）评估可视化。

输入：
  results/eval_latest.json （由 scripts/run_unified_eval.py 生成）

输出到 results/：
  fig_unseen_success_rate.png
  fig_unseen_mean_return.png
  fig_unseen_mean_length.png
  fig_unseen_collision_rate_episodes.png

默认只画三种未见场景，不画 default。
"""
import os
import json
import argparse
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_latest(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_rows(rows: List[Dict], scenarios: List[str], methods: List[str]):
    out = []
    for r in rows:
        if r.get("scenario") in scenarios and r.get("method") in methods:
            out.append(r)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latest", type=str, default="results/eval_latest.json")
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--include_default", action="store_true", help="同时画 default（不建议用于“未见”页）")
    p.add_argument("--save_prefix", type=str, default="fig_unseen")
    args = p.parse_args()

    rows = _load_latest(args.latest)
    if rows is None:
        raise FileNotFoundError(f"未找到 {args.latest}，请先运行统一评估生成评估日志。")

    unseen_scenarios = ["near_uavs", "near_border", "noisy_target"]
    scenarios = unseen_scenarios + (["default"] if args.include_default else [])
    methods = ["expert_v2", "bc", "ppo_finetune", "bc_mixed"]

    # 用到哪些 method（存在于数据里）
    methods_present = []
    for m in methods:
        if any(r.get("method") == m for r in rows):
            methods_present.append(m)
    if not methods_present:
        raise ValueError("评估日志里没有可用 method。")

    os.makedirs(args.out_dir, exist_ok=True)

    # index
    scen_idx = {s: i for i, s in enumerate(scenarios)}
    m_idx = {m: i for i, m in enumerate(methods_present)}

    def get_metric(metric: str, default_val=np.nan):
        # shape: (n_methods, n_scenarios)
        arr = np.full((len(methods_present), len(scenarios)), default_val, dtype=np.float64)
        for r in rows:
            s = r.get("scenario")
            m = r.get("method")
            if s in scen_idx and m in m_idx:
                arr[m_idx[m], scen_idx[s]] = float(r.get(metric, default_val))
        return arr

    success_rate = get_metric("success_rate", default_val=np.nan)  # 0-1
    mean_return = get_metric("mean_return", default_val=np.nan)
    std_return = get_metric("std_return", default_val=np.nan)
    mean_length = get_metric("mean_length", default_val=np.nan)
    std_length = get_metric("std_length", default_val=np.nan)
    coll_ep = get_metric("collision_rate_episodes", default_val=np.nan)

    x = np.arange(len(scenarios))

    # 1) Success rate bar
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    width = 0.8 / max(1, len(methods_present))
    for j, m in enumerate(methods_present):
        ax.bar(x + (j - (len(methods_present) - 1) / 2) * width, success_rate[j] * 100.0, width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Unseen conditions: success rate")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{args.save_prefix}_success_rate.png"), dpi=150)
    plt.close(fig)

    # 2) Mean return line (with error bars)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for j, m in enumerate(methods_present):
        ax.plot(x, mean_return[j], marker="o", label=m)
        # error bars (std)
        if np.isfinite(std_return[j]).any():
            ax.errorbar(x, mean_return[j], yerr=std_return[j], fmt="none", ecolor=ax.lines[-1].get_color(), alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Mean return")
    ax.set_title("Unseen conditions: mean return")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{args.save_prefix}_mean_return.png"), dpi=150)
    plt.close(fig)

    # 3) Mean length line (with error bars)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for j, m in enumerate(methods_present):
        ax.plot(x, mean_length[j], marker="o", label=m)
        if np.isfinite(std_length[j]).any():
            ax.errorbar(x, mean_length[j], yerr=std_length[j], fmt="none", ecolor=ax.lines[-1].get_color(), alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Mean length (steps)")
    ax.set_title("Unseen conditions: mean episode length")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{args.save_prefix}_mean_length.png"), dpi=150)
    plt.close(fig)

    # 4) Collision rate episodes bar
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    for j, m in enumerate(methods_present):
        ax.bar(x + (j - (len(methods_present) - 1) / 2) * width, coll_ep[j] * 100.0, width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Collision rate (episodes) (%)")
    ax.set_title("Unseen conditions: collision rate per episode")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, f"{args.save_prefix}_collision_rate_episodes.png"), dpi=150)
    plt.close(fig)

    print("未见条件图已导出：")
    for fn in [
        f"{args.save_prefix}_success_rate.png",
        f"{args.save_prefix}_mean_return.png",
        f"{args.save_prefix}_mean_length.png",
        f"{args.save_prefix}_collision_rate_episodes.png",
    ]:
        print("-", os.path.join(args.out_dir, fn))


if __name__ == "__main__":
    main()

