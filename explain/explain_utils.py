"""解释结果绘图与对比工具。"""
from __future__ import annotations

import os
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def save_importance_csv(path: str, feature_names: list[str], attributions: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "attribution": attributions.astype(float),
            "abs_attribution": np.abs(attributions.astype(float)),
        }
    ).sort_values("abs_attribution", ascending=False)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_importance_bar(
    feature_names: list[str],
    attributions: np.ndarray,
    save_path: str,
    title: str = "",
    top_k: int = 12,
) -> None:
    idx = np.argsort(np.abs(attributions))[::-1][:top_k]
    names = [feature_names[i] for i in idx]
    vals = attributions[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(names) + 1.5)))
    colors = ["#2166ac" if v >= 0 else "#b2182b" for v in vals]
    ax.barh(range(len(names)), vals, color=colors, edgecolor="#333333", linewidth=0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="#666666", lw=0.8)
    ax.set_xlabel("Attribution")
    ax.set_title(title or "Feature importance")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compare_success_failure_importance(
    success_importance_path: str,
    failure_importance_path: str,
    save_path: str,
    top_k: int = 10,
) -> None:
    s_df = pd.read_csv(success_importance_path)
    f_df = pd.read_csv(failure_importance_path)
    merged = s_df.merge(f_df, on="feature", suffixes=("_success", "_failure"))
    merged["diff"] = merged["abs_attribution_success"] - merged["abs_attribution_failure"]
    merged["abs_diff"] = np.abs(merged["diff"])
    top = merged.sort_values("abs_diff", ascending=False).head(top_k)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(top) + 1.5)))
    y = np.arange(len(top))
    h = 0.35
    ax.barh(y - h / 2, top["abs_attribution_success"], height=h, label="Success", color="#1a9850", alpha=0.85)
    ax.barh(y + h / 2, top["abs_attribution_failure"], height=h, label="Failure", color="#d73027", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("|Attribution| (episode mean)")
    ax.set_title("Success vs Failure — feature importance")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compare_policy_importance(
    importance_paths: list[str],
    policy_names: list[str],
    save_path: str,
    top_k: int = 10,
) -> None:
    frames = []
    for p, name in zip(importance_paths, policy_names):
        df = pd.read_csv(p)
        df = df.sort_values("abs_attribution", ascending=False).head(top_k)
        df["policy"] = name
        frames.append(df[["policy", "feature", "abs_attribution"]])
    all_df = pd.concat(frames, ignore_index=True)
    pivot = all_df.pivot_table(index="feature", columns="policy", values="abs_attribution", aggfunc="mean")
    pivot = pivot.fillna(0.0)

    fig, ax = plt.subplots(figsize=(max(8, len(policy_names) * 2), max(5, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Policy feature importance comparison")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
