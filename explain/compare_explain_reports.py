"""策略间、成功/失败解释对比报告。"""
from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]

FINE_GRAINED_SCORE_COLS = [
    "target_absolute_score",
    "uav_absolute_score",
    "target_relative_score",
    "inter_uav_coordination_score",
    "safety_score",
    "velocity_score",
    "action_stability_score",
    "task_efficiency_score",
    "scenario_alignment_score",
]

SCORE_SHORT_LABELS = {
    "target_absolute_score": "target_abs",
    "uav_absolute_score": "uav_abs",
    "target_relative_score": "target_rel",
    "inter_uav_coordination_score": "coord",
    "safety_score": "safety",
    "velocity_score": "velocity",
    "action_stability_score": "stability",
    "task_efficiency_score": "efficiency",
    "scenario_alignment_score": "alignment",
}

EXPERT_FOOTNOTE = (
    "Note: expert_v2 scores are rule/trajectory-based, while neural policies use Captum attribution."
)


def _load_bundle_dir(report_dir: str, policy: str, scenario: str, seed: int) -> dict[str, Any] | None:
    d = os.path.join(report_dir, f"{policy}_{scenario}_seed{seed}")
    if not os.path.isdir(d):
        return None
    bundle: dict[str, Any] = {"policy_name": policy, "scenario_name": scenario, "seed": seed}
    scores_p = os.path.join(d, "explain_scores.json")
    if os.path.isfile(scores_p):
        with open(scores_p, "r", encoding="utf-8") as f:
            bundle["scores"] = json.load(f)
    gcsv = os.path.join(d, "feature_group_importance.csv")
    if os.path.isfile(gcsv):
        bundle["group_df"] = pd.read_csv(gcsv)
    mcsv = os.path.join(d, "..", "animations", f"metrics_summary_{scenario}_seed{seed}.csv")
    if os.path.isfile(mcsv):
        mdf = pd.read_csv(mcsv)
        row = mdf[mdf["policy_name"] == policy]
        if len(row):
            bundle["metrics"] = row.iloc[0].to_dict()
    return bundle


def compare_success_failure_explanations(
    success_report_data: dict[str, Any],
    failure_report_data: dict[str, Any],
    save_path: str,
) -> str:
    sp = success_report_data
    fp = failure_report_data
    sm, fm = sp.get("metrics", {}), fp.get("metrics", {})
    ss, fs = sp.get("scores", {}), fp.get("scores", {})

    lines = [
        "# 成功 vs 失败 解释对比",
        "",
        "## 指标差异",
        f"| 指标 | 成功 | 失败 |",
        f"|------|------|------|",
        f"| return | {sm.get('episode_return', 'NA')} | {fm.get('episode_return', 'NA')} |",
        f"| length | {sm.get('episode_length', 'NA')} | {fm.get('episode_length', 'NA')} |",
        f"| min_uav_distance | {sm.get('min_uav_distance', 'NA')} | {fm.get('min_uav_distance', 'NA')} |",
        f"| mean_distance_to_target | {sm.get('mean_distance_to_target', 'NA')} | {fm.get('mean_distance_to_target', 'NA')} |",
        f"| action_switch_rate | {sm.get('action_switch_rate', 'NA')} | {fm.get('action_switch_rate', 'NA')} |",
        "",
        "## 特征类别贡献差异",
    ]
    merged = None
    sg, fg = sp.get("group_df"), fp.get("group_df")
    if sg is not None and fg is not None:
        merged = sg.merge(fg, on="group", suffixes=("_succ", "_fail"))
        for _, r in merged.iterrows():
            lines.append(
                f"- **{r['group']}**: 成功 {_fmt(r.get('normalized_importance_succ'))} vs 失败 {_fmt(r.get('normalized_importance_fail'))}"
            )
        lines.append(
            "\n与成功轨迹相比，失败轨迹中 safety / inter_uav_coordination 贡献可能更低，"
            "说明失败过程中对双机间距和协同关系关注不足。"
        )

    lines.append("\n## 细粒度评分差异")
    for k in FINE_GRAINED_SCORE_COLS:
        lines.append(f"- {k}: 成功 {ss.get(k, 'NA')} vs 失败 {fs.get(k, 'NA')}")

    md = "\n".join(lines)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    md_path = save_path if save_path.endswith(".md") else save_path + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    if sg is not None and fg is not None:
        png_path = md_path[:-3] + ".png"
        _plot_group_compare(merged, png_path)
    return md_path


def _fmt(v) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


def _plot_group_compare(merged: pd.DataFrame, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(merged))
    w = 0.35
    ax.bar(x - w / 2, merged["normalized_importance_succ"], w, label="Success", color="#1a9850")
    ax.bar(x + w / 2, merged["normalized_importance_fail"], w, label="Failure", color="#d73027")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["group"], rotation=20, ha="right")
    ax.legend()
    ax.set_title("Group importance: success vs failure")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "（无数据）"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4g}" if pd.notna(v) else "")
            else:
                cells.append(str(v) if pd.notna(v) else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _load_metrics_table(scenario_name: str, seed: int, root: str | None = None) -> pd.DataFrame | None:
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mpath = os.path.join(root, "results", "animations", f"metrics_summary_{scenario_name}_seed{seed}.csv")
    if os.path.isfile(mpath):
        return pd.read_csv(mpath)
    from visualization.trajectory_metrics import compute_trajectory_metrics
    from visualization.trajectory_recorder import discover_trajectories_for_scenario_seed

    traj_dir = os.path.join(root, "results", "trajectories")
    paths = discover_trajectories_for_scenario_seed(traj_dir, scenario_name, seed)
    if not paths:
        return None
    rows = [compute_trajectory_metrics(p) for p in paths.values()]
    return pd.DataFrame(rows)


def _infer_explanation_source(policy_name: str, scores: dict) -> str:
    if scores.get("explanation_source"):
        return str(scores["explanation_source"])
    if policy_name == "expert_v2":
        return "rule_and_trajectory"
    return "captum_and_trajectory"


def _plot_fine_grained_scores(df: pd.DataFrame, save_path: str, title: str, has_expert: bool) -> None:
    cols = [c for c in FINE_GRAINED_SCORE_COLS if c in df.columns]
    if not cols or df.empty:
        return
    policies = df["policy_name"].tolist()
    x = np.arange(len(policies))
    width = 0.08
    fig, ax = plt.subplots(figsize=(max(10, len(policies) * 1.4), 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(cols)))
    for i, col in enumerate(cols):
        offset = (i - len(cols) / 2) * width + width / 2
        vals = df[col].fillna(0).astype(float).values
        label = SCORE_SHORT_LABELS.get(col, col.replace("_score", ""))
        ax.bar(x + offset, vals, width, label=label, color=colors[i], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    if has_expert:
        fig.text(0.01, 0.01, EXPERT_FOOTNOTE, fontsize=7, color="#555")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_scores_heatmap(df: pd.DataFrame, save_path: str, title: str, has_expert: bool) -> None:
    cols = [c for c in FINE_GRAINED_SCORE_COLS if c in df.columns]
    if not cols or df.empty:
        return
    policies = df["policy_name"].tolist()
    mat = df[cols].fillna(0).astype(float).values
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 0.9), max(3.5, len(policies) * 0.7)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([SCORE_SHORT_LABELS.get(c, c) for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(len(policies)))
    ylabels = []
    for _, row in df.iterrows():
        p = row["policy_name"]
        src = row.get("explanation_source", "")
        if src == "rule_and_trajectory":
            ylabels.append(f"{p} [rule]")
        else:
            ylabels.append(str(p))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    for i in range(len(policies)):
        for j in range(len(cols)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#222")
    fig.colorbar(im, ax=ax, fraction=0.03, label="Score")
    if has_expert:
        fig.text(0.01, 0.01, EXPERT_FOOTNOTE, fontsize=7, color="#555")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_performance_joint(df: pd.DataFrame, save_path: str, title: str) -> None:
    if df.empty or "episode_return" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ret = df["episode_return"].fillna(0).astype(float)
    tr = df.get("target_relative_score", pd.Series([0] * len(df))).fillna(0).astype(float)
    coord = df.get("inter_uav_coordination_score", pd.Series([0] * len(df))).fillna(0).astype(float)
    policies = df["policy_name"].tolist()
    colors = ["#762a83" if p == "expert_v2" else "#2166ac" for p in policies]

    axes[0].scatter(tr, ret, c=colors, s=80, edgecolors="#333")
    for i, p in enumerate(policies):
        axes[0].annotate(p, (tr.iloc[i], ret.iloc[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[0].set_xlabel("target_relative_score")
    axes[0].set_ylabel("episode_return")
    axes[0].set_title("Return vs target-relative focus")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(coord, ret, c=colors, s=80, edgecolors="#333")
    for i, p in enumerate(policies):
        axes[1].annotate(p, (coord.iloc[i], ret.iloc[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("inter_uav_coordination_score")
    axes[1].set_ylabel("episode_return")
    axes[1].set_title("Return vs coordination focus")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _plot_legacy_scores_bars(df: pd.DataFrame, save_path: str, title: str) -> None:
    """兼容旧 _scores.png（使用细粒度子集）。"""
    subset = [
        "target_relative_score",
        "inter_uav_coordination_score",
        "safety_score",
        "velocity_score",
        "scenario_alignment_score",
    ]
    cols = [c for c in subset if c in df.columns]
    if not cols:
        return
    sub = df[["policy_name"] + cols].copy()
    _plot_fine_grained_scores(sub, save_path, title, has_expert=(df["policy_name"] == "expert_v2").any())


def compare_policy_explanations(
    report_paths: list[str] | dict[str, str],
    save_path: str,
    scenario_name: str = "default",
    seed: int = 42,
) -> dict[str, str]:
    """report_paths: policy_name -> report directory。返回 md/csv/png 路径。"""
    rows: list[dict] = []
    group_rows: list[dict] = []

    items: list[tuple[str, Any]] = []
    if isinstance(report_paths, dict):
        for name, path in report_paths.items():
            if isinstance(path, dict):
                items.append((name, path))
            elif os.path.isdir(path):
                b = _load_bundle_from_dir(path)
                if b:
                    items.append((name, b))
            else:
                parent = os.path.dirname(path)
                b = _load_bundle_from_dir(parent)
                if b:
                    items.append((name, b))
    else:
        for p in report_paths:
            b = _load_bundle_from_dir(p if os.path.isdir(p) else os.path.dirname(p))
            if b:
                items.append((b.get("policy_name", "?"), b))

    metrics_df = _load_metrics_table(scenario_name, seed)

    for name, b in items:
        m: dict = {}
        if metrics_df is not None and "policy_name" in metrics_df.columns:
            row = metrics_df[metrics_df["policy_name"] == name]
            if len(row):
                m = row.iloc[0].to_dict()
        m = {**m, **b.get("metrics", {})}
        s = b.get("scores", {})
        exp_src = _infer_explanation_source(name, s)
        row = {
            "policy_name": name,
            "explanation_source": exp_src,
            "success": m.get("success", s.get("success")),
            "episode_return": m.get("episode_return"),
            "episode_length": m.get("episode_length"),
        }
        for col in FINE_GRAINED_SCORE_COLS:
            row[col] = s.get(col)
        rows.append(row)

        gdf = b.get("group_df")
        if gdf is not None:
            for _, r in gdf.iterrows():
                group_rows.append({"policy_name": name, "group": r["group"], "norm": r["normalized_importance"]})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    csv_path = save_path if save_path.endswith(".csv") else save_path.replace(".md", ".csv")
    md_path = save_path if save_path.endswith(".md") else save_path + ".md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    base = md_path[:-3]
    base_name = os.path.basename(base)
    has_expert = (df["policy_name"] == "expert_v2").any() if len(df) else False

    lines = [
        f"# 策略解释对比 — {scenario_name} seed={seed}",
        "",
        _df_to_markdown_table(df),
        "",
        "## 图表",
        f"- 细粒度评分柱状图：`{base_name}_fine_grained_scores.png`",
        f"- 可解释性评分热力图：`{base_name}_heatmap.png`",
        f"- 特征组贡献热力图：`{base_name}_groups.png`",
        f"- 性能-解释联合图：`{base_name}_perf_joint.png`",
        f"- 兼容柱状图：`{base_name}_scores.png`",
        "",
        "## 自动解释",
    ]
    if has_expert:
        lines.append(f"- {EXPERT_FOOTNOTE}")
        lines.append(
            "- expert_v2 解释分数来自规则逻辑与轨迹统计，不与 PPO/BC 的 Captum 梯度归因完全等价。"
        )
    if len(df):
        best = df.loc[df["episode_return"].idxmax()] if df["episode_return"].notna().any() else None
        if best is not None:
            lines.append(
                f"- **{best['policy_name']}** 回报较高（{best['episode_return']:.2f}），"
                f"目标相对关系={best.get('target_relative_score', 'NA')}，"
                f"双机协同={best.get('inter_uav_coordination_score', 'NA')}。"
            )
        bc_row = df[df["policy_name"] == "bc"]
        ppo_row = df[df["policy_name"] == "ppo_baseline"]
        if len(bc_row) and len(ppo_row):
            lines.append(
                "- BC 与 PPO baseline 对比：若 BC 的 inter_uav_coordination_score / safety_score 较低，"
                "说明闭环执行中对双机协同与安全约束利用不足；PPO baseline 通常更关注 target_relative 与双机协同。"
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    out_paths: dict[str, str] = {"md": md_path, "csv": csv_path}
    title_base = f"{scenario_name} seed={seed}"

    if group_rows:
        gdf = pd.DataFrame(group_rows)
        pivot = gdf.pivot(index="group", columns="policy_name", values="norm").fillna(0)
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.5), 5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Feature group importance — {title_base}")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        groups_png = base + "_groups.png"
        legacy_png = base + ".png"
        fig.savefig(groups_png, dpi=150)
        fig.savefig(legacy_png, dpi=150)
        plt.close(fig)
        out_paths["groups_png"] = groups_png
        out_paths["png"] = legacy_png

    fine_png = base + "_fine_grained_scores.png"
    _plot_fine_grained_scores(
        df, fine_png, f"Fine-grained explainability scores — {title_base}", has_expert=has_expert
    )
    if os.path.isfile(fine_png):
        out_paths["fine_grained_scores_png"] = fine_png

    heat_png = base + "_heatmap.png"
    _plot_scores_heatmap(df, heat_png, f"Explainability score heatmap — {title_base}", has_expert=has_expert)
    if os.path.isfile(heat_png):
        out_paths["heatmap_png"] = heat_png

    perf_png = base + "_perf_joint.png"
    _plot_performance_joint(df, perf_png, f"Performance vs explanation — {title_base}")
    if os.path.isfile(perf_png):
        out_paths["perf_joint_png"] = perf_png

    scores_png = base + "_scores.png"
    _plot_legacy_scores_bars(df, scores_png, f"Explainability scores — {title_base}")
    if os.path.isfile(scores_png):
        out_paths["scores_png"] = scores_png

    return out_paths


def _load_bundle_from_dir(d: str) -> dict | None:
    if not os.path.isdir(d):
        return None
    b: dict = {"out_dir": d}
    name = os.path.basename(d)
    parts = name.rsplit("_seed", 1)
    if len(parts) == 2:
        prefix, seed_s = parts
        b["seed"] = int(seed_s)
        for scen in ("default", "near_uavs", "near_border", "noisy_target"):
            if prefix.endswith("_" + scen):
                b["scenario_name"] = scen
                b["policy_name"] = prefix[: -(len(scen) + 1)]
                break
    sp = os.path.join(d, "explain_scores.json")
    if os.path.isfile(sp):
        with open(sp, "r", encoding="utf-8") as f:
            b["scores"] = json.load(f)
    gc = os.path.join(d, "feature_group_importance.csv")
    if os.path.isfile(gc):
        b["group_df"] = pd.read_csv(gc)
    return b if b.get("scores") else None
