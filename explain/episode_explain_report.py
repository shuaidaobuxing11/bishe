"""单 episode 自动解释 Markdown 报告生成。"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _analyze_trajectory_behavior(metrics: dict, thresholds: dict) -> list[str]:
    lines: list[str] = []
    sw = metrics.get("action_switch_rate", 0)
    if sw > thresholds.get("high_action_switch_rate", 0.45):
        lines.append("该策略动作切换频率较高，控制过程存在一定抖动。")
    else:
        lines.append("动作切换频率处于较低水平，控制相对平稳。")

    min_uav = metrics.get("min_uav_distance", 999)
    if min_uav < thresholds.get("unsafe_uav_distance", 0.5):
        lines.append("执行过程中双机间距曾接近碰撞阈值，存在较高碰撞风险。")
    else:
        lines.append("双机间距整体保持在相对安全范围。")

    mean_d = metrics.get("mean_distance_to_target", 999)
    min_d = metrics.get("min_mean_distance", 999)
    if min_d < mean_d * 0.85:
        lines.append("平均目标距离呈下降趋势，策略能够稳定接近目标。")
    elif mean_d > thresholds.get("poor_tracking_distance", 2.5):
        lines.append("与目标的平均距离偏大，存在追踪滞后或绕圈现象。")
    else:
        lines.append("目标距离变化中等，追踪过程尚可。")

    if metrics.get("success") and metrics.get("episode_length", 999) < 150:
        lines.append("在较少步数内完成任务，路径效率较高。")
    elif not metrics.get("success"):
        lines.append("未在限定步数内完成协同追踪任务。")
    return lines


def _feature_importance_text(top_df: pd.DataFrame, explanation_source: str) -> str:
    if top_df.empty or explanation_source == "rule_and_trajectory":
        return "（规则/轨迹策略：解释分数主要来自规则逻辑与轨迹统计，不与神经网络 Captum 归因完全等价。）"
    top = top_df.iloc[0]["feature"]
    if "target" in top and ("x" in top or "y" in top):
        return "该策略主要关注目标位置相关特征，动作选择与追踪目标空间关系密切相关。"
    if "uav" in top and ("vx" in top or "vy" in top):
        return "该策略较多依赖无人机速度/运动状态进行决策。"
    return "该策略在多个空间状态特征上均有贡献，决策依据较为分散。"


def _group_contrib_text(group_df: pd.DataFrame) -> str:
    if group_df.empty:
        return ""
    g = group_df.sort_values("normalized_importance", ascending=False)
    top_g = g.iloc[0]["group"]
    low = g[g["normalized_importance"] < 0.08]["group"].tolist()
    text = f"Captum/轨迹融合后，**{top_g}** 类特征贡献最高。"
    if low:
        text += f" 对 {', '.join(low)} 类特征关注较少。"
    if group_df.get("allow_overlap", pd.Series([True])).iloc[0]:
        text += " （语义分组允许重叠，如双机距离同时计入协同与安全。）"
    return text


def _fine_grained_interpretation(scores: dict, scenario: str) -> list[str]:
    lines: list[str] = []
    tr = scores.get("target_relative_score", 0)
    ta = scores.get("target_absolute_score", 0)
    coord = scores.get("inter_uav_coordination_score", 0)
    safety = scores.get("safety_score", 0)
    vel = scores.get("velocity_score", 0)
    stab = scores.get("action_stability_score", 0)
    eff = scores.get("task_efficiency_score", 0)

    if tr >= 0.15:
        lines.append("该策略较关注 UAV 与目标之间的相对关系，说明其动作选择与追踪任务目标直接相关。")
    if ta >= 0.15 and tr < 0.12:
        lines.append(
            "该策略更多依赖目标绝对位置，而对 UAV 与目标的相对关系利用不足，"
            "可能导致在变化初始条件下泛化较弱。"
        )
    if coord >= 0.15:
        lines.append("该策略较关注双机之间的空间关系，说明其具有一定协同控制倾向。")
    if safety < 0.12 and scenario == "near_uavs":
        lines.append("在近距离初始场景下，该策略对安全距离关注不足，可能增加碰撞风险。")
    if vel < 0.12 and scenario == "noisy_target":
        lines.append("在目标扰动场景下，该策略对速度/运动趋势关注不足，可能导致对目标突变适应能力有限。")
    if stab < 0.5:
        lines.append("该策略动作切换频繁，控制过程可能存在抖动。")
    if eff >= 0.55:
        lines.append("该策略能够较快完成追踪任务，任务执行效率较高。")
    if not lines:
        lines.append("各细粒度维度贡献较为均衡，需结合轨迹与回报综合判断。")
    return lines


def _success_failure_reason(metrics: dict, scores: dict, scenario: str, policy_name: str) -> str:
    if metrics.get("success"):
        return (
            "该策略成功的主要原因可能是能够持续关注目标相对关系或目标位置，"
            "并保持较稳定的接近过程。"
        )
    reasons = []
    if metrics.get("min_uav_distance", 1) < 0.55:
        reasons.append("双机间距控制不足导致碰撞风险")
    if scores.get("velocity_score", 0) < 0.08 and scenario == "noisy_target":
        reasons.append("对目标扰动/速度趋势适应不足")
    if scores.get("inter_uav_coordination_score", 0) < 0.1:
        reasons.append("双机协同/间距特征利用不足")
    if scores.get("target_relative_score", 0) < 0.08:
        reasons.append("对 UAV-目标相对关系利用不足")
    if metrics.get("action_switch_rate", 0) > 0.5:
        reasons.append("动作抖动过大")
    if not reasons:
        reasons.append("追踪滞后或未能同时满足双机成功距离条件")
    return "失败可能原因：" + "；".join(reasons) + "。"


def _policy_specific_notes(policy_name: str, metrics: dict, scores: dict, explanation_source: str) -> str:
    if policy_name == "expert_v2":
        return (
            "expert_v2 为规则策略（explanation_source=rule_and_trajectory），不适用 Captum 梯度归因。"
            "专家策略根据 UAV 与目标相对方向选择转向，并在距离较远时加速靠近目标。"
            "其解释分数主要来自规则逻辑和轨迹统计，不与神经网络策略的梯度归因完全等价。"
        )
    src_note = ""
    if explanation_source == "captum_and_trajectory":
        src_note = "解释来源：Captum Integrated Gradients + 轨迹统计。"
    else:
        src_note = "解释来源：行为/轨迹统计（Captum 不可用或未使用）。"
    if policy_name == "bc":
        return (
            src_note + " BC 策略在闭环执行中模仿离线专家分布；若 inter_uav_coordination_score 或 safety_score 偏低，"
            "可能在 near_uavs 等场景更容易偏离专家轨迹。"
        )
    if policy_name == "ppo_baseline":
        return (
            src_note + " PPO baseline 通过在线交互学习；若 target_relative 与 inter_uav_coordination 较高且回报较高，"
            "说明其有效利用了相对目标关系与双机协同。"
        )
    if policy_name.startswith("ppo_finetune"):
        kl = "kl" in policy_name
        base = src_note + " PPO finetune 仍受 BC 初始化影响；"
        if kl and scores.get("action_stability_score", 0) >= 0.6 and scores.get("task_efficiency_score", 0) < 0.5:
            base += "KL 约束可能提高稳定性但限制探索，效率提升不明显。"
        elif kl:
            base += "KL 约束可能提高稳定性但限制偏离 BC 的探索能力。"
        return base
    return src_note


SCORE_LABELS = [
    ("target_absolute_score", "目标绝对位置"),
    ("uav_absolute_score", "UAV 绝对位置"),
    ("target_relative_score", "UAV-目标相对关系"),
    ("inter_uav_coordination_score", "双机协同"),
    ("safety_score", "安全约束"),
    ("velocity_score", "运动趋势/速度"),
    ("action_stability_score", "动作稳定性"),
    ("task_efficiency_score", "任务效率"),
    ("scenario_alignment_score", "场景匹配度"),
]


def build_report_markdown(bundle: dict[str, Any]) -> str:
    m = bundle["metrics"]
    scores = bundle["scores"]
    align = bundle["alignment"]
    thresholds = bundle.get("thresholds", {})
    policy = bundle["policy_name"]
    scenario = bundle["scenario_name"]
    seed = bundle["seed"]
    explanation_source = bundle.get("explanation_source", scores.get("explanation_source", "captum_and_trajectory"))

    src_label = "规则 + 轨迹统计" if explanation_source == "rule_and_trajectory" else "Captum IG + 轨迹统计"

    lines = [
        "# 策略可解释性报告",
        "",
        "## 1. 基本信息",
        f"- **策略**: {policy}",
        f"- **场景**: {scenario}",
        f"- **seed**: {seed}",
        f"- **解释来源**: {src_label} (`{explanation_source}`)",
        f"- **是否成功**: {'是' if m.get('success') else '否'}",
        f"- **episode length**: {m.get('episode_length')}",
        f"- **episode return**: {m.get('episode_return', 0):.2f}",
        f"- **是否碰撞**: {'是' if m.get('collision') else '否'}",
        f"- **最小 UAV 间距**: {m.get('min_uav_distance', float('nan')):.3f}",
        f"- **平均目标距离**: {m.get('mean_distance_to_target', float('nan')):.3f}",
        f"- **最小目标距离**: {m.get('min_mean_distance', float('nan')):.3f}",
        f"- **action switch rate**: {m.get('action_switch_rate', 0):.3f}",
        "",
        "## 2. 轨迹行为分析",
    ]
    lines.extend(f"- {t}" for t in _analyze_trajectory_behavior(m, thresholds))

    lines += ["", "## 3. 特征重要性分析（Top 10）"]
    top_df = bundle.get("importance_df")
    if top_df is not None and len(top_df) and explanation_source == "captum_and_trajectory":
        for _, r in top_df.head(10).iterrows():
            lines.append(f"- {r['feature']}: {r['abs_attribution']:.4f}")
        lines.append("")
        lines.append(_feature_importance_text(top_df.head(10), explanation_source))
    else:
        lines.append("- （无 Captum 梯度归因；见第 4 节细粒度评分与轨迹分析。）")
        lines.append("")
        lines.append(_feature_importance_text(pd.DataFrame(), explanation_source))

    lines += ["", "## 4. 细粒度特征类别贡献"]
    for key, label in SCORE_LABELS:
        val = scores.get(key, 0)
        try:
            lines.append(f"- **{label}** (`{key}`): {float(val):.3f}")
        except (TypeError, ValueError):
            lines.append(f"- **{label}** (`{key}`): {val}")

    gdf = bundle.get("group_df")
    if gdf is not None and len(gdf):
        lines += ["", "### Captum/轨迹组级归一化贡献"]
        for _, r in gdf.sort_values("normalized_importance", ascending=False).iterrows():
            lines.append(f"- **{r['group']}**: {_pct(r['normalized_importance'])}")
        lines.append("")
        lines.append(_group_contrib_text(gdf))

    lines += ["", "### 自动解释"]
    lines.extend(f"- {t}" for t in _fine_grained_interpretation(scores, scenario))

    gw = bundle.get("group_warnings") or []
    if gw:
        lines += ["", "### 分组提示"]
        lines.extend(f"- {w}" for w in gw)

    lines += [
        "",
        "## 5. 场景匹配解释",
        f"- **场景匹配度**: {scores.get('scenario_alignment_score', align.get('alignment_score', 0)):.2f}",
        f"- {align.get('explanation', '')}",
        "",
        "## 6. 成功/失败原因",
        _success_failure_reason(m, scores, scenario, policy),
        "",
        "## 7. 策略专项说明",
        _policy_specific_notes(policy, m, scores, explanation_source),
        "",
        "## 8. 自动总结",
    ]
    summary = (
        f"总体来看，**{policy}** 在 **{scenario}** 场景下"
        f"{'成功' if m.get('success') else '未能成功'}完成追踪。"
        f" episode return={m.get('episode_return', 0):.2f}，"
        f"目标相对关系={scores.get('target_relative_score', 0):.2f}，"
        f"双机协同={scores.get('inter_uav_coordination_score', 0):.2f}，"
        f"安全={scores.get('safety_score', 0):.2f}，"
        f"场景匹配={scores.get('scenario_alignment_score', 0):.2f}。"
    )
    if scenario == "noisy_target" and scores.get("velocity_score", 0) < 0.1:
        summary += " 对速度类特征关注偏低，在更强扰动下可能存在鲁棒性不足。"
    lines.append(summary)
    return "\n".join(lines)


def generate_episode_explain_report(
    policy_name: str,
    scenario_name: str,
    seed: int,
    trajectory_path: str,
    importance_path: str | None,
    save_path: str,
    bundle: dict[str, Any] | None = None,
) -> str:
    """若已运行 pipeline 可传入 bundle；否则 save_path 由 pipeline 写入。"""
    if bundle is None:
        from explain.auto_explain_pipeline import run_auto_explain_pipeline

        out_dir = save_path if os.path.isdir(save_path) else os.path.dirname(save_path)
        bundle = run_auto_explain_pipeline(
            policy_name, scenario_name, seed, trajectory_path,
            model_path=None, save_dir=out_dir,
        )
    md = build_report_markdown(bundle)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    report_path = save_path if save_path.endswith(".md") else os.path.join(save_path, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    return report_path
