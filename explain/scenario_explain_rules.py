"""场景期望关注点与匹配评估（细粒度分组）。"""
from __future__ import annotations

import numpy as np

SCENARIO_EXPECTED_IMPORTANCE = {
    "default": {
        "expected_high": ["target_relative", "target_absolute"],
        "expected_medium": ["uav_absolute", "inter_uav_coordination"],
        "expected_low": ["velocity"],
        "description": (
            "默认场景中目标运动较平稳，策略应主要关注 UAV 与目标之间的相对关系，"
            "同时结合目标位置和自身位置完成追踪。"
        ),
    },
    "near_uavs": {
        "expected_high": ["inter_uav_coordination", "safety"],
        "expected_medium": ["target_relative", "uav_absolute"],
        "expected_low": [],
        "description": (
            "近距离初始场景中双机碰撞风险较高，策略应重点关注 UAV 间距、安全约束和协同队形。"
        ),
    },
    "near_border": {
        "expected_high": ["target_relative", "uav_absolute", "safety"],
        "expected_medium": ["inter_uav_coordination"],
        "expected_low": ["velocity"],
        "description": (
            "边界场景中策略应在追踪目标的同时关注自身位置和安全约束，避免贴边或越界。"
        ),
    },
    "noisy_target": {
        "expected_high": ["target_relative", "velocity"],
        "expected_medium": ["target_absolute", "uav_absolute", "inter_uav_coordination"],
        "expected_low": [],
        "description": (
            "目标扰动场景中策略除关注 UAV 与目标的相对关系外，"
            "还应关注速度和运动趋势，以适应目标运动变化。"
        ),
    },
}

SCORE_KEY_MAP = {
    "target_absolute": "target_absolute_score",
    "uav_absolute": "uav_absolute_score",
    "target_relative": "target_relative_score",
    "inter_uav_coordination": "inter_uav_coordination_score",
    "safety": "safety_score",
    "velocity": "velocity_score",
}


def _score_from_dict(scores: dict, group: str) -> float:
    key = SCORE_KEY_MAP.get(group, f"{group}_score")
    return float(scores.get(key, scores.get(group, 0.0)) or 0.0)


def _composite_high_score(scores: dict, groups: list[str], scenario_name: str) -> tuple[float, list[str], list[str]]:
    """返回 (composite_high_score, matched, missing)。"""
    if scenario_name == "default" and set(groups) >= {"target_relative", "target_absolute"}:
        tr = _score_from_dict(scores, "target_relative")
        ta = _score_from_dict(scores, "target_absolute")
        composite = max(tr, ta)
        matched, missing = [], []
        if composite >= 0.12:
            matched.extend([g for g in ("target_relative", "target_absolute") if _score_from_dict(scores, g) >= 0.12])
            if not matched:
                matched = ["target_relative" if tr >= ta else "target_absolute"]
        else:
            missing = ["target_relative", "target_absolute"]
        return composite, matched, missing

    matched, missing, vals = [], [], []
    for g in groups:
        s = _score_from_dict(scores, g)
        vals.append(s)
        if s >= 0.12:
            matched.append(g)
        else:
            missing.append(g)
    return (float(np.mean(vals)) if vals else 0.0), matched, missing


def evaluate_scenario_alignment(scores: dict, scenario_name: str) -> dict:
    """scores: compute_explain_scores 输出（含各 *_score 字段）。"""
    rules = SCENARIO_EXPECTED_IMPORTANCE.get(scenario_name, SCENARIO_EXPECTED_IMPORTANCE["default"])
    high = rules["expected_high"]
    med = rules["expected_medium"]

    matched: list[str] = []
    missing: list[str] = []
    weighted: list[float] = []

    high_score, high_matched, high_missing = _composite_high_score(scores, high, scenario_name)
    weighted.append(high_score * 1.0)
    matched.extend(high_matched)
    missing.extend(high_missing)

    for g in med:
        s = _score_from_dict(scores, g)
        weighted.append(s * 0.55)
        if s >= 0.08:
            matched.append(g)

    alignment = float(np.clip(np.mean(weighted) if weighted else 0.0, 0.0, 1.0))

    # default：target_relative 高时不因 target_absolute 单独偏低而过度惩罚
    if scenario_name == "default":
        tr = _score_from_dict(scores, "target_relative")
        if tr >= 0.2:
            alignment = max(alignment, float(np.clip(tr * 0.82, 0.0, 1.0)))
            missing = [m for m in missing if m != "target_relative"]

    explanation_parts = [rules["description"]]
    if missing:
        if scenario_name == "noisy_target" and "velocity" in missing:
            explanation_parts.append(
                "在 noisy_target 场景下，策略对速度/运动趋势关注不足，可能导致对目标突变适应能力有限。"
            )
        elif scenario_name == "near_uavs" and ("safety" in missing or "inter_uav_coordination" in missing):
            explanation_parts.append(
                "在 near_uavs 场景下，策略对 UAV 间距或协同关注不足，可能增加碰撞风险。"
            )
        elif scenario_name == "near_border" and "safety" in missing:
            explanation_parts.append("边界场景下对安全约束关注偏低，边界处理可能不稳定。")
        elif scenario_name == "default" and set(missing) <= {"target_absolute"} and _score_from_dict(scores, "target_relative") >= 0.15:
            explanation_parts.append(
                "该策略在 default 场景中主要关注 UAV 与目标的相对关系，与默认追踪任务需求基本一致。"
            )
        else:
            explanation_parts.append(f"与场景期望相比，以下维度关注不足：{', '.join(sorted(set(missing)))}。")
    else:
        explanation_parts.append(
            f"该策略在 {scenario_name} 场景中主要关注 {', '.join(sorted(set(matched))[:3]) or '核心追踪维度'}，"
            f"与场景需求基本一致。"
        )

    if scenario_name == "default" and _score_from_dict(scores, "inter_uav_coordination") >= 0.2:
        explanation_parts.append(
            "双机协同关注度较高，这对双机追踪任务具有积极意义，不视为偏离场景需求。"
        )

    return {
        "scenario_name": scenario_name,
        "alignment_score": round(alignment, 3),
        "matched_expected_groups": sorted(set(matched)),
        "missing_expected_groups": sorted(set(missing)),
        "explanation": " ".join(explanation_parts),
    }
