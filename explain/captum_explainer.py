"""Captum Integrated Gradients / Saliency 策略解释。"""
from __future__ import annotations

import os
from typing import Literal

import numpy as np
import torch

import online_rl.ppo_bc_finetune as _ppo_bc_fe  # noqa: F401 pickle 兼容

from explain.explain_utils import plot_importance_bar, save_importance_csv
from explain.feature_names import validate_obs_dim
from explain.policy_wrapper import BCLogitsWrapper, SB3PolicyLogitsWrapper
from visualization.trajectory_recorder import load_trajectory


def _load_wrapper(model_path: str, device: str = "cpu"):
    path_lower = model_path.lower()
    if path_lower.endswith(".zip"):
        from stable_baselines3 import PPO

        model = PPO.load(model_path, device=device)
        wrapper = SB3PolicyLogitsWrapper(model).to(device)
        wrapper.eval()
        return wrapper, "ppo"
    if path_lower.endswith(".pt"):
        from offline_rl.behavior_cloning import load_bc_model

        net = load_bc_model(model_path, device=device)
        wrapper = BCLogitsWrapper(net).to(device)
        wrapper.eval()
        return wrapper, "bc"
    raise ValueError(f"不支持的模型格式: {model_path}")


def _predict_action(wrapper, obs: np.ndarray, device: str) -> int:
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = wrapper(x)
        return int(logits.argmax(dim=-1).item())


def _compute_attribution(
    wrapper,
    obs: np.ndarray,
    target_action: int,
    method: str,
    device: str,
) -> np.ndarray:
    from captum.attr import IntegratedGradients, Saliency

    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    obs_t.requires_grad_(True)
    baseline = torch.zeros_like(obs_t)

    def forward_func(x: torch.Tensor) -> torch.Tensor:
        logits = wrapper(x)
        return logits[:, target_action]

    if method == "saliency":
        attr = Saliency(forward_func)(obs_t, abs=False)
    else:
        ig = IntegratedGradients(forward_func)
        attr = ig.attribute(obs_t, baselines=baseline)

    return attr.detach().cpu().numpy().reshape(-1)


def explain_single_state(
    model_path: str,
    obs: np.ndarray,
    target_action: int | None = None,
    method: str = "integrated_gradients",
    save_path: str | None = None,
    policy_name: str = "policy",
    scenario_name: str = "default",
    device: str = "cpu",
    top_k_plot: int = 12,
) -> dict:
    wrapper, _ = _load_wrapper(model_path, device=device)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    names = validate_obs_dim(len(obs))

    if target_action is None:
        target_action = _predict_action(wrapper, obs, device)

    attr = _compute_attribution(wrapper, obs, int(target_action), method, device)

    result = {
        "policy_name": policy_name,
        "scenario_name": scenario_name,
        "method": method,
        "target_action": int(target_action),
        "feature_names": names,
        "attributions": attr.tolist(),
    }

    if save_path:
        if save_path.endswith((".png", ".csv")):
            base_dir = os.path.dirname(save_path) or "."
            stem = os.path.splitext(os.path.basename(save_path))[0]
        else:
            base_dir = save_path
            stem = f"{policy_name}_{scenario_name}_single_state_importance"
        os.makedirs(base_dir, exist_ok=True)
        csv_path = os.path.join(base_dir, stem + ".csv")
        png_path = os.path.join(base_dir, stem + ".png")
        save_importance_csv(csv_path, names, attr)
        plot_importance_bar(
            names,
            attr,
            png_path,
            title=f"{policy_name} @ {scenario_name} | action={target_action} ({method})",
            top_k=top_k_plot,
        )
        result["csv_path"] = csv_path
        result["png_path"] = png_path

    return result


def explain_episode(
    model_path: str,
    trajectory_path: str,
    method: str = "integrated_gradients",
    save_dir: str = "results/explain",
    device: str = "cpu",
    max_steps: int | None = None,
) -> dict:
    data = load_trajectory(trajectory_path)
    meta = data.get("meta", {})
    steps = data.get("steps", [])
    if not steps:
        raise ValueError(f"空轨迹: {trajectory_path}")

    policy_name = meta.get("policy_name", "policy")
    scenario_name = meta.get("scenario_name", "default")
    wrapper, _ = _load_wrapper(model_path, device=device)

    if max_steps is not None:
        steps = steps[:max_steps]

    attrs_list = []
    for s in steps:
        obs = np.asarray(s.get("obs"), dtype=np.float32)
        if obs.size == 0:
            continue
        names = validate_obs_dim(len(obs))
        act = int(s.get("action", _predict_action(wrapper, obs, device)))
        attr = _compute_attribution(wrapper, obs, act, method, device)
        attrs_list.append(np.abs(attr))

    if not attrs_list:
        raise ValueError("轨迹中无 obs 字段，请用新版 record_episode 重新录制。")

    mean_abs = np.mean(np.stack(attrs_list, axis=0), axis=0)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{policy_name}_{scenario_name}_episode_importance.csv")
    png_path = os.path.join(save_dir, f"{policy_name}_{scenario_name}_episode_importance.png")
    save_importance_csv(csv_path, names, mean_abs)
    plot_importance_bar(
        names,
        mean_abs,
        png_path,
        title=f"{policy_name} @ {scenario_name} | episode mean | {method}",
        top_k=12,
    )
    return {
        "policy_name": policy_name,
        "scenario_name": scenario_name,
        "method": method,
        "final_success": meta.get("final_success"),
        "csv_path": csv_path,
        "png_path": png_path,
        "mean_abs_attributions": mean_abs.tolist(),
    }


def compare_success_failure_importance(
    success_importance_path: str,
    failure_importance_path: str,
    save_path: str,
) -> None:
    from explain.explain_utils import compare_success_failure_importance as _cmp

    _cmp(success_importance_path, failure_importance_path, save_path)


def compare_policy_importance(
    importance_paths: list[str],
    policy_names: list[str],
    save_path: str,
) -> None:
    from explain.explain_utils import compare_policy_importance as _cmp

    _cmp(importance_paths, policy_names, save_path)
