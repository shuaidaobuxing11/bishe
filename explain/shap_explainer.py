"""SHAP 策略解释（可选；依赖 shap 包）。"""
from __future__ import annotations

import os

import numpy as np
import torch

from explain.explain_utils import plot_importance_bar, save_importance_csv
from explain.feature_names import validate_obs_dim
from explain.policy_wrapper import BCLogitsWrapper, SB3PolicyLogitsWrapper


def _load_wrapper(model_path: str, device: str = "cpu"):
    path_lower = model_path.lower()
    if path_lower.endswith(".zip"):
        import online_rl.ppo_bc_finetune as _ppo_bc_fe  # noqa: F401

        from stable_baselines3 import PPO

        model = PPO.load(model_path, device=device)
        return SB3PolicyLogitsWrapper(model).to(device).eval(), "ppo"
    if path_lower.endswith(".pt"):
        from offline_rl.behavior_cloning import load_bc_model

        net = load_bc_model(model_path, device=device)
        return BCLogitsWrapper(net).to(device).eval(), "bc"
    raise ValueError(f"不支持的模型: {model_path}")


def explain_with_shap(
    model_path: str,
    background_obs: np.ndarray,
    explain_obs: np.ndarray,
    save_dir: str = "results/explain",
    policy_name: str = "policy",
    scenario_name: str = "default",
    device: str = "cpu",
) -> dict:
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "SHAP explainer is optional. Please install shap (`pip install shap`) "
            "or use Captum explainer first (`scripts/explain_policy_captum.py`)."
        ) from e

    wrapper, _ = _load_wrapper(model_path, device=device)
    explain_obs = np.asarray(explain_obs, dtype=np.float32).reshape(1, -1)
    background_obs = np.asarray(background_obs, dtype=np.float32)
    names = validate_obs_dim(explain_obs.shape[1])

    def model_fn(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.as_tensor(x, dtype=torch.float32, device=device)
            logits = wrapper(t)
            probs = torch.softmax(logits, dim=-1)
            return probs.cpu().numpy()

    explainer = shap.KernelExplainer(model_fn, background_obs[: min(50, len(background_obs))])
    shap_values = explainer.shap_values(explain_obs, nsamples=100)
    if isinstance(shap_values, list):
        pred = int(model_fn(explain_obs)[0].argmax())
        attr = np.asarray(shap_values[pred]).reshape(-1)
    else:
        attr = np.asarray(shap_values).reshape(explain_obs.shape[1], -1).mean(axis=1)

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{policy_name}_{scenario_name}_shap_importance.csv")
    png_path = os.path.join(save_dir, f"{policy_name}_{scenario_name}_shap_importance.png")
    save_importance_csv(csv_path, names, attr)
    plot_importance_bar(names, attr, png_path, title=f"SHAP | {policy_name} @ {scenario_name}")
    return {"csv_path": csv_path, "png_path": png_path, "attributions": attr.tolist()}
