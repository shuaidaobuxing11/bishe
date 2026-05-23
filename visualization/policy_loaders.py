"""统一策略加载：Expert / BC / PPO。"""
from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BasePolicyWrapper(ABC):
    """统一接口：predict(obs) -> joint_action int。"""

    def __init__(self) -> None:
        self._env: Any = None

    def bind_env(self, env: Any) -> None:
        self._env = env

    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int: ...


class ExpertPolicyWrapper(BasePolicyWrapper):
    """rule_policy_v2 专家策略（无需模型文件）。"""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        self.config = config or {}
        from offline_rl.dataset_builder import rule_policy_v2

        self._rule_fn = rule_policy_v2

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        if self._env is None:
            raise RuntimeError("ExpertPolicyWrapper 需要先 bind_env(env)")
        return int(self._rule_fn(self._env, obs))


class BCPolicyWrapper(BasePolicyWrapper):
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        super().__init__()
        from offline_rl.behavior_cloning import load_bc_model

        self.net = load_bc_model(model_path, device=device)
        self.model_path = model_path

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        return int(self.net.predict(obs, deterministic=deterministic))


class PPOPolicyWrapper(BasePolicyWrapper):
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        super().__init__()
        import online_rl.ppo_bc_finetune as _ppo_bc_fe  # noqa: F401 pickle 兼容

        from stable_baselines3 import PPO

        self.model = PPO.load(model_path, device=device)
        self.model_path = model_path

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(np.asarray(action).reshape(-1)[0])


# 向后兼容别名
_SB3PolicyWrapper = PPOPolicyWrapper
_BCPolicyWrapper = BCPolicyWrapper


def _load_yaml_config(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(config: dict | None) -> dict:
    if config is not None:
        return config
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "configs", "visualization_config.yaml")
    return _load_yaml_config(cfg_path)


def load_policy(
    policy_name: str | None = None,
    model_path: str | None = None,
    policy_type: str = "auto",
    config: dict | None = None,
    device: str = "cpu",
) -> BasePolicyWrapper:
    """
    按名称或路径加载策略。
    - policy_name: expert_v2 | bc | ppo_baseline | ppo_finetune | ppo_finetune_kl
    - model_path: 直接指定 .zip / .pt（与 policy_type 配合）
    """
    if policy_name == "expert_v2" or policy_type == "expert":
        return ExpertPolicyWrapper(config=config)

    cfg = _resolve_config_path(config)
    policies_cfg = cfg.get("policies", {})

    if policy_name and policy_name in policies_cfg and model_path is None:
        pc = policies_cfg[policy_name]
        ptype = pc.get("type", "ppo")
        mpath = pc.get("model_path", "")
        if ptype == "expert":
            return ExpertPolicyWrapper(config=config)
        if ptype == "bc":
            if not mpath or not os.path.isfile(mpath):
                raise FileNotFoundError(f"BC 模型不存在: {mpath}")
            return BCPolicyWrapper(mpath, device=device)
        if not mpath or not os.path.isfile(mpath):
            raise FileNotFoundError(f"PPO 模型不存在: {mpath}")
        return PPOPolicyWrapper(mpath, device=device)

    if model_path is None:
        raise ValueError("需要 policy_name 或 model_path")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")

    path_lower = model_path.lower()
    if policy_type == "auto":
        if path_lower.endswith(".zip"):
            policy_type = "ppo"
        elif path_lower.endswith(".pt"):
            policy_type = "bc"
        else:
            raise ValueError(f"无法推断策略类型: {model_path}")

    if policy_type == "ppo":
        return PPOPolicyWrapper(model_path, device=device)
    if policy_type == "bc":
        return BCPolicyWrapper(model_path, device=device)
    if policy_type == "expert":
        return ExpertPolicyWrapper(config=config)
    raise ValueError(f"未知 policy_type: {policy_type}")


def load_default_policies(config: dict | None = None, device: str = "cpu") -> dict[str, BasePolicyWrapper]:
    """从 visualization_config.yaml 加载全部可用策略；缺失模型则 warning 并跳过。"""
    cfg = _resolve_config_path(config)
    policies_cfg = cfg.get("policies", {})
    if not policies_cfg:
        policies_cfg = {
            "expert_v2": {"type": "expert", "enabled": True},
            "bc": {"type": "bc", "enabled": True, "model_path": "models/bc_pretrain_best.pt"},
            "ppo_baseline": {"type": "ppo", "enabled": True, "model_path": "models/ppo_online_baseline.zip"},
            "ppo_finetune": {"type": "ppo", "enabled": True, "model_path": "models/ppo_finetune_from_bc.zip"},
            "ppo_finetune_kl": {"type": "ppo", "enabled": True, "model_path": "models/ppo_finetune_kl.zip"},
        }

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out: dict[str, BasePolicyWrapper] = {}

    for name, pc in policies_cfg.items():
        if not pc.get("enabled", True):
            continue
        ptype = pc.get("type", "ppo")
        try:
            if ptype == "expert":
                out[name] = ExpertPolicyWrapper(config=pc)
                continue
            mpath = pc.get("model_path", "")
            if not os.path.isabs(mpath):
                mpath = os.path.join(root, mpath)
            if not os.path.isfile(mpath):
                alt = None
                if name == "bc":
                    for cand in ("models/bc_pretrain.pt", "models/bc_pretrain_best.pt"):
                        p = os.path.join(root, cand)
                        if os.path.isfile(p):
                            alt = p
                            break
                if alt:
                    mpath = alt
                else:
                    warnings.warn(f"[load_default_policies] 跳过 {name}：模型不存在 {mpath}")
                    continue
            if ptype == "bc":
                out[name] = BCPolicyWrapper(mpath, device=device)
            else:
                out[name] = PPOPolicyWrapper(mpath, device=device)
        except Exception as exc:
            warnings.warn(f"[load_default_policies] 跳过 {name}: {exc}")

    return out
