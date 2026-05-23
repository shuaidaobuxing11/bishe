"""
双机协同追踪环境 rollout：供 Streamlit 等展示用，不修改 envs/offline_rl 训练逻辑。
观测为归一化坐标 obs[0:2]=uav1/s, [4:6]=uav2/s, [8:10]=target/s。
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd


PolicyFn = Callable[[Any, np.ndarray], int]


def obs_to_positions(obs: np.ndarray, arena_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    s = float(arena_size)
    u1 = o[0:2] * s
    u2 = o[4:6] * s
    tgt = o[8:10] * s
    return u1, u2, tgt


def distances_m(u1: np.ndarray, u2: np.ndarray, tgt: np.ndarray) -> tuple[float, float, float]:
    d1 = float(np.linalg.norm(u1 - tgt))
    d2 = float(np.linalg.norm(u2 - tgt))
    d12 = float(np.linalg.norm(u1 - u2))
    return d1, d2, d12


@dataclass
class EpisodeTrace:
    """单回合逐步记录。"""
    df: pd.DataFrame
    return_sum: float
    length: int
    win: bool
    collision: bool


def rollout_one_episode(
    env: Any,
    act_fn: PolicyFn,
    max_steps: int,
    seed: Optional[int],
) -> EpisodeTrace:
    obs, _ = env.reset(seed=seed)
    rows: List[dict] = []
    cum_r = 0.0
    win = False
    collision = False
    arena = float(getattr(env, "arena_size", 10.0))

    # t=0：初始状态（尚未执行动作）
    u1, u2, tgt = obs_to_positions(obs, arena)
    d1, d2, d12 = distances_m(u1, u2, tgt)
    rows.append(
        {
            "step": 0,
            "action": -1,
            "reward": 0.0,
            "cum_reward": 0.0,
            "u1_x": u1[0],
            "u1_y": u1[1],
            "u2_x": u2[0],
            "u2_y": u2[1],
            "tgt_x": tgt[0],
            "tgt_y": tgt[1],
            "d1": d1,
            "d2": d2,
            "d12": d12,
        }
    )

    for t in range(max_steps):
        a = int(act_fn(env, obs))
        next_obs, r, terminated, truncated, info = env.step(a)
        cum_r += float(r)
        # 与 env.step 内奖励一致：用后验状态 next_obs 与 info 中的距离
        u1, u2, tgt = obs_to_positions(next_obs, arena)
        fd1, fd2, fd12 = distances_m(u1, u2, tgt)
        d1 = float(info.get("d1", fd1))
        d2 = float(info.get("d2", fd2))
        d12 = float(info.get("d12", fd12))
        win = bool(info.get("win", False))
        collision = bool(info.get("collision", False))
        rows.append(
            {
                "step": t + 1,
                "action": a,
                "reward": float(r),
                "cum_reward": cum_r,
                "u1_x": u1[0],
                "u1_y": u1[1],
                "u2_x": u2[0],
                "u2_y": u2[1],
                "tgt_x": tgt[0],
                "tgt_y": tgt[1],
                "d1": d1,
                "d2": d2,
                "d12": d12,
            }
        )
        obs = next_obs
        if terminated or truncated:
            break

    df = pd.DataFrame(rows)
    length = max(0, len(rows) - 1)  # env.step 调用次数（不含初始行）
    return EpisodeTrace(
        df=df,
        return_sum=cum_r,
        length=length,
        win=win,
        collision=collision,
    )


def make_random_policy() -> PolicyFn:
    def _fn(env: Any, obs: np.ndarray) -> int:
        return int(env.action_space.sample())

    return _fn


def _rule_policy_v2_standalone():
    """直接加载 dataset_builder，避免 `import offline_rl` 触发 behavior_cloning → torch。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "offline_rl", "dataset_builder.py")
    spec = importlib.util.spec_from_file_location("coop_dataset_builder_demo", path)
    if spec is None or spec.loader is None:
        raise ImportError("无法加载 dataset_builder")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rule_policy_v2


def make_expert_v2_policy() -> PolicyFn:
    rule_policy_v2 = _rule_policy_v2_standalone()

    def _fn(env: Any, obs: np.ndarray) -> int:
        return int(rule_policy_v2(env, obs))

    return _fn


def make_bc_policy(model_path: str, device: str = "cpu") -> PolicyFn:
    from offline_rl.behavior_cloning import load_bc_model

    model = load_bc_model(model_path, device=device)

    def _fn(env: Any, obs: np.ndarray) -> int:
        return int(model.predict(obs, deterministic=True))

    return _fn


def make_ppo_policy(zip_path: str) -> PolicyFn:
    from stable_baselines3 import PPO

    model = PPO.load(zip_path)

    def _fn(env: Any, obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])

    return _fn
