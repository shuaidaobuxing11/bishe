"""
CoopTrackingEnvV3 rollout：供 Streamlit 展示；观测为扁平向量。
策略函数签名为 (env, obs) -> shape (3,) int64（MultiDiscrete）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


PolicyFnV3 = Callable[[Any, np.ndarray], np.ndarray]


def _denorm_positions_from_obs(obs: np.ndarray, arena_size: float) -> tuple:
    """从归一化 obs 中取 UAV1–3、目标 XY（与 coop_tracking_env_v3._get_obs 布局一致）。"""
    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    A = float(arena_size)
    u1 = o[0:2] * A
    u2 = o[7:9] * A
    u3 = o[14:16] * A
    tgt = o[21:23] * A
    return u1, u2, u3, tgt


@dataclass
class EpisodeTraceV3:
    df: pd.DataFrame
    return_sum: float
    length: int
    win: bool
    collision: bool


def rollout_one_episode_v3(
    env: Any,
    act_fn: PolicyFnV3,
    max_steps: int,
    seed: int | None,
) -> EpisodeTraceV3:
    obs, _ = env.reset(seed=seed)
    rows = []
    cum_r = 0.0
    win_e = False
    coll_e = False
    arena = float(getattr(env, "arena_size", 10.0))

    u1, u2, u3, tgt = _denorm_positions_from_obs(obs, arena)

    rows.append(
        {
            "step": 0,
            "action": "-1,-1,-1",
            "reward": 0.0,
            "cum_reward": 0.0,
            "u1_x": u1[0],
            "u1_y": u1[1],
            "u2_x": u2[0],
            "u2_y": u2[1],
            "u3_x": u3[0],
            "u3_y": u3[1],
            "tgt_x": tgt[0],
            "tgt_y": tgt[1],
        }
    )

    for t in range(max_steps):
        action = np.asarray(act_fn(env, obs), dtype=np.int64).reshape(3)
        next_obs, r, terminated, truncated, info = env.step(action)
        cum_r += float(r)
        u1, u2, u3, tgt = _denorm_positions_from_obs(next_obs, arena)
        coll_e = coll_e or bool(info.get("collision"))
        win_e = bool(info.get("win")) or win_e

        rows.append(
            {
                "step": t + 1,
                "reward": float(r),
                "cum_reward": cum_r,
                "action": ",".join(map(str, action.tolist())),
                "u1_x": u1[0],
                "u1_y": u1[1],
                "u2_x": u2[0],
                "u2_y": u2[1],
                "u3_x": u3[0],
                "u3_y": u3[1],
                "tgt_x": tgt[0],
                "tgt_y": tgt[1],
                "d1": info.get("d1"),
                "d2": info.get("d2"),
                "d3": info.get("d3"),
            }
        )
        obs = next_obs
        if terminated or truncated:
            break

    df = pd.DataFrame(rows)
    length = max(0, len(rows) - 1)
    return EpisodeTraceV3(df=df, return_sum=cum_r, length=length, win=win_e, collision=coll_e)


def make_random_policy_v3() -> PolicyFnV3:
    def _p(env: Any, obs: np.ndarray) -> np.ndarray:
        return np.asarray(env.action_space.sample(), dtype=np.int64).reshape(3)

    return _p


def make_ppo_policy_v3(path: str) -> PolicyFnV3:
    from stable_baselines3 import PPO

    model = PPO.load(path)

    def _p(env: Any, obs: np.ndarray) -> np.ndarray:
        a, _ = model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.int64).reshape(3)

    return _p
