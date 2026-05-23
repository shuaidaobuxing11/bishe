"""单 episode 轨迹记录，保存 JSON / CSV；支持同 seed 多策略对比。"""
from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np

from visualization.policy_loaders import BasePolicyWrapper

ACTION_NAMES = {
    0: "keep",
    1: "turn_left",
    2: "turn_right",
    3: "accelerate",
    4: "decelerate",
}


def decode_joint_action(action: int) -> tuple[int, int]:
    a = int(action)
    return a // 5, a % 5


def action_name(a: int) -> str:
    return ACTION_NAMES.get(int(a), f"action_{a}")


def trajectory_basename(policy_name: str, scenario_name: str, seed: int) -> str:
    return f"{policy_name}_{scenario_name}_seed{seed}"


def _scenario_kwargs(scenario_name: str) -> dict[str, Any]:
    mapping = {
        "default": {},
        "near_uavs": {"spawn_mode": "near_uavs"},
        "near_border": {"spawn_mode": "near_border"},
        "noisy_target": {"noise_sigma": 0.3},
    }
    if scenario_name not in mapping:
        raise ValueError(f"未知场景: {scenario_name}，可选 {list(mapping)}")
    return mapping[scenario_name]


def _read_positions(env) -> tuple[float, float, float, float, float, float]:
    u1 = env._uav1  # noqa: SLF001
    u2 = env._uav2  # noqa: SLF001
    tg = env._target  # noqa: SLF001
    return (
        float(u1[0]),
        float(u1[1]),
        float(u2[0]),
        float(u2[1]),
        float(tg[0]),
        float(tg[1]),
    )


def _dist(x1, y1, x2, y2) -> float:
    return float(np.hypot(x1 - x2, y1 - y2))


def _build_step_row(
    *,
    policy_name: str,
    scenario_name: str,
    seed: int | None,
    step: int,
    action: int,
    reward: float,
    cum_reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
    env,
    obs: np.ndarray,
) -> dict[str, Any]:
    u1x, u1y, u2x, u2y, tx, ty = _read_positions(env)
    success_dist = float(getattr(env, "success_distance", 0.8))
    coll_dist = float(getattr(env, "collision_distance", 0.5))
    arena = float(getattr(env, "arena_size", 10.0))

    d1 = float(info.get("d1", _dist(u1x, u1y, tx, ty)))
    d2 = float(info.get("d2", _dist(u2x, u2y, tx, ty)))
    d12 = float(info.get("d12", _dist(u1x, u1y, u2x, u2y)))
    collision = bool(info.get("collision", d12 < coll_dist))
    success = bool(info.get("win", d1 < success_dist and d2 < success_dist))
    done = bool(terminated or truncated)

    u1a, u2a = decode_joint_action(action)
    mean_d = (d1 + d2) / 2.0

    safe_info = {k: v for k, v in info.items() if isinstance(v, (bool, int, float, str))}

    return {
        "policy_name": policy_name,
        "scenario_name": scenario_name,
        "seed": seed,
        "step": step,
        "uav1_x": u1x,
        "uav1_y": u1y,
        "uav2_x": u2x,
        "uav2_y": u2y,
        "target_x": tx,
        "target_y": ty,
        "action": int(action),
        "uav1_action": u1a,
        "uav2_action": u2a,
        "uav1_action_name": action_name(u1a),
        "uav2_action_name": action_name(u2a),
        "reward": float(reward),
        "cum_reward": float(cum_reward),
        "distance_uav1_target": d1,
        "distance_uav2_target": d2,
        "mean_distance_to_target": mean_d,
        "distance_uav1_uav2": d12,
        "success": success,
        "collision": collision,
        "done": done,
        "truncated": bool(truncated),
        "near_border": (
            abs(u1x) > arena * 0.85
            or abs(u1y) > arena * 0.85
            or abs(u2x) > arena * 0.85
            or abs(u2y) > arena * 0.85
        ),
        "obs": np.asarray(obs, dtype=np.float32).tolist(),
        "info": safe_info,
    }


def record_episode(
    env,
    policy: BasePolicyWrapper,
    policy_name: str,
    scenario_name: str,
    save_path: str,
    max_steps: int = 200,
    deterministic: bool = True,
    seed: int | None = None,
    case_id: int | str | None = None,
    use_seed_filename: bool = False,
) -> dict[str, Any]:
    if hasattr(policy, "bind_env"):
        policy.bind_env(env)

    obs, _ = env.reset(seed=seed)
    rows: list[dict[str, Any]] = []
    cum_reward = 0.0
    final_success = False
    had_collision = False

    success_dist = float(getattr(env, "success_distance", 0.8))
    coll_dist = float(getattr(env, "collision_distance", 0.5))
    arena = float(getattr(env, "arena_size", 10.0))

    for step in range(max_steps):
        action = int(policy.predict(obs, deterministic=deterministic))
        next_obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += float(reward)

        row = _build_step_row(
            policy_name=policy_name,
            scenario_name=scenario_name,
            seed=seed,
            step=step,
            action=action,
            reward=float(reward),
            cum_reward=cum_reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            env=env,
            obs=next_obs,
        )
        rows.append(row)
        obs = next_obs

        if row["collision"]:
            had_collision = True
        if row["success"]:
            final_success = True
        if row["done"]:
            break

    meta = {
        "policy_name": policy_name,
        "scenario_name": scenario_name,
        "case_id": str(case_id) if case_id is not None else None,
        "seed": seed,
        "max_steps": max_steps,
        "total_reward": cum_reward,
        "episode_length": len(rows),
        "final_success": final_success,
        "had_collision": had_collision,
        "arena_size": arena,
        "success_distance": success_dist,
        "collision_distance": coll_dist,
    }
    payload = {"meta": meta, "steps": rows}

    if use_seed_filename and seed is not None:
        base = trajectory_basename(policy_name, scenario_name, int(seed))
        if save_path.endswith(".json"):
            json_path = save_path
            csv_path = save_path[:-5] + ".csv"
        elif os.path.isdir(save_path) or not os.path.splitext(save_path)[1]:
            d = save_path.rstrip("/\\")
            json_path = os.path.join(d, base + ".json")
            csv_path = os.path.join(d, base + ".csv")
        else:
            json_path = save_path + ".json"
            csv_path = save_path + ".csv"
    else:
        cid = case_id if case_id is not None else 0
        json_path, csv_path = _resolve_save_paths(save_path, policy_name, scenario_name, cid)

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    _write_csv(csv_path, rows, meta)
    payload["paths"] = {"json": json_path, "csv": csv_path}
    return payload


def record_all_policies_same_seed(
    policies: dict[str, BasePolicyWrapper],
    scenario_name: str,
    seed: int,
    save_dir: str,
    max_steps: int = 200,
    deterministic: bool = True,
) -> dict[str, str]:
    """各策略独立 env、相同 seed/scenario，保存 {policy}_{scenario}_seed{seed}.json。"""
    os.makedirs(save_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for policy_name, policy in policies.items():
        env = make_env_for_scenario(scenario_name, seed=seed, max_steps=max_steps)
        payload = record_episode(
            env,
            policy,
            policy_name=policy_name,
            scenario_name=scenario_name,
            save_path=save_dir,
            max_steps=max_steps,
            deterministic=deterministic,
            seed=seed,
            use_seed_filename=True,
        )
        env.close()
        paths[policy_name] = payload["paths"]["json"]
        print(
            f"[{policy_name}] seed={seed} success={payload['meta']['final_success']} "
            f"len={payload['meta']['episode_length']} -> {paths[policy_name]}"
        )
    return paths


def _resolve_save_paths(
    save_path: str, policy_name: str, scenario_name: str, case_id: int | str
) -> tuple[str, str]:
    base = f"{policy_name}_{scenario_name}_{case_id}"
    if save_path.endswith(".json"):
        json_path = save_path
        csv_path = save_path[:-5] + ".csv"
    elif save_path.endswith(".csv"):
        csv_path = save_path
        json_path = save_path[:-4] + ".json"
    elif os.path.isdir(save_path) or save_path.endswith(("/", "\\")):
        d = save_path.rstrip("/\\")
        json_path = os.path.join(d, base + ".json")
        csv_path = os.path.join(d, base + ".csv")
    else:
        json_path = save_path + ".json"
        csv_path = save_path + ".csv"
    return json_path, csv_path


def _write_csv(csv_path: str, rows: list[dict], meta: dict) -> None:
    if not rows:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write("# empty episode\n")
        return
    fieldnames = [k for k in rows[0].keys() if k not in ("obs", "info")]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
        f.write(f"\n# meta: {json.dumps(meta, ensure_ascii=False)}\n")


def load_trajectory(path: str) -> dict[str, Any]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if path.endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(path, comment="#")
        return {"meta": {}, "steps": df.to_dict(orient="records")}
    raise ValueError(f"不支持的轨迹格式: {path}")


def make_env_for_scenario(scenario_name: str, seed: int | None = None, max_steps: int = 200):
    from envs import make_coop_tracking

    kw = _scenario_kwargs(scenario_name)
    kw.setdefault("max_episode_steps", max_steps)
    return make_coop_tracking(seed=seed, **kw)


def discover_trajectories_for_scenario_seed(
    trajectory_dir: str, scenario_name: str, seed: int
) -> dict[str, str]:
    """查找 {policy}_{scenario}_seed{seed}.json。"""
    found: dict[str, str] = {}
    if not os.path.isdir(trajectory_dir):
        return found
    suffix = f"_{scenario_name}_seed{seed}.json"
    for fn in os.listdir(trajectory_dir):
        if fn.endswith(suffix):
            pname = fn[: -len(suffix)]
            found[pname] = os.path.join(trajectory_dir, fn)
    return found


__all__ = [
    "ACTION_NAMES",
    "decode_joint_action",
    "action_name",
    "trajectory_basename",
    "record_episode",
    "record_all_policies_same_seed",
    "load_trajectory",
    "make_env_for_scenario",
    "discover_trajectories_for_scenario_seed",
    "_scenario_kwargs",
]
