#!/usr/bin/env python3
"""
评估三机协同 V3 训练策略（SB3 PPO）。
示例:
  python scripts/eval_v3_policy.py --model_path models/ppo_v3/ppo_coop_v3.zip --n_episodes 100
结果写入 results/v3/（JSON）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from stable_baselines3 import PPO

from envs.coop_tracking_env_v3 import CoopTrackingEnvV3


def load_cfg(path: str | None):
    if not path:
        default = ROOT / "configs" / "ppo_v3.yaml"
        path = str(default)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def evaluate(model: PPO, env_cfg: dict, n_episodes: int, seed: int):
    successes = []
    returns = []
    lengths = []
    collisions = []

    for ep in range(n_episodes):
        e = CoopTrackingEnvV3(config=dict(env_cfg), seed=None)
        obs, _ = e.reset(seed=seed + ep)
        cum = 0.0
        le = 0
        collision_ep = False
        win_ep = False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = e.step(action)
            cum += float(r)
            le += 1
            collision_ep = collision_ep or bool(info.get("collision"))
            if bool(info.get("win")):
                win_ep = True
            if terminated or truncated:
                break

        successes.append(float(win_ep))
        returns.append(cum)
        lengths.append(le)
        collisions.append(float(collision_ep))

    return {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "collision_rate": float(np.mean(collisions)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path",
        type=str,
        default=str(ROOT / "models" / "ppo_v3" / "ppo_coop_v3.zip"),
        help=".zip SB3 PPO",
    )
    ap.add_argument("--n_episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--config", type=str, default="", help="与训练一致的 env 段（ppo_v3.yaml）")
    args = ap.parse_args()

    cfg_path = args.config.strip() or None
    cfg = load_cfg(cfg_path)
    env_cfg = cfg.get("env", {})

    path = Path(args.model_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型: {path}")

    model = PPO.load(str(path))
    metrics = evaluate(model, env_cfg, n_episodes=args.n_episodes, seed=args.seed)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    out_dir = ROOT / "results" / "v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "eval_latest.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"已保存: {out_json}")


if __name__ == "__main__":
    main()
