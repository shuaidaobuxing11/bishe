"""
三机协同追踪环境 V3 — SB3 PPO 训练脚本。
配置文件: configs/ppo_v3.yaml

运行（项目根）:
  python scripts/train_ppo_v3.py --config configs/ppo_v3.yaml
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
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.coop_tracking_env_v3 import CoopTrackingEnvV3


def evaluate_episodes(model, env_cfg: dict, n_episodes: int, seed: int):
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
            win_ep = bool(info.get("win")) or win_ep
            if terminated or truncated:
                break

        successes.append(float(win_ep))
        returns.append(cum)
        lengths.append(le)
        collisions.append(float(collision_ep))

    metrics = {
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "collision_rate": float(np.mean(collisions)),
    }
    return metrics


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "ppo_v3.yaml"))
    ap.add_argument("--total_timesteps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--no-curve-logging",
        action="store_true",
        help="不将训练曲线写入 results/v3/training_curves.csv",
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    env_cfg = cfg.get("env", {})
    ppo_cfg = cfg.get("ppo", {})
    tr_cfg = cfg.get("training", {})
    ev_cfg = cfg.get("eval_end", {})
    curve_cfg = cfg.get("curve_eval", {})

    total_timesteps = int(
        args.total_timesteps if args.total_timesteps is not None else tr_cfg.get("total_timesteps", 200_000)
    )
    seed = int(args.seed if args.seed is not None else tr_cfg.get("seed", 42))

    def make_env(seed_i: int | None = None):
        def _init():
            e = CoopTrackingEnvV3(config=dict(env_cfg), seed=seed_i)
            return e

        return _init

    env = DummyVecEnv([make_env(seed)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 256)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
        seed=seed,
    )

    callbacks = []
    if not args.no_curve_logging and curve_cfg.get("enabled", True):
        from online_rl.ppo_v3_training_callback import PPOv3TrainingCurveCallback

        out_rel = curve_cfg.get("out_csv", "results/v3/training_curves.csv")
        callbacks.append(
            PPOv3TrainingCurveCallback(
                env_cfg=dict(env_cfg),
                method_name=str(curve_cfg.get("method_name", "ppo_v3")),
                eval_seed=seed,
                n_episodes=int(curve_cfg.get("n_episodes", 10)),
                eval_freq=int(curve_cfg.get("eval_freq", 20_000)),
                out_csv=str(ROOT / out_rel),
            )
        )

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=int(tr_cfg.get("log_interval", 10)),
        callback=callbacks if callbacks else None,
    )

    model_dir = ROOT / tr_cfg.get("model_dir", "models/ppo_v3")
    model_dir.mkdir(parents=True, exist_ok=True)
    name = tr_cfg.get("model_name", "ppo_coop_v3")
    zip_path = model_dir / f"{name}.zip"
    model.save(str(zip_path))

    n_ep = int(ev_cfg.get("n_episodes", 20))
    metrics = evaluate_episodes(model, env_cfg, n_episodes=n_ep, seed=seed + 9103)

    out_dir = ROOT / "results" / "v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_end_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"模型已保存: {zip_path}")
    print(f"训练结束评估已写入: {metrics_path}")

    env.close()


if __name__ == "__main__":
    main()
