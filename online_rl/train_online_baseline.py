"""
纯在线 PPO 基线：从零开始在环境中训练，用于与「离线+在线」方法对比。
保证与 finetune 使用相同的总交互步数，便于公平对比奖励与胜率。
"""
import os
import sys
import argparse
import yaml
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs import make_coop_tracking, load_env_config
from online_rl.training_curve_callback import TrainingCurveEvalCallback


def main():
    parser = argparse.ArgumentParser(description="纯在线 PPO 基线")
    parser.add_argument("--total_timesteps", type=int, default=200_000, help="总环境交互步数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--curve_eval_freq", type=int, default=20_000, help="训练期间评估间隔（steps）")
    parser.add_argument("--curve_n_episodes", type=int, default=10, help="每次评估的 episode 数")
    parser.add_argument("--curve_out_csv", type=str, default="results/training_curves.csv", help="训练曲线 CSV 输出路径")
    parser.add_argument("--no_curve_logging", action="store_true", help="不记录 success_rate/return 训练曲线")
    args = parser.parse_args()

    config = load_env_config()
    env_fn = lambda: make_coop_tracking(seed=args.seed)
    env = DummyVecEnv([env_fn])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        # 与 BC 网络一致：ReLU 激活，避免把 ReLU 权重加载到默认 Tanh 网络导致初始策略失配
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
        seed=args.seed,
    )

    callbacks = []
    if not args.no_curve_logging:
        callbacks.append(
            TrainingCurveEvalCallback(
                method_name="ppo_baseline",
                eval_seed=args.seed,
                n_episodes=args.curve_n_episodes,
                eval_freq=args.curve_eval_freq,
                out_csv=args.curve_out_csv,
            )
        )
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks[0] if callbacks else None)
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "ppo_online_baseline.zip")
    model.save(path)
    print(f"纯在线基线已保存: {path}")
    env.close()


if __name__ == "__main__":
    main()
