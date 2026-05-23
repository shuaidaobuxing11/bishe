"""
纯在线 PPO 基线：从零开始在环境中训练，用于与「离线+在线」方法对比。
保证与 finetune 使用相同的总交互步数，便于公平对比奖励与胜率。

场景（与统一评估 `run_unified_eval` 一致）：
  --spawn_mode default|near_uavs|near_border
  --noise_sigma 浮点（与 spawn_mode=default 组合即为 noisy_target 类扰动）
"""
import os
import sys
import argparse
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs import make_coop_tracking, load_env_config
from online_rl.training_curve_callback import TrainingCurveEvalCallback


def _scenario_tag(spawn_mode: str, noise_sigma: float, model_tag: str) -> str:
    if model_tag.strip():
        return model_tag.strip()
    if spawn_mode == "default" and noise_sigma <= 0:
        return ""
    t = spawn_mode
    if noise_sigma > 0:
        t = f"{t}_noise{noise_sigma:g}".replace(".", "p")
    return t


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
    parser.add_argument(
        "--spawn_mode",
        type=str,
        default="default",
        choices=["default", "near_uavs", "near_border"],
        help="初始分布",
    )
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=0.0,
        help="目标扰动强度；与统一评估 noisy_target 可用 0.3",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default="",
        help="保存文件名与曲线名后缀；留空则据 spawn_mode+noise 自动生成",
    )
    parser.add_argument(
        "--curve_method_name",
        type=str,
        default="",
        help="CSV 中 method 列；留空则 ppo_baseline_<tag>",
    )
    args = parser.parse_args()

    _ = load_env_config()

    tag = _scenario_tag(args.spawn_mode, args.noise_sigma, args.model_tag)
    curve_name = args.curve_method_name.strip() or (f"ppo_baseline_{tag}" if tag else "ppo_baseline")
    if tag:
        save_path = os.path.join(args.save_dir, f"ppo_online_baseline_{tag}.zip")
    else:
        save_path = os.path.join(args.save_dir, "ppo_online_baseline.zip")

    env_fn = lambda: make_coop_tracking(
        seed=args.seed, spawn_mode=args.spawn_mode, noise_sigma=args.noise_sigma
    )
    env = DummyVecEnv([env_fn])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
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
                method_name=curve_name,
                eval_seed=args.seed,
                n_episodes=args.curve_n_episodes,
                eval_freq=args.curve_eval_freq,
                out_csv=args.curve_out_csv,
                spawn_mode=args.spawn_mode,
                noise_sigma=args.noise_sigma,
            )
        )
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks[0] if callbacks else None)
    os.makedirs(args.save_dir, exist_ok=True)
    model.save(save_path)
    print(f"纯在线基线已保存: {save_path}")
    env.close()


if __name__ == "__main__":
    main()
