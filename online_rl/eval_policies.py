"""
统一评估：在固定测试种子下计算「回合奖励总和」与「胜率」。
用于对比纯在线基线 vs 离线+在线微调策略。
"""
import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 保证 pickle 加载带 BC 正则扩展的微调模型时能解析类路径
import online_rl.ppo_bc_finetune as _ppo_bc_fe  # noqa: F401

from stable_baselines3 import PPO
from envs import make_coop_tracking, load_env_config


def evaluate_policy(model_path, n_episodes=100, seed=2024, deterministic=True):
    """
    评估策略：返回 (平均回合奖励, 胜率)。
    """
    env = make_coop_tracking(seed=seed)
    model = PPO.load(model_path)
    returns = []
    wins = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action) if hasattr(action, "item") else int(action[0]) if np.ndim(action) > 0 else int(action)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            if term or trunc:
                if info.get("win", False):
                    wins += 1
                break
        returns.append(ep_reward)
    env.close()
    return np.mean(returns), wins / n_episodes, np.std(returns)


def main():
    parser = argparse.ArgumentParser(description="评估策略：奖励总和与胜率")
    parser.add_argument("--baseline", type=str, default="models/ppo_online_baseline.zip", help="纯在线基线模型")
    parser.add_argument("--finetune", type=str, default="models/ppo_finetune_from_bc.zip", help="离线+在线微调模型")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    print("=" * 50)
    print("策略评估（回合奖励均值 ± 标准差，胜率）")
    print("=" * 50)

    if os.path.isfile(args.baseline):
        mean_ret, win_rate, std_ret = evaluate_policy(args.baseline, n_episodes=args.n_episodes, seed=args.seed)
        print(f"纯在线基线:   mean_return = {mean_ret:.2f} ± {std_ret:.2f},  win_rate = {win_rate:.2%}")
    else:
        print(f"纯在线基线:   未找到 {args.baseline}，跳过")

    if os.path.isfile(args.finetune):
        mean_ret, win_rate, std_ret = evaluate_policy(args.finetune, n_episodes=args.n_episodes, seed=args.seed)
        print(f"离线+在线:    mean_return = {mean_ret:.2f} ± {std_ret:.2f},  win_rate = {win_rate:.2%}")
    else:
        print(f"离线+在线:    未找到 {args.finetune}，跳过")

    print("=" * 50)


if __name__ == "__main__":
    main()
