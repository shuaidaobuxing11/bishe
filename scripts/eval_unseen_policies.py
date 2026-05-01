"""
在未见条件场景下统一评估多种策略：
- 随机策略
- 规则专家 v2
- BC (行为克隆) 最优模型
- PPO 在线基线
- BC+PPO 微调

未见条件场景示例（基于用户选择）：
2) 更近的双机初始间距        -> spawn_mode = "near_uavs"
4) 更靠近边界的位置          -> spawn_mode = "near_border"
5) 更强的随机扰动（目标噪声） -> noise_sigma > 0

用法示例：
python scripts/eval_unseen_policies.py --n_episodes 500 --seed 2024
"""
import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import online_rl.ppo_bc_finetune as _ppo_bc_fe  # noqa: F401 pickle 兼容微调模型

from envs import make_coop_tracking, load_env_config
from offline_rl.behavior_cloning import load_bc_model
from offline_rl.dataset_builder import rule_policy_v2
from stable_baselines3 import PPO


SCENARIOS = {
    # 场景 2：更近的双机初始间距
    "near_uavs": {"spawn_mode": "near_uavs"},
    # 场景 4：更靠近边界的位置
    "near_border": {"spawn_mode": "near_border"},
    # 场景 5：更强的随机扰动（目标高斯噪声）
    "noisy_target": {"noise_sigma": 0.3},
}


def eval_with_policy(make_env_kwargs, n_episodes, seed, act_fn):
    """
    通用评估循环：给定动作函数 act_fn(env, obs)->int，统计指标。
    """
    env = make_coop_tracking(seed=seed, **make_env_kwargs)

    returns = []
    lengths = []
    wins = 0
    collision_steps = 0
    total_steps = 0
    collision_episodes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        ep_len = 0
        ep_collision = False
        while True:
            a = int(act_fn(env, obs))
            obs, r, term, trunc, info = env.step(a)
            ep_ret += float(r)
            ep_len += 1
            total_steps += 1
            if info.get("collision", False):
                collision_steps += 1
                ep_collision = True
            if term or trunc:
                if info.get("win", False):
                    wins += 1
                if ep_collision:
                    collision_episodes += 1
                break
        returns.append(ep_ret)
        lengths.append(ep_len)

    env.close()
    returns = np.asarray(returns, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)

    return {
        "success_rate": wins / max(n_episodes, 1),
        "mean_return": float(np.mean(returns)) if len(returns) else 0.0,
        "std_return": float(np.std(returns)) if len(returns) else 0.0,
        "mean_length": float(np.mean(lengths)) if len(lengths) else 0.0,
        "std_length": float(np.std(lengths)) if len(lengths) else 0.0,
        "collision_rate_steps": collision_steps / max(total_steps, 1),
        "collision_rate_episodes": collision_episodes / max(n_episodes, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--bc_path", type=str, default="models/bc_pretrain_best.pt")
    p.add_argument("--ppo_baseline", type=str, default="models/ppo_online_baseline.zip")
    p.add_argument("--ppo_finetune", type=str, default="models/ppo_finetune_from_bc.zip")
    args = p.parse_args()

    n_episodes = args.n_episodes
    seed = args.seed

    # 预加载 BC / PPO 模型（若存在）
    bc_model = None
    if os.path.isfile(args.bc_path):
        bc_model = load_bc_model(args.bc_path, device="cpu")
        print(f"Loaded BC model from {args.bc_path}")
    else:
        print(f"BC model {args.bc_path} not found, 将跳过 BC 评估。")

    ppo_baseline = None
    if os.path.isfile(args.ppo_baseline):
        ppo_baseline = PPO.load(args.ppo_baseline)
        print(f"Loaded PPO baseline from {args.ppo_baseline}")
    else:
        print(f"PPO baseline {args.ppo_baseline} not found, 将跳过 PPO baseline。")

    ppo_finetune = None
    if os.path.isfile(args.ppo_finetune):
        ppo_finetune = PPO.load(args.ppo_finetune)
        print(f"Loaded PPO finetune from {args.ppo_finetune}")
    else:
        print(f"PPO finetune {args.ppo_finetune} not found, 将跳过 PPO finetune。")

    methods = ["random", "expert_v2", "bc"]
    if ppo_baseline is not None:
        methods.append("ppo_baseline")
    if ppo_finetune is not None:
        methods.append("ppo_finetune")

    print("=" * 80)
    print("未见条件场景评估 (n_episodes={}, seed={})".format(n_episodes, seed))
    print("场景包含: 2) near_uavs, 4) near_border, 5) noisy_target")
    print("=" * 80)

    for scen_name, scen_kwargs in SCENARIOS.items():
        print("\n" + "-" * 80)
        print(f"Scenario: {scen_name}  (kwargs={scen_kwargs})")
        print("-" * 80)
        for method in methods:
            if method == "random":
                def act_fn(env, obs):
                    return env.action_space.sample()
            elif method == "expert_v2":
                def act_fn(env, obs):
                    return rule_policy_v2(env, obs)
            elif method == "bc":
                if bc_model is None:
                    continue
                def act_fn(env, obs):
                    return bc_model.predict(obs, deterministic=True)
            elif method == "ppo_baseline":
                def act_fn(env, obs):
                    a, _ = ppo_baseline.predict(obs, deterministic=True)
                    return a
            elif method == "ppo_finetune":
                def act_fn(env, obs):
                    a, _ = ppo_finetune.predict(obs, deterministic=True)
                    return a
            else:
                continue

            metrics = eval_with_policy(scen_kwargs, n_episodes, seed, act_fn)
            print(
                f"{method:12s} | "
                f"succ={metrics['success_rate']*100:5.1f}% | "
                f"R={metrics['mean_return']:7.2f}±{metrics['std_return']:6.2f} | "
                f"L={metrics['mean_length']:6.1f}±{metrics['std_length']:5.1f} | "
                f"coll_step={metrics['collision_rate_steps']*100:5.1f}% | "
                f"coll_ep={metrics['collision_rate_episodes']*100:5.1f}%"
            )


if __name__ == "__main__":
    main()

