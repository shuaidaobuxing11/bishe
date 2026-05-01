"""
随机策略评估：在环境中用随机动作跑 N 局，统计基线指标。

指标：
- success_rate（胜率，info['win']）
- mean_return（平均回合回报）
- mean_length（平均步长）
- collision_rate_steps（碰撞步比例）
- collision_rate_episodes（发生过碰撞的回合比例）

用法：
python scripts/eval_random_policy.py --n_episodes 500 --seed 2024
"""
import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from envs import make_coop_tracking


def eval_random(n_episodes=200, seed=2024):
    env = make_coop_tracking(seed=seed)

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
            a = env.action_space.sample()
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

    metrics = {
        "success_rate": wins / max(n_episodes, 1),
        "mean_return": float(np.mean(returns)) if len(returns) else 0.0,
        "std_return": float(np.std(returns)) if len(returns) else 0.0,
        "mean_length": float(np.mean(lengths)) if len(lengths) else 0.0,
        "std_length": float(np.std(lengths)) if len(lengths) else 0.0,
        "collision_rate_steps": collision_steps / max(total_steps, 1),
        "collision_rate_episodes": collision_episodes / max(n_episodes, 1),
    }
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()

    m = eval_random(n_episodes=args.n_episodes, seed=args.seed)

    print("=" * 60)
    print("随机策略环境回放评估")
    print("=" * 60)
    print(f"success_rate          = {m['success_rate']:.2%}")
    print(f"mean_return ± std     = {m['mean_return']:.2f} ± {m['std_return']:.2f}")
    print(f"mean_length ± std     = {m['mean_length']:.2f} ± {m['std_length']:.2f}")
    print(f"collision_rate_steps  = {m['collision_rate_steps']:.2%}")
    print(f"collision_rate_episodes = {m['collision_rate_episodes']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()

