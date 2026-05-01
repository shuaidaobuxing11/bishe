"""
环境回放测试：加载 BC 策略，单独跑 N 局并统计指标。

指标：
- success_rate（胜率，info['win']）
- mean_return（平均回合回报）
- mean_length（平均步长）
- collision_rate_steps（碰撞步比例，info['collision'] 为 True 的步数 / 总步数）
- collision_rate_episodes（发生过碰撞的回合比例）

用法：
python scripts/eval_bc_policy.py --bc_path models/bc_pretrain.pt --n_episodes 200
"""
import os
import sys
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Windows 下常见：libomp.dll 与 libiomp5md.dll 冲突导致 OMP Error #15
# 这里用官方提示的应急开关，保证脚本可跑（如需更“干净”的修复，见终端设置/重装建议）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

from envs import make_coop_tracking
from offline_rl.behavior_cloning import load_bc_model


def eval_bc(
    bc_path: str,
    n_episodes=200,
    seed=2024,
    deterministic=True,
    device="cpu",
    save_episodes_path: str | None = None,
):
    env = make_coop_tracking(seed=seed)
    model = load_bc_model(bc_path, device=device)

    returns = []
    lengths = []
    wins = 0
    collision_steps = 0
    total_steps = 0
    collision_episodes = 0

    # 如需保存完整 episode 数据，则收集轨迹
    episodes = [] if save_episodes_path is not None else None

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        ep_len = 0
        ep_collision = False
        ep_obs = []
        ep_actions = []
        ep_rewards = []
        while True:
            a = int(model.predict(obs, deterministic=deterministic))
            next_obs, r, term, trunc, info = env.step(a)
            if episodes is not None:
                ep_obs.append(obs.copy())
                ep_actions.append(int(a))
                ep_rewards.append(float(r))
            obs = next_obs
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
        if episodes is not None:
            episodes.append(
                {
                    "ep_index": ep,
                    "obs": np.asarray(ep_obs, dtype=np.float32),
                    "actions": np.asarray(ep_actions, dtype=np.int64),
                    "rewards": np.asarray(ep_rewards, dtype=np.float32),
                    "win": bool(info.get("win", False)),
                    "collision_any": bool(ep_collision),
                    "length": ep_len,
                    "return": ep_ret,
                }
            )

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
    if save_episodes_path is not None and episodes is not None:
        os.makedirs(os.path.dirname(save_episodes_path) or ".", exist_ok=True)
        # 将每个 episode 的统计展开为数组，便于后处理
        ep_win = np.array([int(e["win"]) for e in episodes], dtype=np.int32)
        ep_ret = np.array([float(e["return"]) for e in episodes], dtype=np.float32)
        ep_len = np.array([int(e["length"]) for e in episodes], dtype=np.int32)
        ep_coll = np.array([int(e["collision_any"]) for e in episodes], dtype=np.int32)
        np.savez_compressed(
            save_episodes_path,
            episodes=episodes,
            episode_win=ep_win,
            episode_return=ep_ret,
            episode_length=ep_len,
            episode_collision_any=ep_coll,
        )
        print(f"BC 评估 episode 数据已保存到 {save_episodes_path}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bc_path", type=str, default="models/bc_pretrain.pt")
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--stochastic", action="store_true", help="使用随机采样动作（默认 greedy）")
     # 若指定则保存完整 episode 轨迹，便于后续提取成功/失败案例
    p.add_argument(
        "--save_episodes",
        type=str,
        default=None,
        help="可选：保存评估 episode 数据的 npz 路径，如 results/bc_eval_episodes.npz",
    )
    args = p.parse_args()

    m = eval_bc(
        args.bc_path,
        n_episodes=args.n_episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        device=args.device,
        save_episodes_path=args.save_episodes,
    )

    print("=" * 60)
    print("BC 策略环境回放评估")
    print("=" * 60)
    print(f"success_rate          = {m['success_rate']:.2%}")
    print(f"mean_return ± std     = {m['mean_return']:.2f} ± {m['std_return']:.2f}")
    print(f"mean_length ± std     = {m['mean_length']:.2f} ± {m['std_length']:.2f}")
    print(f"collision_rate_steps  = {m['collision_rate_steps']:.2%}")
    print(f"collision_rate_episodes = {m['collision_rate_episodes']:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()

