"""
用规则策略在环境中采集轨迹，生成离线数据集。
规则：两机始终朝目标加速/保持，简单避碰（太近时一侧减速）。
"""
import os
import sys
import numpy as np

# 项目根目录加入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import CoopTrackingEnv, load_env_config


def _norm(x, y):
    return np.sqrt(x * x + y * y)


def rule_policy(env: CoopTrackingEnv, obs: np.ndarray) -> int:
    """规则策略（旧版）：仅加速/保持/减速，无转向。"""
    s = env.arena_size
    u1 = obs[:4] * s   # x,y,vx,vy
    u2 = obs[4:8] * s
    target = obs[8:10] * s
    d1 = _norm(target[0] - u1[0], target[1] - u1[1])
    d2 = _norm(target[0] - u2[0], target[1] - u2[1])
    d12 = _norm(u2[0] - u1[0], u2[1] - u1[1])
    n = env.N_ACTIONS_PER_UAV  # 5
    # 0=保持 1=左转 2=右转 3=加速朝目标 4=减速
    def one_action(toward_target_xy, speed, too_close_to_other):
        if too_close_to_other:
            return 4  # 减速
        if _norm(toward_target_xy[0], toward_target_xy[1]) < 0.01:
            return 0
        if speed < (env.uav_speed_min + env.uav_speed_max) / 2:
            return 3  # 加速朝目标
        return 0
    toward1 = target - u1[:2]
    toward2 = target - u2[:2]
    too_close = d12 < env.collision_distance * 1.5
    a1 = one_action(toward1, _norm(u1[2], u1[3]), too_close)
    a2 = one_action(toward2, _norm(u2[2], u2[3]), too_close)
    return int(a1 * n + a2)


def rule_policy_v2(env: CoopTrackingEnv, obs: np.ndarray) -> int:
    """
    带转向 + 协同追踪 + 避碰的规则专家。

    UAV1：直接追踪目标 t
      - 若与目标方向夹角偏左/右 -> 左转(1)/右转(2)
      - 若方向已基本对准，且距离大 -> 加速(3)
      - 若方向已对准，且距离很近 -> 减速(4)
      - 其余情况 -> 保持(0)

    UAV2：协同追踪
      - 先计算 UAV1 指向目标的方向向量 d1
      - 在目标 t 左/右侧生成一个偏移目标点 t'（距离约 1.5m）
      - UAV2 追踪 t' 而不是 t
      - 若与 UAV1 太近，则优先减速（简单避碰）
    """
    s = env.arena_size
    u1 = obs[:4] * s   # (x1, y1, vx1, vy1)
    u2 = obs[4:8] * s  # (x2, y2, vx2, vy2)
    target = obs[8:10] * s  # (xt, yt)

    p1 = u1[:2]
    v1 = u1[2:]
    p2 = u2[:2]
    v2 = u2[2:]

    d1 = _norm(target[0] - p1[0], target[1] - p1[1])
    d2 = _norm(target[0] - p2[0], target[1] - p2[1])
    d12 = _norm(p2[0] - p1[0], p2[1] - p1[1])

    n = env.N_ACTIONS_PER_UAV  # 5
    far_thresh = 2.0 * env.success_distance
    near_thresh = 0.7 * env.success_distance
    danger_close = d12 < env.collision_distance * 1.2

    def chase_point(pos, vel, goal, danger=False):
        # 避碰优先：过近时简单减速
        if danger:
            return 4
        gx, gy = goal[0] - pos[0], goal[1] - pos[1]
        dist = _norm(gx, gy)
        if dist < 1e-6:
            return 0
        vx, vy = vel[0], vel[1]
        speed = _norm(vx, vy)
        # 静止时先获得一个朝向目标的速度
        if speed < 1e-6:
            return 3
        tx, ty = gx / dist, gy / dist
        # 叉积判断左/右
        cross = vx * ty - vy * tx
        align_thresh = 0.1
        if abs(cross) > align_thresh:
            return 1 if cross > 0 else 2
        # 已基本对准，用距离决定加速/减速/保持
        if dist > far_thresh:
            return 3
        if dist < near_thresh:
            return 4
        return 0

    # UAV1：直接追 t
    a1 = chase_point(p1, v1, target, danger=danger_close)

    # UAV2：协同追踪，追偏移目标点 t'
    d1_vec = np.array([target[0] - p1[0], target[1] - p1[1]], dtype=np.float32)
    d1_norm = _norm(d1_vec[0], d1_vec[1])
    if d1_norm < 1e-6:
        offset_goal = target.copy()
    else:
        # 取 d1 的左侧方向作为偏移方向
        perp = np.array([-d1_vec[1] / d1_norm, d1_vec[0] / d1_norm], dtype=np.float32)
        offset_goal = target + 1.5 * perp

    a2 = chase_point(p2, v2, offset_goal, danger=danger_close)

    return int(a1 * n + a2)


def _get_expert_policy(name):
    """name in ('v1', 'v2') -> 对应规则策略函数。"""
    if name == "v2":
        return rule_policy_v2
    return rule_policy


def collect_offline_data(
    n_episodes=500,
    max_steps=200,
    save_dir=None,
    config=None,
    seed=42,
    expert="v2",
):
    """
    expert: "v1" 原规则（仅加速/保持/减速），"v2" 带转向+更合理避碰（默认）。
    """
    config = config or load_env_config()
    config["max_episode_steps"] = config.get("max_episode_steps", max_steps)
    env = CoopTrackingEnv(config=config, seed=seed)
    policy_fn = _get_expert_policy(expert)
    buffer = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(max_steps):
            a = policy_fn(env, obs)
            next_obs, r, term, trunc, _ = env.step(a)
            buffer.append((obs.copy(), a, r, next_obs.copy(), term or trunc))
            obs = next_obs
            if term or trunc:
                break
    # 转为 ReplayBuffer 并保存
    from offline_rl.replay_buffer import ReplayBuffer
    rb = ReplayBuffer(capacity=len(buffer), obs_shape=(10,), action_dim=())
    for (o, a, r, no, d) in buffer:
        rb.add(o, a, r, no, d)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "offline_trajectories.npz")
        rb.save(path)
        print(f"Saved {len(rb)} transitions to {path}")
    return rb


def collect_episodes_with_stats(
    n_episodes=100,
    max_steps=200,
    save_dir=None,
    save_name="expert_episodes.npz",
    config=None,
    seed=42,
    expert="v2",
):
    """
    循环采样环境，保存 episode 级数据与统计。
    expert: "v1" 原规则，"v2" 带转向+更合理避碰（默认）。

    输出 npz 字段：
    - obs, actions, rewards, next_obs, dones: transition 级数组
    - episode_id, timestep: 每条 transition 所属回合与步号
    - episode_returns, episode_lengths, episode_wins: episode 级统计
    """
    config = config or load_env_config()
    config["max_episode_steps"] = config.get("max_episode_steps", max_steps)
    env = CoopTrackingEnv(config=config, seed=seed)
    policy_fn = _get_expert_policy(expert)

    obs_list, act_list, rew_list, next_obs_list, done_list = [], [], [], [], []
    ep_id_list, t_list = [], []
    ep_returns, ep_lengths, ep_wins = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        ep_len = 0
        won = False

        for t in range(max_steps):
            a = policy_fn(env, obs)
            next_obs, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)

            obs_list.append(obs.copy())
            act_list.append(int(a))
            rew_list.append(float(r))
            next_obs_list.append(next_obs.copy())
            done_list.append(done)
            ep_id_list.append(ep)
            t_list.append(t)

            ep_ret += float(r)
            ep_len += 1
            obs = next_obs

            if done:
                won = bool(info.get("win", False))
                break

        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)
        ep_wins.append(1 if won else 0)

    out = {
        "obs": np.asarray(obs_list, dtype=np.float32),
        "actions": np.asarray(act_list, dtype=np.int64),
        "rewards": np.asarray(rew_list, dtype=np.float32),
        "next_obs": np.asarray(next_obs_list, dtype=np.float32),
        "dones": np.asarray(done_list, dtype=np.float32),
        "episode_id": np.asarray(ep_id_list, dtype=np.int32),
        "timestep": np.asarray(t_list, dtype=np.int32),
        "episode_returns": np.asarray(ep_returns, dtype=np.float32),
        "episode_lengths": np.asarray(ep_lengths, dtype=np.int32),
        "episode_wins": np.asarray(ep_wins, dtype=np.int32),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, save_name)
        np.savez_compressed(path, **out)
        print(f"Saved episodes to {path}")

    stats = {
        "success_rate": float(np.mean(out["episode_wins"])) if len(ep_wins) else 0.0,
        "return_mean": float(np.mean(out["episode_returns"])) if len(ep_returns) else 0.0,
        "return_std": float(np.std(out["episode_returns"])) if len(ep_returns) else 0.0,
        "len_mean": float(np.mean(out["episode_lengths"])) if len(ep_lengths) else 0.0,
        "len_std": float(np.std(out["episode_lengths"])) if len(ep_lengths) else 0.0,
    }
    return out, stats


if __name__ == "__main__":
    collect_offline_data(n_episodes=100, save_dir="data/offline")
