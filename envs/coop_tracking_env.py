"""
双机协同追踪单目标 — Gymnasium 环境（离散动作）
动作空间：每架机 5 种离散动作，组合为 0~24 的单一离散动作。
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _norm(x, y):
    return np.sqrt(x * x + y * y)


class CoopTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # 每架机 5 种动作: 0=保持, 1=左转, 2=右转, 3=加速朝目标, 4=减速
    N_ACTIONS_PER_UAV = 5

    def __init__(self, config=None, max_episode_steps=200, arena_size=10.0,
                 uav_speed_max=0.5, uav_speed_min=0.1, target_speed=0.3,
                 success_distance=0.8, collision_distance=0.5,
                 seed=None, **kwargs):
        super().__init__(**kwargs)
        config = config or {}
        self.max_episode_steps = config.get("max_episode_steps", max_episode_steps)
        self.arena_size = config.get("arena_size", arena_size)
        self.uav_speed_max = config.get("uav_speed_max", uav_speed_max)
        self.uav_speed_min = config.get("uav_speed_min", uav_speed_min)
        self.target_speed = config.get("target_speed", target_speed)
        self.success_distance = config.get("success_distance", success_distance)
        self.collision_distance = config.get("collision_distance", collision_distance)
        # 未见条件测试相关参数
        # spawn_mode: "default" | "near_uavs" | "near_border"
        self.spawn_mode = config.get("spawn_mode", "default")
        # 噪声强度（用于目标随机扰动）
        self.noise_sigma = float(config.get("noise_sigma", 0.0))
        self._step_count = 0

        # 状态: [uav1_x, uav1_y, uav1_vx, uav1_vy, uav2_x, uav2_y, uav2_vx, uav2_vy, target_x, target_y] 归一化到 [-1,1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(10,), dtype=np.float32
        )
        # 离散动作 0~24: a1*5 + a2
        self.action_space = spaces.Discrete(self.N_ACTIONS_PER_UAV ** 2)
        self.np_random = np.random.default_rng(seed)

        self._uav1 = np.zeros(4)   # x, y, vx, vy
        self._uav2 = np.zeros(4)
        self._target = np.zeros(2) # x, y

    def _get_obs(self):
        s = self.arena_size
        obs = np.concatenate([
            self._uav1 / s,
            self._uav2 / s,
            self._target / s
        ]).astype(np.float32)
        return obs

    def _spawn(self):
        s = self.arena_size * 0.8
        # 默认：三者在 [-0.8A, 0.8A] 均匀随机
        if self.spawn_mode == "near_uavs":
            # 双机初始间距更近：中心附近，间距略大于碰撞半径
            center = self.np_random.uniform(-s * 0.5, s * 0.5, 2)
            angle = self.np_random.uniform(0, 2 * np.pi)
            dist = self.np_random.uniform(self.collision_distance * 1.05,
                                          self.collision_distance * 1.5)
            offset = np.array([np.cos(angle), np.sin(angle)]) * dist
            self._uav1[:2] = center - 0.5 * offset
            self._uav2[:2] = center + 0.5 * offset
            self._target[:] = self.np_random.uniform(-s, s, 2)
        elif self.spawn_mode == "near_border":
            # 更靠近边界的位置：目标靠近某一侧边界，双机在其内侧附近
            side = self.np_random.integers(0, 4)
            if side == 0:  # 右边界 x = +A
                self._target[0] = self.arena_size * 0.9
                self._target[1] = self.np_random.uniform(-s, s)
                normal = np.array([-1.0, 0.0])
            elif side == 1:  # 左边界 x = -A
                self._target[0] = -self.arena_size * 0.9
                self._target[1] = self.np_random.uniform(-s, s)
                normal = np.array([1.0, 0.0])
            elif side == 2:  # 上边界 y = +A
                self._target[1] = self.arena_size * 0.9
                self._target[0] = self.np_random.uniform(-s, s)
                normal = np.array([0.0, -1.0])
            else:  # 下边界 y = -A
                self._target[1] = -self.arena_size * 0.9
                self._target[0] = self.np_random.uniform(-s, s)
                normal = np.array([0.0, 1.0])
            base = self._target - normal * 2.0
            jitter = self.np_random.uniform(-1.0, 1.0, 2)
            self._uav1[:2] = base + jitter
            self._uav2[:2] = base - jitter
        else:
            self._uav1[:2] = self.np_random.uniform(-s, s, 2)
            self._uav2[:2] = self.np_random.uniform(-s, s, 2)
            self._target[:] = self.np_random.uniform(-s, s, 2)
        # 初始速度朝向目标
        for uav, pos in [(self._uav1, self._uav1[:2]), (self._uav2, self._uav2[:2])]:
            to = self._target - pos
            d = _norm(to[0], to[1])
            if d > 1e-6:
                v = (self.uav_speed_min + self.uav_speed_max) / 2
                uav[2] = to[0] / d * v
                uav[3] = to[1] / d * v
            else:
                uav[2] = uav[3] = 0.0

    def _decode_action(self, action):
        a1 = action // self.N_ACTIONS_PER_UAV
        a2 = action % self.N_ACTIONS_PER_UAV
        return a1, a2

    def _apply_uav_action(self, uav, a, is_uav1):
        x, y, vx, vy = uav[0], uav[1], uav[2], uav[3]
        to_t = self._target - uav[:2]
        d_t = _norm(to_t[0], to_t[1])
        v = _norm(vx, vy)
        if a == 0:  # 保持
            pass
        elif a == 1:  # 左转
            if v > 1e-6:
                c, s = -vy / v, vx / v
                uav[2], uav[3] = vx * c - vy * s, vx * s + vy * c
        elif a == 2:  # 右转
            if v > 1e-6:
                c, s = vy / v, -vx / v
                uav[2], uav[3] = vx * c - vy * s, vx * s + vy * c
        elif a == 3:  # 加速朝目标
            if d_t > 1e-6:
                v_new = min(self.uav_speed_max, v + 0.05)
                uav[2] = to_t[0] / d_t * v_new
                uav[3] = to_t[1] / d_t * v_new
        else:  # 4 减速
            v_new = max(self.uav_speed_min, v - 0.05)
            if v > 1e-6:
                uav[2] *= v_new / v
                uav[3] *= v_new / v
        # 边界
        uav[0] = np.clip(uav[0] + uav[2], -self.arena_size, self.arena_size)
        uav[1] = np.clip(uav[1] + uav[3], -self.arena_size, self.arena_size)

    def _move_target(self):
        # 简单随机游走 + 可选高斯扰动（用于未见条件测试）
        self._target[:] += self.np_random.uniform(-self.target_speed, self.target_speed, 2)
        if self.noise_sigma > 0.0:
            self._target[:] += self.np_random.normal(0.0, self.noise_sigma, 2)
        self._target[0] = np.clip(self._target[0], -self.arena_size, self.arena_size)
        self._target[1] = np.clip(self._target[1], -self.arena_size, self.arena_size)

    def step(self, action):
        a1, a2 = self._decode_action(action)
        self._apply_uav_action(self._uav1, a1, True)
        self._apply_uav_action(self._uav2, a2, False)
        self._move_target()
        self._step_count += 1

        d1 = _norm(self._uav1[0] - self._target[0], self._uav1[1] - self._target[1])
        d2 = _norm(self._uav2[0] - self._target[0], self._uav2[1] - self._target[1])
        d12 = _norm(self._uav1[0] - self._uav2[0], self._uav1[1] - self._uav2[1])

        reward = - (d1 + d2) * 0.1
        collision = d12 < self.collision_distance
        if collision:
            reward -= 1.0
        if d1 < self.success_distance and d2 < self.success_distance:
            reward += 2.0

        terminated = (d1 < self.success_distance and d2 < self.success_distance) or self._step_count >= self.max_episode_steps
        truncated = self._step_count >= self.max_episode_steps
        won = d1 < self.success_distance and d2 < self.success_distance

        return self._get_obs(), float(reward), terminated, truncated, {"win": won, "collision": collision, "d1": d1, "d2": d2, "d12": d12}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._step_count = 0
        self._spawn()
        return self._get_obs(), {}
