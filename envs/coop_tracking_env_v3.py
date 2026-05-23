"""
三机协同追踪单目标 — Gymnasium（MultiDiscrete 独立决策）。
动作：每架机 5 种离散动作 MultiDiscrete([5,5,5])，禁止使用 5³ 单一联合离散。
动力学：航向角 + 标量速度，左转/右转/加减速后与双机语义一致。
观测：扁平 float32 向量（未使用 dict）。
"""
from __future__ import annotations

import warnings

import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _norm2(dx: float, dy: float) -> float:
    return float(np.sqrt(dx * dx + dy * dy))


def _wrap_angle(a: float) -> float:
    return float(((a + np.pi) % (2 * np.pi)) - np.pi)


class CoopTrackingEnvV3(gym.Env):
    metadata = {"render_modes": ["human"]}

    N_ACTIONS_PER_UAV = 5
    SUCCESS_BONUS = 100.0
    COLLISION_PENALTY = 30.0
    BORDER_PENALTY = 5.0

    def __init__(
        self,
        config=None,
        max_episode_steps=200,
        arena_size=10.0,
        num_uavs=3,
        uav_speed_max=0.5,
        uav_speed_min=0.1,
        target_speed=0.3,
        success_distance=0.8,
        collision_distance=0.5,
        turn_angle=0.12,
        accel=0.05,
        target_heading_noise_std=0.05,
        desired_pair_distance=2.5,
        formation_coef=0.08,
        seed=None,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.render_mode = render_mode
        config = config or {}
        self.max_episode_steps = int(config.get("max_episode_steps", max_episode_steps))
        self.arena_size = float(config.get("arena_size", arena_size))
        self.num_uavs = int(config.get("num_uavs", num_uavs))
        assert self.num_uavs == 3
        self.uav_speed_max = float(config.get("uav_speed_max", uav_speed_max))
        self.uav_speed_min = float(config.get("uav_speed_min", uav_speed_min))
        self.target_speed = float(config.get("target_speed", target_speed))
        self.success_distance = float(config.get("success_distance", success_distance))
        self.collision_distance = float(config.get("collision_distance", collision_distance))
        self.turn_angle = float(config.get("turn_angle", turn_angle))
        self.accel = float(config.get("accel", accel))
        self.target_heading_noise_std = float(
            config.get("target_heading_noise_std", target_heading_noise_std)
        )
        self.desired_pair_distance = float(
            config.get("desired_pair_distance", desired_pair_distance)
        )
        self.formation_coef = float(config.get("formation_coef", formation_coef))

        self.np_random = np.random.default_rng(seed)

        obs_dim = self.num_uavs * 7 + 5 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            np.full(self.num_uavs, self.N_ACTIONS_PER_UAV, dtype=np.int64)
        )

        self._step_count = 0
        # UAV: [x,y,heading,speed]
        self._uav = np.zeros((self.num_uavs, 4), dtype=np.float64)
        self._target = np.zeros(3, dtype=np.float64)  # x,y,heading
        self._last_info: dict = {}

        self._trail_uavs: list[list[np.ndarray]] = []
        self._trail_tgt: list[np.ndarray] = []
        self._last_outcome: str = ""

    def _pairwise_distances(self) -> tuple[float, float, float]:
        p = self._uav[:, :2]
        d12 = _norm2(p[0, 0] - p[1, 0], p[0, 1] - p[1, 1])
        d13 = _norm2(p[0, 0] - p[2, 0], p[0, 1] - p[2, 1])
        d23 = _norm2(p[1, 0] - p[2, 0], p[1, 1] - p[2, 1])
        return d12, d13, d23

    def _distances_to_target(self) -> tuple[float, float, float]:
        tx, ty = self._target[0], self._target[1]
        d = []
        for i in range(self.num_uavs):
            d.append(_norm2(self._uav[i, 0] - tx, self._uav[i, 1] - ty))
        return float(d[0]), float(d[1]), float(d[2])

    def _get_obs(self) -> np.ndarray:
        A = self.arena_size
        vmax = max(self.uav_speed_max, 1e-6)
        parts: list[np.ndarray] = []
        tx, ty, th = self._target
        tvx = np.cos(th) * self.target_speed
        tvy = np.sin(th) * self.target_speed

        dists = np.array(self._distances_to_target(), dtype=np.float32)
        for i in range(self.num_uavs):
            x, y, h, sp = self._uav[i]
            vx = np.cos(h) * sp
            vy = np.sin(h) * sp
            parts.append(
                np.array(
                    [
                        x / A,
                        y / A,
                        h / np.pi,
                        sp / vmax,
                        vx / vmax,
                        vy / vmax,
                        float(dists[i]) / A,
                    ],
                    dtype=np.float32,
                )
            )
        parts.append(
            np.array(
                [tx / A, ty / A, tvx / vmax, tvy / vmax, th / np.pi],
                dtype=np.float32,
            )
        )
        d12, d13, d23 = self._pairwise_distances()
        parts.append(
            np.array([d12 / A, d13 / A, d23 / A], dtype=np.float32)
        )
        return np.concatenate(parts, axis=0)

    def _spawn(self) -> None:
        s = self.arena_size * 0.8
        for i in range(self.num_uavs):
            self._uav[i, 0] = self.np_random.uniform(-s, s)
            self._uav[i, 1] = self.np_random.uniform(-s, s)
        self._target[0] = self.np_random.uniform(-s, s)
        self._target[1] = self.np_random.uniform(-s, s)
        self._target[2] = float(self.np_random.uniform(-np.pi, np.pi))

        tx, ty = self._target[0], self._target[1]
        for i in range(self.num_uavs):
            dx, dy = tx - self._uav[i, 0], ty - self._uav[i, 1]
            d = _norm2(dx, dy)
            if d > 1e-6:
                h = float(np.arctan2(dy, dx))
                self._uav[i, 2] = h
                self._uav[i, 3] = (self.uav_speed_min + self.uav_speed_max) / 2.0
            else:
                self._uav[i, 2] = float(self.np_random.uniform(-np.pi, np.pi))
                self._uav[i, 3] = self.uav_speed_min

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._step_count = 0
        self._last_outcome = ""
        self._spawn()
        self._trail_uavs = [[] for _ in range(self.num_uavs)]
        self._trail_tgt = []
        for i in range(self.num_uavs):
            self._trail_uavs[i].append(self._uav[i, :2].copy())
        self._trail_tgt.append(np.array([self._target[0], self._target[1]], copy=True))

        obs = self._get_obs()
        self._last_info = {}
        return obs, {}

    def _outside_arena(self, x: float, y: float) -> bool:
        return abs(x) > self.arena_size or abs(y) > self.arena_size

    def _apply_action_uav(self, idx: int, a: int) -> None:
        x, y, h, sp = self._uav[idx]
        if int(a) == 0:
            pass
        elif int(a) == 1:
            h = _wrap_angle(h - self.turn_angle)
        elif int(a) == 2:
            h = _wrap_angle(h + self.turn_angle)
        elif int(a) == 3:
            sp = min(self.uav_speed_max, sp + self.accel)
        else:
            sp = max(self.uav_speed_min, sp - self.accel)
        vx = np.cos(h) * sp
        vy = np.sin(h) * sp
        x = x + vx
        y = y + vy
        self._uav[idx] = np.array([x, y, h, sp], dtype=np.float64)

    def _move_target(self) -> None:
        th = self._target[2]
        th = _wrap_angle(th + float(self.np_random.normal(0.0, self.target_heading_noise_std)))
        self._target[2] = th
        self._target[0] += np.cos(th) * self.target_speed
        self._target[1] += np.sin(th) * self.target_speed
        self._target[0] = float(np.clip(self._target[0], -self.arena_size, self.arena_size))
        self._target[1] = float(np.clip(self._target[1], -self.arena_size, self.arena_size))

    def step(self, action):
        self._step_count += 1
        act = np.asarray(action, dtype=np.int64).reshape(-1)
        if act.shape[0] != self.num_uavs:
            raise ValueError(f"action length {act.shape[0]} != num_uavs {self.num_uavs}")

        for i in range(self.num_uavs):
            self._apply_action_uav(i, int(act[i]))
        self._move_target()

        for i in range(self.num_uavs):
            self._trail_uavs[i].append(self._uav[i, :2].copy())
        self._trail_tgt.append(np.array([self._target[0], self._target[1]], copy=True))

        d1, d2, d3 = self._distances_to_target()
        mean_d = (d1 + d2 + d3) / 3.0
        reward = -mean_d * 0.1

        d12, d13, d23 = self._pairwise_distances()
        pair_list = [(0, 1, d12), (0, 2, d13), (1, 2, d23)]
        collision_any = any(d_ij < self.collision_distance for _, _, d_ij in pair_list)
        if collision_any:
            reward -= self.COLLISION_PENALTY

        severe_collision = min(d12, d13, d23) < self.collision_distance * 0.35

        border_hits = 0
        for i in range(self.num_uavs):
            xi, yi = float(self._uav[i, 0]), float(self._uav[i, 1])
            if self._outside_arena(xi, yi):
                border_hits += 1
                reward -= self.BORDER_PENALTY

        form_pen = (
            abs(d12 - self.desired_pair_distance)
            + abs(d13 - self.desired_pair_distance)
            + abs(d23 - self.desired_pair_distance)
        ) / 3.0
        reward -= self.formation_coef * form_pen

        close = sum(1 for d in (d1, d2, d3) if d < self.success_distance)
        success = close >= 2
        win = False
        if success:
            reward += self.SUCCESS_BONUS
            win = True
            self._last_outcome = "success"

        all_outside = border_hits == self.num_uavs
        if all_outside and not success:
            self._last_outcome = "fail_all_out"

        terminated = False
        if success:
            terminated = True
        elif severe_collision:
            terminated = True
            self._last_outcome = "fail_severe_collision"
        elif all_outside:
            terminated = True
            self._last_outcome = "fail_all_out"

        truncated = self._step_count >= self.max_episode_steps and not terminated
        if truncated:
            self._last_outcome = "timeout"

        obs = self._get_obs()
        info = {
            "win": win,
            "collision": collision_any,
            "d1": d1,
            "d2": d2,
            "d3": d3,
            "d12": d12,
            "d13": d13,
            "d23": d23,
            "mean_distance": mean_d,
        }
        self._last_info = info
        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            warnings.warn(
                "CoopTrackingEnvV3: render_mode is not 'human'; call with render_mode='human'",
                stacklevel=2,
            )
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 7.2))
        lim = float(self.arena_size) * 1.08
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.35)
        ax.axhline(0.0, color="0.88", lw=0.9)
        ax.axvline(0.0, color="0.88", lw=0.9)

        colors = ["#1f77b4", "#2ca02c", "#9467bd"]
        for i, c in enumerate(colors):
            tr = np.array(self._trail_uavs[i]) if len(self._trail_uavs[i]) else None
            if tr is not None and len(tr) > 1:
                ax.plot(tr[:, 0], tr[:, 1], "-", color=c, lw=1.8, alpha=0.85, label=f"UAV{i+1}")

        tg = np.array(self._trail_tgt) if len(self._trail_tgt) else None
        if tg is not None and len(tg) > 1:
            ax.plot(tg[:, 0], tg[:, 1], "-", color="#d62728", lw=1.6, label="Target")

        for i, c in enumerate(colors):
            if len(self._trail_uavs[i]):
                xy = self._trail_uavs[i][-1]
                ax.scatter([xy[0]], [xy[1]], c=c, s=55, zorder=6, edgecolors="k", linewidths=0.4)
        if len(self._trail_tgt):
            txy = self._trail_tgt[-1]
            ax.scatter([txy[0]], [txy[1]], c="#d62728", s=70, zorder=6, marker="*", edgecolors="k")

        outcome = (
            getattr(self, "_last_outcome", "")
            or ("success" if self._last_info.get("win") else "")
        )
        title = (
            f"Step {self._step_count}"
            + (f" — {outcome}" if outcome else "")
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=9)
        fig.tight_layout()
        plt.pause(0.001)
        return fig


__all__ = ["CoopTrackingEnvV3"]
