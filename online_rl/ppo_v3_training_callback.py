"""
三机 CoopTrackingEnvV3 — 训练曲线记录（定期评估写入 CSV）。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from envs.coop_tracking_env_v3 import CoopTrackingEnvV3


class PPOv3TrainingCurveCallback(BaseCallback):
    """周期性评估并把 success_rate、mean_return 等追加到 CSV。"""

    def __init__(
        self,
        *,
        env_cfg: dict,
        method_name: str,
        eval_seed: int,
        n_episodes: int = 10,
        eval_freq: int = 20_000,
        out_csv: str = "results/v3/training_curves.csv",
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self._env_cfg = dict(env_cfg)
        self.method_name = method_name
        self.eval_seed = int(eval_seed)
        self.n_episodes = int(n_episodes)
        self.eval_freq = int(eval_freq)
        self.out_csv = Path(out_csv)
        self.deterministic = deterministic
        self._next_eval_step: Optional[int] = None
        self._last_logged_steps: Optional[int] = None

    def _init_callback(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        if not self.out_csv.is_file():
            with self.out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "method",
                        "training_steps",
                        "success_rate",
                        "mean_return",
                        "mean_length",
                        "collision_rate",
                    ]
                )
        self._next_eval_step = self.eval_freq

    def _evaluate_once(self, training_steps: int) -> None:
        assert self.model is not None
        model = self.model
        successes: list[float] = []
        returns: list[float] = []
        lengths: list[float] = []
        collisions: list[float] = []

        for ep in range(self.n_episodes):
            e = CoopTrackingEnvV3(config=dict(self._env_cfg), seed=None)
            obs, _ = e.reset(seed=self.eval_seed + ep)
            cum = 0.0
            le = 0
            coll_ep = False
            win_ep = False
            while True:
                action, _ = model.predict(obs, deterministic=self.deterministic)
                a = np.asarray(action, dtype=np.int64).reshape(-1)
                obs, r, terminated, truncated, info = e.step(a)
                cum += float(r)
                le += 1
                coll_ep = coll_ep or bool(info.get("collision"))
                win_ep = bool(info.get("win")) or win_ep
                if terminated or truncated:
                    break
            successes.append(float(win_ep))
            returns.append(cum)
            lengths.append(le)
            collisions.append(float(coll_ep))

        row = [
            self.method_name,
            training_steps,
            float(np.mean(successes)),
            float(np.mean(returns)),
            float(np.mean(lengths)),
            float(np.mean(collisions)),
        ]
        with self.out_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        self._last_logged_steps = int(training_steps)

    def _on_step(self) -> bool:
        if self._next_eval_step is None:
            return True
        while self.num_timesteps >= self._next_eval_step:
            self._evaluate_once(training_steps=self.num_timesteps)
            self._next_eval_step += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        ts = int(self.num_timesteps)
        if ts <= 0:
            return
        if self._last_logged_steps == ts:
            return
        self._evaluate_once(training_steps=ts)
