import csv
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from envs import make_coop_tracking

CSV_HEADER_FULL = [
    "method",
    "training_steps",
    "success_rate",
    "mean_return",
    "mean_length",
    "collision_rate",
]


def _migrate_csv_to_full(path: str) -> None:
    """将旧版 4 列表头升级为 6 列，旧行补空占位（需重训后才有有效长度/碰撞率）。"""
    import os

    if not os.path.isfile(path):
        return

    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header = rows[0]
    if header == CSV_HEADER_FULL:
        return
    if header != ["method", "training_steps", "success_rate", "mean_return"]:
        return
    new_rows = [CSV_HEADER_FULL]
    for row in rows[1:]:
        if len(row) >= 4:
            new_rows.append(row[:4] + ["", ""])
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(new_rows)
    os.replace(tmp, path)


class TrainingCurveEvalCallback(BaseCallback):
    """
    在训练过程中定期评估当前策略，并把：
    - success_rate、mean_return、mean_length、collision_rate（按评估 episode 聚合）
    写入 CSV，横轴 training_steps（= num_timesteps）。
    """

    def __init__(
        self,
        *,
        method_name: str,
        eval_seed: int,
        n_episodes: int = 10,
        eval_freq: int = 20_000,
        out_csv: str = "results/training_curves.csv",
        deterministic: bool = True,
        spawn_mode: str = "default",
        noise_sigma: float = 0.0,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.method_name = method_name
        self.eval_seed = eval_seed
        self.n_episodes = int(n_episodes)
        self.eval_freq = int(eval_freq)
        self.out_csv = out_csv
        self.deterministic = deterministic
        self.spawn_mode = spawn_mode
        self.noise_sigma = float(noise_sigma)

        self._next_eval_step: Optional[int] = None
        self._eval_env = None

    def _init_callback(self) -> None:
        self._eval_env = make_coop_tracking(
            seed=self.eval_seed,
            spawn_mode=self.spawn_mode,
            noise_sigma=self.noise_sigma,
        )

        import os

        out_dir = os.path.dirname(self.out_csv) or "."
        os.makedirs(out_dir, exist_ok=True)
        file_exists = os.path.isfile(self.out_csv)

        if file_exists:
            _migrate_csv_to_full(self.out_csv)

        if not file_exists:
            with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(CSV_HEADER_FULL)

        self._next_eval_step = self.eval_freq

    def _evaluate_once(self, training_steps: int) -> None:
        env = self._eval_env
        model = self.model
        assert env is not None
        assert model is not None

        returns: list[float] = []
        lengths: list[int] = []
        wins = 0
        collision_episodes = 0

        for ep in range(self.n_episodes):
            obs, _ = env.reset(seed=self.eval_seed + ep)
            ep_reward = 0.0
            steps = 0
            ep_collision = False
            while True:
                action, _ = model.predict(obs, deterministic=self.deterministic)
                action = (
                    int(action)
                    if np.ndim(action) == 0
                    else int(action[0]) if np.ndim(action) > 0 else int(action)
                )
                obs, reward, term, trunc, info = env.step(action)
                ep_reward += float(reward)
                steps += 1
                ep_collision = ep_collision or bool(info.get("collision", False))
                if term or trunc:
                    if info.get("win", False):
                        wins += 1
                    break

            returns.append(ep_reward)
            lengths.append(steps)
            if ep_collision:
                collision_episodes += 1

        mean_return = float(np.mean(returns)) if returns else 0.0
        success_rate = wins / max(self.n_episodes, 1)
        mean_length = float(np.mean(lengths)) if lengths else 0.0
        collision_rate = collision_episodes / max(self.n_episodes, 1)

        with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    self.method_name,
                    training_steps,
                    success_rate,
                    mean_return,
                    mean_length,
                    collision_rate,
                ]
            )

    def _on_step(self) -> bool:
        if self._next_eval_step is None:
            return True
        if self.num_timesteps >= self._next_eval_step:
            self._evaluate_once(training_steps=self.num_timesteps)
            self._next_eval_step += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None
