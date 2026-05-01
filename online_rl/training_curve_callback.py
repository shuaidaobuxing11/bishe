import csv
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from envs import make_coop_tracking


class TrainingCurveEvalCallback(BaseCallback):
    """
    在训练过程中定期评估当前策略，并把：
    - success_rate（info["win"] 的占比）
    - mean_return（平均回合回报）
    记录到 CSV，横轴用 training_steps（= num_timesteps）。
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
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.method_name = method_name
        self.eval_seed = eval_seed
        self.n_episodes = int(n_episodes)
        self.eval_freq = int(eval_freq)
        self.out_csv = out_csv
        self.deterministic = deterministic

        self._next_eval_step: Optional[int] = None
        self._eval_env = None

    def _init_callback(self) -> None:
        # 评估环境固定使用默认 spawn_mode（与训练基线一致）。
        self._eval_env = make_coop_tracking(seed=self.eval_seed)

        # 准备输出 CSV（追加写入）。
        # 表头只在文件不存在时写。
        import os

        out_dir = os.path.dirname(self.out_csv) or "."
        os.makedirs(out_dir, exist_ok=True)
        file_exists = os.path.isfile(self.out_csv)

        if not file_exists:
            with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["method", "training_steps", "success_rate", "mean_return"])

        # 第一次评估：从 eval_freq 开始。
        self._next_eval_step = self.eval_freq

    def _evaluate_once(self, training_steps: int) -> None:
        env = self._eval_env
        model = self.model
        assert env is not None
        assert model is not None

        returns = []
        wins = 0

        for ep in range(self.n_episodes):
            obs, _ = env.reset(seed=self.eval_seed + ep)
            ep_reward = 0.0
            while True:
                action, _ = model.predict(obs, deterministic=self.deterministic)
                action = (
                    int(action)
                    if np.ndim(action) == 0
                    else int(action[0]) if np.ndim(action) > 0 else int(action)
                )
                obs, reward, term, trunc, info = env.step(action)
                ep_reward += float(reward)
                if term or trunc:
                    if info.get("win", False):
                        wins += 1
                    break

            returns.append(ep_reward)

        mean_return = float(np.mean(returns)) if returns else 0.0
        success_rate = wins / max(self.n_episodes, 1)

        with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([self.method_name, training_steps, success_rate, mean_return])

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

