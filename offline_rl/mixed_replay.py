"""
离线/在线双经验池与混合回放。
- 离线池：预加载专家数据，只读。
- 在线池：训练时实时写入。
- 混合回放：按比例 offline_ratio 从离线池、(1-offline_ratio) 从在线池抽样。
"""
import numpy as np
from typing import Tuple, Optional

from offline_rl.replay_buffer import ReplayBuffer


class DualReplayBuffer:
    """
    双经验池：offline（只读）+ online（可写）。
    支持混合抽样与单独从某一池抽样。
    """

    def __init__(
        self,
        obs_shape=(10,),
        action_dim=(),
        offline_capacity: int = 500_000,
        online_capacity: int = 200_000,
    ):
        self.offline = ReplayBuffer(capacity=offline_capacity, obs_shape=obs_shape, action_dim=action_dim)
        self.online = ReplayBuffer(capacity=online_capacity, obs_shape=obs_shape, action_dim=action_dim)
        self.obs_shape = obs_shape
        self.action_dim = action_dim

    def load_offline(self, path: str) -> int:
        """从 npz 加载离线数据到 offline 池，返回加载的 transition 数。"""
        self.offline.load(path)
        return len(self.offline)

    def add_online(self, obs, action, reward, next_obs, done):
        """向在线池写入一条 transition。"""
        self.online.add(obs, action, reward, next_obs, done)

    def add_online_batch(self, obs, actions, rewards, next_obs, dones):
        """批量写入在线池。"""
        self.online.add_batch(obs, actions, rewards, next_obs, dones)

    def sample_mixed(
        self,
        batch_size: int,
        offline_ratio: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        混合抽样：offline_ratio 来自离线池，其余来自在线池。
        返回 (obs, actions, rewards, next_obs, dones, is_offline)。
        is_offline[i]=1 表示该条来自离线池，用于保守约束加权。
        """
        rng = rng or np.random.default_rng()
        n_off = min(int(batch_size * offline_ratio), len(self.offline))
        n_on = min(batch_size - n_off, len(self.online))
        if n_off + n_on == 0:
            raise ValueError("离线池与在线池均为空，无法抽样。")

        if n_off > 0 and n_on > 0:
            o_obs, o_a, o_r, o_no, o_d = self.offline.sample(n_off)
            on_obs, on_a, on_r, on_no, on_d = self.online.sample(n_on)
            obs = np.concatenate([o_obs, on_obs], axis=0)
            actions = np.concatenate([o_a, on_a], axis=0)
            rewards = np.concatenate([o_r, on_r], axis=0)
            next_obs = np.concatenate([o_no, on_no], axis=0)
            dones = np.concatenate([o_d, on_d], axis=0)
            is_offline = np.concatenate([np.ones(n_off, dtype=np.float32), np.zeros(n_on, dtype=np.float32)])
        elif n_off > 0:
            obs, actions, rewards, next_obs, dones = self.offline.sample(n_off)
            is_offline = np.ones(n_off, dtype=np.float32)
        else:
            obs, actions, rewards, next_obs, dones = self.online.sample(n_on)
            is_offline = np.zeros(n_on, dtype=np.float32)

        return obs, actions, rewards, next_obs, dones, is_offline

    def sample_offline(self, batch_size: int):
        """仅从离线池抽样。"""
        return self.offline.sample(batch_size)

    def sample_online(self, batch_size: int):
        """仅从在线池抽样。"""
        return self.online.sample(batch_size)

    def __len__(self) -> int:
        return len(self.offline) + len(self.online)

    @property
    def n_offline(self) -> int:
        return len(self.offline)

    @property
    def n_online(self) -> int:
        return len(self.online)
