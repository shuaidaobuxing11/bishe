"""简单离线经验池：保存 (s, a, r, s', done) 用于行为克隆/离线 RL。"""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=int(1e6), obs_shape=(10,), action_dim=1):
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_dim) if isinstance(action_dim, tuple) else (capacity,), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.capacity = capacity

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs, actions, rewards, next_obs, dones):
        n = len(obs)
        for i in range(n):
            self.add(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=min(batch_size, self.size))
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size

    def save(self, path):
        np.savez_compressed(
            path,
            obs=self.obs[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_obs=self.next_obs[:self.size],
            dones=self.dones[:self.size],
        )

    def load(self, path):
        """
        从 npz 向量化加载，避免逐条 add 导致非常慢。

        期望 npz 字段：obs/actions/rewards/next_obs/dones
        """
        data = np.load(path, allow_pickle=False)
        obs = data["obs"]
        actions = data["actions"]
        rewards = data["rewards"]
        next_obs = data["next_obs"]
        dones = data["dones"]

        n = int(obs.shape[0])
        if n > self.capacity:
            raise ValueError(f"数据集大小 {n} 超过 ReplayBuffer 容量 {self.capacity}")

        # 拷贝到预分配数组（保持 dtype 兼容）
        self.obs[:n] = obs
        self.actions[:n] = actions.reshape(self.actions[:n].shape)
        self.rewards[:n] = rewards
        self.next_obs[:n] = next_obs
        self.dones[:n] = dones

        self.size = n
        self.ptr = n % self.capacity
