"""
行为克隆：用离线数据 (s,a) 训练策略网络，用于离线预训练。
网络与 SB3 的 MlpPolicy 兼容（仅取 policy 部分，离散动作）。
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BCNet(nn.Module):
    """与 SB3 MlpPolicy 一致的 feature_extractor + mlp 离散头。"""
    def __init__(self, obs_dim=10, n_actions=25, net_arch=[64, 64], lr=1e-3):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        layers = []
        in_d = obs_dim
        for d in net_arch:
            layers += [nn.Linear(in_d, d), nn.ReLU()]
            in_d = d
        self.feature = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_d, n_actions)

    def forward(self, x):
        feat = self.feature(x)
        logits = self.policy_head(feat)
        return logits

    def predict(self, obs, deterministic=True):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            logits = self.forward(x)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
            return action.cpu().numpy().squeeze()


def _split_indices_episode_level(
    episode_ids: np.ndarray,
    split: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按整条轨迹划分 train/val/test，避免相邻 transition 同时出现在 train 与 val。"""
    from offline_rl.mixed_dataset import allocate_episode_counts

    ep = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    unique_eps = np.unique(ep)
    n_ep = int(len(unique_eps))
    counts = allocate_episode_counts(n_ep, list(split))
    perm = rng.permutation(n_ep)
    shuffled = unique_eps[perm]
    cut1, cut2 = counts[0], counts[0] + counts[1]
    tr_ep = set(shuffled[:cut1].tolist())
    va_ep = set(shuffled[cut1:cut2].tolist())
    te_ep = set(shuffled[cut2:].tolist())
    m_tr = np.isin(ep, list(tr_ep))
    m_va = np.isin(ep, list(va_ep))
    m_te = np.isin(ep, list(te_ep))
    return np.flatnonzero(m_tr), np.flatnonzero(m_va), np.flatnonzero(m_te)


def train_bc(
    replay_buffer=None,
    *,
    mixed_dataset_path=None,
    bc_train=None,
    obs_dim=10,
    n_actions=25,
    batch_size=64,
    epochs=50,
    lr=1e-3,
    device="cpu",
    save_path=None,
    split=(0.8, 0.1, 0.1),
    split_seed=42,
    metrics_path=None,
    split_by_episode: bool = True,
):
    bc_train = bc_train or {}

    net = BCNet(obs_dim=obs_dim, n_actions=n_actions, lr=lr).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    if mixed_dataset_path:
        from offline_rl.mixed_dataset import filter_for_bc, load_mixed_bundle

        merged, _ = load_mixed_bundle(mixed_dataset_path)
        modes = tuple(bc_train.get("use_data_modes", ("expert", "recovery")))
        label_mode = str(bc_train.get("label_mode", "expert_action"))
        use_ep_split = bool(split_by_episode) and ("episode_id" in merged)
        if use_ep_split:
            obs_np, act_np, ep_np = filter_for_bc(
                merged, modes, label_mode, return_episode_ids=True
            )
        else:
            obs_np, act_np = filter_for_bc(merged, modes, label_mode)
            ep_np = None
            if split_by_episode and "episode_id" not in merged:
                print(
                    "BC: split_by_episode=True 但数据无 episode_id，已退回按 transition 随机划分。"
                )
        obs_np = obs_np.astype(np.float32)
        act_np = act_np.astype(np.int64)
    elif replay_buffer is not None:
        obs_np = replay_buffer.obs[:replay_buffer.size].astype(np.float32)
        act_np = replay_buffer.actions[:replay_buffer.size].astype(np.int64)
        ep_np = None
    else:
        raise ValueError("train_bc 需要 replay_buffer 或 mixed_dataset_path。")

    n = len(obs_np)
    if n == 0:
        raise ValueError("ReplayBuffer 为空，无法训练 BC。")

    p_train, p_val, p_test = split
    if not np.isclose(p_train + p_val + p_test, 1.0):
        raise ValueError(f"split 需要和为 1.0，当前为 {split}")

    rng = np.random.default_rng(split_seed)
    if ep_np is not None:
        idx_train, idx_val, idx_test = _split_indices_episode_level(ep_np, split, rng)
        print(
            f"BC 划分：按整条轨迹（episode），"
            f"train_ep transitions={len(idx_train)}, val_ep={len(idx_val)}, test_ep={len(idx_test)}, "
            f"uniq_episodes={len(np.unique(ep_np))}"
        )
    else:
        idx = rng.permutation(n)
        n_train = int(n * p_train)
        n_val = int(n * p_val)
        idx_train = idx[:n_train]
        idx_val = idx[n_train:n_train + n_val]
        idx_test = idx[n_train + n_val:]
        print(
            f"BC 划分：按单步 transition 随机（无 episode_id 或未启用 split_by_episode），"
            f"n_train/n_val/n_test={len(idx_train)}/{len(idx_val)}/{len(idx_test)}"
        )

    used_episode_split = ep_np is not None

    def make_loader(indices, shuffle):
        x = torch.as_tensor(obs_np[indices], dtype=torch.float32)
        y = torch.as_tensor(act_np[indices], dtype=torch.long)
        ds = TensorDataset(x, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(idx_train, shuffle=True)
    val_loader = make_loader(idx_val, shuffle=False) if len(idx_val) else None
    test_loader = make_loader(idx_test, shuffle=False) if len(idx_test) else None

    def eval_loader(loader):
        if loader is None:
            return None, None
        net.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = net(x)
                loss = nn.functional.cross_entropy(logits, y)
                bs = int(y.shape[0])
                total_loss += float(loss.item()) * bs
                pred = logits.argmax(dim=-1)
                correct += int((pred == y).sum().item())
                total += bs
        net.train()
        return total_loss / max(total, 1), correct / max(total, 1)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    best_state = None
    for ep in range(epochs):
        net.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = int(y.shape[0])
            total_loss += float(loss.item()) * bs
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += bs

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = eval_loader(val_loader)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(np.nan if val_loss is None else float(val_loss))
        history["val_acc"].append(np.nan if val_acc is None else float(val_acc))

        # 记录最优验证模型（用于论文基线）
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

        if (ep + 1) % 10 == 0 or ep == 0 or (ep + 1) == epochs:
            if val_loss is None:
                print(f"BC epoch {ep+1}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.2%}")
            else:
                print(
                    f"BC epoch {ep+1}/{epochs} "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
                )

    test_loss, test_acc = eval_loader(test_loader)
    history["test_loss"] = np.nan if test_loss is None else float(test_loss)
    history["test_acc"] = np.nan if test_acc is None else float(test_acc)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        # 保存最后一轮模型
        torch.save(net.state_dict(), save_path)
        print(f"BC model saved to {save_path}")
        # 保存验证集最优模型
        if best_state is not None:
            root, ext = os.path.splitext(save_path)
            best_path = root + "_best" + (ext or ".pt")
            torch.save(best_state, best_path)
            print(f"BC best-val model saved to {best_path} (val_loss={best_val_loss:.4f})")
    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        used_episode_split = ep_np is not None
        np.savez_compressed(
            metrics_path,
            **history,
            split=np.asarray(split, dtype=np.float32),
            n=n,
            n_train=len(idx_train),
            n_val=len(idx_val),
            n_test=len(idx_test),
            split_episode_level=np.asarray([1 if used_episode_split else 0], dtype=np.uint8),
        )
        print(f"BC metrics saved to {metrics_path}")
    return net


def train_bc_mixed(
    dual_buffer,
    obs_dim=10,
    n_actions=25,
    batch_size=64,
    steps=5000,
    lr=1e-3,
    device="cpu",
    save_path=None,
    offline_ratio=0.5,
    conservative_coef=1.0,
    metrics_path=None,
    log_every=200,
    seed=42,
):
    """
    混合回放 + 保守约束：从双经验池抽样，对离线条目的 BC 损失加权（保守约束），
    使策略在离线数据上不易偏离专家。
    dual_buffer: DualReplayBuffer，已加载离线数据，在线池可空或已有数据。
    offline_ratio: 混合抽样时来自离线池的比例。
    conservative_coef: 离线条目的损失权重 = 1 + conservative_coef，即离线样本权重更高。
    """
    rng = np.random.default_rng(seed)
    net = BCNet(obs_dim=obs_dim, n_actions=n_actions, lr=lr).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    log_steps = []
    log_loss = []
    log_acc = []

    for step in range(steps):
        obs, actions, _, _, _, is_offline = dual_buffer.sample_mixed(
            batch_size, offline_ratio=offline_ratio, rng=rng
        )
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        y = torch.as_tensor(actions, dtype=torch.long, device=device)
        w = torch.as_tensor(1.0 + conservative_coef * is_offline, dtype=torch.float32, device=device)

        logits = net(x)
        ce = nn.functional.cross_entropy(logits, y, reduction="none")
        loss = (ce * w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        do_log = ((step + 1) % log_every == 0) or (step == 0) or ((step + 1) == steps)
        if do_log:
            acc = (logits.argmax(dim=-1) == y).float().mean().item()
            log_steps.append(step + 1)
            log_loss.append(float(loss.item()))
            log_acc.append(float(acc))
            print(f"BC mixed step {step+1}/{steps} loss={loss.item():.4f} acc={acc:.2%}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(net.state_dict(), save_path)
        print(f"BC mixed model saved to {save_path}")

    if metrics_path:
        os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
        np.savez_compressed(
            metrics_path,
            steps=np.asarray(log_steps, dtype=np.int32),
            loss=np.asarray(log_loss, dtype=np.float32),
            acc=np.asarray(log_acc, dtype=np.float32),
            offline_ratio=np.asarray([offline_ratio], dtype=np.float32),
            conservative_coef=np.asarray([conservative_coef], dtype=np.float32),
        )
        print(f"BC mixed metrics saved to {metrics_path}")
    return net


def load_bc_model(path, obs_dim=10, n_actions=25, device="cpu"):
    net = BCNet(obs_dim=obs_dim, n_actions=n_actions).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net
