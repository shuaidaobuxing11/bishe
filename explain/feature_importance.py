"""
可解释性：基于策略网络梯度或权重，计算各状态维度的相对重要性。
状态维度含义：0-3 uav1(x,y,vx,vy), 4-7 uav2, 8-9 target(x,y)。
"""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import online_rl.ppo_bc_finetune as _ppo_bc_fe  # noqa: F401 pickle 兼容微调模型

FEATURE_NAMES = [
    "uav1_x", "uav1_y", "uav1_vx", "uav1_vy",
    "uav2_x", "uav2_y", "uav2_vx", "uav2_vy",
    "target_x", "target_y",
]


def feature_importance_from_weights(policy):
    """
    用策略网络第一层权重的绝对值之和作为各输入维度的代理重要性。
    """
    net = policy.mlp_extractor.policy_net
    # 第一层 Linear(obs_dim, hidden)
    w = net[0].weight.detach().cpu().numpy()
    imp = np.abs(w).sum(axis=0)
    return dict(zip(FEATURE_NAMES, imp.tolist()))


def feature_importance_from_gradient(policy, obs, device="cpu"):
    """
    对单条观测计算 |d(logits)/d(obs)| 作为各维重要性（可选，需完整前向+反向）。
    """
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    obs_t.requires_grad_(True)
    features = policy.extract_features(obs_t)
    latent_pi, _ = policy.mlp_extractor(features)
    logits = policy.action_net(latent_pi)
    logits.sum().backward()
    imp = obs_t.grad.abs().squeeze().cpu().numpy()
    return dict(zip(FEATURE_NAMES, imp.tolist()))


def print_feature_importance(model_path, n_samples=50, seed=42):
    """加载 PPO 模型，在随机观测上汇总特征重要性并打印。"""
    from stable_baselines3 import PPO
    from envs import make_coop_tracking
    env = make_coop_tracking(seed=seed)
    model = PPO.load(model_path)
    policy = model.policy
    device = next(policy.parameters()).device
    importance_sum = np.zeros(10)
    try:
        for _ in range(n_samples):
            obs, _ = env.reset()
            imp = feature_importance_from_gradient(policy, obs, device=device)
            importance_sum += np.array(list(imp.values()))
    except Exception:
        imp = feature_importance_from_weights(policy)
        importance_sum = np.array(list(imp.values()))
        n_samples = 1
    env.close()
    importance_sum /= n_samples
    print("特征重要性（平均）:")
    for name, v in zip(FEATURE_NAMES, importance_sum):
        print(f"  {name}: {v:.4f}")
    return dict(zip(FEATURE_NAMES, importance_sum.tolist()))
