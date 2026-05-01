"""
可解释性：用决策树拟合「状态 -> 动作」，抽取可读规则。
便于在论文中展示「在什么状态下两机采取何种动作」。
"""
import os
import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import online_rl.ppo_bc_finetune as _ppo_bc_fe  # noqa: F401 pickle 兼容微调模型

FEATURE_NAMES = [
    "uav1_x", "uav1_y", "uav1_vx", "uav1_vy",
    "uav2_x", "uav2_y", "uav2_vx", "uav2_vy",
    "target_x", "target_y",
]


def collect_state_actions(model_path, env, n_episodes=100, max_steps=200, seed=42):
    """用策略在环境中采样，收集 (state, action) 对。"""
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    X, y = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action) if hasattr(action, "item") else int(action[0])
            X.append(obs)
            y.append(action)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break
    return np.array(X), np.array(y)


def fit_and_export_rules(model_path, n_episodes=80, max_depth=5, seed=42):
    """用决策树拟合策略并导出文本规则。"""
    from envs import make_coop_tracking
    env = make_coop_tracking(seed=seed)
    X, y = collect_state_actions(model_path, env, n_episodes=n_episodes, max_steps=200, seed=seed)
    env.close()
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    tree.fit(X, y)
    text = export_text(tree, feature_names=FEATURE_NAMES)
    print("决策树规则（动作 0~24 = a1*5+a2）:")
    print(text)
    return tree, text
