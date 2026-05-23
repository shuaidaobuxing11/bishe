"""SB3 PPO / BC 策略包装，供 Captum / SHAP 调用。"""
from __future__ import annotations

import torch
import torch.nn as nn


class SB3PolicyLogitsWrapper(nn.Module):
    """将 SB3 ActorCriticPolicy 包装为 forward(obs)->logits [B, n_actions]。"""

    def __init__(self, sb3_model) -> None:
        super().__init__()
        self.policy = sb3_model.policy

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        obs_tensor = obs_tensor.float()
        features = self.policy.extract_features(obs_tensor)
        if self.policy.share_features_extractor:
            latent_pi, _ = self.policy.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        dist = self.policy._get_action_dist_from_latent(latent_pi)
        if hasattr(dist.distribution, "logits"):
            return dist.distribution.logits
        return dist.distribution.probs


class BCLogitsWrapper(nn.Module):
    """BCNet -> logits [B, n_actions]。"""

    def __init__(self, bc_net: nn.Module) -> None:
        super().__init__()
        self.net = bc_net

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(obs_tensor.float())
