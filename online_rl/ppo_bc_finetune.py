"""
PPO finetune 扩展：仅在 train() 的 loss 处加入
- KL(pi_theta || pi_BC)（BC  logits 不参与梯度）
- 行为克隆辅助：CrossEntropy(student_logits, argmax_bc)
- 系数随 _current_progress_remaining 线性衰减（前期大、后期小）

不改变 SB3 learn / rollout / buffer / evaluate；仅重写 train() 内梯度步。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


class PPOWithBCFinetuneExtras(PPO):
    """
    在 attach_bc_regularizers(...) 设置参考 BC 后，train() 使用带 KL + BC aux 的损失；
    未设置 bc_ref_net 时，退化为原版 PPO.train()。
    """

    bc_ref_net: Optional[nn.Module] = None
    kl_coef_start: float = 0.1
    kl_coef_end: float = 0.01
    bc_coef_start: float = 0.5
    bc_coef_end: float = 0.05

    def attach_bc_regularizers(
        self,
        bc_ref_net: nn.Module,
        *,
        kl_coef_start: float = 0.1,
        kl_coef_end: float = 0.01,
        bc_coef_start: float = 0.5,
        bc_coef_end: float = 0.05,
    ) -> None:
        """冻结参考 BC，用于正则与 BC 辅助项。"""
        self.bc_ref_net = bc_ref_net
        self.bc_ref_net.eval()
        for p in self.bc_ref_net.parameters():
            p.requires_grad_(False)
        self.bc_ref_net.to(self.device)

        self.kl_coef_start = float(kl_coef_start)
        self.kl_coef_end = float(kl_coef_end)
        self.bc_coef_start = float(bc_coef_start)
        self.bc_coef_end = float(bc_coef_end)

    def _coeff_schedule(self, start: float, end: float) -> float:
        """与训练进度一致的线性调度：remaining=1 → start；remaining=0 → end"""
        remaining = float(self._current_progress_remaining)
        return end + (start - end) * remaining

    def _student_action_logits(self, obs: th.Tensor) -> th.Tensor:
        features = self.policy.extract_features(obs)
        if self.policy.share_features_extractor:
            latent_pi, _ = self.policy.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.policy.mlp_extractor.forward_actor(pi_features)
        return self.policy.action_net(latent_pi)

    @staticmethod
    def _kl_discrete_logits(student_logits: th.Tensor, ref_logits_detached: th.Tensor) -> th.Tensor:
        """KL( softmax(student) || softmax(ref) )，ref 无梯度。"""
        log_p = F.log_softmax(student_logits, dim=-1)
        log_q = F.log_softmax(ref_logits_detached.detach(), dim=-1)
        p = log_p.exp()
        return (p * (log_p - log_q)).sum(dim=-1).mean()

    def train(self) -> None:
        if self.bc_ref_net is None:
            return super().train()

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        w_kl = self._coeff_schedule(self.kl_coef_start, self.kl_coef_end)
        w_bc = self._coeff_schedule(self.bc_coef_start, self.bc_coef_end)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        kl_ref_losses = []
        bc_aux_losses = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                obs_mb = rollout_data.observations
                if isinstance(obs_mb, dict):
                    raise NotImplementedError("Dict obs 暂未支持 BC 正则；当前环境为 Box Flatten。")

                student_logits = self._student_action_logits(obs_mb)
                with th.no_grad():
                    ref_logits = self.bc_ref_net(obs_mb.float())

                kl_ref = self._kl_discrete_logits(student_logits, ref_logits)
                bc_tgt = ref_logits.argmax(dim=-1).long().flatten()
                bc_aux = F.cross_entropy(student_logits, bc_tgt)

                kl_ref_losses.append(float(kl_ref.item()))
                bc_aux_losses.append(float(bc_aux.item()))

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + w_kl * kl_ref
                    + w_bc * bc_aux
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)

        self.logger.record("train/bc_ref_kl", np.mean(kl_ref_losses))
        self.logger.record("train/bc_aux_ce", np.mean(bc_aux_losses))
        self.logger.record("train/bc_w_kl", w_kl)
        self.logger.record("train/bc_w_bc", w_bc)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
