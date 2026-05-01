"""
将 BC 预训练权重加载到 SB3 PPO 的 MlpPolicy 中。
要求 policy 的 net_arch 与 BC 一致：pi 流 10->64->64->25，即 dict(pi=[64,64], vf=[64,64])。
"""
import torch


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def load_bc_into_policy(policy, bc_state_dict, device=None):
    """
    把 BCNet 的 state_dict 拷贝到 SB3 ActorCriticPolicy。
    BC: feature (Sequential 10->64->64) + policy_head (64->25)
    SB3: mlp_extractor.policy_net (obs->64->64) + action_net (64->25)
    """
    if device is None:
        device = next(policy.parameters()).device
    policy_net = policy.mlp_extractor.policy_net
    action_net = policy.action_net

    # Sequential: [0]=Linear(10,64), [1]=ReLU, [2]=Linear(64,64), [3]=ReLU
    policy_net[0].weight.data.copy_(_to_device(bc_state_dict["feature.0.weight"], device))
    policy_net[0].bias.data.copy_(_to_device(bc_state_dict["feature.0.bias"], device))
    policy_net[2].weight.data.copy_(_to_device(bc_state_dict["feature.2.weight"], device))
    policy_net[2].bias.data.copy_(_to_device(bc_state_dict["feature.2.bias"], device))
    action_net.weight.data.copy_(_to_device(bc_state_dict["policy_head.weight"], device))
    action_net.bias.data.copy_(_to_device(bc_state_dict["policy_head.bias"], device))
    return policy
