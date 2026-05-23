"""观测特征名称（与 CoopTrackingEnv 观测维度一致）。"""
from __future__ import annotations


def get_feature_names(env_name: str = "coop_tracking") -> list[str]:
    """
    CoopTrackingEnv 观测为 10 维（arena 归一化）：
      uav1(x,y,vx,vy), uav2(x,y,vx,vy), target(x,y)
    """
    if env_name in ("coop_tracking", "default", "CoopTrackingEnv"):
        return [
            "uav1_x_norm",
            "uav1_y_norm",
            "uav1_vx_norm",
            "uav1_vy_norm",
            "uav2_x_norm",
            "uav2_y_norm",
            "uav2_vx_norm",
            "uav2_vy_norm",
            "target_x_norm",
            "target_y_norm",
        ]
    raise ValueError(f"未知 env_name: {env_name}")


def validate_obs_dim(obs_dim: int, env_name: str = "coop_tracking") -> list[str]:
    names = get_feature_names(env_name)
    if len(names) != obs_dim:
        raise ValueError(
            f"特征名数量 ({len(names)}) 与观测维度 ({obs_dim}) 不一致。"
            f" env={env_name}, names={names}"
        )
    return names
