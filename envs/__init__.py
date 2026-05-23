from envs.coop_tracking_env import CoopTrackingEnv
from envs.coop_tracking_env_v3 import CoopTrackingEnvV3

def load_env_config():
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "env_config.yaml")
    if os.path.isfile(path):
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError:
            print("未安装 PyYAML，无法读取 configs/env_config.yaml，将使用环境默认参数。")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def make_coop_tracking(**kwargs):
    config = load_env_config()
    config.update(kwargs)
    return CoopTrackingEnv(config=config)


def make_coop_tracking_v3(**kwargs):
    """三机 MultiDiscrete 环境；不在此合并 env_config.yaml，避免与双机参数混用。"""
    return CoopTrackingEnvV3(config=dict(kwargs))


__all__ = [
    "CoopTrackingEnv",
    "CoopTrackingEnvV3",
    "make_coop_tracking",
    "make_coop_tracking_v3",
    "load_env_config",
]
