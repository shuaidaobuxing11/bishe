from envs.coop_tracking_env import CoopTrackingEnv

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

__all__ = ["CoopTrackingEnv", "make_coop_tracking", "load_env_config"]
