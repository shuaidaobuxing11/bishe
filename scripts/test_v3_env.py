"""
快速验证 CoopTrackingEnvV3（MultiDiscrete）可 reset/step/render。
项目根目录: python scripts/test_v3_env.py [--no-render]
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt

from envs.coop_tracking_env_v3 import CoopTrackingEnvV3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-render", action="store_true", help="不进行 matplotlib 绘图")
    p.add_argument("--max-steps", type=int, default=200)
    args = p.parse_args()

    env = CoopTrackingEnvV3(render_mode="human" if not args.no_render else None)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.action_space.shape == (3,)

    if not args.no_render:
        plt.ion()

    for step in range(args.max_steps):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        assert next_obs.dtype == obs.dtype

        if not args.no_render:
            fig = env.render()
            plt.close(fig)
        obs = next_obs
        if done or truncated:
            break

    env.close()
    print("CoopTrackingEnvV3 smoke test OK. last reward:", reward, "info:", info)


if __name__ == "__main__":
    main()
