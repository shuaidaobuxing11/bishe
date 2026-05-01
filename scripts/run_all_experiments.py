"""
一键运行：生成离线数据 -> BC 预训练 -> 纯在线基线 -> 离线+在线微调 -> 评估对比。
保证同等训练条件（相同 total_timesteps）下对比奖励与胜率。
"""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

TOTAL_TIMESTEPS = 200_000  # 与 configs/online_config.yaml 一致，便于公平对比


def run(cmd, desc):
    print(f"\n{'='*60}\n{desc}\n{'='*60}")
    ret = subprocess.run(cmd, shell=True, cwd=ROOT)
    if ret.returncode != 0:
        print(f"命令失败: {cmd}")
        return False
    return True


def main():
    # 1. 生成离线数据并训练 BC
    if not run("python scripts/generate_offline_data.py", "1. 生成离线数据并训练 BC"):
        return
    # 2. 纯在线基线
    if not run(
        f"python online_rl/train_online_baseline.py --total_timesteps {TOTAL_TIMESTEPS}",
        "2. 纯在线 PPO 基线",
    ):
        return
    # 3. 离线+在线微调（相同步数）
    if not run(
        f"python online_rl/train_online_finetune.py --total_timesteps {TOTAL_TIMESTEPS}",
        "3. 离线 BC + 在线 PPO 微调",
    ):
        return
    # 4. 评估
    run("python online_rl/eval_policies.py --n_episodes 100", "4. 评估：奖励与胜率对比")
    print("\n全部实验完成。对比上述评估结果即可验证「离线+在线」不低于纯在线。")


if __name__ == "__main__":
    main()
