"""
使用「带转向 + 更合理避碰」的规则专家 v2：
  1) 用 v2 专家重新采集离线数据并训练 BC
  2) 在环境中回放 BC 策略 500 局并输出指标

用法（在项目根目录）：
  python scripts/run_bc_v2_pipeline.py
  python scripts/run_bc_v2_pipeline.py --n_episodes 300   # 仅改评估局数
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=500, help="BC 回放评估局数")
    p.add_argument("--skip_collect", action="store_true", help="跳过采集与训练，仅做 BC 回放评估")
    args = p.parse_args()

    if not args.skip_collect:
        print("=" * 60)
        print("1. 使用规则专家 v2 采集离线数据并训练 BC")
        print("=" * 60)
        ret = subprocess.run("python scripts/generate_offline_data.py", shell=True, cwd=ROOT)
        if ret.returncode != 0:
            print("采集/训练失败，已退出。")
            return

    print("\n" + "=" * 60)
    print("2. BC 策略环境回放评估（{} 局）".format(args.n_episodes))
    print("=" * 60)
    ret = subprocess.run(
        f"python scripts/eval_bc_policy.py --bc_path models/bc_pretrain.pt --n_episodes {args.n_episodes}",
        shell=True,
        cwd=ROOT,
    )
    if ret.returncode != 0:
        print("BC 回放评估失败。")
        return
    print("\nBC v2 全流程完成：采集+训练+评估 已跑完。")


if __name__ == "__main__":
    main()
