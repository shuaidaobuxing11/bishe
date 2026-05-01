"""
离线预训练 + 在线微调：先加载 BC 权重，再用 PPO 在相同步数预算下微调。
与纯在线基线使用相同的 total_timesteps，便于公平对比奖励与胜率。
"""
import os
import sys
import argparse
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from stable_baselines3.common.vec_env import DummyVecEnv
from envs import make_coop_tracking, load_env_config
from offline_rl.behavior_cloning import load_bc_model
from online_rl.load_bc_weights import load_bc_into_policy
from online_rl.ppo_bc_finetune import PPOWithBCFinetuneExtras
from online_rl.training_curve_callback import TrainingCurveEvalCallback


def main():
    parser = argparse.ArgumentParser(description="离线 BC + 在线 PPO 微调")
    parser.add_argument("--bc_path", type=str, default="models/bc_pretrain.pt", help="BC 预训练权重路径")
    parser.add_argument("--total_timesteps", type=int, default=200_000, help="在线微调总步数（与基线一致）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="models")
    # 稳定化：避免把 BC 初始化的策略“破坏/卡死”
    parser.add_argument("--finetune_lr", type=float, default=1e-4, help="PPO 微调学习率（比 baseline 小）")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="PPO 熵系数（增强探索）")
    # BC 参考约束（仅 finetune，在 train() loss 中加 KL + BC CE + 分段权重）
    parser.add_argument("--no_bc_regularizers", action="store_true", help="关闭 KL/BC 辅助项（退化为原版 PPO）")
    parser.add_argument("--kl_coef_start", type=float, default=0.1, help="KL(pi||pi_BC) 系数起点（remaining=1）")
    parser.add_argument("--kl_coef_end", type=float, default=0.01, help="KL 系数终点（remaining=0）")
    parser.add_argument("--bc_coef_start", type=float, default=0.5, help="BC 交叉熵辅助系数起点")
    parser.add_argument("--bc_coef_end", type=float, default=0.05, help="BC 交叉熵辅助系数终点")
    parser.add_argument("--curve_eval_freq", type=int, default=20_000, help="训练期间评估间隔（steps）")
    parser.add_argument("--curve_n_episodes", type=int, default=10, help="每次评估的 episode 数")
    parser.add_argument("--curve_out_csv", type=str, default="results/training_curves.csv", help="训练曲线 CSV 输出路径")
    parser.add_argument("--no_curve_logging", action="store_true", help="不记录 success_rate/return 训练曲线")
    args = parser.parse_args()

    if not os.path.isfile(args.bc_path):
        raise FileNotFoundError(f"请先运行脚本生成离线数据并训练 BC，得到 {args.bc_path}")

    config = load_env_config()
    env_fn = lambda: make_coop_tracking(seed=args.seed)
    env = DummyVecEnv([env_fn])

    algo_cls = PPOWithBCFinetuneExtras
    model = algo_cls(
        "MlpPolicy",
        env,
        learning_rate=args.finetune_lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=args.ent_coef,
        # 与 BC 网络一致：ReLU 激活
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
        seed=args.seed,
    )

    bc_state = torch.load(args.bc_path, map_location=model.device)
    load_bc_into_policy(model.policy, bc_state, device=model.device)
    print("已将 BC state_dict 加载到 PPO 的 Actor 部分（共享结构）")
    if not args.no_bc_regularizers:
        bc_ref = load_bc_model(args.bc_path, device=str(model.device))
        model.attach_bc_regularizers(
            bc_ref,
            kl_coef_start=args.kl_coef_start,
            kl_coef_end=args.kl_coef_end,
            bc_coef_start=args.bc_coef_start,
            bc_coef_end=args.bc_coef_end,
        )
        print(
            "已在 loss 中加入 KL(pi||BC) + CE(BC)：分段权重调度 "
            f"kl[{args.kl_coef_start}->{args.kl_coef_end}], "
            f"bce[{args.bc_coef_start}->{args.bc_coef_end}] "
            "(随 _current_progress_remaining 衰减)"
        )
    print("开始在线 PPO 微调...")

    callbacks = []
    if not args.no_curve_logging:
        callbacks.append(
            TrainingCurveEvalCallback(
                method_name="ppo_finetune",
                eval_seed=args.seed,
                n_episodes=args.curve_n_episodes,
                eval_freq=args.curve_eval_freq,
                out_csv=args.curve_out_csv,
            )
        )
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks[0] if callbacks else None)
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "ppo_finetune_from_bc.zip")
    # 推理与评估仅需 policy；不落盘参考 BC 子网络，减小 zip、避免冗余权重
    if getattr(model, "bc_ref_net", None) is not None:
        model.bc_ref_net = None
    model.save(path)
    print(f"微调策略已保存: {path}")
    env.close()


if __name__ == "__main__":
    main()
