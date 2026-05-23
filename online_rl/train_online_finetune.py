"""
离线预训练 + 在线微调：先加载 BC 权重，再用 PPO 在相同步数预算下微调。
与纯在线基线使用相同的 total_timesteps，便于公平对比奖励与胜率。

场景参数与 `train_online_baseline.py` 一致（spawn_mode / noise_sigma / model_tag）。
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


def _scenario_tag(spawn_mode: str, noise_sigma: float, model_tag: str) -> str:
    if model_tag.strip():
        return model_tag.strip()
    if spawn_mode == "default" and noise_sigma <= 0:
        return ""
    t = spawn_mode
    if noise_sigma > 0:
        t = f"{t}_noise{noise_sigma:g}".replace(".", "p")
    return t


def main():
    parser = argparse.ArgumentParser(description="离线 BC + 在线 PPO 微调")
    parser.add_argument("--bc_path", type=str, default="models/bc_pretrain.pt", help="BC 预训练权重路径")
    parser.add_argument("--total_timesteps", type=int, default=200_000, help="在线微调总步数（与基线一致）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--finetune_lr", type=float, default=1e-4, help="PPO 微调学习率（比 baseline 小）")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="PPO 熵系数（增强探索）")
    parser.add_argument("--no_bc_regularizers", action="store_true", help="关闭 KL/BC 辅助项（退化为原版 PPO）")
    parser.add_argument("--kl_coef_start", type=float, default=0.1, help="KL(pi||pi_BC) 系数起点（remaining=1）")
    parser.add_argument("--kl_coef_end", type=float, default=0.01, help="KL 系数终点（remaining=0）")
    parser.add_argument("--bc_coef_start", type=float, default=0.5, help="BC 交叉熵辅助系数起点")
    parser.add_argument("--bc_coef_end", type=float, default=0.05, help="BC 交叉熵辅助系数终点")
    parser.add_argument("--curve_eval_freq", type=int, default=20_000, help="训练期间评估间隔（steps）")
    parser.add_argument("--curve_n_episodes", type=int, default=10, help="每次评估的 episode 数")
    parser.add_argument("--curve_out_csv", type=str, default="results/training_curves.csv", help="训练曲线 CSV 输出路径")
    parser.add_argument("--no_curve_logging", action="store_true", help="不记录 success_rate/return 训练曲线")
    parser.add_argument(
        "--spawn_mode",
        type=str,
        default="default",
        choices=["default", "near_uavs", "near_border"],
        help="初始分布",
    )
    parser.add_argument("--noise_sigma", type=float, default=0.0, help="目标扰动强度")
    parser.add_argument(
        "--model_tag",
        type=str,
        default="",
        help="保存文件名后缀；留空则据 spawn_mode+noise 自动生成",
    )
    parser.add_argument(
        "--curve_method_name",
        type=str,
        default="",
        help="CSV method 列；留空则 (ppo_finetune|ppo_kl)_<tag>",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.bc_path):
        raise FileNotFoundError(f"请先运行脚本生成离线数据并训练 BC，得到 {args.bc_path}")

    _ = load_env_config()

    tag = _scenario_tag(args.spawn_mode, args.noise_sigma, args.model_tag)
    base_curve = "ppo_finetune" if args.no_bc_regularizers else "ppo_kl"
    if args.curve_method_name.strip():
        curve_name = args.curve_method_name.strip()
    elif tag:
        curve_name = f"{base_curve}_{tag}"
    else:
        curve_name = base_curve
    if tag:
        save_path = os.path.join(args.save_dir, f"ppo_finetune_from_bc_{tag}.zip")
    else:
        save_path = os.path.join(args.save_dir, "ppo_finetune_from_bc.zip")

    env_fn = lambda: make_coop_tracking(
        seed=args.seed, spawn_mode=args.spawn_mode, noise_sigma=args.noise_sigma
    )
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
                method_name=curve_name,
                eval_seed=args.seed,
                n_episodes=args.curve_n_episodes,
                eval_freq=args.curve_eval_freq,
                out_csv=args.curve_out_csv,
                spawn_mode=args.spawn_mode,
                noise_sigma=args.noise_sigma,
            )
        )
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks[0] if callbacks else None)
    os.makedirs(args.save_dir, exist_ok=True)
    if getattr(model, "bc_ref_net", None) is not None:
        model.bc_ref_net = None
    model.save(save_path)
    print(f"微调策略已保存: {save_path}")
    env.close()


if __name__ == "__main__":
    main()
