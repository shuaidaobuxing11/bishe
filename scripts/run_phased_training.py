"""
分阶段训练脚本：离线 -> 混合回放+保守约束 -> 可选 PPO 微调。

阶段1：生成离线数据并训练 BC（仅离线池）。
阶段2：加载离线池 + 可选收集在线数据，混合回放 + 保守约束训练 BC，保存 bc_mixed.pt。
阶段3：可选，从 bc_mixed 初始化 PPO 并微调。

用法：
  python scripts/run_phased_training.py                    # 执行阶段1+2+3
  python scripts/run_phased_training.py --phase 1         # 仅阶段1
  python scripts/run_phased_training.py --phase 1 2       # 阶段1与2
  python scripts/run_phased_training.py --skip-ppo        # 阶段1+2，不跑阶段3
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def load_phased_config():
    path = os.path.join(ROOT, "configs", "phased_config.yaml")
    if not os.path.isfile(path):
        return None
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def run_phase1(cfg):
    """阶段1：离线数据 + BC（仅离线）。"""
    from offline_rl.dataset_builder import collect_offline_data
    from offline_rl.replay_buffer import ReplayBuffer
    from offline_rl.behavior_cloning import train_bc

    p1 = cfg.get("phase1_offline", {})
    n_episodes = p1.get("n_episodes", 500)
    max_steps = p1.get("max_steps", 200)
    save_dir = p1.get("save_dir", "data/offline")
    seed = p1.get("seed", 42)
    expert = p1.get("expert", "v2")

    print("[Phase 1] 生成离线数据并训练 BC（仅离线）...")
    rb = collect_offline_data(
        n_episodes=n_episodes,
        max_steps=max_steps,
        save_dir=save_dir,
        seed=seed,
        expert=expert,
    )
    data_path = os.path.join(save_dir, "offline_trajectories.npz")
    if not os.path.isfile(data_path) and len(rb) > 0:
        rb.save(data_path)

    bc_cfg = p1
    train_bc(
        rb,
        batch_size=bc_cfg.get("bc_batch_size", 64),
        epochs=bc_cfg.get("bc_epochs", 50),
        lr=bc_cfg.get("bc_lr", 1e-3),
        save_path=bc_cfg.get("bc_save_path", "models/bc_pretrain.pt"),
        split=(0.8, 0.1, 0.1),
        split_seed=seed,
        metrics_path="results/bc_metrics.npz",
    )
    print("[Phase 1] 完成。")
    return data_path


def run_phase2(cfg, offline_data_path):
    """阶段2：双经验池 + 混合回放 + 保守约束。"""
    from offline_rl.mixed_replay import DualReplayBuffer
    from offline_rl.behavior_cloning import train_bc_mixed, load_bc_model

    p2 = cfg.get("phase2_mixed", {})
    dual = DualReplayBuffer(obs_shape=(10,), action_dim=())
    n_off = dual.load_offline(offline_data_path)
    print(f"[Phase 2] 已加载离线池: {n_off} 条。")

    # 可选：用当前 BC 策略收集在线数据到在线池
    collect_steps = p2.get("collect_online_steps", 0)
    if collect_steps > 0:
        from envs import make_coop_tracking
        bc_path = cfg.get("phase1_offline", {}).get("bc_save_path", "models/bc_pretrain.pt")
        if os.path.isfile(bc_path):
            model = load_bc_model(bc_path, device="cpu")
            env = make_coop_tracking(seed=p2.get("seed", 43))
            step = 0
            obs, _ = env.reset(seed=43)
            while step < collect_steps:
                a = int(model.predict(obs, deterministic=True))
                next_obs, r, term, trunc, _ = env.step(a)
                dual.add_online(obs, a, float(r), next_obs, term or trunc)
                step += 1
                obs = next_obs
                if term or trunc:
                    obs, _ = env.reset(seed=43 + step)
            env.close()
            print(f"[Phase 2] 已收集在线数据: {len(dual.online)} 条。")

    train_bc_mixed(
        dual,
        batch_size=p2.get("mixed_batch_size", 64),
        steps=p2.get("mixed_steps", 5000),
        lr=p2.get("mixed_lr", 1e-3),
        save_path=p2.get("bc_mixed_save_path", "models/bc_mixed.pt"),
        offline_ratio=p2.get("offline_ratio", 0.5),
        conservative_coef=p2.get("conservative_coef", 1.0),
        metrics_path=p2.get("mixed_metrics_path", None),
        log_every=p2.get("mixed_log_every", 200),
        seed=p2.get("seed", 42),
    )
    print("[Phase 2] 完成。")


def run_phase3(cfg):
    """阶段3：PPO 微调（从 bc_mixed 或 bc_pretrain 初始化）。"""
    p3 = cfg.get("phase3_ppo", {})
    bc_path = p3.get("bc_init_path", "models/bc_mixed.pt")
    if not os.path.isfile(bc_path):
        bc_path = cfg.get("phase1_offline", {}).get("bc_save_path", "models/bc_pretrain.pt")
    if not os.path.isfile(bc_path):
        print("[Phase 3] 未找到 BC 权重，跳过 PPO 微调。")
        return
    total = p3.get("total_timesteps", 200_000)
    save_path = p3.get("save_path", "models/ppo_finetune_from_bc.zip")
    print(f"[Phase 3] PPO 微调: bc={bc_path}, total_timesteps={total}")
    subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "online_rl", "train_online_finetune.py"),
            "--bc_path", bc_path,
            "--total_timesteps", str(total),
            "--save_dir", os.path.dirname(save_path),
        ],
        cwd=ROOT,
        check=True,
    )
    print("[Phase 3] 完成。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, nargs="+", default=[1, 2, 3], help="要执行的阶段 1 2 3")
    ap.add_argument("--skip-ppo", action="store_true", help="不执行阶段3（等价于 --phase 1 2）")
    ap.add_argument("--config", type=str, default=None, help="phased_config.yaml 路径")
    args = ap.parse_args()

    if args.skip_ppo:
        args.phase = [1, 2]

    cfg = load_phased_config()
    if cfg is None:
        cfg = {}
        print("未找到 configs/phased_config.yaml，使用默认参数。")

    offline_data_path = None
    if 1 in args.phase:
        offline_data_path = run_phase1(cfg)
        if offline_data_path is None:
            offline_data_path = cfg.get("phase1_offline", {}).get("offline_data_path", "data/offline/offline_trajectories.npz")

    if 2 in args.phase:
        if offline_data_path is None:
            offline_data_path = cfg.get("phase1_offline", {}).get("offline_data_path", "data/offline/offline_trajectories.npz")
        if not os.path.isfile(offline_data_path):
            print(f"未找到离线数据 {offline_data_path}，请先执行阶段1。")
        else:
            run_phase2(cfg, offline_data_path)

    if 3 in args.phase:
        run_phase3(cfg)

    print("分阶段训练脚本执行完毕。")


if __name__ == "__main__":
    main()
