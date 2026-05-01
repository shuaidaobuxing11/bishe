"""
一键生成离线轨迹数据（多场景 mixed / 可选 recovery + suboptimal），并可训练 BC。

典型用法：
  python scripts/generate_offline_data.py
  python scripts/generate_offline_data.py --config configs/offline_data_mixed.yaml

产出（mixed 模式下，见 YAML）：
  - mixed_offline_dataset.pkl（arrays + meta，含 expert_action、data_mode、标签等）
  - dataset_summary.json（与 pickle 同目录，除非另行指定）
  - 可选 *_full.npz（全字段）
  - offline_trajectories.npz（仅用「实际执行动作」五元组，兼容旧 ReplayBuffer/DualReplay）
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from offline_rl.dataset_builder import collect_offline_data
from offline_rl.replay_buffer import ReplayBuffer
from offline_rl.behavior_cloning import train_bc


def load_yaml_cfg(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        raise RuntimeError("请安装 PyYAML 以读取配置：pip install pyyaml") from None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser(description="离线数据生成（mixed）并可选 BC 预训练")
    ap.add_argument(
        "--config",
        type=str,
        default=os.path.join(ROOT, "configs", "offline_config.yaml"),
        help="YAML 配置路径（含 offline_data / behavior_cloning / bc_train）",
    )
    ap.add_argument("--skip_bc", action="store_true", help="仅生成数据，跳过 BC")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isfile(args.config) else os.path.join(ROOT, "configs", "offline_config.yaml")
    if not os.path.isfile(cfg_path):
        print(f"未找到配置文件：{cfg_path}")
        return

    cfg = load_yaml_cfg(cfg_path)
    offline_cfg = cfg.get("offline_data", {})
    bc_cfg = cfg.get("behavior_cloning", {})
    bc_train_cfg = cfg.get("bc_train", {})

    save_dir = offline_cfg.get("save_dir", "data/offline")
    os.makedirs(save_dir, exist_ok=True)

    data_path_legacy = offline_cfg.get("legacy_npz_path") or os.path.join(save_dir, "offline_trajectories.npz")

    mix_env_variants = offline_cfg.get("mix_env_variants")

    if mix_env_variants:
        from offline_rl.mixed_dataset import (
            collect_mixed_offline_dataset,
            save_mixed_dataset,
            build_replay_buffer_from_merged,
        )

        merged, meta = collect_mixed_offline_dataset(offline_cfg)
        n_eps = int(offline_cfg.get("n_episodes", 500))

        pkl_out = offline_cfg.get("save_path") or os.path.join(save_dir, "mixed_offline_dataset.pkl")
        summary_out = offline_cfg.get("summary_path") or os.path.join(
            os.path.dirname(pkl_out) or ".", "dataset_summary.json"
        )
        npz_extra = offline_cfg.get("extended_npz_path") or os.path.join(save_dir, "offline_trajectories_full.npz")

        save_mixed_dataset(
            merged,
            meta,
            save_path=pkl_out,
            summary_path=summary_out,
            npz_extra_path=npz_extra,
            total_episodes=n_eps,
        )
        print(f"混合离线 pickle 已保存: {pkl_out}")
        print(f"数据集统计: {summary_out}")
        print(f"扩展字段 npz: {npz_extra}")

        rb_full = build_replay_buffer_from_merged(merged, mask=None)
        rb_full.save(data_path_legacy)
        print(f"兼容 ReplayBuffer npz（全量 executed action）已保存: {data_path_legacy} (size={rb_full.size})")

        if not args.skip_bc:
            train_bc(
                mixed_dataset_path=pkl_out,
                bc_train=bc_train_cfg,
                batch_size=bc_cfg.get("batch_size", 64),
                epochs=bc_cfg.get("epochs", 50),
                lr=bc_cfg.get("learning_rate", 1e-3),
                save_path=bc_cfg.get("save_path", "models/bc_pretrain.pt"),
                split=tuple(bc_cfg.get("split", (0.8, 0.1, 0.1))),
                split_seed=int(bc_cfg.get("split_seed", 42)),
                metrics_path=bc_cfg.get("metrics_path", "results/bc_metrics.npz"),
            )
        print("完成（mixed）。")

    else:
        expert = offline_cfg.get("expert", "v2")
        rb = collect_offline_data(
            n_episodes=offline_cfg.get("n_episodes", 500),
            max_steps=offline_cfg.get("max_steps", 200),
            save_dir=save_dir,
            seed=offline_cfg.get("seed", 42),
            expert=expert,
        )
        data_path_legacy = os.path.join(save_dir, "offline_trajectories.npz")
        if not os.path.isfile(data_path_legacy) and len(rb) > 0:
            rb.save(data_path_legacy)

        if len(rb) == 0:
            if os.path.isfile(data_path_legacy):
                rb = ReplayBuffer(capacity=500 * 200, obs_shape=(10,), action_dim=())
                rb.load(data_path_legacy)
            else:
                print("无离线数据，无法训练 BC")
                return

        if not args.skip_bc:
            train_bc(
                rb,
                batch_size=bc_cfg.get("batch_size", 64),
                epochs=bc_cfg.get("epochs", 50),
                lr=bc_cfg.get("learning_rate", 1e-3),
                save_path=bc_cfg.get("save_path", "models/bc_pretrain.pt"),
                split=tuple(bc_cfg.get("split", (0.8, 0.1, 0.1))),
                split_seed=int(bc_cfg.get("split_seed", 42)),
                metrics_path=bc_cfg.get("metrics_path", "results/bc_metrics.npz"),
            )
        print("完成（单机专家采集）。")


if __name__ == "__main__":
    main()
