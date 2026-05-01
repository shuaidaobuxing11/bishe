"""
仅从混合离线 pickle 训练 BC（与 generate_offline_data 解耦）。

用法：
  python scripts/train_bc.py --config configs/bc_config.yaml
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from offline_rl.behavior_cloning import train_bc


def load_yaml(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(ROOT, "configs", "bc_config.yaml"))
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    off = cfg.get("offline_data", {})
    pkl = off.get("save_path", os.path.join("data", "offline", "mixed_offline_dataset.pkl"))
    pkl_abs = pkl if os.path.isabs(pkl) else os.path.join(ROOT, pkl)
    if not os.path.isfile(pkl_abs):
        raise FileNotFoundError(f"未找到 mixed 数据集 pickle，请先生成：{pkl_abs}")

    bc_train = cfg.get("bc_train", {})
    bc = cfg.get("behavior_cloning", {})
    train_bc(
        mixed_dataset_path=pkl_abs,
        bc_train=bc_train,
        batch_size=bc.get("batch_size", 64),
        epochs=bc.get("epochs", 50),
        lr=bc.get("learning_rate", 1e-3),
        save_path=bc.get("save_path", "models/bc_pretrain.pt"),
        split=tuple(bc.get("split", (0.8, 0.1, 0.1))),
        split_seed=int(bc.get("split_seed", 42)),
        metrics_path=bc.get("metrics_path", "results/bc_metrics.npz"),
    )
    print("BC 训练完成。")


if __name__ == "__main__":
    main()
