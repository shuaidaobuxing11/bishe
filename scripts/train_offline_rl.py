"""
从混合离线 pickle 按 data_mode 导出「实际执行动作」的 npz，供 DualReplayBuffer.load_offline。

（本仓库若未接 CQL/IQL，本脚本仅负责数据管线；PORL phase2 可直接 load 导出文件。）

用法：
  python scripts/train_offline_rl.py --config configs/offline_rl_training.yaml
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from offline_rl.mixed_dataset import load_mixed_bundle, save_core_npz_by_mask, filter_for_offline_rl_mask


def load_yaml(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=os.path.join(ROOT, "configs", "offline_rl_training.yaml"))
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    orl = cfg.get("offline_rl", {})
    pkl = orl.get("dataset_path", "data/offline/mixed_offline_dataset.pkl")
    pkl_abs = os.path.join(ROOT, pkl) if not os.path.isabs(pkl) else pkl
    if not os.path.isfile(pkl_abs):
        raise FileNotFoundError(f"未找到数据集：{pkl_abs}")

    modes = tuple(orl.get("use_data_modes", ("expert", "recovery", "suboptimal")))
    out_npz = orl.get("export_npz", "data/offline/offline_trajectories_rl.npz")
    out_abs = os.path.join(ROOT, out_npz) if not os.path.isabs(out_npz) else out_npz
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)

    merged, _ = load_mixed_bundle(pkl_abs)
    mask = filter_for_offline_rl_mask(merged, modes)
    save_core_npz_by_mask(merged, mask=mask, path=out_abs)
    print(f"已导出离线 RL 用 npz: {out_abs} (transitions={int(mask.sum())}, modes={modes})")


if __name__ == "__main__":
    main()
