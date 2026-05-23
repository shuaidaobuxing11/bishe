import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Windows / 常见环境中文图例
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _read_training_curves(csv_path: str):
    """同一 method 下相同 training_steps 若出现多次（多次训练追加 CSV），保留最后一次。"""
    data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            method = row.get("method", "").strip()
            if not method:
                continue
            ts = int(float(row.get("training_steps", "0")))
            succ = float(row.get("success_rate", "0"))
            ret = float(row.get("mean_return", "0"))
            data.setdefault(method, []).append((ts, succ, ret))
    for method in list(data.keys()):
        by_step = {}
        for ts, succ, ret in data[method]:
            by_step[ts] = (succ, ret)
        data[method] = sorted((ts, v[0], v[1]) for ts, v in by_step.items())
    return data


def _plot_series(ax, arr: np.ndarray, label: str, **kwargs):
    if arr.size == 0:
        return
    ax.plot(arr[:, 0], arr[:, 1] * 100.0, marker="o", label=label, **kwargs)


def _plot_series_return(ax, arr: np.ndarray, label: str, **kwargs):
    if arr.size == 0:
        return
    ax.plot(arr[:, 0], arr[:, 2], marker="o", label=label, **kwargs)


def _plot_dual(data: dict, baseline_key: str, second_key: str, second_label: str, out_path: str):
    if baseline_key not in data:
        raise ValueError(f"CSV 中未找到 baseline 方法：{baseline_key}")
    if second_key not in data:
        raise ValueError(f"CSV 中未找到对比方法：{second_key}")

    base = np.asarray(data[baseline_key], dtype=np.float64)
    sec = np.asarray(data[second_key], dtype=np.float64)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    _plot_series(ax1, base, "PPO baseline", color="C0")
    _plot_series(ax1, sec, second_label, color="C1")
    ax1.set_title("Success Rate vs Training Steps")
    ax1.set_xlabel("Training steps")
    ax1.set_ylabel("Success rate (%)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    _plot_series_return(ax2, base, "PPO baseline", color="C0")
    _plot_series_return(ax2, sec, second_label, color="C1")
    ax2.set_title("Return vs Training Steps")
    ax2.set_xlabel("Training steps")
    ax2.set_ylabel("Mean return")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    os.makedirs(os.path.dirname(out_path) or "results", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"图表已保存: {out_path}")


def _resolve_dual_second_method(data: dict, finetune_method: str):
    """
    双路对比：第二根曲线优先用显式 --finetune_method；
    若为默认 ppo_finetune 且 CSV 中已有 ppo_kl，则自动用 ppo_kl 作为「改进」曲线（新训练脚本写入）。
    """
    if finetune_method != "ppo_finetune":
        return finetune_method, "PPO finetune" if "finetune" in finetune_method else finetune_method
    if "ppo_kl" in data:
        return "ppo_kl", "PPO + KL（改进）"
    if "ppo_finetune" in data:
        return "ppo_finetune", "PPO finetune"
    raise ValueError("CSV 中未找到 ppo_kl 或 ppo_finetune，无法绘制第二根曲线")


def _plot_triplet(data: dict, out_path: str, baseline_key: str, finetune_key: str, kl_key: str):
    """
    三路：PPO baseline、PPO finetune（无 KL）、PPO + KL（改进）。
    兼容旧 CSV：仅有 ppo_finetune、无 ppo_kl 时，将该列仅作为「PPO + KL」绘制，并提示缺少无 KL 微调数据。
    """
    warnings = []
    base = np.asarray(data[baseline_key], dtype=np.float64) if baseline_key in data else np.empty((0, 3))

    kl_raw = data.get(kl_key)
    fin_raw = data.get(finetune_key)

    if kl_raw is None and fin_raw is not None:
        kl_raw = fin_raw
        fin_raw = None
        warnings.append(
            f"未找到 {kl_key}，将历史列 {finetune_key} 视为「PPO + KL（改进）」；"
            f"请用新版 train_online_finetune 分别训练以写入 {finetune_key} 与 {kl_key}。"
        )

    fin = np.asarray(fin_raw, dtype=np.float64) if fin_raw is not None else np.empty((0, 3))
    kl = np.asarray(kl_raw, dtype=np.float64) if kl_raw is not None else np.empty((0, 3))

    if base.size == 0:
        raise ValueError(f"CSV 中未找到 baseline：{baseline_key}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))

    _plot_series(ax1, base, "PPO baseline", color="C0")
    _plot_series(ax1, fin, "PPO finetune", color="C1")
    _plot_series(ax1, kl, "PPO + KL（改进）", color="C2")
    ax1.set_title("Success Rate vs Training Steps")
    ax1.set_xlabel("Training steps")
    ax1.set_ylabel("Success rate (%)")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    _plot_series_return(ax2, base, "PPO baseline", color="C0")
    _plot_series_return(ax2, fin, "PPO finetune", color="C1")
    _plot_series_return(ax2, kl, "PPO + KL（改进）", color="C2")
    ax2.set_title("Return vs Training Steps")
    ax2.set_xlabel("Training steps")
    ax2.set_ylabel("Mean return")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    os.makedirs(os.path.dirname(out_path) or "results", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"图表已保存: {out_path}")
    for w in warnings:
        print(f"[提示] {w}")


def main():
    ap = argparse.ArgumentParser(description="训练曲线：双路 baseline vs 改进，或三路 baseline / finetune / KL")
    ap.add_argument("--csv", type=str, default="results/training_curves.csv", help="训练曲线 CSV")
    ap.add_argument("--out", type=str, default=None, help="输出图片路径（双路默认 fig8，三路默认 fig8_three）")
    ap.add_argument("--triplet", action="store_true", help="绘制 PPO baseline / PPO finetune / PPO+KL 三路对比")
    ap.add_argument("--baseline_method", type=str, default="ppo_baseline")
    ap.add_argument("--finetune_method", type=str, default="ppo_finetune", help="双路模式下第二曲线键；默认时若存在 ppo_kl 则优先用 ppo_kl")
    ap.add_argument("--method_finetune", type=str, default="ppo_finetune", help="三路：无 KL 微调的 CSV 方法名")
    ap.add_argument("--method_kl", type=str, default="ppo_kl", help="三路：含 KL/BC 辅助微调的 CSV 方法名")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"未找到训练曲线 CSV：{args.csv}")

    data = _read_training_curves(args.csv)

    if args.triplet:
        out = args.out or "results/fig8_training_curves_three_way.png"
        _plot_triplet(
            data,
            out,
            baseline_key=args.baseline_method,
            finetune_key=args.method_finetune,
            kl_key=args.method_kl,
        )
        return

    out = args.out or "results/fig8_training_curves.png"
    second_key, second_label = _resolve_dual_second_method(data, args.finetune_method)
    _plot_dual(data, args.baseline_method, second_key, second_label, out)


if __name__ == "__main__":
    main()
