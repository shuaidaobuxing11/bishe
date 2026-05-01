"""
多随机种子在线训练与评估：
- 对 PPO baseline / PPO finetune 各训练多次（默认 5 个 seed）
- 每个 seed 保存独立模型文件，避免覆盖
- 统一评估并输出逐 seed 结果 + mean/std 汇总

用法示例：
  python scripts/run_multiseed_online.py --seeds 42 52 62 72 82 --total_timesteps 200000 --eval_episodes 200
"""
import os
import sys
import csv
import argparse
import shutil
import subprocess
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from online_rl.eval_policies import evaluate_policy


def _run(cmd: str, desc: str) -> bool:
    print("\n" + "=" * 70)
    print(desc)
    print("=" * 70)
    ret = subprocess.run(cmd, shell=True, cwd=ROOT)
    return ret.returncode == 0


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="baseline/finetune 多 seed 训练与评估")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82], help="训练随机种子列表")
    ap.add_argument("--total_timesteps", type=int, default=200_000)
    ap.add_argument("--bc_path", type=str, default="models/bc_pretrain.pt")
    ap.add_argument("--eval_episodes", type=int, default=200)
    ap.add_argument("--eval_seed", type=int, default=2024, help="评估环境种子（固定用于公平对比）")
    ap.add_argument("--out_dir", type=str, default="results/multiseed")
    ap.add_argument("--models_dir", type=str, default="models/multiseed")
    ap.add_argument("--finetune_lr", type=float, default=1e-4)
    ap.add_argument("--ent_coef", type=float, default=0.01)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    rows = []
    for seed in args.seeds:
        # baseline
        ok = _run(
            f"python online_rl/train_online_baseline.py --total_timesteps {args.total_timesteps} --seed {seed} --no_curve_logging",
            f"[seed={seed}] 训练 PPO baseline",
        )
        if not ok:
            print(f"[WARN] baseline 训练失败，跳过 seed={seed}")
            continue
        baseline_src = os.path.join(ROOT, "models", "ppo_online_baseline.zip")
        baseline_dst = os.path.join(ROOT, args.models_dir, f"ppo_online_baseline_seed{seed}.zip")
        _ensure_parent(baseline_dst)
        shutil.copy2(baseline_src, baseline_dst)

        # finetune
        ok = _run(
            (
                f"python online_rl/train_online_finetune.py --total_timesteps {args.total_timesteps} "
                f"--seed {seed} --bc_path {args.bc_path} --finetune_lr {args.finetune_lr} "
                f"--ent_coef {args.ent_coef} --no_curve_logging"
            ),
            f"[seed={seed}] 训练 PPO finetune",
        )
        if not ok:
            print(f"[WARN] finetune 训练失败，跳过 seed={seed}")
            continue
        finetune_src = os.path.join(ROOT, "models", "ppo_finetune_from_bc.zip")
        finetune_dst = os.path.join(ROOT, args.models_dir, f"ppo_finetune_from_bc_seed{seed}.zip")
        _ensure_parent(finetune_dst)
        shutil.copy2(finetune_src, finetune_dst)

        # evaluate both on same eval seed
        b_ret, b_win, b_std = evaluate_policy(
            baseline_dst, n_episodes=args.eval_episodes, seed=args.eval_seed, deterministic=True
        )
        f_ret, f_win, f_std = evaluate_policy(
            finetune_dst, n_episodes=args.eval_episodes, seed=args.eval_seed, deterministic=True
        )

        rows.append(
            {
                "seed": seed,
                "baseline_mean_return": float(b_ret),
                "baseline_std_return": float(b_std),
                "baseline_win_rate": float(b_win),
                "finetune_mean_return": float(f_ret),
                "finetune_std_return": float(f_std),
                "finetune_win_rate": float(f_win),
                "delta_return_f_minus_b": float(f_ret - b_ret),
                "delta_win_rate_f_minus_b": float(f_win - b_win),
            }
        )
        print(
            f"[seed={seed}] baseline: R={b_ret:.2f}, win={b_win:.2%} | "
            f"finetune: R={f_ret:.2f}, win={f_win:.2%}"
        )

    if not rows:
        print("没有可用结果，未生成汇总文件。")
        return

    # write per-seed csv
    per_seed_csv = os.path.join(ROOT, args.out_dir, "online_multiseed_results.csv")
    fieldnames = list(rows[0].keys())
    with open(per_seed_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # summary
    def arr(k):
        return np.asarray([r[k] for r in rows], dtype=np.float64)

    summary = {
        "n_seeds": len(rows),
        "baseline_mean_return_mean": float(arr("baseline_mean_return").mean()),
        "baseline_mean_return_std": float(arr("baseline_mean_return").std()),
        "baseline_win_rate_mean": float(arr("baseline_win_rate").mean()),
        "baseline_win_rate_std": float(arr("baseline_win_rate").std()),
        "finetune_mean_return_mean": float(arr("finetune_mean_return").mean()),
        "finetune_mean_return_std": float(arr("finetune_mean_return").std()),
        "finetune_win_rate_mean": float(arr("finetune_win_rate").mean()),
        "finetune_win_rate_std": float(arr("finetune_win_rate").std()),
        "delta_return_mean": float(arr("delta_return_f_minus_b").mean()),
        "delta_return_std": float(arr("delta_return_f_minus_b").std()),
        "delta_win_rate_mean": float(arr("delta_win_rate_f_minus_b").mean()),
        "delta_win_rate_std": float(arr("delta_win_rate_f_minus_b").std()),
    }

    summary_csv = os.path.join(ROOT, args.out_dir, "online_multiseed_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print("\n多 seed 结果已保存：")
    print(f"- {per_seed_csv}")
    print(f"- {summary_csv}")
    print("\n汇总（mean ± std）：")
    print(
        f"baseline return: {summary['baseline_mean_return_mean']:.2f} ± {summary['baseline_mean_return_std']:.2f}, "
        f"win: {summary['baseline_win_rate_mean']:.2%} ± {summary['baseline_win_rate_std']:.2%}"
    )
    print(
        f"finetune return: {summary['finetune_mean_return_mean']:.2f} ± {summary['finetune_mean_return_std']:.2f}, "
        f"win: {summary['finetune_win_rate_mean']:.2%} ± {summary['finetune_win_rate_std']:.2%}"
    )
    print(
        f"delta(fin-b) return: {summary['delta_return_mean']:.2f} ± {summary['delta_return_std']:.2f}, "
        f"win: {summary['delta_win_rate_mean']:.2%} ± {summary['delta_win_rate_std']:.2%}"
    )


if __name__ == "__main__":
    main()

