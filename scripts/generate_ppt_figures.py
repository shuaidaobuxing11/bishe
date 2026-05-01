"""
一键生成答辩 PPT 的 8 张图（输出到 results/）：
1) fig1_env_overview.png        环境示意 + 代表性轨迹
2) fig2_dataset_quality.png     离线数据质量（专家）
3) bc_curves.png                BC 训练曲线（loss/acc）
4) bc_trajs.png                 BC 成功/失败轨迹对比
5) fig5_baseline_table.png      baseline 表格图（default）
6) fig6_unseen_generalization.png 未见条件表格图（2/4/5）
7) fig7_bc_mixed_curves.png     PORL/phase2：混合回放+保守约束训练曲线（如已生成 bc_mixed_metrics）
8) fig8_training_curves.png     PPO baseline vs finetune：训练曲线对比（success_rate/return）

该脚本会在缺少前置文件时，尽量自动调用对应脚本生成；若仍缺失则提示你先跑哪一步。
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def _run(cmd: str, desc: str):
    print("\n" + "=" * 60)
    print(desc)
    print("=" * 60)
    return subprocess.run(cmd, shell=True, cwd=ROOT).returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--expert_episodes", type=int, default=500, help="生成 expert_episodes.npz 用的局数")
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("data/offline", exist_ok=True)

    # (A) 确保专家 episode 数据存在（用于 fig2/fig1 fallback）
    if not os.path.isfile("data/offline/expert_episodes.npz"):
        _run(
            f"python scripts/collect_expert_episodes.py --expert v2 --n_episodes {args.expert_episodes} --save_dir data/offline --save_name expert_episodes.npz --seed {args.seed}",
            "生成 expert_episodes.npz（规则专家 v2）",
        )

    # (B) 确保 BC metrics 存在（用于 fig3）
    if not os.path.isfile("results/bc_metrics.npz"):
        _run("python scripts/generate_offline_data.py", "生成离线数据并训练 BC（产出 bc_metrics.npz）")

    # (C) 确保 BC 评估 episode 数据存在（用于 fig4）
    if not os.path.isfile("results/bc_eval_episodes.npz"):
        # 优先 best 模型，否则用 last
        bc_path = "models/bc_pretrain_best.pt" if os.path.isfile("models/bc_pretrain_best.pt") else "models/bc_pretrain.pt"
        _run(
            f"python scripts/eval_bc_policy.py --bc_path {bc_path} --n_episodes {args.n_episodes} --seed {args.seed} --save_episodes results/bc_eval_episodes.npz",
            "评估 BC 并保存 results/bc_eval_episodes.npz",
        )

    # (D) 抽取成功/失败 10 条（用于 fig4/fig1）
    if not os.path.isfile("results/bc_success_episodes_10.npz") or not os.path.isfile("results/bc_failure_episodes_10.npz"):
        _run(
            "python scripts/extract_bc_episodes.py --eval_npz results/bc_eval_episodes.npz --n_success 10 --n_failure 10 --out_dir results",
            "抽取 10 条成功 + 10 条失败轨迹",
        )

    # (E) 确保统一评估日志存在（用于 fig5/fig6）
    if not os.path.isfile("results/eval_latest.json"):
        _run(
            f"python scripts/run_unified_eval.py --n_episodes {args.n_episodes} --seed {args.seed}",
            "运行统一评估并生成 results/eval_latest.json",
        )

    # 1) fig1：环境示意
    _run("python scripts/plot_env_overview.py --save results/fig1_env_overview.png", "绘制 fig1_env_overview.png")

    # 2) fig2：离线数据质量（专家）
    _run("python scripts/plot_dataset_quality.py --expert_npz data/offline/expert_episodes.npz --save results/fig2_dataset_quality.png", "绘制 fig2_dataset_quality.png")

    # 3) fig3：BC 曲线
    _run("python scripts/plot_bc_metrics.py --metrics results/bc_metrics.npz --save results/bc_curves.png", "绘制 bc_curves.png")

    # 4) fig4：BC 轨迹对比
    _run("python scripts/plot_bc_trajectories.py --success_npz results/bc_success_episodes_10.npz --failure_npz results/bc_failure_episodes_10.npz --save results/bc_trajs.png --max_plots 3", "绘制 bc_trajs.png")

    # 5-6) fig5/fig6：表格图
    _run("python scripts/plot_eval_tables.py --latest results/eval_latest.json --save_baseline results/fig5_baseline_table.png --save_unseen results/fig6_unseen_generalization.png", "绘制 fig5/fig6 表格图")

    # 7) fig7：BC mixed 曲线（PORL/phase2）
    if os.path.isfile("results/bc_mixed_metrics.npz"):
        _run(
            "python scripts/plot_bc_mixed_metrics.py --metrics results/bc_mixed_metrics.npz --save results/fig7_bc_mixed_curves.png",
            "绘制 fig7_bc_mixed_curves.png",
        )

    # 8) fig8：PPO 训练曲线对比（baseline vs finetune）
    if os.path.isfile("results/training_curves.csv"):
        _run(
            "python scripts/plot_training_curves.py --csv results/training_curves.csv --out results/fig8_training_curves.png",
            "绘制 fig8_training_curves.png",
        )

    print("\n全部图已输出到 results/：")
    for f in [
        "results/fig1_env_overview.png",
        "results/fig2_dataset_quality.png",
        "results/bc_curves.png",
        "results/bc_trajs.png",
        "results/fig5_baseline_table.png",
        "results/fig6_unseen_generalization.png",
        "results/fig7_bc_mixed_curves.png",
        "results/fig8_training_curves.png",
    ]:
        print("-", f)


if __name__ == "__main__":
    main()
