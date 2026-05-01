"""
将统一评估日志（results/eval_latest.json 或 results/eval_log.csv）渲染成图片表格。

输出两张图：
- results/fig5_baseline_table.png：default 场景下的 baseline 表
- results/fig6_unseen_generalization.png：未见场景（near_uavs/near_border/noisy_target）对比表
"""
import os
import argparse
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_latest(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_row(r):
    return [
        r["method"],
        f'{r["success_rate"]*100:.1f}%',
        f'{r["mean_return"]:.2f} ± {r["std_return"]:.2f}',
        f'{r["mean_length"]:.1f} ± {r["std_length"]:.1f}',
        f'{r["collision_rate_steps"]*100:.1f}%',
        f'{r["collision_rate_episodes"]*100:.1f}%',
    ]

def _format_row_compact(r, with_std: bool = True):
    if with_std:
        ret = f'{r["mean_return"]:.2f} ± {r["std_return"]:.2f}'
        length = f'{r["mean_length"]:.1f} ± {r["std_length"]:.1f}'
    else:
        ret = f'{r["mean_return"]:.2f}'
        length = f'{r["mean_length"]:.1f}'
    return [
        r["method"],
        f'{r["success_rate"]*100:.1f}%',
        ret,
        length,
    ]


def _render_table(rows, title, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 0.6 + 0.45 * (len(rows) + 1)))
    ax.axis("off")
    col_labels = ["Method", "Succ", "Return", "Length", "Coll(step)", "Coll(ep)"]
    cell_text = [_format_row(r) for r in rows]
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.2)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"表格图已保存: {save_path}")

def _render_table_compact(rows, title, save_path, with_std: bool = True):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 0.6 + 0.45 * (len(rows) + 1)))
    ax.axis("off")
    col_labels = ["Method", "Succ", "Return", "Length"]
    cell_text = [_format_row_compact(r, with_std=with_std) for r in rows]
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.2)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"精简表格图已保存: {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latest", type=str, default="results/eval_latest.json")
    p.add_argument("--save_baseline", type=str, default="results/fig5_baseline_table.png")
    p.add_argument("--save_unseen", type=str, default="results/fig6_unseen_generalization.png")
    p.add_argument("--compact", action="store_true", help="输出精简版表格（只含成功率/回报/步长）")
    p.add_argument("--no_std", action="store_true", help="精简版表格不显示标准差")
    args = p.parse_args()

    data = _load_latest(args.latest)
    if data is None:
        print(f"未找到 {args.latest}，请先运行 scripts/run_unified_eval.py 生成评估日志。")
        return

    # baseline: default
    base = [r for r in data if r.get("scenario") == "default"]
    # unseen: other scenarios
    unseen = [r for r in data if r.get("scenario") in ("near_uavs", "near_border", "noisy_target")]

    # 排序：让 baseline 按 method 排，unseen 按 scenario+method
    method_order = {"random": 0, "expert_v2": 1, "bc": 2, "bc_mixed": 3, "ppo_baseline": 4, "ppo_finetune": 5}
    base.sort(key=lambda r: method_order.get(r["method"], 999))
    unseen.sort(key=lambda r: (r["scenario"], method_order.get(r["method"], 999)))

    if len(base):
        title = f"Baseline (scenario=default)  n={base[0]['n_episodes']} seed={base[0]['seed']}"
        if args.compact:
            _render_table_compact(base, title, args.save_baseline, with_std=not args.no_std)
        else:
            _render_table(base, title, args.save_baseline)
    else:
        print("baseline(default) 结果为空，跳过 fig5。")

    if len(unseen):
        # 为未见场景加一列 scenario：先拷贝并塞进 method 字段前缀，渲染更紧凑
        rows = []
        for r in unseen:
            rr = dict(r)
            rr["method"] = f'{r["scenario"]} / {r["method"]}'
            rows.append(rr)
        title = f"Unseen conditions (2/4/5)  n={rows[0]['n_episodes']} seed={rows[0]['seed']}"
        if args.compact:
            _render_table_compact(rows, title, args.save_unseen, with_std=not args.no_std)
        else:
            _render_table(rows, title, args.save_unseen)
    else:
        print("unseen 结果为空，跳过 fig6。")


if __name__ == "__main__":
    main()

