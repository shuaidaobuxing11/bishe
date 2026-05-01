"""
统一评估与日志：在默认及未见场景下评估所有基线，写入同一份 CSV/JSON 日志。
便于论文复现与对比。

输出：
  results/eval_log.csv   - 每次运行追加一行（或按 run_id 区分）
  results/eval_log.jsonl  - 每行一个 JSON 对象（便于按 run_id 筛选）
  results/eval_latest.json - 本次运行全部结果的 JSON 数组

用法：
  python scripts/run_unified_eval.py --n_episodes 500 --seed 2024
  python scripts/run_unified_eval.py --scenarios default near_uavs --n_episodes 200
"""
import os
import sys
import argparse
import json
import csv
import numpy as np
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import online_rl.ppo_bc_finetune as _ppo_bc_fe  # noqa: F401 pickle 兼容微调模型保存类

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from envs import make_coop_tracking
from offline_rl.behavior_cloning import load_bc_model
from offline_rl.dataset_builder import rule_policy_v2


SCENARIOS = {
    "default": {},
    "near_uavs": {"spawn_mode": "near_uavs"},
    "near_border": {"spawn_mode": "near_border"},
    "noisy_target": {"noise_sigma": 0.3},
}


def eval_with_policy(make_env_kwargs, n_episodes, seed, act_fn):
    env = make_coop_tracking(seed=seed, **make_env_kwargs)
    returns = []
    lengths = []
    wins = 0
    collision_steps = 0
    total_steps = 0
    collision_episodes = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret = 0.0
        ep_len = 0
        ep_collision = False
        while True:
            a = int(act_fn(env, obs))
            obs, r, term, trunc, info = env.step(a)
            ep_ret += float(r)
            ep_len += 1
            total_steps += 1
            if info.get("collision", False):
                collision_steps += 1
                ep_collision = True
            if term or trunc:
                if info.get("win", False):
                    wins += 1
                if ep_collision:
                    collision_episodes += 1
                break
        returns.append(ep_ret)
        lengths.append(ep_len)

    env.close()
    returns = np.asarray(returns, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    return {
        "success_rate": wins / max(n_episodes, 1),
        "mean_return": float(np.mean(returns)) if len(returns) else 0.0,
        "std_return": float(np.std(returns)) if len(returns) else 0.0,
        "mean_length": float(np.mean(lengths)) if len(lengths) else 0.0,
        "std_length": float(np.std(lengths)) if len(lengths) else 0.0,
        "collision_rate_steps": collision_steps / max(total_steps, 1),
        "collision_rate_episodes": collision_episodes / max(n_episodes, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--scenarios", type=str, nargs="+", default=["default", "near_uavs", "near_border", "noisy_target"])
    ap.add_argument("--bc_path", type=str, default="models/bc_pretrain_best.pt")
    ap.add_argument("--bc_mixed_path", type=str, default="models/bc_mixed.pt")
    ap.add_argument("--ppo_baseline", type=str, default="models/ppo_online_baseline.zip")
    ap.add_argument("--ppo_finetune", type=str, default="models/ppo_finetune_from_bc.zip")
    ap.add_argument("--include_random", action="store_true", help="额外评估随机策略（默认关闭）")
    ap.add_argument("--include_ppo_baseline", action="store_true", help="额外评估 PPO baseline（默认关闭）")
    ap.add_argument("--include_bc_mixed", action="store_true", help="额外评估 bc_mixed（默认关闭）")
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--no_append", action="store_true", help="不追加到 eval_log，仅写 eval_latest")
    args = ap.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)

    # 预加载模型
    bc_model = load_bc_model(args.bc_path, device="cpu") if os.path.isfile(args.bc_path) else None
    bc_mixed_model = load_bc_model(args.bc_mixed_path, device="cpu") if os.path.isfile(args.bc_mixed_path) else None
    ppo_baseline = None
    if args.include_ppo_baseline and os.path.isfile(args.ppo_baseline):
        from stable_baselines3 import PPO
        ppo_baseline = PPO.load(args.ppo_baseline)
    ppo_finetune = None
    if os.path.isfile(args.ppo_finetune):
        from stable_baselines3 import PPO
        ppo_finetune = PPO.load(args.ppo_finetune)

    methods = []
    if args.include_random:
        methods.append(("random", lambda env, obs: env.action_space.sample()))
    methods.append(("expert_v2", lambda env, obs: rule_policy_v2(env, obs)))
    if bc_model is not None:
        methods.append(("bc", lambda env, obs: bc_model.predict(obs, deterministic=True)))
    if args.include_bc_mixed and bc_mixed_model is not None:
        methods.append(("bc_mixed", lambda env, obs: bc_mixed_model.predict(obs, deterministic=True)))
    if ppo_baseline is not None:
        methods.append(("ppo_baseline", lambda env, obs: ppo_baseline.predict(obs, deterministic=True)[0]))
    if ppo_finetune is not None:
        methods.append(("ppo_finetune", lambda env, obs: ppo_finetune.predict(obs, deterministic=True)[0]))

    rows = []
    for scen_name in args.scenarios:
        if scen_name not in SCENARIOS:
            print(f"未知场景 {scen_name}，跳过。")
            continue
        kw = SCENARIOS[scen_name]
        for method_name, act_fn in methods:
            m = eval_with_policy(kw, args.n_episodes, args.seed, act_fn)
            row = {
                "run_id": run_id,
                "method": method_name,
                "scenario": scen_name,
                "n_episodes": args.n_episodes,
                "seed": args.seed,
                **m,
            }
            rows.append(row)
            print(f"{method_name:14} @ {scen_name:12} | succ={m['success_rate']*100:5.1f}% | R={m['mean_return']:7.2f}")

    # 写 results/eval_latest.json
    latest_path = os.path.join(args.out_dir, "eval_latest.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"已写入 {latest_path}")

    # 追加到 results/eval_log.jsonl 和 results/eval_log.csv
    if not args.no_append:
        log_jsonl = os.path.join(args.out_dir, "eval_log.jsonl")
        with open(log_jsonl, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"已追加到 {log_jsonl}")

        log_csv = os.path.join(args.out_dir, "eval_log.csv")
        fieldnames = ["run_id", "method", "scenario", "n_episodes", "seed",
                      "success_rate", "mean_return", "std_return", "mean_length", "std_length",
                      "collision_rate_steps", "collision_rate_episodes"]
        file_exists = os.path.isfile(log_csv)
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"已追加到 {log_csv}")
    print("统一评估完成。")


if __name__ == "__main__":
    main()
