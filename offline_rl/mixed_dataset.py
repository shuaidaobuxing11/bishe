"""
多场景混合离线数据集：专家 / recovery（扰动→恢复）/ suboptimal（noisy expert）。
导出：pickle（数组 + meta）、扩展 npz、dataset_summary.json。
BC：可选仅 expert + recovery，`label_mode=expert_action`；离线 RL 使用实际执行的动作。
"""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from envs import CoopTrackingEnv, load_env_config
from offline_rl.dataset_builder import rule_policy_v2


def allocate_episode_counts(n_episodes: int, ratios: Sequence[float]) -> List[int]:
    ratio_sum = float(sum(ratios))
    if ratio_sum <= 0:
        raise ValueError("ratio 总和必须 > 0")
    ratios = [r / ratio_sum for r in ratios]
    raw = [n_episodes * r for r in ratios]
    counts = [int(x) for x in raw]
    remainder = n_episodes - sum(counts)
    if remainder != 0:
        frac_sorted = sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        for k in range(abs(remainder)):
            idx = frac_sorted[k % len(frac_sorted)]
            counts[idx] += 1 if remainder > 0 else -1
    counts = [max(0, c) for c in counts]
    diff = n_episodes - sum(counts)
    if diff != 0:
        j = int(np.argmax(ratios)) if ratios else 0
        counts[j] = max(0, counts[j] + diff)
    return counts


def _infer_data_mode(variant: Mapping[str, Any]) -> str:
    if variant.get("data_mode"):
        return str(variant["data_mode"]).lower()
    policy = variant.get("policy", "expert_v2")
    if isinstance(policy, str) and policy.lower() in ("noisy_expert", "suboptimal_expert"):
        return "suboptimal"
    return "expert"


def _scenario_name_fallback(i: int) -> str:
    return f"variant_{i}"


def normalize_mix_variants(offline_cfg: Mapping[str, Any], default_expert: str = "v2") -> List[Dict[str, Any]]:
    """兼容旧 YAML（仅有 ratio/config）与新 YAML（name/policy/data_mode/recovery/suboptimal）。"""
    mix = offline_cfg.get("mix_env_variants")
    base_expert_yaml = offline_cfg.get("expert", default_expert)
    if isinstance(base_expert_yaml, str) and base_expert_yaml.startswith("v"):
        expert_key = base_expert_yaml
    else:
        expert_key = default_expert

    out: List[Dict[str, Any]] = []
    if not mix:
        return out
    for i, v in enumerate(mix):
        v = dict(v)
        name = v.get("name") or _scenario_name_fallback(i)
        policy = v.get("policy", "expert_v2")
        dm = _infer_data_mode({**v, "policy": policy})
        v.setdefault("name", name)
        v.setdefault("policy", policy)
        v.setdefault("data_mode", dm)
        v.setdefault("config", {})
        if not isinstance(v["config"], dict):
            v["config"] = {}
        v["_default_expert_key"] = expert_key
        out.append(v)
    return out


def expert_action_from_obs(env: CoopTrackingEnv, obs: np.ndarray) -> int:
    return int(rule_policy_v2(env, obs))


def _quality_label(win: bool, ep_collision_steps: int, ep_len: int) -> str:
    if not win:
        return "low"
    if ep_collision_steps >= max(12, max(ep_len // 3, 1)):
        return "low"
    if ep_collision_steps > 0:
        return "medium"
    return "high"


ACTION_SOURCE_TO_INT = {"expert": 0, "perturbed": 1, "expert_recovery": 2, "noisy": 3}

DATA_MODE_TO_INT = {"expert": 0, "recovery": 1, "suboptimal": 2}

INT_TO_DATA_MODE = {v: k for k, v in DATA_MODE_TO_INT.items()}


def collect_transition_batch_for_variant(
    variant: Mapping[str, Any],
    n_episodes: int,
    max_steps: int,
    seed_base: int,
    global_episode_start: int,
    scenario_uid: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """采集单个 scenario 的一批 episode，global_episode_id 递增。"""
    base_cfg = load_env_config()
    vcfg = dict(variant.get("config", {}) or {})
    env_cfg = {**base_cfg, **vcfg}
    env_cfg["max_episode_steps"] = env_cfg.get("max_episode_steps", max_steps)
    rng = np.random.default_rng(seed_base)

    scenario = str(variant["name"])
    data_mode = str(variant["data_mode"]).lower()
    policy_name = str(variant["policy"]).lower()

    policy_noise_prob = float(vcfg.get("policy_noise_prob", 0.5))
    perturb_prob = float(vcfg.get("perturb_prob", 0.3))
    p_steps_min = int(vcfg.get("perturb_steps_min", 5))
    p_steps_max = int(vcfg.get("perturb_steps_max", 20))
    n_act = int(env_cfg.get("n_actions_per_uav", 5))
    joint_n_actions = n_act ** 2

    obs_l, act_l, exp_act_l = [], [], []
    rew_l, no_l, done_l = [], [], []
    ep_id_l, step_l = [], []
    scenario_uid_l: List[int] = []
    dm_int_l: List[int] = []
    src_l = []
    collision_l = []
    succ_l = []
    ep_coll_any_rep = []
    ep_ret_rep = []
    ep_len_rep = []
    qual_l = []

    episodic_stats_win: List[int] = []
    episodic_collision_tot: List[int] = []
    episodic_returns: List[float] = []
    episodic_lens: List[int] = []

    gid = global_episode_start

    for ep_ix in range(n_episodes):
        env = CoopTrackingEnv(config=env_cfg, seed=int(seed_base + ep_ix))
        obs, _ = env.reset(seed=int(seed_base + ep_ix))

        perturb_steps = 0
        if data_mode == "recovery":
            perturb_steps = int(rng.integers(p_steps_min, p_steps_max + 1))

        ep_ret = 0.0
        ep_collision_steps = 0
        win_final = False
        done_ep = False
        episode_transitions = []
        last_info: Dict[str, Any] = {}

        for step in range(max_steps):
            ex_a = expert_action_from_obs(env, obs)

            if policy_name == "noisy_expert":
                if rng.random() < policy_noise_prob:
                    act = int(rng.integers(0, joint_n_actions))
                    a_src = "noisy"
                else:
                    act = ex_a
                    a_src = "expert"
            elif data_mode == "recovery":
                if step < perturb_steps and rng.random() < perturb_prob:
                    act = int(rng.integers(0, joint_n_actions))
                    a_src = "perturbed"
                else:
                    act = ex_a
                    a_src = "expert_recovery"
            else:
                act = ex_a
                a_src = "expert"

            next_obs, r, term, trunc, info = env.step(act)
            last_info = dict(info)
            done = bool(term or trunc)
            coll = bool(info.get("collision", False))
            ep_collision_steps += int(coll)
            ep_ret += float(r)

            episode_transitions.append(
                (
                    obs.copy(),
                    int(act),
                    int(ex_a),
                    float(r),
                    next_obs.copy(),
                    float(done),
                    step,
                    a_src,
                    1.0 if coll else 0.0,
                )
            )
            obs = next_obs
            if done:
                win_final = bool(info.get("win", False))
                done_ep = True
                break

        if not episode_transitions:
            continue

        if not done_ep:
            win_final = bool(last_info.get("win", False))

        ep_len_actual = len(episode_transitions)
        qlabel = _quality_label(win_final, ep_collision_steps, ep_len_actual)
        episodic_stats_win.append(1 if win_final else 0)
        episodic_collision_tot.append(ep_collision_steps)
        episodic_returns.append(ep_ret)
        episodic_lens.append(ep_len_actual)

        for tup in episode_transitions:
            (
                ob,
                a,
                exh,
                rw,
                no,
                done_f,
                t_idx,
                a_src_inner,
                col_f,
            ) = tup

            obs_l.append(ob)
            act_l.append(a)
            exp_act_l.append(exh)
            rew_l.append(rw)
            no_l.append(no)
            done_l.append(done_f)
            ep_id_l.append(int(gid))
            step_l.append(int(t_idx))
            scenario_uid_l.append(int(scenario_uid))

            dm_int_l.append(int(DATA_MODE_TO_INT[data_mode]))
            src_l.append(ACTION_SOURCE_TO_INT[a_src_inner])
            collision_l.append(col_f)
            succ_l.append(1.0 if win_final else 0.0)
            ep_coll_any_rep.append(1.0 if ep_collision_steps > 0 else 0.0)
            ep_ret_rep.append(ep_ret)
            ep_len_rep.append(float(ep_len_actual))
            qual_l.append(
                {"high": np.uint8(0), "medium": np.uint8(1), "low": np.uint8(2)}[qlabel]
            )

        gid += 1

    if len(obs_l) == 0:
        return {}, {
            "scenario": scenario,
            "scenario_uid": int(scenario_uid),
            "ep_win_rate": float("nan"),
            "collision_mean_steps": float("nan"),
            "mean_return": float("nan"),
            "mean_length": float("nan"),
            "n_episodes": 0,
        }

    pack: Dict[str, np.ndarray] = {
        "obs": np.asarray(obs_l, dtype=np.float32),
        "actions": np.asarray(act_l, dtype=np.int64),
        "expert_actions": np.asarray(exp_act_l, dtype=np.int64),
        "rewards": np.asarray(rew_l, dtype=np.float32),
        "next_obs": np.asarray(no_l, dtype=np.float32),
        "dones": np.asarray(done_l, dtype=np.float32),
        "episode_id": np.asarray(ep_id_l, dtype=np.int32),
        "timestep": np.asarray(step_l, dtype=np.int32),
        "scenario_id": np.asarray(scenario_uid_l, dtype=np.uint16),
        "data_mode_id": np.asarray(dm_int_l, dtype=np.uint8),
        "action_source_id": np.asarray(src_l, dtype=np.uint8),
        "collision": np.asarray(collision_l, dtype=np.float32),
        "episode_success": np.asarray(succ_l, dtype=np.float32),
        "episode_collision_any": np.asarray(ep_coll_any_rep, dtype=np.float32),
        "episode_return": np.asarray(ep_ret_rep, dtype=np.float32),
        "episode_length": np.asarray(ep_len_rep, dtype=np.float32),
        "quality_id": np.asarray(qual_l, dtype=np.uint8),
    }

    ew = episodic_stats_win
    ct = episodic_collision_tot

    agg = {
        "scenario": scenario,
        "scenario_uid": int(scenario_uid),
        "ep_win_rate": float(np.mean(ew)) if ew else 0.0,
        "collision_rate_episodes_any": float(np.mean([min(1, c) for c in ct])) if ct else 0.0,
        "collision_mean_steps": float(np.mean(ct)) if ct else 0.0,
        "mean_return": float(np.mean(episodic_returns)) if episodic_returns else 0.0,
        "mean_length": float(np.mean(episodic_lens)) if episodic_lens else 0.0,
        "n_episodes": int(n_episodes),
    }

    return pack, agg


def concatenate_pack(lists_dict: Mapping[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, lst in lists_dict.items():
        if not lst:
            continue
        out[k] = np.concatenate(lst, axis=0)
    return out


def collect_mixed_offline_dataset(offline_cfg: Mapping[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """按 mix_env_variants 生成完整混合数据集。"""
    base_n = int(offline_cfg.get("n_episodes", 500))
    max_steps = int(offline_cfg.get("max_steps", 200))
    base_seed = int(offline_cfg.get("seed", 42))

    variants = normalize_mix_variants(offline_cfg)
    if not variants:
        raise ValueError("mix_env_variants 为空；请配置 offline_data.mix_env_variants")

    ratios = [float(v.get("ratio", 0.0)) for v in variants]
    counts = allocate_episode_counts(base_n, ratios)

    scenario_names = [str(v["name"]) for v in variants]
    scenario_name_to_id = {n: i for i, n in enumerate(scenario_names)}

    lists: Dict[str, List[np.ndarray]] = {}
    scenario_aggs: List[Dict[str, Any]] = []
    global_ep = 0

    for var, n_eps in zip(variants, counts):
        if n_eps <= 0:
            continue
        sid = int(scenario_name_to_id[str(var["name"])])
        pack, agg = collect_transition_batch_for_variant(
            var,
            n_episodes=n_eps,
            max_steps=max_steps,
            seed_base=_mix_variant_seed(base_seed, sid, global_ep),
            global_episode_start=global_ep,
            scenario_uid=sid,
        )
        global_ep += n_eps
        scenario_aggs.append(agg)
        if not pack:
            continue
        for k, arr in pack.items():
            lists.setdefault(k, []).append(arr)

    merged = concatenate_pack(lists)
    meta = {
        "schema_version": 1,
        "data_mode_vocab": {k: int(v) for k, v in DATA_MODE_TO_INT.items()},
        "action_source_vocab": {k: int(v) for k, v in ACTION_SOURCE_TO_INT.items()},
        "quality_vocab": {"high": 0, "medium": 1, "low": 2},
        "scenario_name_to_id": scenario_name_to_id,
        "scenario_id_to_name": {i: n for n, i in scenario_name_to_id.items()},
        "per_scenario": scenario_aggs,
    }

    return merged, meta


def _mix_variant_seed(base_seed: int, scenario_uid: int, global_ep: int) -> int:
    return int(base_seed + scenario_uid * 10_000 + global_ep * 33)


def build_dataset_summary(
    merged: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
    total_episodes: int,
) -> Dict[str, Any]:
    n_tr = int(merged["obs"].shape[0]) if "obs" in merged else 0

    scenario_counts_ep: Dict[str, int] = {}
    trans_by_scenario: Dict[str, int] = defaultdict(int)
    success_rate_by_scenario: Dict[str, float] = {}
    collision_rate_by_scenario: Dict[str, float] = {}
    mean_return_by_scenario: Dict[str, float] = {}
    mean_length_by_scenario: Dict[str, float] = {}

    for a in meta.get("per_scenario", []):
        sc = a["scenario"]
        scenario_counts_ep[sc] = int(a.get("n_episodes", 0))
        success_rate_by_scenario[sc] = float(a["ep_win_rate"])
        collision_rate_by_scenario[sc] = float(a["collision_rate_episodes_any"])
        mean_return_by_scenario[sc] = float(a["mean_return"])
        mean_length_by_scenario[sc] = float(a["mean_length"])

    id_to_name = meta.get("scenario_id_to_name", {})
    if "episode_id" in merged and len(merged["episode_id"]):
        ep_ids = merged["episode_id"]
        scen_ids = merged["scenario_id"].reshape(-1)
        for eid in np.unique(ep_ids):
            mask = ep_ids == eid
            sid = int(scen_ids[np.flatnonzero(mask)[0]])
            name = id_to_name.get(sid, str(sid))
            trans_by_scenario[name] += int(mask.sum())

    quality_counts: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    if "quality_id" in merged:
        q = merged["quality_id"]
        quality_counts["high"] = int((q == 0).sum())
        quality_counts["medium"] = int((q == 1).sum())
        quality_counts["low"] = int((q == 2).sum())

    return {
        "total_episodes": total_episodes,
        "total_transitions": n_tr,
        "scenario_counts": scenario_counts_ep,
        "transition_counts_by_scenario": dict(trans_by_scenario),
        "success_rate_by_scenario": success_rate_by_scenario,
        "collision_rate_by_scenario": collision_rate_by_scenario,
        "mean_return_by_scenario": mean_return_by_scenario,
        "mean_length_by_scenario": mean_length_by_scenario,
        "quality_counts": quality_counts,
    }


def save_mixed_dataset(
    merged: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
    save_path: str,
    summary_path: Optional[str] = None,
    npz_extra_path: Optional[str] = None,
    total_episodes: int = 0,
) -> None:
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    bundle = {"arrays": dict(merged), "meta": dict(meta)}
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    summ = build_dataset_summary(merged, meta, total_episodes=total_episodes)
    sp = summary_path or os.path.join(os.path.dirname(save_path), "dataset_summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)

    if npz_extra_path:
        np.savez_compressed(npz_extra_path, **merged)


def load_mixed_bundle(path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=False)
        merged = {k: data[k] for k in data.files}
        return merged, {}
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["arrays"], bundle.get("meta", {})


def filter_for_bc(
    merged: Mapping[str, np.ndarray],
    use_data_modes: Sequence[str],
    label_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (obs, labels) 用于 BC。"""
    mode_set = {m.lower() for m in use_data_modes}
    allowed_ids = [DATA_MODE_TO_INT[m] for m in mode_set if m in DATA_MODE_TO_INT]
    if not allowed_ids:
        raise ValueError(f"use_data_modes 无有效条目: {use_data_modes}")
    dm_ids = merged["data_mode_id"].reshape(-1)
    dm_mask = np.isin(dm_ids, np.asarray(allowed_ids, dtype=np.uint8))

    obs = merged["obs"][dm_mask].astype(np.float32)
    if label_mode == "expert_action":
        y = merged["expert_actions"][dm_mask].astype(np.int64)
    elif label_mode == "action":
        y = merged["actions"][dm_mask].astype(np.int64)
    else:
        raise ValueError(f"未知 label_mode: {label_mode}")
    return obs, y


def filter_for_offline_rl_mask(merged: Mapping[str, np.ndarray], use_data_modes: Sequence[str]) -> np.ndarray:
    mode_set = {m.lower() for m in use_data_modes}
    allowed_ids = [DATA_MODE_TO_INT[m] for m in mode_set if m in DATA_MODE_TO_INT]
    if not allowed_ids:
        raise ValueError(f"use_data_modes 无有效条目: {use_data_modes}")
    dm_ids = merged["data_mode_id"].reshape(-1)
    return np.isin(dm_ids, np.asarray(allowed_ids, dtype=np.uint8))


def build_replay_buffer_from_merged(
    merged: Mapping[str, np.ndarray], mask: Optional[np.ndarray] = None
):
    """从混合数据集构造仅含 (obs, executed action, ...) 的经典 ReplayBuffer，供 DualReplay/PORL 等加载。"""
    from offline_rl.replay_buffer import ReplayBuffer

    n_all = len(merged["obs"])
    if mask is None:
        mask = np.ones(n_all, dtype=bool)
        n_tr = n_all
    else:
        n_tr = int(np.sum(mask))
    if n_tr == 0:
        raise ValueError("mask 后无可用 transition。")

    obs_s = merged["obs"][mask]
    act_s = merged["actions"][mask]
    rew_s = merged["rewards"][mask]
    next_s = merged["next_obs"][mask]
    done_s = merged["dones"][mask]

    rb = ReplayBuffer(capacity=max(n_tr, 1), obs_shape=(obs_s.shape[1],), action_dim=())
    rb.obs[:n_tr] = obs_s
    rb.actions[:n_tr] = act_s.reshape(rb.actions[:n_tr].shape)
    rb.rewards[:n_tr] = rew_s
    rb.next_obs[:n_tr] = next_s
    rb.dones[:n_tr] = done_s
    rb.size = n_tr
    rb.ptr = n_tr % rb.capacity
    return rb


def save_core_npz_by_mask(
    merged: Mapping[str, np.ndarray], mask: Optional[np.ndarray], path: str
) -> None:
    rb = build_replay_buffer_from_merged(merged, mask=mask)
    rb.save(path)
