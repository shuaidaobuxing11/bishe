"""
Streamlit Demo：双机协同追踪（Gym 环境）+ 可选 JSBSim/Stub 单机仿真。
在项目根目录 code/ 下运行:
  pip install -r requirements-demo.txt
  # 若使用 BC / PPO 策略，建议同时: pip install -r requirements.txt
  streamlit run demo/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st

from demo.jsbsim_minimal import create_minimal_simulator, is_jsbsim_available


def _plot_trajectory_2d(df: pd.DataFrame, arena_size: float) -> "object":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    lim = float(arena_size) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    ax.grid(True, alpha=0.35)

    if len(df) == 0:
        ax.set_title("无轨迹数据")
        return fig

    ax.plot(df["u1_x"], df["u1_y"], "-", color="#1f77b4", lw=1.8, label="UAV1")
    ax.plot(df["u2_x"], df["u2_y"], "-", color="#ff7f0e", lw=1.8, label="UAV2")
    ax.plot(df["tgt_x"], df["tgt_y"], "-", color="#2ca02c", lw=1.5, label="目标")
    ax.scatter([df["u1_x"].iloc[0]], [df["u1_y"].iloc[0]], c="#1f77b4", s=40, zorder=5, marker="o")
    ax.scatter([df["u2_x"].iloc[0]], [df["u2_y"].iloc[0]], c="#ff7f0e", s=40, zorder=5, marker="o")
    ax.scatter([df["tgt_x"].iloc[0]], [df["tgt_y"].iloc[0]], c="#2ca02c", s=40, zorder=5, marker="s")
    ax.scatter([df["tgt_x"].iloc[-1]], [df["tgt_y"].iloc[-1]], c="#2ca02c", s=80, zorder=5, marker="*", label="目标末点")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("平面轨迹（与 env 内状态一致，按 arena_size 反归一化）")
    fig.tight_layout()
    return fig


def _plot_trajectory_v3(df: pd.DataFrame, arena_size: float) -> "object":
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal", adjustable="box")
    lim = float(arena_size) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    ax.grid(True, alpha=0.35)

    if len(df) == 0:
        ax.set_title("无轨迹数据（V3）")
        return fig

    ax.plot(df["u1_x"], df["u1_y"], "-", color="#1f77b4", lw=1.8, label="UAV1")
    ax.plot(df["u2_x"], df["u2_y"], "-", color="#2ca02c", lw=1.8, label="UAV2")
    ax.plot(df["u3_x"], df["u3_y"], "-", color="#9467bd", lw=1.8, label="UAV3")
    ax.plot(df["tgt_x"], df["tgt_y"], "-", color="#d62728", lw=1.5, label="目标")

    ax.scatter([df["u1_x"].iloc[0]], [df["u1_y"].iloc[0]], c="#1f77b4", s=42, zorder=5)
    ax.scatter([df["u2_x"].iloc[0]], [df["u2_y"].iloc[0]], c="#2ca02c", s=42, zorder=5)
    ax.scatter([df["u3_x"].iloc[0]], [df["u3_y"].iloc[0]], c="#9467bd", s=42, zorder=5)
    ax.scatter([df["tgt_x"].iloc[0]], [df["tgt_y"].iloc[0]], c="#d62728", s=42, marker="s", zorder=5)
    ax.scatter([df["tgt_x"].iloc[-1]], [df["tgt_y"].iloc[-1]], c="#d62728", s=80, marker="*", zorder=6)

    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("三机协同轨迹（CoopTrackingEnvV3 · MultiDiscrete）")
    fig.tight_layout()
    return fig


def _run_coop_tracking_ui():
    st.subheader("双机协同追踪（`CoopTrackingEnv`）")
    st.caption(
        "与 `offline_rl` / `online_rl` 训练使用同一 Gym 环境；曲线由逐步 `obs` 反归一化得到，与仿真一致。"
        " JSBSim 页签为单机六自由度演示，动力学与本环境不同。"
    )

    try:
        from envs import make_coop_tracking
    except ModuleNotFoundError as e:
        st.error(f"无法导入 `envs`（需 gymnasium 等依赖）: {e}")
        st.info("请执行: `pip install -r requirements-demo.txt` 或完整 `pip install -r requirements.txt`")
        return

    from demo.coop_tracking_rollout import (
        EpisodeTrace,
        make_bc_policy,
        make_expert_v2_policy,
        make_ppo_policy,
        make_random_policy,
        rollout_one_episode,
    )

    col_l, col_r = st.columns([1, 2])
    with col_l:
        spawn_mode = st.selectbox(
            "初始分布 spawn_mode",
            ["default", "near_uavs", "near_border"],
            index=0,
        )
        noise_sigma = st.slider("目标扰动 noise_sigma", 0.0, 0.5, 0.0, 0.01)
        max_steps = st.slider("单回合最大步数", 50, 500, 200, 10)
        n_episodes = st.slider("回合数", 1, 50, 3, 1)
        base_seed = st.number_input("基础随机种子", value=0, step=1)
        policy_name = st.selectbox(
            "策略",
            ["规则专家 rule_policy_v2", "随机", "BC (.pt)", "PPO (.zip)"],
        )
        bc_path = st.text_input("BC 模型路径", value="models/bc_pretrain.pt")
        ppo_path = st.text_input("PPO 模型路径", value="models/ppo_coop.zip")
        run = st.button("运行双机追踪仿真", type="primary", key="run_coop")

    def build_policy():
        if policy_name.startswith("规则"):
            return make_expert_v2_policy()
        if policy_name == "随机":
            return make_random_policy()
        if policy_name.startswith("BC"):
            try:
                return make_bc_policy(bc_path)
            except Exception as e:
                st.error(f"加载 BC 失败: {e}")
                return None
        try:
            return make_ppo_policy(ppo_path)
        except Exception as e:
            st.error(f"加载 PPO 失败: {e}")
            return None

    with col_r:
        if not run:
            st.info("左侧配置后点击「运行双机追踪仿真」。")
            return

        act_fn = build_policy()
        if act_fn is None:
            return

        traces: list[EpisodeTrace] = []
        arena = 10.0
        for ep in range(int(n_episodes)):
            env = make_coop_tracking(
                spawn_mode=spawn_mode,
                noise_sigma=float(noise_sigma),
                max_episode_steps=int(max_steps),
            )
            arena = float(getattr(env, "arena_size", arena))
            seed_ep = int(base_seed) + ep * 9973
            tr = rollout_one_episode(env, act_fn, int(max_steps), seed_ep)
            traces.append(tr)

        summ = pd.DataFrame(
            {
                "episode": range(len(traces)),
                "return": [t.return_sum for t in traces],
                "length": [t.length for t in traces],
                "win": [t.win for t in traces],
                "collision": [t.collision for t in traces],
            }
        )
        st.metric("成功率", f"{summ['win'].mean():.1%}")
        st.dataframe(summ, use_container_width=True)

        pick = st.selectbox("查看轨迹与曲线的回合", options=list(range(len(traces))), format_func=lambda i: f"回合 {i}")
        df = traces[int(pick)].df

        if len(df) > 0:
            st.markdown("**距离与回报（逐步）**")
            chart_df = df.set_index("step")[["d1", "d2", "d12", "cum_reward"]]
            st.line_chart(chart_df)

            st.markdown("**2D 轨迹（等比例坐标）**")
            fig = _plot_trajectory_2d(df, arena)
            st.pyplot(fig)
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "下载本回合逐步 CSV",
                data=csv_bytes,
                file_name=f"coop_trace_ep{pick}.csv",
                mime="text/csv",
            )
        else:
            st.warning("该回合无数据（可能立即终止）。")


def _run_coop_tracking_v3_ui():
    st.subheader("三机协同追踪 · `CoopTrackingEnvV3`")
    st.caption(
        "MultiDiscrete([5,5,5]) 独立离散策略；与环境 `envs/coop_tracking_env_v3.py`、"
        "脚本 `scripts/train_ppo_v3.py` / `scripts/eval_v3_policy.py` 一致。"
    )

    try:
        from demo.coop_tracking_rollout_v3 import (
            EpisodeTraceV3,
            make_ppo_policy_v3,
            make_random_policy_v3,
            rollout_one_episode_v3,
        )
        from envs import make_coop_tracking_v3
    except ModuleNotFoundError as e:
        st.error(f"导入失败: {e}")
        st.info("`pip install -r requirements-demo.txt` 并包含 gymnasium / numpy / matplotlib；PPO 需 SB3。")
        return

    col_l, col_r = st.columns([1, 2])
    with col_l:
        max_steps = st.slider("单回合最大步数（V3）", 50, 500, 200, 10, key="v3_steps")
        n_episodes = st.slider("回合数（V3）", 1, 30, 5, 1, key="v3_eps")
        base_seed = st.number_input("基础随机种子（V3）", value=101, step=1, key="v3_seed")
        policy_pick = st.selectbox("策略（V3）", ["随机 MultiDiscrete", "PPO (.zip)"], key="v3_pol")
        ppo_v3_path = st.text_input("V3 PPO 模型路径", value="models/ppo_v3/ppo_coop_v3.zip", key="v3_zip")
        run_v3 = st.button("运行三机协同仿真", type="primary", key="run_v3")

    def build_policy():
        if policy_pick.startswith("随机"):
            return make_random_policy_v3()
        try:
            return make_ppo_policy_v3(ppo_v3_path)
        except Exception as exc:
            st.error(f"加载 PPO-V3 失败: {exc}")
            return None

    with col_r:
        if not run_v3:
            st.info("左侧配置后点击「运行三机协同仿真」。若尚无模型，可先 `python scripts/train_ppo_v3.py`。")
            return

        act_fn = build_policy()
        if act_fn is None:
            return

        traces: list[EpisodeTraceV3] = []
        arena = 10.0
        for ep in range(int(n_episodes)):
            env = make_coop_tracking_v3(max_episode_steps=int(max_steps))
            arena = float(getattr(env, "arena_size", arena))
            tr = rollout_one_episode_v3(env, act_fn, int(max_steps), int(base_seed) + ep * 9137)
            traces.append(tr)

        summ = pd.DataFrame(
            {
                "episode": range(len(traces)),
                "return": [t.return_sum for t in traces],
                "length": [t.length for t in traces],
                "win": [t.win for t in traces],
                "collision": [t.collision for t in traces],
            }
        )
        st.metric("成功率", f"{summ['win'].mean():.1%}")
        st.metric("平均回报", f"{summ['return'].mean():.2f}")
        st.dataframe(summ, use_container_width=True)

        success_only = [i for i, t in enumerate(traces) if t.win]
        fail_only = [i for i, t in enumerate(traces) if not t.win]
        view_mode = st.radio("案例筛选", ["全部回合", "仅成功", "仅失败"], horizontal=True, key="v3_view")

        indices = list(range(len(traces)))
        if view_mode == "仅成功":
            indices = success_only if success_only else indices
            if not success_only:
                st.warning("本批无成功案例，退回显示全部回合。")
        elif view_mode == "仅失败":
            indices = fail_only if fail_only else indices
            if not fail_only:
                st.warning("本批无失败案例，退回显示全部回合。")

        pick_local = st.selectbox(
            "查看轨迹（按筛选列表）",
            options=list(range(len(indices))),
            format_func=lambda j: f"列表项 {j} → episode {indices[j]}",
            key="v3_pick",
        )
        pick = int(indices[int(pick_local)])
        df = traces[pick].df

        if len(df) > 0:
            st.markdown("**回报累计（逐步）**")
            chart_df = df.set_index("step")[["cum_reward"]]
            st.line_chart(chart_df)

            fig = _plot_trajectory_v3(df, arena)
            st.pyplot(fig)
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception:
                pass

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "下载本回合 CSV（V3）",
                data=csv_bytes,
                file_name=f"coop_v3_trace_ep{pick}.csv",
                mime="text/csv",
            )
        else:
            st.warning("该回合无数据。")


def _run_trajectory_viz_ui():
    st.subheader("统一策略轨迹可视化")
    st.caption("Expert / BC / PPO 同 seed 对比 · 支持逐步动画与指标表")

    anim_dir = os.path.join(_ROOT, "results", "animations")
    traj_dir = os.path.join(_ROOT, "results", "trajectories")
    os.makedirs(anim_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)

    col_l, col_r = st.columns([1, 2])
    with col_l:
        scenario = st.selectbox(
            "场景",
            ["default", "near_uavs", "near_border", "noisy_target"],
            key="traj_scenario",
        )
        seed = st.selectbox("seed", [42, 123, 2026], key="traj_seed")
        policy_pick = st.multiselect(
            "策略（展示逐步表）",
            ["expert_v2", "bc", "ppo_baseline", "ppo_finetune", "ppo_finetune_kl"],
            default=["expert_v2", "bc", "ppo_baseline", "ppo_finetune"],
            key="traj_policies",
        )
        btn_record = st.button("录制所有策略轨迹", type="primary", key="btn_rec_all")
        btn_anim = st.button("生成同步对比动画+汇总表", key="btn_make_all")
        view_type = st.radio(
            "展示内容",
            ["同步对比 GIF", "静态对比 PNG", "单策略 stepwise GIF", "step_table", "metrics_summary"],
            key="traj_view",
        )

    suffix = f"_{scenario}_seed{seed}"
    compare_gif = os.path.join(anim_dir, f"compare_all_policies{suffix}.gif")
    compare_png = os.path.join(anim_dir, f"compare_all_policies{suffix}.png")
    step_csv = os.path.join(anim_dir, f"step_table{suffix}.csv")
    metrics_csv = os.path.join(anim_dir, f"metrics_summary{suffix}.csv")

    with col_r:
        if btn_record:
            import subprocess

            cmd = [
                sys.executable,
                os.path.join(_ROOT, "scripts", "record_all_policy_trajectories.py"),
                "--scenario",
                scenario,
                "--seed",
                str(seed),
                "--save_dir",
                traj_dir,
            ]
            with st.spinner("录制中…"):
                r = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                st.success(r.stdout or "录制完成")
            else:
                st.error(r.stderr or r.stdout)

        if btn_anim:
            import subprocess

            cmd = [
                sys.executable,
                os.path.join(_ROOT, "scripts", "make_all_policy_comparison_animation.py"),
                "--scenario",
                scenario,
                "--seed",
                str(seed),
                "--trajectory_dir",
                traj_dir,
                "--save_dir",
                anim_dir,
            ]
            with st.spinner("生成动画与汇总…"):
                r = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
            if r.returncode == 0:
                st.success(r.stdout or "生成完成")
            else:
                st.error(r.stderr or r.stdout)

        if view_type == "同步对比 GIF" and os.path.isfile(compare_gif):
            st.image(compare_gif)
        elif view_type == "静态对比 PNG" and os.path.isfile(compare_png):
            st.image(compare_png)
        elif view_type.startswith("单策略"):
            gifs = sorted(
                f
                for f in os.listdir(anim_dir)
                if f.endswith("_stepwise.gif") and suffix in f
            )
            if gifs:
                pick = st.selectbox("stepwise 文件", gifs, key="pick_stepwise")
                st.image(os.path.join(anim_dir, pick))
            else:
                st.info("尚无 stepwise GIF，请先点击「生成同步对比动画+汇总表」。")
        elif view_type == "step_table":
            if os.path.isfile(step_csv):
                df = pd.read_csv(step_csv)
                if policy_pick:
                    df = df[df["policy_name"].isin(policy_pick)]
                st.dataframe(df, use_container_width=True, height=400)
            else:
                st.info(f"未找到 {step_csv}")
        elif view_type == "metrics_summary":
            if os.path.isfile(metrics_csv):
                mdf = pd.read_csv(metrics_csv)
                cols = [
                    "policy_name",
                    "success",
                    "episode_return",
                    "episode_length",
                    "collision",
                    "min_uav_distance",
                    "action_switch_rate",
                ]
                show = [c for c in cols if c in mdf.columns]
                st.dataframe(mdf[show], use_container_width=True)
            else:
                st.info(f"未找到 {metrics_csv}")

        st.markdown("**轨迹 JSON（当前 seed）**")
        if os.path.isdir(traj_dir):
            jsons = sorted(f for f in os.listdir(traj_dir) if f.endswith(suffix + ".json"))
            st.code("\n".join(jsons) if jsons else "（无）")


def _run_explain_ui():
    st.subheader("自动解释报告")
    st.caption("轨迹 → 特征分组 → 场景匹配 → Markdown 报告（Expert 为规则解释，BC/PPO 可用 Captum）")

    report_root = os.path.join(_ROOT, "results", "explain_reports")
    comp_dir = os.path.join(_ROOT, "results", "explain_comparisons")
    traj_dir = os.path.join(_ROOT, "results", "trajectories")

    col_l, col_r = st.columns([1, 2])
    with col_l:
        policy = st.selectbox(
            "策略",
            ["expert_v2", "bc", "ppo_baseline", "ppo_finetune", "ppo_finetune_kl"],
            key="exp_policy",
        )
        scenario = st.selectbox(
            "场景",
            ["default", "near_uavs", "near_border", "noisy_target"],
            key="exp_scenario",
        )
        seed = st.selectbox("seed", [42, 123, 2026], key="exp_seed")
        btn_one = st.button("生成单策略解释报告", type="primary", key="btn_exp_one")
        btn_batch = st.button("批量生成所有策略报告", key="btn_exp_batch")
        btn_cmp_pol = st.button("对比不同策略解释", key="btn_cmp_pol")
        view = st.selectbox(
            "展示",
            ["report.md", "fig_feature_importance.png", "fig_group_importance.png", "explain_scores.json", "keyframes.md", "counterfactual.csv"],
            key="exp_view",
        )

    report_dir = os.path.join(report_root, f"{policy}_{scenario}_seed{seed}")

    with col_r:
        if btn_one:
            import subprocess

            traj = os.path.join(traj_dir, f"{policy}_{scenario}_seed{seed}.json")
            cmd = [
                sys.executable,
                os.path.join(_ROOT, "scripts", "generate_episode_explain_report.py"),
                "--policy_name",
                policy,
                "--scenario",
                scenario,
                "--seed",
                str(seed),
                "--trajectory_path",
                traj,
            ]
            with st.spinner("生成报告…"):
                r = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
            st.success(r.stdout) if r.returncode == 0 else st.error(r.stderr or r.stdout)

        if btn_batch:
            import subprocess

            cmd = [
                sys.executable,
                os.path.join(_ROOT, "scripts", "batch_generate_explain_reports.py"),
                "--scenario",
                scenario,
                "--seed",
                str(seed),
            ]
            with st.spinner("批量生成…"):
                r = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
            st.success(r.stdout) if r.returncode == 0 else st.error(r.stderr or r.stdout)

        if btn_cmp_pol:
            import subprocess

            cmd = [
                sys.executable,
                os.path.join(_ROOT, "scripts", "compare_policy_explanations.py"),
                "--scenario",
                scenario,
                "--seed",
                str(seed),
            ]
            with st.spinner("策略对比…"):
                r = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
            st.success(r.stdout) if r.returncode == 0 else st.error(r.stderr or r.stdout)

        if os.path.isdir(report_dir):
            fpath = os.path.join(report_dir, view)
            if view == "report.md" and os.path.isfile(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    st.markdown(f.read())
            elif view.endswith(".png") and os.path.isfile(fpath):
                st.image(fpath)
            elif view.endswith(".json") and os.path.isfile(fpath):
                st.json(json.load(open(fpath, "r", encoding="utf-8")))
            elif view.endswith(".md") and os.path.isfile(fpath):
                st.markdown(open(fpath, "r", encoding="utf-8").read())
            elif view.endswith(".csv") and os.path.isfile(fpath):
                st.dataframe(pd.read_csv(fpath), use_container_width=True)
            else:
                st.info(f"未找到 {fpath}，请先生成报告。")
        else:
            st.info(f"报告目录不存在: {report_dir}")

        cmp_md = os.path.join(comp_dir, f"policy_compare_{scenario}_seed{seed}.md")
        cmp_groups = os.path.join(comp_dir, f"policy_compare_{scenario}_seed{seed}_groups.png")
        cmp_fine = os.path.join(comp_dir, f"policy_compare_{scenario}_seed{seed}_fine_grained_scores.png")
        cmp_heat = os.path.join(comp_dir, f"policy_compare_{scenario}_seed{seed}_heatmap.png")
        cmp_scores = os.path.join(comp_dir, f"policy_compare_{scenario}_seed{seed}_scores.png")
        cmp_perf = os.path.join(comp_dir, f"policy_compare_{scenario}_seed{seed}_perf_joint.png")
        if os.path.isfile(cmp_md):
            st.markdown("---")
            st.markdown("**策略对比摘要**")
            st.markdown(open(cmp_md, "r", encoding="utf-8").read())
        if os.path.isfile(cmp_fine):
            st.image(cmp_fine, caption="细粒度可解释性评分柱状图")
        if os.path.isfile(cmp_heat):
            st.image(cmp_heat, caption="可解释性评分热力图")
        if os.path.isfile(cmp_groups):
            st.image(cmp_groups, caption="特征组贡献热力图")
        if os.path.isfile(cmp_scores):
            st.image(cmp_scores, caption="可解释性评分柱状图（兼容）")
        if os.path.isfile(cmp_perf):
            st.image(cmp_perf, caption="性能-解释联合图")

        if policy == "expert_v2":
            st.info(
                "expert_v2 解释分数来自规则逻辑与轨迹统计（rule_and_trajectory），"
                "不与 PPO/BC 的 Captum 梯度归因完全等价。"
            )

        st.markdown("---")
        st.caption(
            "**解释说明**：target_absolute 表示对目标绝对位置的关注；"
            "target_relative 表示对 UAV 与目标相对关系的关注；"
            "inter_uav_coordination 表示对双机协同关系的关注；"
            "safety 表示对碰撞/边界风险的关注；"
            "velocity 表示对运动趋势的关注。"
        )


def main():
    st.set_page_config(page_title="毕设 Demo", layout="wide")
    st.title("双机协同追踪 RL 项目 · 可视化 Demo")

    tab_coop, tab_v3, tab_traj, tab_explain, tab_sim, tab_rl = st.tabs(
        [
            "双机协同追踪仿真",
            "三机协同 (V3)",
            "轨迹动画",
            "策略可解释性",
            "JSBSim / Stub 单机",
            "RL 工程说明",
        ]
    )

    with tab_coop:
        _run_coop_tracking_ui()

    with tab_v3:
        _run_coop_tracking_v3_ui()

    with tab_traj:
        _run_trajectory_viz_ui()

    with tab_explain:
        _run_explain_ui()

    with tab_sim:
        st.caption("单机最小接口：用于展示 JSBSim 或 Stub 状态曲线，与双机 2D 追踪环境非同一动力学模型。")
        col_l, col_r = st.columns([1, 2])
        with col_l:
            use_jsb = st.checkbox(
                "优先使用真实 JSBSim",
                value=False,
                help="需 pip install jsbsim 且配置 JSBSIM_ROOT；失败时自动使用 Stub。",
            )
            if use_jsb and not is_jsbsim_available():
                st.warning("当前环境未检测到 `jsbsim` 包，将使用 Stub。")
            dt = st.slider("步长 dt (s)", 0.01, 0.2, 0.05, 0.01, key="dt_jsb")
            n_steps = st.slider("仿真步数", 20, 2000, 400, 20, key="n_jsb")
            throttle = st.slider("油门 throttle", 0.0, 1.0, 0.65, 0.01, key="thr")
            pitch_cmd = st.slider("俯仰指令 pitch_cmd_rad", -0.2, 0.2, 0.0, 0.01, key="pch")
            seed = st.number_input("随机种子（Stub reset）", value=42, step=1, key="seed_jsb")
            run = st.button("运行单机仿真", type="primary", key="run_jsb")

        with col_r:
            if run:
                sim = create_minimal_simulator(prefer_jsbsim=use_jsb)
                sim.reset(seed=int(seed))
                for _ in range(int(n_steps) - 1):
                    sim.step(float(dt), {"throttle": float(throttle), "pitch_cmd_rad": float(pitch_cmd)})
                df = sim.get_history_dataframe()
                st.subheader("状态曲线")
                chart_cols = [c for c in df.columns if c not in ("mode",)]
                if "time_s" in df.columns:
                    ycols = [c for c in chart_cols if c != "time_s"]
                    df_plot = df.set_index("time_s")[ycols]
                else:
                    df_plot = df
                st.line_chart(df_plot)
                st.subheader("末帧状态")
                st.json(df.iloc[-1].to_dict())
                st.download_button(
                    "下载 CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="demo_flight_trace.csv",
                    mime="text/csv",
                )
            else:
                st.info("左侧设置参数后点击「运行单机仿真」。")

    with tab_rl:
        st.markdown(
            """
### 与主工程的关系

- **训练 / 评估**：在 `envs/`、`offline_rl/`、`online_rl/`、`scripts/` 中，按根目录 `README.md` 执行。
- **双机追踪页**：直接 `make_coop_tracking` + rollout，与训练环境一致。
- **JSBSim 页**：独立单机演示，可选安装 `jsbsim`。

### 常用命令（不在此页执行）

```bash
python scripts/generate_offline_data.py
python online_rl/train_online_baseline.py --total_timesteps 200000
python online_rl/train_online_finetune.py --total_timesteps 200000 --bc_path models/bc_pretrain.pt
python online_rl/eval_policies.py --n_episodes 100
python scripts/train_ppo_v3.py --config configs/ppo_v3.yaml
python scripts/eval_v3_policy.py --model_path models/ppo_v3/ppo_coop_v3.zip --n_episodes 50
```
            """
        )


if __name__ == "__main__":
    main()
