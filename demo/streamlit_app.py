"""
Streamlit Demo：双机协同追踪（Gym 环境）+ 可选 JSBSim/Stub 单机仿真。
在项目根目录 code/ 下运行:
  pip install -r requirements-demo.txt
  # 若使用 BC / PPO 策略，建议同时: pip install -r requirements.txt
  streamlit run demo/streamlit_app.py
"""
from __future__ import annotations

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


def main():
    st.set_page_config(page_title="毕设 Demo", layout="wide")
    st.title("双机协同追踪 RL 项目 · 可视化 Demo")

    tab_coop, tab_v3, tab_sim, tab_rl = st.tabs(
        [
            "双机协同追踪仿真",
            "三机协同 (V3)",
            "JSBSim / Stub 单机",
            "RL 工程说明",
        ]
    )

    with tab_coop:
        _run_coop_tracking_ui()

    with tab_v3:
        _run_coop_tracking_v3_ui()

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
