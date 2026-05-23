# 三机协同追踪（CoopTrackingEnvV3）

在现有「双机 + 离散联合动作」框架外，最小扩展为三机、`MultiDiscrete([5,5,5])` 独立离散动作、`stable-baselines3` PPO 训练与评估，用于演示框架可扩展性。

## 依赖

```bash
pip install -r requirements.txt
```

需包含：`gymnasium`、`numpy`、`matplotlib`、`stable-baselines3`、`torch`、`pyyaml`。

## 快速自检环境

```bash
python scripts/test_v3_env.py
python scripts/test_v3_env.py --no-render
```

`--no-render` 跳过 Matplotlib，适合无显示环境 CI。

## 训练 PPO

```bash
python scripts/train_ppo_v3.py --config configs/ppo_v3.yaml
```

可选：`--total_timesteps`、`--seed`。

- 默认保存：`models/ppo_v3/ppo_coop_v3.zip`
- 训练结束小规模评估：`results/v3/train_end_metrics.json`

## 评估策略

```bash
python scripts/eval_v3_policy.py --model_path models/ppo_v3/ppo_coop_v3.zip --n_episodes 100
```

输出控制台 JSON，并写入 `results/v3/eval_latest.json`。

## Streamlit Demo

在项目根：

```bash
streamlit run demo/streamlit_app.py
```

打开 **「三机协同 (V3)」** 页签：可跑随机/PPO、`success_rate`、`mean_return`、2D 轨迹与成功案例筛选。

## 动作与观测概要

| 项目 | 说明 |
|------|------|
| 动作 | `spaces.MultiDiscrete([5,5,5])`，每架 0–4：保持 / 左转 / 右转 / 加速 / 减速 |
| 观测 | 扁平 `float32`，三机状态（×7） + 目标（×5） + 两两间距（×3） |

环境与双机版本文件隔离：`envs/coop_tracking_env.py` **未改动**；三机实现在 `envs/coop_tracking_env_v3.py`。

## Render / 截图

训练或测试中使用 `render_mode="human"` 时，`CoopTrackingEnvV3.render()` 会绘制蓝/绿/紫三架 UAV 轨迹与红色目标，并在标题栏显示当前步数与成败状态。

无显示器环境可直接生成 PNG：

```bash
python scripts/render_v3_oneframe.py --out results/v3/v3_trajectory_demo.png --steps 150
```
