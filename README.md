# 可解释的双机协同追踪单目标：离线数据转在线学习策略

毕设目标：在**同等训练条件**下，离线预训练 + 在线微调策略的**回合奖励总和**与**胜率**不低于同场景**纯在线强化学习**方法。

## 环境要求

- Python 3.11（推荐 Conda）
- 见 `requirements.txt` 或 `environment.yml`

### Conda 创建环境

```bash
conda env create -f environment.yml
conda activate bishe
```

或手动：

```bash
conda create -n bishe python=3.11 -y
conda activate bishe
pip install -r requirements.txt
```

## 项目结构

```
code/
├── envs/                    # 双机协同追踪环境（离散动作 0~24）
├── offline_rl/              # 离线数据采集、ReplayBuffer、行为克隆(BC)
├── online_rl/               # 纯在线 PPO 基线、BC+在线微调、统一评估
├── explain/                 # 可解释性：特征重要性、决策树规则
├── configs/                 # 环境 / 离线 / 在线 配置
├── scripts/                 # 一键生成离线数据、跑全流程、绘图
├── data/offline/            # 离线轨迹数据（脚本生成）
├── models/                  # 保存的模型（基线、BC、微调）
├── environment.yml
├── requirements.txt
└── README.md
```

## 可视化 Demo（Streamlit，独立模块）

不依赖训练脚本，用于答辩/系统展示入口：

```bash
pip install -r requirements-demo.txt
streamlit run demo/streamlit_app.py
```

说明见 `demo/README.md`。默认使用内置 **Stub** 飞行仿真；可选接入 **JSBSim**（需自行安装并配置 `JSBSIM_ROOT`）。

## 快速运行

在项目根目录 `code/` 下执行。

### 1. 生成离线数据并训练 BC

**默认配置**（`configs/offline_config.yaml`，含混合场景权重）：

```bash
python scripts/generate_offline_data.py
```

会生成：`data/offline/mixed_offline_dataset.pkl`（含 `expert_action`、`data_mode`、质量等元信息）、`dataset_summary.json`、`offline_trajectories_full.npz`，以及兼容旧代码的 **`data/offline/offline_trajectories.npz`**（仅五元组，动作为环境中**实际执行**的动作），并完成 BC（默认仅用 `bc_train.use_data_modes` 中的 **`expert` + `recovery`**，标签为 **`expert_action`**）。

**完整「多场景 + recovery + suboptimal」示例（800 局）**：

```bash
python scripts/generate_offline_data.py --config configs/offline_data_mixed.yaml
```

仅训练 BC（数据已生成时）：

```bash
python scripts/train_bc.py --config configs/bc_config.yaml
```

为离线 RL / PORL 导出含 `suboptimal` 的 npz（实际执行动作）：

```bash
python scripts/train_offline_rl.py --config configs/offline_rl_training.yaml
```

### 2. 纯在线 PPO 基线（从零训练）

```bash
python online_rl/train_online_baseline.py --total_timesteps 200000
```

输出模型：`models/ppo_online_baseline.zip`。

### 3. 离线 + 在线微调（相同步数）

```bash
python online_rl/train_online_finetune.py --total_timesteps 200000 --bc_path models/bc_pretrain.pt
```

输出模型：`models/ppo_finetune_from_bc.zip`。

### 4. 评估对比（奖励与胜率）

```bash
python online_rl/eval_policies.py --n_episodes 100
```

在相同测试种子下对比两种策略的**平均回合奖励**和**胜率**，用于验证「离线+在线」不低于纯在线。

### 一键跑全流程

```bash
python scripts/run_all_experiments.py
```

依次执行：生成离线数据 → BC 预训练 → 纯在线基线 → 离线+在线微调 → 评估。

## 可解释性

- **特征重要性**（基于策略网络权重或梯度）：
  ```bash
  python -c "from explain.feature_importance import print_feature_importance; print_feature_importance('models/ppo_finetune_from_bc.zip')"
  ```
- **决策树规则**（状态→动作）：
  ```bash
  python -c "from explain.rule_extraction import fit_and_export_rules; fit_and_export_rules('models/ppo_finetune_from_bc.zip')"
  ```

## 公平对比说明

- `total_timesteps` 在基线与微调中设为相同（如 200_000），保证**同等训练条件**。
- 评估时使用相同 `n_episodes` 与 `seed`，比较**平均回合奖励**与**胜率**，从而验证离线转在线策略不低于纯在线。

## 配置

- `configs/env_config.yaml`：环境参数（步数、区域、速度、成功/碰撞距离等）。
- `configs/offline_config.yaml`：混合离线比例、`bc_train`（BC 用哪些 `data_mode`、监督标签）。
- `configs/offline_data_mixed.yaml`：论文级混合数据示例（default / 难 spawn / 噪声 / recovery / suboptimal）。
- `configs/bc_config.yaml`：仅从 pickle 训练 BC。
- `configs/offline_rl_training.yaml`：按 `data_mode` 过滤并导出 RL 用 npz。
- `configs/online_config.yaml`：PPO 超参（步数、学习率、网络结构等）。

## 许可与引用

仅供毕业设计使用；若引用请注明出处。
