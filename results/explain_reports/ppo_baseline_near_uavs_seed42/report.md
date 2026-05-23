# 策略可解释性报告

## 1. 基本信息
- **策略**: ppo_baseline
- **场景**: near_uavs
- **seed**: 42
- **解释来源**: Captum IG + 轨迹统计 (`captum_and_trajectory`)
- **是否成功**: 是
- **episode length**: 23
- **episode return**: -26.12
- **是否碰撞**: 否
- **最小 UAV 间距**: 0.673
- **平均目标距离**: 6.114
- **最小目标距离**: 0.542
- **action switch rate**: 0.000

## 2. 轨迹行为分析
- 动作切换频率处于较低水平，控制相对平稳。
- 双机间距整体保持在相对安全范围。
- 平均目标距离呈下降趋势，策略能够稳定接近目标。
- 在较少步数内完成任务，路径效率较高。

## 3. 特征重要性分析（Top 10）
- target_y_norm: 0.1616
- uav2_y_norm: 0.0826
- target_x_norm: 0.0655
- uav1_x_norm: 0.0342
- uav2_x_norm: 0.0310
- uav1_y_norm: 0.0201
- uav2_vx_norm: 0.0065
- uav1_vy_norm: 0.0054
- uav1_vx_norm: 0.0028
- uav2_vy_norm: 0.0002

该策略主要关注目标位置相关特征，动作选择与追踪目标空间关系密切相关。

## 4. 细粒度特征类别贡献
- **目标绝对位置** (`target_absolute_score`): 0.012
- **UAV 绝对位置** (`uav_absolute_score`): 0.009
- **UAV-目标相对关系** (`target_relative_score`): 0.823
- **双机协同** (`inter_uav_coordination_score`): 0.197
- **安全约束** (`safety_score`): 0.157
- **运动趋势/速度** (`velocity_score`): 0.001
- **动作稳定性** (`action_stability_score`): 1.000
- **任务效率** (`task_efficiency_score`): 0.721
- **场景匹配度** (`scenario_alignment_score`): 0.212

### Captum/轨迹组级归一化贡献
- **target_relative**: 84.4%
- **inter_uav_coordination**: 12.2%
- **target_absolute**: 1.9%
- **uav_absolute**: 1.4%
- **velocity**: 0.1%

Captum/轨迹融合后，**target_relative** 类特征贡献最高。 对 target_absolute, uav_absolute, velocity 类特征关注较少。 （语义分组允许重叠，如双机距离同时计入协同与安全。）

### 自动解释
- 该策略较关注 UAV 与目标之间的相对关系，说明其动作选择与追踪任务目标直接相关。
- 该策略较关注双机之间的空间关系，说明其具有一定协同控制倾向。
- 该策略能够较快完成追踪任务，任务执行效率较高。

### 分组提示
- 观测中无相对距离特征，target_relative / inter_uav 将结合 trajectory 距离字段补充。

## 5. 场景匹配解释
- **场景匹配度**: 0.21
- 近距离初始场景中双机碰撞风险较高，策略应重点关注 UAV 间距、安全约束和协同队形。 该策略在 near_uavs 场景中主要关注 inter_uav_coordination, safety, target_relative，与场景需求基本一致。

## 6. 成功/失败原因
该策略成功的主要原因可能是能够持续关注目标相对关系或目标位置，并保持较稳定的接近过程。

## 7. 策略专项说明
解释来源：Captum Integrated Gradients + 轨迹统计。 PPO baseline 通过在线交互学习；若 target_relative 与 inter_uav_coordination 较高且回报较高，说明其有效利用了相对目标关系与双机协同。

## 8. 自动总结
总体来看，**ppo_baseline** 在 **near_uavs** 场景下成功完成追踪。 episode return=-26.12，目标相对关系=0.82，双机协同=0.20，安全=0.16，场景匹配=0.21。