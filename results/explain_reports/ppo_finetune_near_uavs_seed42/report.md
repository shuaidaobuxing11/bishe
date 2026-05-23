# 策略可解释性报告

## 1. 基本信息
- **策略**: ppo_finetune
- **场景**: near_uavs
- **seed**: 42
- **解释来源**: Captum IG + 轨迹统计 (`captum_and_trajectory`)
- **是否成功**: 是
- **episode length**: 23
- **episode return**: -26.13
- **是否碰撞**: 否
- **最小 UAV 间距**: 0.676
- **平均目标距离**: 6.115
- **最小目标距离**: 0.573
- **action switch rate**: 0.045

## 2. 轨迹行为分析
- 动作切换频率处于较低水平，控制相对平稳。
- 双机间距整体保持在相对安全范围。
- 平均目标距离呈下降趋势，策略能够稳定接近目标。
- 在较少步数内完成任务，路径效率较高。

## 3. 特征重要性分析（Top 10）
- uav2_y_norm: 14.7658
- uav1_y_norm: 13.6706
- target_x_norm: 6.9560
- target_y_norm: 5.4274
- uav2_x_norm: 4.2333
- uav1_x_norm: 2.3669
- uav1_vx_norm: 0.7679
- uav1_vy_norm: 0.6950
- uav2_vx_norm: 0.2793
- uav2_vy_norm: 0.1800

该策略在多个空间状态特征上均有贡献，决策依据较为分散。

## 4. 细粒度特征类别贡献
- **目标绝对位置** (`target_absolute_score`): 0.132
- **UAV 绝对位置** (`uav_absolute_score`): 0.373
- **UAV-目标相对关系** (`target_relative_score`): 0.379
- **双机协同** (`inter_uav_coordination_score`): 0.134
- **安全约束** (`safety_score`): 0.158
- **运动趋势/速度** (`velocity_score`): 0.020
- **动作稳定性** (`action_stability_score`): 0.955
- **任务效率** (`task_efficiency_score`): 0.721
- **场景匹配度** (`scenario_alignment_score`): 0.187

### Captum/轨迹组级归一化贡献
- **uav_absolute**: 57.4%
- **target_absolute**: 20.3%
- **target_relative**: 16.8%
- **velocity**: 3.1%
- **inter_uav_coordination**: 2.4%

Captum/轨迹融合后，**uav_absolute** 类特征贡献最高。 对 velocity, inter_uav_coordination 类特征关注较少。 （语义分组允许重叠，如双机距离同时计入协同与安全。）

### 自动解释
- 该策略较关注 UAV 与目标之间的相对关系，说明其动作选择与追踪任务目标直接相关。
- 该策略能够较快完成追踪任务，任务执行效率较高。

### 分组提示
- 观测中无相对距离特征，target_relative / inter_uav 将结合 trajectory 距离字段补充。

## 5. 场景匹配解释
- **场景匹配度**: 0.19
- 近距离初始场景中双机碰撞风险较高，策略应重点关注 UAV 间距、安全约束和协同队形。 该策略在 near_uavs 场景中主要关注 inter_uav_coordination, safety, target_relative，与场景需求基本一致。

## 6. 成功/失败原因
该策略成功的主要原因可能是能够持续关注目标相对关系或目标位置，并保持较稳定的接近过程。

## 7. 策略专项说明
解释来源：Captum Integrated Gradients + 轨迹统计。 PPO finetune 仍受 BC 初始化影响；

## 8. 自动总结
总体来看，**ppo_finetune** 在 **near_uavs** 场景下成功完成追踪。 episode return=-26.13，目标相对关系=0.38，双机协同=0.13，安全=0.16，场景匹配=0.19。