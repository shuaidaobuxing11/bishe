# 策略可解释性报告

## 1. 基本信息
- **策略**: bc
- **场景**: near_uavs
- **seed**: 42
- **解释来源**: Captum IG + 轨迹统计 (`captum_and_trajectory`)
- **是否成功**: 否
- **episode length**: 200
- **episode return**: -152.91
- **是否碰撞**: 是
- **最小 UAV 间距**: 0.110
- **平均目标距离**: 3.223
- **最小目标距离**: 0.996
- **action switch rate**: 0.618

## 2. 轨迹行为分析
- 该策略动作切换频率较高，控制过程存在一定抖动。
- 执行过程中双机间距曾接近碰撞阈值，存在较高碰撞风险。
- 平均目标距离呈下降趋势，策略能够稳定接近目标。
- 未在限定步数内完成协同追踪任务。

## 3. 特征重要性分析（Top 10）
- uav1_y_norm: 40.8530
- target_y_norm: 38.2467
- uav1_x_norm: 35.5727
- target_x_norm: 32.4398
- uav2_y_norm: 13.8567
- uav2_x_norm: 11.4469
- uav1_vy_norm: 2.9229
- uav2_vy_norm: 2.3164
- uav1_vx_norm: 1.9973
- uav2_vx_norm: 1.1829

该策略在多个空间状态特征上均有贡献，决策依据较为分散。

## 4. 细粒度特征类别贡献
- **目标绝对位置** (`target_absolute_score`): 0.233
- **UAV 绝对位置** (`uav_absolute_score`): 0.335
- **UAV-目标相对关系** (`target_relative_score`): 0.230
- **双机协同** (`inter_uav_coordination_score`): 0.054
- **安全约束** (`safety_score`): 0.026
- **运动趋势/速度** (`velocity_score`): 0.028
- **动作稳定性** (`action_stability_score`): 0.382
- **任务效率** (`task_efficiency_score`): 0.000
- **场景匹配度** (`scenario_alignment_score`): 0.117

### Captum/轨迹组级归一化贡献
- **uav_absolute**: 51.5%
- **target_absolute**: 35.8%
- **inter_uav_coordination**: 5.4%
- **velocity**: 4.3%
- **target_relative**: 3.0%

Captum/轨迹融合后，**uav_absolute** 类特征贡献最高。 对 inter_uav_coordination, velocity, target_relative 类特征关注较少。 （语义分组允许重叠，如双机距离同时计入协同与安全。）

### 自动解释
- 该策略较关注 UAV 与目标之间的相对关系，说明其动作选择与追踪任务目标直接相关。
- 在近距离初始场景下，该策略对安全距离关注不足，可能增加碰撞风险。
- 该策略动作切换频繁，控制过程可能存在抖动。

### 分组提示
- 观测中无相对距离特征，target_relative / inter_uav 将结合 trajectory 距离字段补充。

## 5. 场景匹配解释
- **场景匹配度**: 0.12
- 近距离初始场景中双机碰撞风险较高，策略应重点关注 UAV 间距、安全约束和协同队形。 在 near_uavs 场景下，策略对 UAV 间距或协同关注不足，可能增加碰撞风险。

## 6. 成功/失败原因
失败可能原因：双机间距控制不足导致碰撞风险；双机协同/间距特征利用不足；动作抖动过大。

## 7. 策略专项说明
解释来源：Captum Integrated Gradients + 轨迹统计。 BC 策略在闭环执行中模仿离线专家分布；若 inter_uav_coordination_score 或 safety_score 偏低，可能在 near_uavs 等场景更容易偏离专家轨迹。

## 8. 自动总结
总体来看，**bc** 在 **near_uavs** 场景下未能成功完成追踪。 episode return=-152.91，目标相对关系=0.23，双机协同=0.05，安全=0.03，场景匹配=0.12。