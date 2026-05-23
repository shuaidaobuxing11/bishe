# 策略可解释性报告

## 1. 基本信息
- **策略**: bc
- **场景**: noisy_target
- **seed**: 42
- **解释来源**: 规则 + 轨迹统计 (`rule_and_trajectory`)
- **是否成功**: 否
- **episode length**: 200
- **episode return**: -119.50
- **是否碰撞**: 是
- **最小 UAV 间距**: 0.131
- **平均目标距离**: 2.912
- **最小目标距离**: 0.630
- **action switch rate**: 0.724

## 2. 轨迹行为分析
- 该策略动作切换频率较高，控制过程存在一定抖动。
- 执行过程中双机间距曾接近碰撞阈值，存在较高碰撞风险。
- 平均目标距离呈下降趋势，策略能够稳定接近目标。
- 未在限定步数内完成协同追踪任务。

## 3. 特征重要性分析（Top 10）
- （无 Captum 梯度归因；见第 4 节细粒度评分与轨迹分析。）

（规则/轨迹策略：解释分数主要来自规则逻辑与轨迹统计，不与神经网络 Captum 归因完全等价。）

## 4. 细粒度特征类别贡献
- **目标绝对位置** (`target_absolute_score`): 0.002
- **UAV 绝对位置** (`uav_absolute_score`): 0.019
- **UAV-目标相对关系** (`target_relative_score`): 0.631
- **双机协同** (`inter_uav_coordination_score`): 0.227
- **安全约束** (`safety_score`): 0.057
- **运动趋势/速度** (`velocity_score`): 0.000
- **动作稳定性** (`action_stability_score`): 0.276
- **任务效率** (`task_efficiency_score`): 0.007
- **场景匹配度** (`scenario_alignment_score`): 0.113

### Captum/轨迹组级归一化贡献
- **inter_uav_coordination**: 52.6%
- **target_relative**: 41.4%
- **uav_absolute**: 5.4%
- **target_absolute**: 0.6%
- **velocity**: 0.1%

Captum/轨迹融合后，**inter_uav_coordination** 类特征贡献最高。 对 uav_absolute, target_absolute, velocity 类特征关注较少。 （语义分组允许重叠，如双机距离同时计入协同与安全。）

### 自动解释
- 该策略较关注 UAV 与目标之间的相对关系，说明其动作选择与追踪任务目标直接相关。
- 该策略较关注双机之间的空间关系，说明其具有一定协同控制倾向。
- 在目标扰动场景下，该策略对速度/运动趋势关注不足，可能导致对目标突变适应能力有限。
- 该策略动作切换频繁，控制过程可能存在抖动。

### 分组提示
- 观测中无相对距离特征，target_relative / inter_uav 将结合 trajectory 距离字段补充。

## 5. 场景匹配解释
- **场景匹配度**: 0.11
- 目标扰动场景中策略除关注 UAV 与目标的相对关系外，还应关注速度和运动趋势，以适应目标运动变化。 在 noisy_target 场景下，策略对速度/运动趋势关注不足，可能导致对目标突变适应能力有限。

## 6. 成功/失败原因
失败可能原因：双机间距控制不足导致碰撞风险；对目标扰动/速度趋势适应不足；动作抖动过大。

## 7. 策略专项说明
解释来源：行为/轨迹统计（Captum 不可用或未使用）。 BC 策略在闭环执行中模仿离线专家分布；若 inter_uav_coordination_score 或 safety_score 偏低，可能在 near_uavs 等场景更容易偏离专家轨迹。

## 8. 自动总结
总体来看，**bc** 在 **noisy_target** 场景下未能成功完成追踪。 episode return=-119.50，目标相对关系=0.63，双机协同=0.23，安全=0.06，场景匹配=0.11。 对速度类特征关注偏低，在更强扰动下可能存在鲁棒性不足。