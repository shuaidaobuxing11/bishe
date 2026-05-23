# 策略可解释性报告

## 1. 基本信息
- **策略**: expert_v2
- **场景**: default
- **seed**: 42
- **解释来源**: 规则 + 轨迹统计 (`rule_and_trajectory`)
- **是否成功**: 否
- **episode length**: 200
- **episode return**: -166.94
- **是否碰撞**: 是
- **最小 UAV 间距**: 0.141
- **平均目标距离**: 2.599
- **最小目标距离**: 0.672
- **action switch rate**: 0.407

## 2. 轨迹行为分析
- 动作切换频率处于较低水平，控制相对平稳。
- 执行过程中双机间距曾接近碰撞阈值，存在较高碰撞风险。
- 平均目标距离呈下降趋势，策略能够稳定接近目标。
- 未在限定步数内完成协同追踪任务。

## 3. 特征重要性分析（Top 10）
- （无 Captum 梯度归因；见第 4 节细粒度评分与轨迹分析。）

（规则/轨迹策略：解释分数主要来自规则逻辑与轨迹统计，不与神经网络 Captum 归因完全等价。）

## 4. 细粒度特征类别贡献
- **目标绝对位置** (`target_absolute_score`): 0.001
- **UAV 绝对位置** (`uav_absolute_score`): 0.020
- **UAV-目标相对关系** (`target_relative_score`): 0.624
- **双机协同** (`inter_uav_coordination_score`): 0.225
- **安全约束** (`safety_score`): 0.061
- **运动趋势/速度** (`velocity_score`): 0.000
- **动作稳定性** (`action_stability_score`): 0.593
- **任务效率** (`task_efficiency_score`): 0.033
- **场景匹配度** (`scenario_alignment_score`): 0.512

### Captum/轨迹组级归一化贡献
- **inter_uav_coordination**: 51.2%
- **target_relative**: 42.6%
- **uav_absolute**: 5.8%
- **target_absolute**: 0.3%
- **velocity**: 0.0%

Captum/轨迹融合后，**inter_uav_coordination** 类特征贡献最高。 对 uav_absolute, target_absolute, velocity 类特征关注较少。 （语义分组允许重叠，如双机距离同时计入协同与安全。）

### 自动解释
- 该策略较关注 UAV 与目标之间的相对关系，说明其动作选择与追踪任务目标直接相关。
- 该策略较关注双机之间的空间关系，说明其具有一定协同控制倾向。

### 分组提示
- 观测中无相对距离特征，target_relative / inter_uav 将结合 trajectory 距离字段补充。

## 5. 场景匹配解释
- **场景匹配度**: 0.51
- 默认场景中目标运动较平稳，策略应主要关注 UAV 与目标之间的相对关系，同时结合目标位置和自身位置完成追踪。 该策略在 default 场景中主要关注 inter_uav_coordination, target_relative，与场景需求基本一致。 双机协同关注度较高，这对双机追踪任务具有积极意义，不视为偏离场景需求。

## 6. 成功/失败原因
失败可能原因：双机间距控制不足导致碰撞风险。

## 7. 策略专项说明
expert_v2 为规则策略（explanation_source=rule_and_trajectory），不适用 Captum 梯度归因。专家策略根据 UAV 与目标相对方向选择转向，并在距离较远时加速靠近目标。其解释分数主要来自规则逻辑和轨迹统计，不与神经网络策略的梯度归因完全等价。

## 8. 自动总结
总体来看，**expert_v2** 在 **default** 场景下未能成功完成追踪。 episode return=-166.94，目标相对关系=0.62，双机协同=0.23，安全=0.06，场景匹配=0.51。