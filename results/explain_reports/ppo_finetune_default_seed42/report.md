# 策略可解释性报告

## 1. 基本信息
- **策略**: ppo_finetune
- **场景**: default
- **seed**: 42
- **解释来源**: 规则 + 轨迹统计 (`rule_and_trajectory`)
- **是否成功**: 是
- **episode length**: 26
- **episode return**: -33.38
- **是否碰撞**: 否
- **最小 UAV 间距**: 0.906
- **平均目标距离**: 6.805
- **最小目标距离**: 0.453
- **action switch rate**: 0.000

## 2. 轨迹行为分析
- 动作切换频率处于较低水平，控制相对平稳。
- 双机间距整体保持在相对安全范围。
- 平均目标距离呈下降趋势，策略能够稳定接近目标。
- 在较少步数内完成任务，路径效率较高。

## 3. 特征重要性分析（Top 10）
- （无 Captum 梯度归因；见第 4 节细粒度评分与轨迹分析。）

（规则/轨迹策略：解释分数主要来自规则逻辑与轨迹统计，不与神经网络 Captum 归因完全等价。）

## 4. 细粒度特征类别贡献
- **目标绝对位置** (`target_absolute_score`): 0.000
- **UAV 绝对位置** (`uav_absolute_score`): 0.024
- **UAV-目标相对关系** (`target_relative_score`): 0.808
- **双机协同** (`inter_uav_coordination_score`): 0.345
- **安全约束** (`safety_score`): 0.393
- **运动趋势/速度** (`velocity_score`): 0.000
- **动作稳定性** (`action_stability_score`): 1.000
- **任务效率** (`task_efficiency_score`): 0.718
- **场景匹配度** (`scenario_alignment_score`): 0.663

### Captum/轨迹组级归一化贡献
- **target_relative**: 78.7%
- **inter_uav_coordination**: 14.6%
- **uav_absolute**: 6.7%
- **target_absolute**: 0.0%
- **velocity**: 0.0%

Captum/轨迹融合后，**target_relative** 类特征贡献最高。 对 uav_absolute, target_absolute, velocity 类特征关注较少。 （语义分组允许重叠，如双机距离同时计入协同与安全。）

### 自动解释
- 该策略较关注 UAV 与目标之间的相对关系，说明其动作选择与追踪任务目标直接相关。
- 该策略较关注双机之间的空间关系，说明其具有一定协同控制倾向。
- 该策略能够较快完成追踪任务，任务执行效率较高。

### 分组提示
- 观测中无相对距离特征，target_relative / inter_uav 将结合 trajectory 距离字段补充。

## 5. 场景匹配解释
- **场景匹配度**: 0.66
- 默认场景中目标运动较平稳，策略应主要关注 UAV 与目标之间的相对关系，同时结合目标位置和自身位置完成追踪。 该策略在 default 场景中主要关注 inter_uav_coordination, target_relative，与场景需求基本一致。 双机协同关注度较高，这对双机追踪任务具有积极意义，不视为偏离场景需求。

## 6. 成功/失败原因
该策略成功的主要原因可能是能够持续关注目标相对关系或目标位置，并保持较稳定的接近过程。

## 7. 策略专项说明
解释来源：行为/轨迹统计（Captum 不可用或未使用）。 PPO finetune 仍受 BC 初始化影响；

## 8. 自动总结
总体来看，**ppo_finetune** 在 **default** 场景下成功完成追踪。 episode return=-33.38，目标相对关系=0.81，双机协同=0.34，安全=0.39，场景匹配=0.66。