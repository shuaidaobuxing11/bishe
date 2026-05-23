# 策略解释对比 — default seed=42

| policy_name | explanation_source | success | episode_return | episode_length | target_absolute_score | uav_absolute_score | target_relative_score | inter_uav_coordination_score | safety_score | velocity_score | action_stability_score | task_efficiency_score | scenario_alignment_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc | rule_and_trajectory | True | -73.64 | 117 | 0.001 | 0.025 | 0.741 | 0.235 | 0.195 | 0 | 0.293 | 0.604 | 0.608 |
| expert_v2 | rule_and_trajectory | False | -166.9 | 200 | 0.001 | 0.02 | 0.624 | 0.225 | 0.061 | 0 | 0.593 | 0.033 | 0.512 |
| ppo_baseline | rule_and_trajectory | True | -33.38 | 26 | 0 | 0.024 | 0.808 | 0.345 | 0.393 | 0 | 1 | 0.718 | 0.663 |
| ppo_finetune | rule_and_trajectory | True | -33.38 | 26 | 0 | 0.024 | 0.808 | 0.345 | 0.393 | 0 | 1 | 0.718 | 0.663 |

## 图表
- 细粒度评分柱状图：`policy_compare_default_seed42_fine_grained_scores.png`
- 可解释性评分热力图：`policy_compare_default_seed42_heatmap.png`
- 特征组贡献热力图：`policy_compare_default_seed42_groups.png`
- 性能-解释联合图：`policy_compare_default_seed42_perf_joint.png`
- 兼容柱状图：`policy_compare_default_seed42_scores.png`

## 自动解释
- Note: expert_v2 scores are rule/trajectory-based, while neural policies use Captum attribution.
- expert_v2 解释分数来自规则逻辑与轨迹统计，不与 PPO/BC 的 Captum 梯度归因完全等价。
- **ppo_baseline** 回报较高（-33.38），目标相对关系=0.808，双机协同=0.345。
- BC 与 PPO baseline 对比：若 BC 的 inter_uav_coordination_score / safety_score 较低，说明闭环执行中对双机协同与安全约束利用不足；PPO baseline 通常更关注 target_relative 与双机协同。