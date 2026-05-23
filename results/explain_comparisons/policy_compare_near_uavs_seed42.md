# 策略解释对比 — near_uavs seed=42

| policy_name | explanation_source | success | episode_return | episode_length | target_absolute_score | uav_absolute_score | target_relative_score | inter_uav_coordination_score | safety_score | velocity_score | action_stability_score | task_efficiency_score | scenario_alignment_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc | rule_and_trajectory | False | -152.9 | 200 | 0.001 | 0.018 | 0.508 | 0.249 | 0.048 | 0 | 0.382 | 0 | 0.146 |
| expert_v2 | rule_and_trajectory | True | -73.11 | 115 | 0.001 | 0.027 | 0.707 | 0.309 | 0.288 | 0 | 0.228 | 0.606 | 0.234 |

## 图表
- 细粒度评分柱状图：`policy_compare_near_uavs_seed42_fine_grained_scores.png`
- 可解释性评分热力图：`policy_compare_near_uavs_seed42_heatmap.png`
- 特征组贡献热力图：`policy_compare_near_uavs_seed42_groups.png`
- 性能-解释联合图：`policy_compare_near_uavs_seed42_perf_joint.png`
- 兼容柱状图：`policy_compare_near_uavs_seed42_scores.png`

## 自动解释
- Note: expert_v2 scores are rule/trajectory-based, while neural policies use Captum attribution.
- expert_v2 解释分数来自规则逻辑与轨迹统计，不与 PPO/BC 的 Captum 梯度归因完全等价。
- **expert_v2** 回报较高（-73.11），目标相对关系=0.707，双机协同=0.309。