# 策略解释对比 — noisy_target seed=42

| policy_name | explanation_source | success | episode_return | episode_length | target_absolute_score | uav_absolute_score | target_relative_score | inter_uav_coordination_score | safety_score | velocity_score | action_stability_score | task_efficiency_score | scenario_alignment_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bc | rule_and_trajectory | False | -119.5 | 200 | 0.002 | 0.019 | 0.631 | 0.227 | 0.057 | 0 | 0.276 | 0.007 | 0.113 |
| expert_v2 | rule_and_trajectory | False | -196.6 | 200 | 0 | 0 | 0.465 | 0.346 | 0 | 0 | 0.487 | 0 | 0.106 |

## 图表
- 细粒度评分柱状图：`policy_compare_noisy_target_seed42_fine_grained_scores.png`
- 可解释性评分热力图：`policy_compare_noisy_target_seed42_heatmap.png`
- 特征组贡献热力图：`policy_compare_noisy_target_seed42_groups.png`
- 性能-解释联合图：`policy_compare_noisy_target_seed42_perf_joint.png`
- 兼容柱状图：`policy_compare_noisy_target_seed42_scores.png`

## 自动解释
- Note: expert_v2 scores are rule/trajectory-based, while neural policies use Captum attribution.
- expert_v2 解释分数来自规则逻辑与轨迹统计，不与 PPO/BC 的 Captum 梯度归因完全等价。
- **bc** 回报较高（-119.50），目标相对关系=0.631，双机协同=0.227。