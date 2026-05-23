"""轨迹记录、动画与多策略对比可视化。"""

from visualization.policy_loaders import (
    BasePolicyWrapper,
    BCPolicyWrapper,
    ExpertPolicyWrapper,
    PPOPolicyWrapper,
    load_default_policies,
    load_policy,
)
from visualization.trajectory_recorder import (
    record_all_policies_same_seed,
    record_episode,
    trajectory_basename,
)
from visualization.trajectory_animation import make_animation, make_stepwise_animation
from visualization.compare_trajectories import (
    plot_trajectory_comparison,
    make_multi_policy_comparison_animation,
    export_step_table,
    compare_all_policies_outputs,
)
from visualization.trajectory_metrics import compute_trajectory_metrics, summarize_all_policy_metrics

__all__ = [
    "BasePolicyWrapper",
    "ExpertPolicyWrapper",
    "BCPolicyWrapper",
    "PPOPolicyWrapper",
    "load_policy",
    "load_default_policies",
    "record_episode",
    "record_all_policies_same_seed",
    "trajectory_basename",
    "make_animation",
    "make_stepwise_animation",
    "plot_trajectory_comparison",
    "make_multi_policy_comparison_animation",
    "export_step_table",
    "compare_all_policies_outputs",
    "compute_trajectory_metrics",
    "summarize_all_policy_metrics",
]
