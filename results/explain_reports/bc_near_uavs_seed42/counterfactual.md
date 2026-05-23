# 反事实解释

- **target_x_norm** (+0.1): 当 target_x_norm 增加 0.1 后，动作由 keep/turn_left 变为 turn_right/turn_left，说明策略对该特征较敏感。

- **target_y_norm** (+0.1): 当 target_y_norm 增加 0.1 后，动作由 keep/turn_left 变为 turn_right/accelerate，说明策略对该特征较敏感。

- **uav1_x_norm** (+0.1): 当 uav1_x_norm 增加 0.1 后，动作由 keep/turn_left 变为 turn_left/turn_left，说明策略对该特征较敏感。

- **uav2_x_norm** (+0.1): 当 uav2_x_norm 增加 0.1 后，动作由 keep/turn_left 变为 keep/turn_left，动作未变，说明对该扰动不敏感。

- **uav1_vx_norm** (+0.1): 当 uav1_vx_norm 增加 0.1 后，动作由 keep/turn_left 变为 turn_left/turn_left，说明策略对该特征较敏感。

- **uav2_vx_norm** (+0.1): 当 uav2_vx_norm 增加 0.1 后，动作由 keep/turn_left 变为 keep/accelerate，说明策略对该特征较敏感。
