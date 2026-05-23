# 关键帧动作解释

## initial (step=0)

在第 0 步，UAV1 距目标 11.191，UAV2 距目标 11.869，双机间距 0.682。策略选择 action=18（UAV1: accelerate，UAV2: accelerate）。 特征归因显示 target_y_norm(+3.6125), target_x_norm(+2.7217), uav2_y_norm(+1.5085), uav1_x_norm(+1.1159), uav1_vx_norm(+0.3985) 对该动作贡献较高。

## mid (step=11)

在第 11 步，UAV1 距目标 5.500，UAV2 距目标 6.178，双机间距 0.679。策略选择 action=18（UAV1: accelerate，UAV2: accelerate）。 特征归因显示 uav2_y_norm(-11.0785), uav1_y_norm(+10.2254), target_x_norm(+6.0788), target_y_norm(+5.2880), uav2_x_norm(-1.1973) 对该动作贡献较高。

## critical_min_uav_distance (step=21)

在第 21 步，UAV1 距目标 0.558，UAV2 距目标 1.231，双机间距 0.676。策略选择 action=18（UAV1: accelerate，UAV2: accelerate）。 特征归因显示 uav1_y_norm(+39.0964), uav2_y_norm(-35.9200), target_x_norm(+15.2668), uav2_x_norm(-11.8240), uav1_vy_norm(+3.9953) 对该动作贡献较高。

## critical_min_distance (step=22)

在第 22 步，UAV1 距目标 0.406，UAV2 距目标 0.741，双机间距 0.677。策略选择 action=3（UAV1: keep，UAV2: accelerate）。 特征归因显示 uav2_y_norm(-41.9613), uav1_y_norm(+32.7915), uav1_x_norm(+28.9197), uav2_x_norm(-15.6105), target_y_norm(+13.0441) 对该动作贡献较高。
