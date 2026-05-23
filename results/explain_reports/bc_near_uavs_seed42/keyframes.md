# 关键帧动作解释

## initial (step=0)

在第 0 步，UAV1 距目标 11.191，UAV2 距目标 11.869，双机间距 0.682。策略选择 action=18（UAV1: accelerate，UAV2: accelerate）。 特征归因显示 target_y_norm(+9.9559), target_x_norm(+7.8219), uav1_x_norm(+1.4720), uav2_x_norm(+0.9360), uav1_vy_norm(-0.3163) 对该动作贡献较高。

## mid (step=100)

在第 100 步，UAV1 距目标 1.454，UAV2 距目标 2.164，双机间距 1.174。策略选择 action=18（UAV1: accelerate，UAV2: accelerate）。 特征归因显示 target_x_norm(+29.8948), uav1_x_norm(-24.0784), target_y_norm(+20.9183), uav1_y_norm(-18.1959), uav1_vy_norm(+2.4521) 对该动作贡献较高。

## critical_min_uav_distance (step=159)

在第 159 步，UAV1 距目标 3.622，UAV2 距目标 3.514，双机间距 0.110。策略选择 action=24（UAV1: decelerate，UAV2: decelerate）。 特征归因显示 uav2_y_norm(+64.6176), uav1_y_norm(-56.3514), target_y_norm(-7.9751), uav2_x_norm(+7.3840), uav1_x_norm(-6.2258) 对该动作贡献较高。

## critical_min_distance (step=149)

在第 149 步，UAV1 距目标 0.674，UAV2 距目标 1.318，双机间距 0.939。策略选择 action=13（UAV1: turn_right，UAV2: accelerate）。 特征归因显示 target_x_norm(-117.0051), uav1_x_norm(+114.9176), uav1_y_norm(+65.0475), target_y_norm(-59.5678), uav1_vx_norm(+6.7643) 对该动作贡献较高。

## pre_collision (step=92)

在第 92 步，UAV1 距目标 1.404，UAV2 距目标 2.647，双机间距 1.264。策略选择 action=1（UAV1: keep，UAV2: turn_left）。 特征归因显示 target_y_norm(-31.4439), uav1_y_norm(+16.5536), uav2_y_norm(+11.3583), uav2_vx_norm(+2.8080), uav2_vy_norm(-1.6865) 对该动作贡献较高。

## action_switch (step=89)

在第 89 步，UAV1 距目标 0.496，UAV2 距目标 3.978，双机间距 3.482。策略选择 action=12（UAV1: turn_right，UAV2: turn_right）。 特征归因显示 uav1_x_norm(+134.7176), target_x_norm(-122.5393), uav1_y_norm(-83.7247), target_y_norm(+73.7086), uav2_x_norm(-10.5475) 对该动作贡献较高。

## final (step=199)

在第 199 步，UAV1 距目标 0.215，UAV2 距目标 2.712，双机间距 2.848。策略选择 action=11（UAV1: turn_right，UAV2: turn_left）。 特征归因显示 target_y_norm(+143.2311), uav1_y_norm(-139.5137), uav1_x_norm(+82.9330), target_x_norm(-72.8436), uav2_x_norm(-6.6891) 对该动作贡献较高。
