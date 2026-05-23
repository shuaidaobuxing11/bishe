"""
最小占位飞行仿真：不依赖 JSBSim，用于 Demo 与无 C++ 绑定环境。
状态为简化标量，便于 Streamlit 曲线展示；接口与 JsbsimBackend 对齐。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class StubFlightSimulator:
    """极简纵向通道：高度 h、空速 V、航迹角 gamma（rad）。"""

    dt_default: float = 0.05
    mass_kg: float = 800.0
    g: float = 9.81
    t: float = 0.0
    h_m: float = 1000.0
    V_mps: float = 60.0
    gamma_rad: float = 0.02
    theta_rad: float = 0.05
    history: list = field(default_factory=list)

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, float]:
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.h_m = float(rng.uniform(800.0, 1200.0))
            self.V_mps = float(rng.uniform(50.0, 70.0))
        else:
            self.h_m = 1000.0
            self.V_mps = 60.0
        self.gamma_rad = 0.02
        self.theta_rad = 0.05
        self.t = 0.0
        self.history.clear()
        s = self._state_dict()
        self.history.append(s)
        return s

    def _state_dict(self) -> Dict[str, float]:
        return {
            "time_s": float(self.t),
            "altitude_m": float(self.h_m),
            "airspeed_mps": float(self.V_mps),
            "gamma_rad": float(self.gamma_rad),
            "theta_rad": float(self.theta_rad),
            "mode": "stub",
        }

    def step(self, dt: float, controls: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        controls 可选:
          throttle: 0~1 推力系数
          pitch_cmd_rad: 目标俯仰增量（小量）
        """
        controls = controls or {}
        throttle = float(np.clip(controls.get("throttle", 0.65), 0.0, 1.0))
        pitch_cmd = float(controls.get("pitch_cmd_rad", 0.0))

        # 极简：推力 ∝ throttle，阻力 ∝ V^2
        T = 6000.0 * throttle
        D = 0.35 * self.V_mps**2
        a_long = (T - D) / self.mass_kg - self.g * np.sin(self.gamma_rad)
        self.V_mps = float(np.clip(self.V_mps + a_long * dt, 25.0, 120.0))

        self.gamma_rad = float(np.clip(self.gamma_rad + 0.02 * (self.theta_rad - self.gamma_rad) * dt, -0.25, 0.25))
        self.theta_rad = float(np.clip(self.theta_rad + pitch_cmd * dt * 0.5, -0.35, 0.35))
        self.h_m = float(self.h_m + self.V_mps * np.sin(self.gamma_rad) * dt)
        self.h_m = float(np.clip(self.h_m, 0.0, 20_000.0))

        self.t += dt
        s = self._state_dict()
        self.history.append(s)
        return s

    def get_history_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.history)
