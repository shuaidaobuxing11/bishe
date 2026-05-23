"""
可选真实 JSBSim 后端：若未安装或加载模型失败，由上层退回 Stub。
需: pip install jsbsim，并通常需设置 JSBSIM_ROOT 指向含 aircraft/ 的数据根目录。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np


def _guess_jsbsim_root() -> Optional[str]:
    env = os.environ.get("JSBSIM_ROOT")
    if env and os.path.isdir(env):
        return env
    try:
        import jsbsim  # type: ignore

        root = getattr(jsbsim, "get_data_dir", None)
        if callable(root):
            p = root()
            if p and os.path.isdir(p):
                return str(p)
        import pathlib

        cand = pathlib.Path(jsbsim.__file__).resolve().parent / "data"
        if cand.is_dir():
            return str(cand)
    except Exception:
        pass
    return None


class JsbsimBackendSimulator:
    """最小封装：单步 fdm.run()，导出常用标量。"""

    def __init__(self, model_name: str = "c172p"):
        import jsbsim  # type: ignore

        root = _guess_jsbsim_root()
        if not root:
            raise RuntimeError(
                "无法定位 JSBSim 数据目录。请设置环境变量 JSBSIM_ROOT，或确认 pip 安装的 jsbsim 含 data。"
            )
        self._jsbsim = jsbsim
        self.fdm = jsbsim.FGFDMExec(root)
        ok = self.fdm.load_model(str(model_name))
        if ok is False or ok == 0:
            raise RuntimeError(f"JSBSim 加载模型失败: {model_name} (root={root})")
        try:
            self.fdm.set_dt(1.0 / 120.0)
        except Exception:
            pass
        self._model_name = model_name
        self.history: List[Dict[str, float]] = []

    def reset(self, *, seed: Optional[int] = None) -> Dict[str, float]:
        try:
            self.fdm.reset_to_initial_conditions(0)
        except Exception:
            try:
                self.fdm.reset()
            except Exception:
                pass
        if seed is not None:
            try:
                ic = self.fdm.get_property_value("ic/h-sl-ft")
                self.fdm.set_property_value("ic/h-sl-ft", float(ic) + (seed % 100))
            except Exception:
                pass
        self.history.clear()
        s = self._read_state()
        self.history.append(s)
        return s

    def _read_state(self) -> Dict[str, float]:
        fdm = self.fdm
        t = float(fdm.get_sim_time())
        try:
            h_m = float(fdm.get_property_value("position/h-sl-meters"))
        except Exception:
            try:
                h_m = float(fdm.get_property_value("position/h-sl-ft")) * 0.3048
            except Exception:
                h_m = 0.0
        try:
            V = float(fdm.get_property_value("velocities/vt-fps")) * 0.3048
        except Exception:
            try:
                V = float(fdm.get_property_value("velocities/u-fps")) * 0.3048
            except Exception:
                V = 0.0
        try:
            gamma = float(fdm.get_property_value("flight-path/gamma-rad"))
        except Exception:
            gamma = 0.0
        try:
            theta = float(fdm.get_property_value("attitude/theta-rad"))
        except Exception:
            theta = 0.0
        return {
            "time_s": t,
            "altitude_m": h_m,
            "airspeed_mps": V,
            "gamma_rad": gamma,
            "theta_rad": theta,
            "mode": f"jsbsim:{self._model_name}",
        }

    def step(self, dt: float, controls: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        controls = controls or {}
        # 常见属性名（机型不同可能需调整；Demo 仅作最小示例）
        if "throttle" in controls:
            for prop in ("fcs/throttle-cmd-norm", "propulsion/engine/set-throttle"):
                try:
                    self.fdm.set_property_value(prop, float(controls["throttle"]))
                    break
                except Exception:
                    continue
        try:
            h = float(self.fdm.get_delta_t())
        except Exception:
            h = 1.0 / 120.0
        n = max(1, int(round(float(dt) / max(h, 1e-6))))
        for _ in range(n):
            if not self.fdm.run():
                break
        s = self._read_state()
        self.history.append(s)
        return s

    def get_history_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self.history)
