"""
工厂：选择 JSBSim 真机或 Stub，不依赖 RL 训练代码。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Union


class _SimProto(Protocol):
    def reset(self, *, seed: Optional[int] = None) -> Dict[str, float]: ...
    def step(self, dt: float, controls: Optional[Dict[str, float]] = None) -> Dict[str, float]: ...
    def get_history_dataframe(self): ...


def is_jsbsim_available() -> bool:
    try:
        import jsbsim  # noqa: F401

        return True
    except Exception:
        return False


def create_minimal_simulator(
    *,
    prefer_jsbsim: bool = False,
    model_name: str = "c172p",
) -> Union[_SimProto, Any]:
    """
    :param prefer_jsbsim: True 时优先尝试真实 JSBSim，失败则退回 Stub。
    """
    from .stub_simulator import StubFlightSimulator

    if prefer_jsbsim and is_jsbsim_available():
        try:
            from .jsbsim_backend import JsbsimBackendSimulator

            return JsbsimBackendSimulator(model_name=model_name)
        except Exception:
            pass
    return StubFlightSimulator()


__all__ = ["create_minimal_simulator", "is_jsbsim_available"]
