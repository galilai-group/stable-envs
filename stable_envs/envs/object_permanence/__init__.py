"""Source-authored replay/rendering and optional legacy MuJoCo interfaces."""

from importlib.util import find_spec

from .replay import G34Replay
from .trajectory import RigidTrajectory

__all__ = ["G34Replay", "RigidTrajectory"]

if find_spec("mujoco") is not None:
    from .occlusion_ball import MuJoCoOcclusionBall

    __all__.append("MuJoCoOcclusionBall")
