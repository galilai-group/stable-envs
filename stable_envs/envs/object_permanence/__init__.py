"""Source-authored replay/rendering and optional legacy MuJoCo interfaces."""

from importlib.util import find_spec

from .replay import G34Replay

__all__ = ["G34Replay"]

if find_spec("mujoco") is not None:
    from .occlusion_ball import MuJoCoOcclusionBall

    __all__.append("MuJoCoOcclusionBall")
