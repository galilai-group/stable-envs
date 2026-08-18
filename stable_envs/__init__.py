"""stable-envs: famous research environments as GPU-batched, state-driven renderers.

The claim the repo is built on: for a large fraction of the environments the field trains on, a frame
is a pure function of a compact state vector. If that is true, an image dataset is redundant with a
state dataset that is orders of magnitude smaller, and the frames can be regenerated on the GPU faster
than they could be read from disk. PushT is 5 floats per step against a 96x96x3 frame — 1,382x at 96px, 7,526x at 224px — and
renders at 3.2 M frames/s end-to-end on one H200 (5.6 M for the rasterizer alone).

Public surface:
    from stable_envs import make, REGISTRY
    env = make("PushT", batch=4096, size=96)
    frames = env.render(states)          # (B,96,96,3) uint8, on device
"""
from .core.env import (FIDELITY_TIERS, REGISTRY, RENDER_DECODERS, StableEnv,  # noqa: F401
                       check_states, format_registry, make, register, registry_table)
from .render.scene import Scene2D, SceneBuilder                  # noqa: F401
from .render import raster2d                                     # noqa: F401
from .render import raycast3d                                    # noqa: F401  (3-D primitive caster)
from . import spaces                                             # noqa: F401
from .data import StateDataset, TrajectoryDataset, collate_batched, render_states  # noqa: F401
from .rollout import BasePolicy, RandomPolicy, ScriptedPolicy, sample  # noqa: F401
from .core.style import StyleTensor                               # noqa: F401
from . import envs                                               # noqa: F401  (auto-discovers and registers)

__version__ = "0.1.0"
__all__ = ["REGISTRY", "StableEnv", "make", "register", "registry_table", "format_registry",
           "FIDELITY_TIERS", "RENDER_DECODERS", "Scene2D", "SceneBuilder", "spaces",
           "raster2d", "render_states", "StateDataset", "TrajectoryDataset", "collate_batched",
           "check_states",
           "StyleTensor",
           "BasePolicy", "RandomPolicy", "ScriptedPolicy", "sample"]
