"""Physics Variation leaves per simulator, with the ranges and defaults of the registered wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import torch

from ...._core import physics as _physics


@dataclass(frozen=True)
class Leaf:
    path: str
    low: float
    high: float
    default: float
    knobs: tuple[_physics.Knob, ...]


def _gravity():
    axes = (("x", -5.0, 5.0, 0.0), ("y", -5.0, 5.0, 0.0), ("z", -20.0, 0.0, -9.81))
    return tuple(
        Leaf(f"gravity.{a}", lo, hi, default, (_physics.ModelField(f"gravity.{a}", "opt.gravity", index=i),))
        for i, (a, lo, hi, default) in enumerate(axes)
    )


def _density(path, *geoms):
    return Leaf(path, 500.0, 1500.0, 1000.0, tuple(_physics.GeomMass(path, g, 1000.0) for g in geoms))


def _floor_friction(geom):
    return Leaf(
        "floor.friction", 0.0, 1.0, 1.0, (_physics.ModelField("floor.friction", "geom_friction", "geom", geom, 0),)
    )


_CART_MASS = Leaf("agent.cart_mass", 0.5, 1.5, 1.0, (_physics.GeomMass("agent.cart_mass", "cart", 1.0, per="mass"),))

LEAVES: dict[str, tuple[Leaf, ...]] = {
    "acrobot": _gravity(),
    "pendulum": _gravity(),
    "cartpole": (*_gravity(), _CART_MASS),
    "cartpole-balance": (*_gravity(), _CART_MASS),
    "ball_in_cup": (
        *_gravity(),
        _density("agent.density", *(f"cup_part_{i}" for i in range(5))),
        _density("ball.density", "ball"),
    ),
    "finger": (
        _density("agent.proximal_density", "proximal"),
        _density("agent.fingertip_density", "fingertip"),
        _density("spinner.density", "cap1", "cap2"),
        Leaf(
            "spinner.friction",
            0.0,
            1.0,
            0.1,
            (_physics.ModelField("spinner.friction", "dof_frictionloss", "joint", "hinge"),),
        ),
    ),
    "manipulator": (_density("agent.upper_arm_density", "upper_arm"), _density("agent.hand_density", "hand")),
    "reacher": (_density("agent.arm_density", "arm"), _density("agent.finger_density", "finger")),
    "quadruped": (
        *_gravity(),
        _density("agent.torso_density", "torso"),
        _density("agent.foot_back_left_density", "foot_back_left"),
        _floor_friction("floor"),
    ),
    "cheetah": (
        *_gravity(),
        _density("agent.torso_density", "torso"),
        _density("agent.back_foot_density", "bfoot"),
        _floor_friction("ground"),
    ),
    "hopper": (
        *_gravity(),
        _density("agent.torso_density", "torso"),
        _density("agent.foot_density", "foot"),
        _floor_friction("floor"),
    ),
    "walker": (
        *_gravity(),
        _density("agent.torso_density", "torso"),
        _density("agent.left_foot_density", "left_foot"),
        _floor_friction("floor"),
    ),
    "humanoid": (
        *_gravity(),
        _density("agent.torso_density", "torso"),
        _density("agent.right_lower_arm_density", "right_lower_arm"),
        _floor_friction("floor"),
    ),
}


def knobs(key: str) -> tuple[_physics.Knob, ...]:
    return tuple(k for leaf in LEAVES.get(key, ()) for k in leaf.knobs)


def state_fields(key: str) -> dict:
    return {k: (width, torch.float64) for k, width in _physics.state_fields(knobs(key)).items()}


@cache
def compile_options(key: str) -> dict:
    """Cheetah compiles with settotalmass, so a mass change rescales every body."""
    if key != "cheetah":
        return {}
    import mujoco
    from dm_control.suite import cheetah

    xml, assets = cheetah.get_model_and_assets()
    unscaled = mujoco.MjModel.from_xml_string(xml.replace(b'settotalmass="14"', b""), assets)
    return {"total_mass": 14.0, "unscaled_total": float(unscaled.body_mass[1:].sum())}


def make_lanes(key: str, models):
    declared = knobs(key)
    return _physics.PhysicsLanes(declared, models, **compile_options(key)) if declared else None
