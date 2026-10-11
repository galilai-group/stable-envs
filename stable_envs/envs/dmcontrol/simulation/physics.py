"""Physics Variation leaves per simulator, with the ranges and defaults of the registered wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import torch

from ...._core import physics as _physics


@dataclass(frozen=True)
class SphereRadius:
    """A sphere geom's radius: its collision bounds and, with its density, its body's mass and inertia."""

    leaf: str
    geom: str
    width = 1


@dataclass(frozen=True)
class SphereDensity:
    """The density of a sphere geom that is its body's only mass, read together with its radius."""

    leaf: str
    geom: str
    nominal: float
    width = 1


class SphereKnobs(_physics.PhysicsKnobs):
    """PhysicsKnobs plus sphere geoms whose radius and density set their body, as a recompile does."""

    def __init__(self, knobs, model, **options):
        self._spheres = {}
        for knob in knobs:
            if isinstance(knob, (SphereRadius, SphereDensity)):
                import mujoco

                geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, knob.geom)
                if geom < 0 or int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    raise KeyError(f"{knob.leaf}: no sphere geom named {knob.geom!r}")
                body = int(model.geom_bodyid[geom])
                if sum(int(b) == body for b in model.geom_bodyid) != 1:
                    raise ValueError(f"{knob.leaf}: {knob.geom!r} must be its body's only geom")
                self._spheres.setdefault(geom, {})[type(knob)] = knob
        super().__init__(knobs, model, **options)
        self._derived = True

    def _read_nominal(self, m):
        row = super()._read_nominal(m)
        for geom, pair in self._spheres.items():
            row[self.offsets[self.leaves.index(pair[SphereRadius].leaf)]] = float(m.geom_size[geom, 0])
            row[self.offsets[self.leaves.index(pair[SphereDensity].leaf)]] = float(pair[SphereDensity].nominal)
        return row

    def _apply_masses(self, model, row) -> None:
        super()._apply_masses(model, row)
        n = self._nominal_model
        for geom, pair in self._spheres.items():
            radius = float(row[self.offsets[self.leaves.index(pair[SphereRadius].leaf)]])
            density = float(row[self.offsets[self.leaves.index(pair[SphereDensity].leaf)]])
            if not (radius > 0 and density > 0):
                raise ValueError(f"{pair[SphereRadius].leaf} and {pair[SphereDensity].leaf} must be positive")
            body = int(n.geom_bodyid[geom])
            if radius == float(n.geom_size[geom, 0]) and density == float(pair[SphereDensity].nominal):
                for name in ("body_mass", "body_inertia"):
                    getattr(model, name)[body] = getattr(n, name)[body]
                for name in ("geom_size", "geom_rbound", "geom_aabb"):
                    getattr(model, name)[geom] = getattr(n, name)[geom]
                adr, num = int(n.body_bvhadr[body]), int(n.body_bvhnum[body])
                model.bvh_aabb[adr : adr + num] = n.bvh_aabb[adr : adr + num]
                continue
            mass = density * 4.0 * np.pi * radius**3 / 3.0
            model.body_mass[body] = mass
            model.body_inertia[body] = 0.4 * mass * radius**2
            model.geom_size[geom, 0] = radius
            model.geom_rbound[geom] = radius
            model.geom_aabb[geom, 3:] = radius
            _physics._refresh_bvh(model, body)


class SphereLanes(_physics.PhysicsLanes):
    def __init__(self, knobs, models, **options):
        super().__init__([k for k in knobs if not isinstance(k, (SphereRadius, SphereDensity))], models, **options)
        self.spec = SphereKnobs(knobs, models[0], **options)
        self.values = np.tile(self.spec.nominal, (len(self.models), 1))


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
        Leaf("ball.density", 500.0, 1500.0, 1000.0, (SphereDensity("ball.density", "ball", 1000.0),)),
        Leaf("ball.size", 0.01, 0.05, 0.025, (SphereRadius("ball.size", "ball"),)),
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
    if not declared:
        return None
    spheres = any(isinstance(k, (SphereRadius, SphereDensity)) for k in declared)
    return (SphereLanes if spheres else _physics.PhysicsLanes)(declared, models, **compile_options(key))
