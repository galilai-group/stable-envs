"""DM Control appearance leaves rendered against the pinned stable_worldmodel wrappers.

Both sides reset with the same seed and the same leaf values (ours through the variation map),
step the same actions, and render every State: SWM through MuJoCo's renderer on its own model,
ours through the cell. The residual is reported as normalized MSE (pixels in [0, 1]) next to the
residual at SWM's registered default, and next to what ignoring the leaf would cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from tests.helpers.reference import style_from_variation
from tests.helpers.reference.dmc_semantics import DMCReference
from tests.helpers.reference.variation_maps.dmc_control import swm_defaults

DOMAINS = {
    "DMCBallInCupCatch": "ball_in_cup",
    "DMCFingerTurnHard": "finger",
    "DMCReacherHard": "reacher",
    "DMCCheetahRun": "cheetah",
    "DMCHopperHop": "hopper",
    "DMCWalkerWalk": "walker",
    "DMCHumanoidWalk": "humanoid",
    "DMCQuadrupedWalk": "quadruped",
}
SWM_IDS = {
    "ball_in_cup": "swm/BallInCupDMControl-v0",
    "finger": "swm/FingerDMControl-v0",
    "reacher": "swm/ReacherDMControl-v0",
    "cheetah": "swm/CheetahDMControl-v0",
    "hopper": "swm/HopperDMControl-v0",
    "walker": "swm/WalkerDMControl-v0",
    "humanoid": "swm/HumanoidDMControl-v0",
    "quadruped": "swm/QuadrupedDMControl-v0",
}
_SHOWN = {"rendering.render_target": 1}
CASES = {
    "DMCBallInCupCatch": [
        {"gravity.x": 2.5},
        {"gravity.y": -2.5},
        {"gravity.z": -2.5},
        {"ball.size": 0.04},
        {"ball.size": 0.015},
    ],
    "DMCFingerTurnHard": [
        {"spinner.color": [0.1, 0.8, 0.2]},
        {**_SHOWN, "target.shape": 0},
        {**_SHOWN, "target.shape": 1},
    ],
    "DMCReacherHard": [{**_SHOWN, "target.shape": 0}, {**_SHOWN, "target.shape": 1}],
    **{
        cell: [{f"agent.{part}_density": value} for part in ("torso", foot) for value in (550.0, 1450.0)]
        for cell, foot in (
            ("DMCCheetahRun", "back_foot"),
            ("DMCHopperHop", "foot"),
            ("DMCWalkerWalk", "left_foot"),
            ("DMCHumanoidWalk", "right_lower_arm"),
            ("DMCQuadrupedWalk", "foot_back_left"),
        )
    },
}
TRACKCOM = ("DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk", "DMCHumanoidWalk", "DMCQuadrupedWalk")
#: Checker colours for the trackcom floor: a saturated pair and a dim one.
FLOOR_COLORS = ([[0.9, 0.1, 0.1], [0.1, 0.8, 0.2]], [[0.3, 0.25, 0.05], [0.05, 0.1, 0.35]])
#: The trackcom floor leaves. Walker's floor turns about x = 248, so a fraction of a degree already
#: lifts or sinks it under the walker.
FLOOR_CASES = {cell: [{"floor.color": c} for c in FLOOR_COLORS] for cell in TRACKCOM}
FLOOR_CASES["DMCWalkerWalk"] += [{"floor.rotation_y": 0.125}, {"floor.rotation_y": -0.25}]


def _swm_value(path, value):
    # SWM's float32 leaves refuse float64 values.
    wide = path.startswith("gravity.") or path.endswith("color") or path == "floor.rotation_y"
    dtype = np.float64 if wide else np.float32
    if isinstance(value, int):
        return value
    value = np.asarray(value, dtype)
    return value if value.ndim > 1 else value.reshape(-1)


class _Renderer:
    """MuJoCo's renderer on SWM's live model. It and SWM's wrapper stay open for the process:
    tearing a GL context down corrupts the contexts the cells' renderers bake from."""

    _alive: list = []

    def __init__(self, ref, size, camera):
        import mujoco

        self.physics, self.camera = ref.envs[0].env.physics, camera
        self.renderer = mujoco.Renderer(self.physics.model.ptr, height=size, width=size)
        self._alive.append((self.renderer, ref))

    def __call__(self):
        self.renderer.update_scene(self.physics.data.ptr, camera=self.camera)
        return self.renderer.render().copy()


def _mse(a, b):
    err = ((a.astype(np.float64) - b.astype(np.float64)) / 255.0) ** 2
    per_frame = err.reshape(len(a), -1).mean(1)
    return {"mean": float(per_frame.mean()), "worst": float(per_frame.max())}


def _swm_texture_frames(env, ref, states, default_row):
    """Render ``states`` at the default style on our model compiled with SWM's grid texture."""
    import mujoco

    from stable_envs.envs.dmcontrol._model import _SCENE_CACHE, _TC_CACHE

    swm = ref.envs[0].env.physics.model
    swm = swm.ptr if hasattr(swm, "ptr") else swm
    m = type(env)._model()

    def grid(model):
        t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "grid")
        adr = int(model.tex_adr[t])
        return slice(adr, adr + int(model.tex_width[t]) * int(model.tex_height[t]) * int(model.tex_nchannel[t]))

    ours_span, swm_span = grid(m), grid(swm)
    assert ours_span.stop - ours_span.start == swm_span.stop - swm_span.start
    kept = np.array(m.tex_data[ours_span], copy=True)

    def evict():
        env.__dict__.pop("_cpu_scenes", None)
        _TC_CACHE.clear()
        _SCENE_CACHE.clear()

    m.tex_data[ours_span] = swm.tex_data[swm_span]
    evict()
    try:
        return np.stack([env.render(s, style=torch.as_tensor(default_row[None]))[0].cpu().numpy() for s in states])
    finally:
        m.tex_data[ours_span] = kept
        evict()


def frames(cell, cfg, seed=13, steps=8, device="cpu", substituted=False):
    """Render both sides along one trajectory; return SWM's frames and ours with and without ``cfg``.

    With ``substituted``, also return our default-style frames on a model carrying SWM's compiled grid texture."""
    domain = DOMAINS[cell]
    swm_cfg = {k: _swm_value(k, v) for k, v in cfg.items()}
    ref = DMCReference(domain, 1, options={"variation_values": swm_cfg})
    with se.make(cell, num_envs=1, device=device) as env:
        row = style_from_variation(env, {**swm_defaults(SWM_IDS[domain]), **swm_cfg})
        default_row = style_from_variation(env, swm_defaults(SWM_IDS[domain]))
        env.set_variation(row)
        ref.reset(seed=seed)
        # SWM compiles a new model on reset when the options change it.
        renderer = _Renderer(ref, env.size, env.CAMERA)
        env.reset(seed=seed)
        actions = np.random.default_rng(seed).uniform(-1, 1, (steps, 1, env.action_dim))
        theirs, ours, ignored, states = [], [], [], []
        for action in actions:
            state = env.step(action, states=None, frames=None)["state"]
            states.append(state)
            ref.step(action)
            theirs.append(renderer())
            ours.append(env.render(state)[0].cpu().numpy())
            ignored.append(env.render(state, style=torch.as_tensor(default_row[None]))[0].cpu().numpy())
        out = np.stack(theirs), np.stack(ours), np.stack(ignored)
        if substituted:
            out += (_swm_texture_frames(env, ref, states, default_row),)
    return out


def leaf_gate(cell, cfg, **kw):
    theirs, ours, ignored = frames(cell, cfg, **kw)
    return {
        "values": {k: np.asarray(v).tolist() for k, v in cfg.items()},
        "mse": _mse(ours, theirs),
        "mse_ignoring_leaf": _mse(ignored, theirs),
    }


def floor_gate(cell, cfg, **kw):
    """Our recolour against our renderer on SWM's substituted texture, and against SWM's frames."""
    theirs, ours, _, built = frames(cell, cfg, substituted=True, **kw)
    return {
        "values": {k: np.asarray(v).tolist() for k, v in cfg.items()},
        "max_abs_vs_substituted": int(np.abs(ours.astype(np.int16) - built.astype(np.int16)).max()),
        "mse": _mse(ours, theirs),
    }


def run_gate(cells=None, device="cpu", cases=CASES) -> dict:
    out = {}
    for cell in cells or cases:
        theirs, ours, _ = frames(cell, {}, device=device)
        out[cell] = {
            "registered_default_mse": _mse(ours, theirs),
            "cases": [leaf_gate(cell, c, device=device) for c in cases[cell]],
        }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("cells", nargs="*")
    args = parser.parse_args()
    result = run_gate(args.cells or None, args.device)
    floor = [c for c in args.cells or FLOOR_CASES if c in FLOOR_CASES]
    result["floor"] = run_gate(floor, args.device, FLOOR_CASES) if floor else {}
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
