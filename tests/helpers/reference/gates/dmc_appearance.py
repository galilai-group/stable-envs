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


def _swm_value(path, value):
    # SWM's float32 leaves refuse float64 values.
    dtype = np.float64 if path.startswith("gravity.") or path.endswith("color") else np.float32
    return np.asarray(value, dtype).reshape(-1) if not isinstance(value, int) else value


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


def frames(cell, cfg, seed=13, steps=8, device="cpu"):
    """Render both sides along one trajectory; return SWM's frames and ours with and without ``cfg``."""
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
        theirs, ours, ignored = [], [], []
        for action in actions:
            state = env.step(action, states=None, frames=None)["state"]
            ref.step(action)
            theirs.append(renderer())
            ours.append(env.render(state)[0].cpu().numpy())
            ignored.append(env.render(state, style=torch.as_tensor(default_row[None]))[0].cpu().numpy())
    return np.stack(theirs), np.stack(ours), np.stack(ignored)


def leaf_gate(cell, cfg, **kw):
    theirs, ours, ignored = frames(cell, cfg, **kw)
    return {
        "values": {k: np.asarray(v).tolist() for k, v in cfg.items()},
        "mse": _mse(ours, theirs),
        "mse_ignoring_leaf": _mse(ignored, theirs),
    }


def run_gate(cells=None, device="cpu") -> dict:
    out = {}
    for cell in cells or CASES:
        theirs, ours, _ = frames(cell, {}, device=device)
        out[cell] = {
            "registered_default_mse": _mse(ours, theirs),
            "cases": [leaf_gate(cell, c, device=device) for c in CASES[cell]],
        }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("cells", nargs="*")
    args = parser.parse_args()
    result = run_gate(args.cells or None, args.device)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
