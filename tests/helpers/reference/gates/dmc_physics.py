"""Every DM Control physics Variation leaf against the pinned stable_worldmodel wrappers.

For each leaf, three lanes run the registered value, a low and a high value. SWM sets each
lane's value through ``reset(options={"variation_values": ...})`` and recompiles or writes
the model; ours writes the same value into each lane's compiled model. The reset State and a
short rollout must agree within ``PHYSICS_TOL``, and the varied lanes must move SWM's own
trajectory far beyond it. Then every leaf is sampled per lane at once, and a restored State
keeps the physics it was produced under.

Errors are ``|ours - swm| / (1 + |swm|)`` over every floating State leaf, so the solver
warm-start, of order 1e3, is held to the same relative precision as positions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from tests.helpers.reference.dmc_semantics import PHYSICS_LEAVES, DMCReference
from tests.helpers.reference.semantics import compare_state

#: stable-envs cell -> SWM domain.
CELLS = {
    "DMCAcrobotBalance": "acrobot",
    "DMCPendulumSwingup": "pendulum",
    "DMCCartpoleSwingup": "cartpole",
    "DMCCartpoleBalance": "cartpole",
    "DMCBallInCupCatch": "ball_in_cup",
    "DMCFingerTurnHard": "finger",
    "DMCManipulatorBringBall": "manipulator",
    "DMCReacherHard": "reacher",
    "DMCCheetahRun": "cheetah",
    "DMCHopperHop": "hopper",
    "DMCWalkerWalk": "walker",
    "DMCHumanoidWalk": "humanoid",
    "DMCQuadrupedWalk": "quadruped",
}
TRACKCOM = {"DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk", "DMCHumanoidWalk", "DMCQuadrupedWalk"}
#: State tolerance over a rollout. Direct model writes (gravity, friction) are exact; an analytic mass
#: change agrees with SWM's recompile to 1e-12 in the model, which contact-rich rollouts amplify.
PHYSICS_TOL = 1e-6
#: The leaf must move SWM's trajectory at least this many times the tolerance.
SENSITIVITY_FACTOR = 1e3
#: Leaves SWM itself never lets reach the trajectory, with the reason.
INERT_UPSTREAM = {
    **{
        (cell, "gravity.y"): "planar model: every joint moves in the x-z plane, so y gravity does no work"
        for cell in CELLS
        if cell not in ("DMCHumanoidWalk", "DMCQuadrupedWalk")
    },
    ("DMCHopperHop", "floor.friction"): "every hopper geom has friction 1.0 and MuJoCo combines by max",
    ("DMCQuadrupedWalk", "floor.friction"): "the feet have friction 1.5 and MuJoCo combines by max",
}
#: Rollouts long enough for the floor to be reached before friction can act.
STEPS = {("DMCHopperHop", "floor.friction"): 200, ("DMCQuadrupedWalk", "floor.friction"): 200}
#: Spinner angular velocity both sides start from in the finger cases.
KICK = 8.0


def _balance_wrapper():
    """SWM's cartpole wrapper on dm_control's Balance task, which DMCCartpoleBalance simulates."""
    import os
    import tempfile

    from dm_control import mjcf
    from dm_control.rl import control
    from dm_control.suite import cartpole
    from dm_control.suite.wrappers import action_scale
    from stable_worldmodel.envs.dmcontrol.cartpole import CartpoleDMControlWrapper

    class BalanceWrapper(CartpoleDMControlWrapper):
        def compile_model(self, seed=None, environment_kwargs=None):
            self._mjcf_tempdir = tempfile.TemporaryDirectory()
            mjcf.export_with_assets(self._mjcf_model, self._mjcf_tempdir.name, out_file_name="cartpole.xml")
            physics = cartpole.Physics.from_xml_path(os.path.join(self._mjcf_tempdir.name, "cartpole.xml"))
            task = cartpole.Balance(swing_up=False, sparse=False, random=seed)
            env = control.Environment(physics, task, time_limit=10, **(environment_kwargs or {}))
            self.env = action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)
            self._dirty = False

    return BalanceWrapper


def _make(name):
    return se.make(name, num_envs=3, device="cpu", **({"backend": "opengl"} if name in TRACKCOM else {}))


def _swm_dtype(leaf, value):
    # SWM holds gravity and the floor tilt in float64 and every other physics leaf in float32.
    wide = leaf.startswith("gravity.") or leaf == "floor.rotation_y"
    return np.asarray(value, np.float64 if wide else np.float32).reshape(-1)


class _Lanes:
    """One SWM wrapper per lane, each reset with its own explicit physics values."""

    def __init__(self, name, values):
        wrapper = _balance_wrapper() if name == "DMCCartpoleBalance" else None
        self.refs = [
            DMCReference(
                CELLS[name],
                1,
                options={"variation_values": {k: _swm_dtype(k, v[lane]) for k, v in values.items()}},
                wrapper=wrapper,
            )
            for lane in range(3)
        ]

    def reset(self, seed):
        return torch.cat([ref.reset(seed=seed + lane) for lane, ref in enumerate(self.refs)])

    def step(self, actions):
        return torch.cat([ref.step(actions[lane : lane + 1])["state"] for lane, ref in enumerate(self.refs)])

    def close(self):
        for ref in self.refs:
            ref.close()


def _swm_values(name, leaf):
    """SWM's registered value and two interior values of its range, in SWM's dtype for the leaf."""
    ref = _Lanes(name, {}).refs
    try:
        node = ref[0].envs[0].variation_space
        for part in leaf.split("."):
            node = node[part]
        lo, hi = float(np.asarray(node.low).reshape(-1)[0]), float(np.asarray(node.high).reshape(-1)[0])
        init = np.asarray(node.init_value).reshape(-1)[:1]
        values = np.concatenate([init, np.asarray([lo + 0.25 * (hi - lo), lo + 0.9 * (hi - lo)], init.dtype)])
    finally:
        for r in ref:
            r.close()
    return values.reshape(3, 1)


def _compare(ours, theirs) -> float:
    """Largest relative floating error; integer and boolean leaves must match exactly."""
    assert set(ours.keys(True, True)) == set(theirs.keys(True, True))
    error = 0.0
    for key in theirs.keys(True, True):
        a, e = ours[key].detach().cpu(), theirs[key].detach().cpu()
        if a.is_floating_point():
            error = max(error, float(((a - e).abs() / (1.0 + e.abs())).max()))
        else:
            assert torch.equal(a, e), key
    return error


def _rollout(env, lanes, actions):
    error = 0.0
    states = []
    for action in actions:
        ours, theirs = env.step(action, states=None, frames=None)["state"], lanes.step(action)
        error = max(error, _compare(ours, theirs))
        states.append(theirs)
    return error, states


def _kick(name, env, lanes):
    """Spin the finger's spinner on both sides; random actions rarely reach it within a short rollout."""
    if name != "DMCFingerTurnHard":
        return
    state = env.get_state()
    state["qvel"][:, -1] = KICK
    env.set_state(state)
    for ref in lanes.refs:
        physics = ref.envs[0].env.physics
        warmstart = physics.data.qacc_warmstart.copy()
        physics.named.data.qvel["hinge"] = KICK
        # dm_control steps from derived quantities of the current state, as our set_state leaves them.
        physics.forward()
        physics.data.qacc_warmstart[:] = warmstart


def leaf_gate(name, leaf, seed=11):
    steps = STEPS.get((name, leaf), 40)
    with _make(name) as env:
        actions = np.random.default_rng(seed).uniform(-1, 1, (steps, 3, env.action_dim))
        values = _swm_values(name, leaf)
        reset = env.reset(seed=seed, options={"variation_values": {leaf: values}})
        lanes = _Lanes(name, {leaf: values})
        try:
            _compare(reset, lanes.reset(seed))
            _kick(name, env, lanes)
            error, varied = _rollout(env, lanes, actions)
        finally:
            lanes.close()
        default = _Lanes(name, {})
        try:
            env.reset(seed=seed)
            default.reset(seed)
            _kick(name, env, default)
            base = [default.step(action)["qpos"] for action in actions]
        finally:
            default.close()
    moved = torch.stack([v["qpos"] - b for v, b in zip(varied, base, strict=True)]).abs().amax(dim=(0, 2))
    assert float(moved[0]) == 0.0, "lane 0 runs the registered value on both sides"
    return {
        "values": values.reshape(-1).tolist(),
        "max_state_error": error,
        "swm_max_qpos_shift": moved[1:].tolist(),
        "sensitive": bool((moved[1:] > SENSITIVITY_FACTOR * PHYSICS_TOL).all()),
    }


def sampled_and_restored_gate(name, seed=29, steps=20):
    """Sample every physics leaf per lane, step against SWM, then restore a mid-rollout State."""
    with _make(name) as env:
        actions = np.random.default_rng(seed).uniform(-1, 1, (steps, 3, env.action_dim))
        leaves = list(env.physics_variation_leaves)
        drawn = env.reset(seed=seed, options={"variation": leaves})
        values = {leaf: drawn[("physics", *leaf.split("."))].numpy() for leaf in leaves}
        for leaf, value in values.items():
            assert len(set(value.reshape(-1).tolist())) == 3, (leaf, value)
            if not (leaf.startswith("gravity.") or leaf == "floor.rotation_y"):
                # SWM holds these leaves in float32; our draw is a float32 style value.
                assert np.array_equal(value, value.astype(np.float32).astype(np.float64)), (leaf, value)
        lanes = _Lanes(name, values)
        try:
            _compare(drawn, lanes.reset(seed))
            error, _ = _rollout(env, lanes, actions[: steps // 2])
            saved = env.get_state()
            env.reset(seed=seed + 1)
            env.set_state(saved)
            compare_state(env.get_state(), saved)
            restored, _ = _rollout(env, lanes, actions[steps // 2 :])
        finally:
            lanes.close()
    return {"leaves": leaves, "max_state_error": error, "restored_max_state_error": restored}


def run_gate(cells=None) -> dict:
    out = {}
    for name in cells or CELLS:
        leaves = PHYSICS_LEAVES[CELLS[name]]
        out[name] = {
            "leaves": {leaf: leaf_gate(name, leaf) for leaf in leaves},
            "sampled_and_restored": sampled_and_restored_gate(name),
        }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("cells", nargs="*")
    args = parser.parse_args()
    result = run_gate(args.cells or None)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
