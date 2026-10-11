"""Gates for the two variation hooks against pinned stable_worldmodel.

``reset_leaf_gate`` checks a per-lane reset distribution: the same explicit start on
both sides must give the same reset State and first frame, and the same seed must
draw the same sampled start. ``physics_gate`` checks a physics leaf: our trajectory
must stay within tolerance of SWM stepping the same knob, while moving the knob must
move SWM's trajectory far beyond that tolerance and ours the same way.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from stable_envs.envs.classic_control.simulation.local import _PHYSICS_SPACES
from tests.helpers.reference.dmc_semantics import DMCReference
from tests.helpers.reference.native_acceptance import CASES as CLASSIC_CASES
from tests.helpers.reference.native_acceptance import _actions as _classic_actions
from tests.helpers.reference.native_acceptance import compare_state as _classic_compare
from tests.helpers.reference.native_classic_semantics import NativeClassicReference
from tests.helpers.reference.native_navigation_semantics import NativeNavigationReference
from tests.helpers.reference.semantics import compare_state

NAVIGATION = {"two_room": "TwoRoom", "simple_point_maze": "SimplePointMaze"}
STARTS = {
    "two_room": np.asarray([[31.25, 47.5], [180.0, 60.0], [40.0, 200.0]], np.float32),
    "simple_point_maze": np.asarray([[0.3, 4.1], [4.6, 0.4], [2.75, 0.2]], np.float32),
}
SAMPLED = {"variation": ["agent.position"]}
#: Pixels within this many of the start's pixel are reported separately.
WINDOW = 16


def _pixel(domain, position):
    x, y = (float(v) for v in position)
    if domain == "simple_point_maze":
        return int(round(500 - 100 * y)), int(round(100 * x))
    return int(round(y)), int(round(x))


def _start_frames(domain, starts, ours, theirs, floor) -> dict:
    """First-frame agreement with SWM under the cell's declared parity floor, plus a window at each start.

    The floor applies to ``interior_mask`` pixels, the parity contract of these cells: antialiased
    shape edges are not interior, and a dot's edge is where the renderers' residual lives.
    """
    from stable_envs.testing.conformance import interior_mask, pixel_agreement

    agreement = pixel_agreement(ours, theirs)
    diff = np.abs(ours.astype(np.int16) - theirs.astype(np.int16)).max(-1)
    interior = interior_mask(theirs)
    window = []
    for lane, start in enumerate(starts):
        r, c = _pixel(domain, start)
        rows, cols = slice(max(r - WINDOW, 0), r + WINDOW + 1), slice(max(c - WINDOW, 0), c + WINDOW + 1)
        patch, flat = diff[lane, rows, cols], interior[lane, rows, cols]
        window.append(
            {
                "pixels_differing": int((patch > 0).sum()),
                "interior_pixels_differing": int((patch[: flat.shape[0], : flat.shape[1]][flat] > 0).sum()),
                "max_abs": int(patch.max()),
            }
        )
    return {
        "interior_exact_frac": agreement["interior_exact_frac"],
        "interior_bad_px": agreement["interior_bad_px"],
        "parity_interior_min": floor,
        "exact_frac": agreement["exact_frac"],
        "max_abs": agreement["max_abs"],
        "window": window,
    }


def _frames_agree(frames: dict) -> bool:
    return frames["interior_exact_frac"] is not None and frames["interior_exact_frac"] >= frames["parity_interior_min"]


def _navigation_reference(domain: str, lanes: np.ndarray | None, sampled: bool):
    """SWM per-lane references; an explicit start goes through SWM's own mechanism for that env."""
    if domain == "simple_point_maze":
        # SimplePointMazeEnv ignores options['variation_values']; its start is the leaf's value.
        refs = []
        for lane in range(3):
            values = None if lanes is None else {"agent.position": lanes[lane]}
            options = SAMPLED if sampled else {"variation": []}
            refs.append(NativeNavigationReference(domain, 1, options=options, persistent_values=values))
        return refs
    refs = []
    for lane in range(3):
        options = SAMPLED if sampled else {"variation_values": {"agent.position": lanes[lane]}}
        refs.append(NativeNavigationReference(domain, 1, options=options))
    return refs


def _reset_refs(refs, seed):
    states = [ref.reset(seed=seed + lane) for lane, ref in enumerate(refs)]
    frames = np.stack([np.asarray(ref.envs[0].render()) for ref in refs])
    return torch.cat(states), frames


def _our_options(domain, lanes):
    if domain == "simple_point_maze":
        return {"variation": [], "variation_values": {"agent.position": lanes}}
    return {"variation_values": {"agent.position": lanes}}


def reset_leaf_gate(domain: str, seed: int = 17) -> dict:
    """``agent.position`` as a per-lane reset distribution on a navigation cell."""
    evidence = {"domain": domain, "stable_env": NAVIGATION[domain], "leaf": "agent.position", "seed": seed}
    lanes = STARTS[domain]
    with se.make(NAVIGATION[domain], num_envs=3, device="cpu") as env:
        refs = _navigation_reference(domain, lanes, sampled=False)
        try:
            expected, their_frames = _reset_refs(refs, seed)
        finally:
            for ref in refs:
                ref.close()
        actual = env.reset(seed=seed, options=_our_options(domain, lanes))
        compare_state(actual, expected)
        assert np.array_equal(actual["agent", "position"].cpu().numpy().astype(np.float32), lanes)
        frames = _start_frames(domain, lanes, env.render().cpu().numpy(), their_frames, env.parity_interior_min)
        assert _frames_agree(frames), frames
        evidence["explicit"] = {"state_exact": True, "first_frame": frames}

        refs = _navigation_reference(domain, None, sampled=True)
        try:
            expected, their_frames = _reset_refs(refs, seed)
        finally:
            for ref in refs:
                ref.close()
        actual = env.reset(seed=seed, options=SAMPLED)
        compare_state(actual, expected)
        starts = actual["agent", "position"].cpu().numpy()
        assert len({tuple(row) for row in starts.tolist()}) == 3, starts
        frames = _start_frames(domain, starts, env.render().cpu().numpy(), their_frames, env.parity_interior_min)
        assert _frames_agree(frames), frames
        evidence["sampled"] = {"state_exact": True, "distinct_lanes": 3, "first_frame": frames}
    return evidence


class _Lanes:
    """SWM DMC wrappers, one per lane, each reset with its own explicit physics values.

    ``DMCReference`` reads gravity from each SWM model into State, so the comparison
    covers the physics SWM actually ran, not the values asked for.
    """

    def __init__(self, domain, values: dict[str, np.ndarray]):
        batch = len(next(iter(values.values())))
        self.refs = [
            DMCReference(
                domain,
                1,
                options={
                    "variation_values": {k: np.asarray(v[lane], np.float64).reshape(-1) for k, v in values.items()}
                },
            )
            for lane in range(batch)
        ]

    def reset(self, seed):
        return torch.cat([ref.reset(seed=seed + lane) for lane, ref in enumerate(self.refs)])

    def step(self, actions):
        return torch.cat([ref.step(actions[lane : lane + 1])["state"] for lane, ref in enumerate(self.refs)])

    def close(self):
        for ref in self.refs:
            ref.close()


def _rollout_ours(env, actions):
    states = []
    for action in actions:
        states.append(env.step(action, frames=None)["state"])
    return states


def _rollout_theirs(lanes, actions):
    return [lanes.step(action) for action in actions]


def _max_state_error(ours, theirs):
    return max(max(compare_state(a, b, atol=PHYSICS_ATOL).values()) for a, b in zip(ours, theirs, strict=True))


#: Per-step State tolerance against SWM stepping the same physics values.
PHYSICS_ATOL = 1e-9
#: The knob must move SWM's trajectory at least this many times the tolerance.
SENSITIVITY_FACTOR = 1e6


def physics_gate(stable_env="DMCCartpoleSwingup", domain="cartpole", seed=11, steps=40) -> dict:
    """``gravity.z`` (and a combined gravity vector) through the physics hook against SWM."""
    evidence = {"stable_env": stable_env, "domain": domain, "seed": seed, "steps": steps, "atol": PHYSICS_ATOL}
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-1, 1, (steps, 3, 1))
    with se.make(stable_env, num_envs=3, device="cpu") as env:
        leaves = env.physics_variation_leaves
        assert {"gravity.x", "gravity.y", "gravity.z"} <= set(leaves), leaves

        # Explicit per-lane gravity.z, including the registered value on lane 0.
        gz = np.asarray([[-9.81], [-3.25], [-17.5]], np.float64)
        ours0 = env.reset(seed=seed, options={"variation_values": {"gravity.z": gz}})
        lanes = _Lanes(domain, {"gravity.z": gz})
        try:
            compare_state(ours0, lanes.reset(seed), atol=0.0)
            ours = _rollout_ours(env, actions)
            theirs = _rollout_theirs(lanes, actions)
        finally:
            lanes.close()
        error = _max_state_error(ours, theirs)
        evidence["explicit_gravity_z"] = {"values": gz.reshape(-1).tolist(), "max_state_error": error}

        # Moving the knob moves SWM far beyond the tolerance, and ours identically.
        default = DMCReference(domain, 3)
        try:
            default.reset(seed=seed)
            base = [default.step(action)["state"]["qpos"] for action in actions]
        finally:
            default.close()
        swm_shift = torch.stack([t["qpos"] - b for t, b in zip(theirs, base, strict=True)])
        our_shift = torch.stack([o["qpos"] - b for o, b in zip(ours, base, strict=True)])
        moved = swm_shift.abs().amax(dim=(0, 2))
        assert float(moved[0]) == 0.0, "lane 0 runs the registered gravity on both sides"
        assert bool((moved[1:] > SENSITIVITY_FACTOR * PHYSICS_ATOL).all()), moved
        agreement = float((our_shift - swm_shift).abs().max())
        assert agreement <= PHYSICS_ATOL, agreement
        evidence["sensitivity"] = {
            "swm_max_qpos_shift_per_lane": moved.tolist(),
            "ours_minus_swm_shift_max": agreement,
            "shift_over_tolerance": float(moved[1:].min() / PHYSICS_ATOL),
        }

        # Per-lane sampled values, then SWM stepping the values we drew.
        drawn = env.reset(seed=seed + 1, options={"variation": ["gravity.z"]})
        gz = drawn["physics", "gravity", "z"].cpu().numpy()
        assert len(set(gz.reshape(-1).tolist())) == 3, gz
        lo, hi = env.variation_spec.low, env.variation_spec.high
        col = {p: o for p, o, _ in env.variation_space.flat_layout()}["gravity.z"]
        assert ((gz >= lo[col]) & (gz <= hi[col])).all()
        lanes = _Lanes(domain, {"gravity.z": gz})
        try:
            compare_state(drawn, lanes.reset(seed + 1), atol=0.0)
            error = _max_state_error(_rollout_ours(env, actions), _rollout_theirs(lanes, actions))
        finally:
            lanes.close()
        evidence["sampled_gravity_z"] = {"values": gz.reshape(-1).tolist(), "max_state_error": error}

        # A restored State carries its physics: varied gravity survives a reset to the registered value.
        vector = {
            "gravity.x": np.asarray([[1.25], [0.0], [-2.0]]),
            "gravity.y": np.asarray([[-0.75], [3.5], [0.0]]),
            "gravity.z": np.asarray([[-7.5], [-12.0], [-1.5]]),
        }
        env.reset(seed=seed + 2, options={"variation_values": vector})
        lanes = _Lanes(domain, vector)
        try:
            lanes.reset(seed + 2)
            prefix = actions[:5]
            _rollout_ours(env, prefix)
            _rollout_theirs(lanes, prefix)
            saved = env.get_state()
            registered = env.reset(seed=seed + 3, options={"variation_values": {"gravity.z": np.asarray([-9.81])}})
            for axis, value in zip("xyz", (0.0, 0.0, -9.81), strict=True):
                assert (registered["physics", "gravity", axis] == value).all()
            assert all(e.physics.model.opt.gravity[2] == -9.81 for e in env._simulator.envs)
            env.set_state(saved)
            assert np.array_equal(
                np.stack([e.physics.model.opt.gravity for e in env._simulator.envs]),
                np.concatenate([vector[f"gravity.{axis}"] for axis in "xyz"], axis=1),
            )
            compare_state(env.get_state(), saved)
            error = _max_state_error(_rollout_ours(env, actions[5:]), _rollout_theirs(lanes, actions[5:]))
        finally:
            lanes.close()
        evidence["restored_gravity_vector"] = {"max_state_error": error}
    return evidence


def reset_rule_gate(stable_env="DMCCartpoleSwingup", domain="cartpole", seed=23, steps=10) -> dict:
    """``reset(options)`` then ``reset()``: leaves the second reset does not name return to their defaults.

    First the registered defaults, then defaults installed by ``set_variation``, which SWM
    spells ``variation_space.set_init_value``. Gravity is read from each SWM model.
    """
    evidence = {"stable_env": stable_env, "domain": domain, "seed": seed, "steps": steps}
    actions = np.random.default_rng(seed).uniform(-1, 1, (steps, 3, 1))
    gz = np.asarray([[-3.25], [-17.5], [-9.81]], np.float64)
    gx = np.asarray([[1.5], [0.0], [-2.0]], np.float64)
    configured = -5.0
    with se.make(stable_env, num_envs=3, device="cpu") as env:
        lanes = _Lanes(domain, {"gravity.z": gz})
        try:
            compare_state(
                env.reset(seed=seed, options={"variation_values": {"gravity.z": gz}}), lanes.reset(seed), atol=0.0
            )
            for ref in lanes.refs:
                ref.options = None
            ours = env.reset(seed=seed + 1)
            compare_state(ours, lanes.reset(seed + 1), atol=0.0)
            assert bool((ours["physics", "gravity", "z"] == -9.81).all())
            registered = _max_state_error(_rollout_ours(env, actions), _rollout_theirs(lanes, actions))

            env.set_variation({"gravity.z": [configured]})
            for lane, ref in enumerate(lanes.refs):
                leaf = ref.envs[0].variation_space["gravity"]["z"]
                leaf.set_init_value(np.asarray([configured], leaf.dtype))
                ref.options = {"variation_values": {"gravity.x": gx[lane]}}
            compare_state(
                env.reset(seed=seed + 2, options={"variation_values": {"gravity.x": gx}}),
                lanes.reset(seed + 2),
                atol=0.0,
            )
            for ref in lanes.refs:
                ref.options = None
            ours = env.reset(seed=seed + 3)
            compare_state(ours, lanes.reset(seed + 3), atol=0.0)
            assert bool((ours["physics", "gravity", "x"] == 0.0).all())
            assert bool((ours["physics", "gravity", "z"] == configured).all())
            configured_error = _max_state_error(_rollout_ours(env, actions), _rollout_theirs(lanes, actions))
        finally:
            lanes.close()
    evidence["registered_defaults"] = {"max_state_error": registered}
    evidence["set_variation_defaults"] = {"value": configured, "max_state_error": configured_error}
    return evidence


def navigation_reset_rule_gate(seed=29, steps=8) -> dict:
    """TwoRoom ``agent.speed`` set by one reset's options, then a plain reset, stepped against SWM."""
    speed = np.asarray([6.5], np.float32)
    actions = np.random.default_rng(seed).uniform(-1, 1, (steps, 3, 2)).astype(np.float32)
    with se.make("TwoRoom", num_envs=3, device="cpu") as env:
        ref = NativeNavigationReference("two_room", 3, options={"variation_values": {"agent.speed": speed}})
        try:
            compare_state(env.reset(seed=seed, options={"variation_values": {"agent.speed": speed}}), ref.reset(seed))
            ref.options = None
            compare_state(env.reset(seed=seed + 1), ref.reset(seed + 1))
            default = float(env.variation_space["agent"]["speed"].value[0])
            assert bool((env._style["agent.speed"] == default).all())
            assert all(float(e.variation_space["agent"]["speed"].value[0]) == default for e in ref.envs)
            for action in actions:
                compare_state(env.step(action, frames=None)["state"], ref.step(action)["state"])
        finally:
            ref.close()
    return {"stable_env": "TwoRoom", "leaf": "agent.speed", "value": float(speed[0]), "steps": steps}


#: Start velocity that sits above every ``max_speed`` the leaf can take, so the clip decides the rollout.
_FAST_FIELD = {"pendulum": "theta_dot", "mountain_car": "velocity", "mountain_car_continuous": "velocity"}
_FAST_SPEED = {"pendulum": 14.0, "mountain_car": 0.2, "mountain_car_continuous": 0.2}


def classic_physics_gate(domain: str, seed: int = 13, steps: int = 24) -> dict:
    """Every physics leaf of a classic-control cell, explicit and sampled, against SWM.

    Lanes sit at 10 %, 50 % and 90 % of each leaf's range; State and the whole step output
    (reward, termination) must equal SWM's exactly. Where the cell's frame draws physics leaves
    (CartPole's pole length, Acrobot's link lengths), every reset and step frame must equal SWM's
    too, as worst-frame normalized MSE 0. A plain reset then restores the defaults.
    """
    batch = 3
    evidence = {"domain": domain, "seed": seed, "steps": steps, "leaves": {}}
    actions = _classic_actions(domain, seed, steps, batch)
    leaves = {f"physics.{name}": (lo, hi) for name, (lo, hi, _) in _PHYSICS_SPACES[domain].items()}

    def frame_mse(env, ref):
        ours = env._render().cpu().numpy().astype(np.float64) / 255
        want = np.stack([wrapper.render() for wrapper in ref.envs]).astype(np.float64) / 255
        return float(((ours - want) ** 2).mean(axis=(1, 2, 3)).max())

    def run(env, options, plain=None, fast=False, frames=False):
        ref = NativeClassicReference(domain, batch, options=options, render_mode="rgb_array" if frames else None)
        worst = [0.0]

        def framed(got, want):
            _classic_compare(got, want)
            if frames:
                worst.append(frame_mse(env, ref))
            return got

        try:
            framed(env.reset(seed=seed, options=plain), ref.reset(seed))
            if fast:
                start = ref.get_state()
                start[_FAST_FIELD[domain]].fill_(_FAST_SPEED[domain])
                env.set_state(start)
                ref.set_state(start)
            states = [
                framed(got, want)["state"]
                for got, want in zip(
                    (env.step(a, states=None, frames=None) for a in actions),
                    (ref.step(a) for a in actions),
                    strict=True,
                )
            ]
            assert max(worst) == 0.0, f"{domain} frames differ from SWM: worst-frame MSE {max(worst)}"
            return states, max(worst)
        finally:
            ref.close()

    with se.make(CLASSIC_CASES[domain]["stable_env"], num_envs=batch, device="cpu") as env:
        assert set(leaves) == set(env.dynamics_variation_leaves)
        drawn_leaves = set(env.geometry_leaves)
        draws = bool(drawn_leaves)
        baselines = {fast: run(env, None, fast=fast, frames=draws)[0][-1] for fast in (False, domain in _FAST_FIELD)}
        for leaf, (lo, hi) in leaves.items():
            values = np.asarray([[lo + f * (hi - lo)] for f in (0.1, 0.5, 0.9)], np.float32)
            per_lane = [{"variation_values": {leaf: v}} for v in values]
            fast = leaf.endswith("max_speed") and domain in _FAST_FIELD
            states, worst = run(
                env, per_lane, {"variation_values": {leaf: values}}, fast=fast, frames=leaf in drawn_leaves
            )
            last, baseline = states[-1], baselines[fast]
            moved = [
                any(
                    not torch.equal(last[k][lane], baseline[k][lane])
                    for k in last.keys(True, True)
                    if k[0] not in ("rng", "physics")
                )
                for lane in (0, 2)
            ]
            evidence["leaves"][leaf] = {"values": values.reshape(-1).tolist(), "moved_vs_default": moved}
            if leaf in drawn_leaves:
                evidence["leaves"][leaf]["frame_worst_mse"] = worst
        drawn = env.reset(seed=seed + 1, options={"variation": list(leaves)})
        sampled = {leaf: env._style[leaf].cpu().numpy().reshape(batch, 1) for leaf in leaves}
        for leaf, (lo, hi) in leaves.items():
            assert len(set(sampled[leaf].reshape(-1).tolist())) == batch, leaf
            assert ((sampled[leaf] >= lo) & (sampled[leaf] <= hi)).all(), leaf
        per_lane = [{"variation_values": {leaf: sampled[leaf][lane] for leaf in leaves}} for lane in range(batch)]
        ref = NativeClassicReference(domain, batch, options=per_lane, render_mode="rgb_array" if draws else None)
        try:
            _classic_compare(drawn, ref.reset(seed + 1))
            worst = [frame_mse(env, ref)] if draws else [0.0]
            for a in actions:
                _classic_compare(env.step(a, states=None, frames=None), ref.step(a))
                if draws:
                    worst.append(frame_mse(env, ref))
            assert max(worst) == 0.0, f"{domain} sampled frames differ from SWM: worst-frame MSE {max(worst)}"
        finally:
            ref.close()
        evidence["sampled"] = {leaf: sampled[leaf].reshape(-1).tolist() for leaf in leaves}
        if draws:
            evidence["sampled_frame_worst_mse"] = max(worst)
        run(env, None, None, frames=draws)
        evidence["plain_reset_restores_defaults"] = True
    return evidence


ROCKET_START = {
    "environment.start_height_ratio": np.asarray([0.9, 0.75, 0.93]),
    "environment.start_horizontal_offset": np.asarray([[0.0, 0.0], [1.5, -2.0], [-0.7, 1.2]], np.float32),
    "environment.start_tilt": np.asarray([[0.0, 0.0], [0.15, -0.1], [-0.2, 0.05]], np.float32),
}


def _swm_leaf(env, path):
    node = env.variation_space
    for part in path.split("."):
        node = node[part]
    return node


class _RocketLanes:
    """One SWM Rocket per lane; ``defaults`` are installed with SWM's ``set_init_value``."""

    def __init__(self, defaults=None):
        from stable_worldmodel.envs.rocket_landing.pyflyt_rocketlanding import RocketLandingEnv

        self.envs = [RocketLandingEnv() for _ in range(3)]
        for lane, env in enumerate(self.envs):
            for path, values in (defaults or {}).items():
                _swm_leaf(env, path).set_init_value(values[lane])

    def reset(self, seed, options=None):
        for lane, env in enumerate(self.envs):
            env.reset(seed=seed + lane, options=options)

    def step(self, actions):
        return [env.step(action) for env, action in zip(self.envs, actions, strict=True)]

    def view(self):
        """The source-visible part of our State: observation, materials and Variation values."""
        out = {"observation": np.stack([env.state for env in self.envs])}
        visuals = [{v[1]: v[7] for v in env.env.getVisualShapeData(env.env.drones[0].Id)} for env in self.envs]
        out["rgba"] = np.stack(
            [
                np.asarray([v[-1], v[2], v[6], v[1], env.env.getVisualShapeData(env.landing_pad_id)[0][7]])
                for v, env in zip(visuals, self.envs, strict=True)
            ]
        )
        for path in ROCKET_VIEWED:
            out[path] = np.stack([np.asarray(_swm_leaf(env, path).value, np.float64) for env in self.envs])
        return out

    def close(self):
        for env in self.envs:
            env.close()


ROCKET_VIEWED = (*ROCKET_START, "rocket.body_color", "rocket.fin_color", "pad.color")


def _rocket_view(state):
    out = {
        "observation": state["current", "wrapper", "state"].cpu().numpy(),
        "rgba": state["render", "rgba"][:, :5].cpu().numpy(),
    }
    for path in ROCKET_VIEWED:
        out[path] = state[("variation", *path.split("."), "value")].cpu().numpy()
    return out


def _rocket_same(ours, theirs):
    for key, value in theirs.items():
        assert np.array_equal(ours[key].reshape(value.shape), value), (key, ours[key], value)


def _rocket_rollout(env, lanes, actions):
    """Per-step observation, reward and termination must be bit-identical."""
    for action in actions:
        out = env.step(action, frames=None)
        theirs = lanes.step(action)
        assert np.array_equal(_rocket_view(out["state"])["observation"], np.stack([r[0] for r in theirs]))
        assert out["reward"].tolist() == [float(r[1]) for r in theirs]
        assert out["terminated"].tolist() == [bool(r[2]) for r in theirs]


def rocket_reset_leaf_gate(seed=31, steps=40) -> dict:
    """Rocket's start leaves per lane, explicit and sampled, and the colours reset can set, against SWM."""
    low = np.asarray([-1, -1, -1, 0, 0, -1, -1], np.float64)
    actions = np.random.default_rng(seed).uniform(low, 1.0, (steps, 3, 7))
    evidence = {"stable_env": "PyFlytRocketLanding", "leaves": sorted(ROCKET_START), "seed": seed, "steps": steps}
    with se.make("PyFlytRocketLanding", num_envs=3, device="cpu") as env:
        # Explicit per-lane values; SWM reads only init values, so its lanes take them as defaults.
        ours = env.reset(seed=seed, options={"variation_values": ROCKET_START})
        lanes, default = _RocketLanes(ROCKET_START), _RocketLanes()
        try:
            lanes.reset(seed)
            default.reset(seed)
            _rocket_same(_rocket_view(ours), lanes.view())
            start = lanes.view()["observation"][:, 10:13] - default.view()["observation"][:, 10:13]
            assert not start[0].any() and np.abs(start[1:]).max(-1).min() > 0.1, start
            _rocket_rollout(env, lanes, actions)
        finally:
            lanes.close()
            default.close()
        evidence["explicit"] = {"start_shift_vs_default_m": np.abs(start).max(-1).tolist(), "exact": True}

        # Sampled per lane under seed + lane, then a plain reset back to the registered defaults.
        options = {"variation": sorted(ROCKET_START)}
        ours = env.reset(seed=seed + 1, options=options)
        lanes = _RocketLanes()
        try:
            lanes.reset(seed + 1, options)
            view = _rocket_view(ours)
            _rocket_same(view, lanes.view())
            assert len({float(h) for h in view["environment.start_height_ratio"]}) == 3
            _rocket_rollout(env, lanes, actions)
            ours = env.reset(seed=seed + 2)
            lanes.reset(seed + 2)
            _rocket_same(_rocket_view(ours), lanes.view())
            _rocket_rollout(env, lanes, actions[:10])
        finally:
            lanes.close()
        evidence["sampled"] = {"heights": view["environment.start_height_ratio"].tolist(), "exact": True}

        # Colours set for one reset are drawn as SWM draws the same values once a colour reset paints.
        colours = {
            "rocket.body_color": np.asarray([[255, 204, 0], [12, 140, 250], [200, 30, 90]], np.uint8),
            "pad.color": np.asarray([[200, 200, 200], [104, 60, 3], [240, 132, 195]], np.uint8),
        }
        options = {"variation": ["rocket.fin_color"]}
        ours = env.reset(seed=seed + 3, options={**options, "variation_values": colours})
        lanes = _RocketLanes(colours)
        try:
            lanes.reset(seed + 3, options)
            _rocket_same(_rocket_view(ours), lanes.view())
        finally:
            lanes.close()
        evidence["colours"] = {"leaves": sorted(colours), "exact": True}
    return evidence


def run_gate() -> dict:
    return {
        "reset_leaf": [reset_leaf_gate(domain) for domain in NAVIGATION],
        "physics": physics_gate(),
        "reset_rule": reset_rule_gate(),
        "navigation_reset_rule": navigation_reset_rule_gate(),
        "classic_physics": [classic_physics_gate(domain) for domain in CLASSIC_CASES],
        "rocket_reset_leaf": rocket_reset_leaf_gate(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_gate()
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
