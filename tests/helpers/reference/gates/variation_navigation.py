"""Navigation variation leaves against pinned stable_worldmodel.

``reset_leaf_gate`` covers a per-lane reset distribution (start and target/goal positions):
explicit values and seeded per-lane draws must give the same reset State and first frame as
SWM, and a short rollout must keep matching. ``appearance_gate`` covers Piecewise leaves that
only the frame reads. ``navigation_reset_rule_gate`` checks that a plain reset restores a leaf the
previous reset's options set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from tests.helpers import source_mse as SM
from tests.helpers.reference.native_navigation_semantics import NativeNavigationReference
from tests.helpers.reference.semantics import compare_state

ENVS = {"two_room": "TwoRoom", "piecewise": "Piecewise", "simple_point_maze": "SimplePointMaze"}
EXPLICIT = {
    ("two_room", "agent.position"): [[31.25, 47.5], [180.0, 60.0], [40.0, 200.0]],
    ("simple_point_maze", "agent.position"): [[0.3, 4.1], [4.6, 0.4], [2.75, 0.2]],
    ("two_room", "target.position"): [[31.25, 47.5], [180.0, 60.0], [40.0, 200.0]],
    ("piecewise", "agent.position"): [[31.25, 47.5], [180.0, 60.0], [40.0, 200.0]],
    ("piecewise", "target.position"): [[100.0, 30.0], [20.5, 150.0], [190.0, 190.0]],
    ("simple_point_maze", "goal.position"): [[0.3, 4.1], [4.6, 0.4], [2.75, 0.2]],
}
STEPS = 8
#: Pixels within this many of the start's pixel are reported separately.
WINDOW = 16


def _pixel(domain, position):
    x, y = (float(v) for v in position)
    if domain == "simple_point_maze":
        return int(round(500 - 100 * y)), int(round(100 * x))
    return int(round(y)), int(round(x))


def _start_frames(domain, starts, ours, theirs, ceiling) -> dict:
    """First-frame agreement with SWM: the worst per-frame MSE against ``ceiling``, plus a window at each start."""
    from stable_envs.testing.conformance import frame_mse, interior_mask, pixel_agreement

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
        "mse_max": float(frame_mse(ours, theirs).max()),
        "mse_ceiling": ceiling,
        "exact_frac": agreement["exact_frac"],
        "max_abs": agreement["max_abs"],
        "window": window,
    }


def _mse_ceiling(env) -> float:
    """The class's recorded worst-frame MSE against the source, with the ratchet's margin."""
    name = type(env).__name__
    rec = SM.record(name)
    if rec is None:
        import pytest

        pytest.skip(f"no MSE record for {name}; run {SM.RECORD}")
    return SM.ceiling(rec["mse_max"])


def _frames_agree(frames: dict) -> bool:
    return frames["mse_max"] <= frames["mse_ceiling"]


def _references(domain, leaf, explicit, options):
    if domain == "simple_point_maze":
        # SimplePointMazeEnv ignores options['variation_values']; an explicit value is the leaf's init value.
        return [
            NativeNavigationReference(
                domain, 1, options=options, persistent_values=None if explicit is None else {leaf: explicit[lane]}
            )
            for lane in range(3)
        ]
    if explicit is None:
        return [NativeNavigationReference(domain, 1, options=options) for _ in range(3)]
    return [
        NativeNavigationReference(domain, 1, options={**options, "variation_values": {leaf: explicit[lane]}})
        for lane in range(3)
    ]


def _swm_reset(refs, seed):
    states = [ref.reset(seed=seed + lane) for lane, ref in enumerate(refs)]
    frames = np.stack(
        [
            np.asarray(ref.envs[0].render(*(("rgb_array",) if ref.domain == "simple_point_maze" else ())))
            for ref in refs
        ]
    )
    return torch.cat(states), frames


def _rollout(env, refs, seed, steps):
    limit = 0.2 if refs[0].domain == "simple_point_maze" else 1.0
    actions = np.random.default_rng(seed).uniform(-limit, limit, (steps, 3, 2)).astype(np.float32)
    for action in actions:
        ours = env.step(action, frames=None)["state"]
        theirs = torch.cat([ref.step(action[lane : lane + 1])["state"] for lane, ref in enumerate(refs)])
        compare_state(ours, theirs)
    return steps


def _position(state, leaf):
    return state[tuple(leaf.split("."))].cpu().numpy().astype(np.float32)


def _case(env, domain, leaf, seed, explicit):
    values = None if explicit is None else np.asarray(explicit, np.float32)
    options = {"variation": [] if values is not None else [leaf]}
    refs = _references(domain, leaf, values, options)
    try:
        expected, their_frames = _swm_reset(refs, seed)
        ours_options = dict(options)
        if values is not None:
            ours_options["variation_values"] = {leaf: values}
        actual = env.reset(seed=seed, options=ours_options)
        compare_state(actual, expected)
        drawn = _position(actual, leaf)
        if values is not None:
            assert np.array_equal(drawn, values), drawn
        else:
            assert len({tuple(row) for row in drawn.tolist()}) == 3, drawn
        frames = _start_frames(domain, drawn, env.render().cpu().numpy(), their_frames, _mse_ceiling(env))
        assert _frames_agree(frames), frames
        steps = _rollout(env, refs, seed, STEPS)
    finally:
        for ref in refs:
            ref.close()
    return {"state_exact": True, "values": drawn.tolist(), "first_frame": frames, "rollout_steps_exact": steps}


def reset_leaf_gate(domain: str, leaf: str, seed: int = 17) -> dict:
    evidence = {"domain": domain, "stable_env": ENVS[domain], "leaf": leaf, "seed": seed}
    with se.make(ENVS[domain], num_envs=3, device="cpu") as env:
        evidence["explicit"] = _case(env, domain, leaf, seed, EXPLICIT[(domain, leaf)])
        evidence["sampled"] = _case(env, domain, leaf, seed + 1, None)
    return evidence


#: One Piecewise configuration per lane: the target alone, the bias arrows alone, then both together.
LAYERS = (
    {"rendering.render_target": 1, "target.color": [10, 20, 200], "target.radius": [11.0]},
    {"rendering.render_bias_field": 1, "zones.bias_0": [3.5, -1.0], "zones.bias_3": [-0.5, 4.0]},
    {"rendering.render_target": 1, "rendering.render_bias_field": 1, "target.radius": [7.5]},
)


def _lane_values(configs):
    paths = sorted({path for config in configs for path in config})
    with se.make("Piecewise", num_envs=1, device="cpu") as probe:
        default = {path: probe._style[path][0].cpu().numpy() for path in paths}
    return {
        path: np.stack([np.asarray(config.get(path, default[path]), np.float32).reshape(-1) for config in configs])
        for path in paths
    }


FLAGS = ("rendering.render_target", "rendering.render_bias_field")


def _swm_value(path, value):
    if path in FLAGS:
        return int(value[0])
    return value.astype(np.uint8) if path.endswith(".color") else value.astype(np.float32)


def appearance_gate(seed: int = 5, steps: int = 4) -> dict:
    """Piecewise target and bias-field leaves, per lane, against SWM's frames through a rollout."""
    values = _lane_values(LAYERS)
    refs = [
        NativeNavigationReference(
            "piecewise",
            1,
            options={"variation": [], "variation_values": {k: _swm_value(k, v[lane]) for k, v in values.items()}},
        )
        for lane in range(3)
    ]
    actions = np.random.default_rng(seed).uniform(-1, 1, (steps, 3, 2)).astype(np.float32)
    evidence = {"stable_env": "Piecewise", "lanes": [sorted(c) for c in LAYERS], "seed": seed, "frames": []}
    with se.make("Piecewise", num_envs=3, device="cpu") as env:
        try:
            expected = _swm_reset(refs, seed)[0]
            compare_state(env.reset(seed=seed, options={"variation": [], "variation_values": values}), expected)
            for t in range(steps + 1):
                if t:
                    theirs = [ref.step(actions[t - 1][lane : lane + 1])["state"] for lane, ref in enumerate(refs)]
                    compare_state(env.step(actions[t - 1], frames=None)["state"], torch.cat(theirs))
                ours = env.render().cpu().numpy()
                theirs = np.stack([np.asarray(ref.envs[0].render()) for ref in refs])
                starts = env.get_state()["agent", "position"].cpu().numpy()
                frames = _start_frames("piecewise", starts, ours, theirs, _mse_ceiling(env))
                assert _frames_agree(frames), frames
                evidence["frames"].append(frames)
            plain = env.render(variation={flag: 0 for flag in FLAGS}).cpu().numpy()
        finally:
            for ref in refs:
                ref.close()
    moved = (np.abs(ours.astype(np.int16) - plain.astype(np.int16)).max(-1) > 0).reshape(3, -1).sum(1)
    assert (moved > 0).all(), moved
    evidence["px_moved_by_leaves_per_lane"] = moved.tolist()
    return evidence


def zones_bias_field_gate(seed: int = 5) -> dict:
    """PiecewiseZones with the bias arrows over the zone tints, against SWM's frame."""
    config = {"rendering.render_bias_field": 1, "zones.bias_2": [0.25, 3.0]}
    ref = NativeNavigationReference(
        "piecewise",
        1,
        options={
            "variation": [],
            "variation_values": {
                "rendering.render_zones": 1,
                "rendering.render_bias_field": 1,
                "zones.bias_2": np.asarray(config["zones.bias_2"], np.float32),
            },
        },
    )
    with se.make("PiecewiseZones", num_envs=1, device="cpu") as env:
        try:
            ref.reset(seed=seed)
            env.set_variation({"rendering.render_bias_field": [1], "zones.bias_2": config["zones.bias_2"]})
            start = ref.envs[0].agent_position.numpy()[None].astype(np.float32)
            env._set_render_state(start)
            frames = _start_frames(
                "piecewise",
                start,
                env.render().cpu().numpy(),
                np.asarray(ref.envs[0].render())[None],
                _mse_ceiling(env),
            )
            assert _frames_agree(frames), frames
        finally:
            ref.close()
    return {"stable_env": "PiecewiseZones", "leaves": sorted(config), "first_frame": frames}


def min_steps_probe(seeds=range(5), steps: int = 8) -> dict:
    """``task.min_steps`` at its extremes against its default, in SWM alone: reset, rollout and frames.

    SWM declares the leaf and reads it only in TwoRoom's ``_constrain_target_by_min_steps``, whose
    use as ``constrain_fn`` is commented out; Piecewise never reads it.
    """
    evidence = {}
    for domain in ("two_room", "piecewise"):
        moved = 0
        for seed in seeds:
            actions = np.random.default_rng(seed).uniform(-1, 1, (steps, 1, 2)).astype(np.float32)
            runs = []
            for value in (None, 15, 114):
                options = {"variation": ["agent.position", "target.position"]}
                if value is not None:
                    options["variation_values"] = {"task.min_steps": value}
                ref = NativeNavigationReference(domain, 1, options=options)
                try:
                    states = [ref.reset(seed=seed)]
                    frames = [np.asarray(ref.envs[0].render())]
                    for action in actions:
                        states.append(ref.step(action)["state"])
                        frames.append(np.asarray(ref.envs[0].render()))
                finally:
                    ref.close()
                runs.append((states, np.stack(frames)))
            for states, frames in runs[1:]:
                for a, b in zip(states, runs[0][0], strict=True):
                    compare_state(a, b)
                moved += int((frames != runs[0][1]).any(-1).sum())
        assert moved == 0, moved
        evidence[domain] = {"seeds": len(seeds), "steps": steps, "values": [15, 114], "px_moved": moved}
    return evidence


GATES = (
    ("two_room", "agent.position"),
    ("simple_point_maze", "agent.position"),
    ("two_room", "target.position"),
    ("piecewise", "agent.position"),
    ("piecewise", "target.position"),
    ("simple_point_maze", "goal.position"),
)


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


def run_gate() -> dict:
    return {
        "reset_leaf": [reset_leaf_gate(domain, leaf) for domain, leaf in GATES],
        "appearance": appearance_gate(),
        "zones_bias_field": zones_bias_field_gate(),
        "min_steps_probe": min_steps_probe(),
        "reset_rule": navigation_reset_rule_gate(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    text = json.dumps(run_gate(), indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
