"""PushT variation leaves against pinned stable_worldmodel.

Each leaf is set per lane (``variation_values`` with one value per lane) and, where SWM
samples it, drawn per lane (``variation``); one SWM PushT per lane is reset with the same seed
and options. The reset bodies, observation, goal and frame must match exactly, and so must a
short rollout's observations, rewards, terminations and frames. A leaf must also move SWM away
from its default, so the comparison is not two defaults agreeing.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

import stable_envs as se

CLASSES = {"PushT": ("env", "PushT"), "PushTDiscrete": ("env_discrete", "PushTDiscrete")}
EXPLICIT = {
    "agent.shape": np.asarray([1, 3, 7]),
    "block.shape": np.asarray([1, 4, 6]),
    "block.angle": np.asarray([0.3, -2.0, 5.5]),
    "agent.angle": np.asarray([0.0, 1.0, -2.5]),
    "agent.start_position": np.asarray([[100.0, 100.0], [300.0, 420.0], [60.0, 250.0]]),
    "block.start_position": np.asarray([[150.0, 150.0], [380.0, 300.0], [250.0, 120.0]]),
    "agent.velocity": np.asarray([[0.0, 0.0], [30.0, 5.0], [200.0, 100.0]]),
}
#: ``agent.angle`` turns only a non-circular agent, so it is gated on the Z-shaped agent.
WITH = {"agent.angle": {"agent.shape": np.asarray([3, 3, 3])}}
SAMPLED = tuple(EXPLICIT)
BODY_FIELDS = ("position", "velocity", "angle", "angular_velocity", "center_of_gravity", "mass", "moment")
STEPS = 6


def _swm(name):
    module, cls = CLASSES[name]
    return getattr(importlib.import_module(f"stable_worldmodel.envs.pusht.{module}"), cls)


def _swm_value(value):
    value = np.asarray(value)
    return int(value) if value.dtype.kind in "iu" and not value.shape else value.copy()


def _swm_lanes(name, seed, options):
    envs = []
    for lane in range(3):
        lane_options = dict(options)
        if "variation_values" in options:
            lane_options["variation_values"] = {k: _swm_value(v[lane]) for k, v in options["variation_values"].items()}
        env = _swm(name)()
        env.reset(seed=seed + lane, options=lane_options)
        envs.append(env)
    return envs


def _body(env, name, field):
    value = np.asarray(getattr(getattr(env, name), field), np.float64)
    return np.where(np.isposinf(value), 0.0, value)


def _compare_reset(state, envs):
    for lane, env in enumerate(envs):
        for name in ("agent", "block"):
            for field in BODY_FIELDS:
                ours = state["current", "body", name, field][lane].cpu().numpy()
                np.testing.assert_array_equal(ours, _body(env, name, field), err_msg=f"{name}.{field} lane {lane}")
        np.testing.assert_array_equal(state["current", "observation"][lane].cpu().numpy(), env._get_obs())
        np.testing.assert_array_equal(state["current", "goal_image"][lane].cpu().numpy(), env._goal)


def _frames(ours, envs):
    theirs = np.stack([env.render() for env in envs])
    mse = ((ours.astype(np.float64) - theirs.astype(np.float64)) ** 2).mean(axis=(1, 2, 3))
    return float(mse.max()), int((ours != theirs).any(-1).sum())


def _actions(name, seed):
    rng = np.random.default_rng(seed)
    if name == "PushTDiscrete":
        return rng.integers(0, 16, (STEPS, 3, 2), dtype=np.int64)
    return rng.uniform(-1, 1, (STEPS, 3, 2)).astype(np.float32)


def _run(env, name, seed, options) -> dict:
    envs = _swm_lanes(name, seed, options)
    try:
        state = env.reset(seed=seed, options=options)
        _compare_reset(state, envs)
        frames = [env.render().cpu().numpy()]
        worst, px = _frames(frames[0], envs)
        for action in _actions(name, seed):
            out = env.step(action, frames=None)
            expected = [e.step(a) for e, a in zip(envs, action, strict=True)]
            for lane, (obs, reward, terminated, _, _) in enumerate(expected):
                np.testing.assert_array_equal(out["state"]["current", "observation"][lane].cpu().numpy(), obs["state"])
                assert float(out["reward"][lane]) == reward and bool(out["terminated"][lane]) == terminated
            frames.append(env.render().cpu().numpy())
            mse, moved = _frames(frames[-1], envs)
            worst, px = max(worst, mse), px + moved
    finally:
        for e in envs:
            e.close()
    return {"state_exact": True, "worst_frame_mse": worst, "px_differing": px, "frames": np.stack(frames, axis=1)}


def leaf_gate(name: str, leaf: str, seed: int = 31) -> dict:
    extra = WITH.get(leaf, {})
    explicit = {"variation": [], "variation_values": {**extra, leaf: EXPLICIT[leaf]}}
    baseline = {"variation": [], "variation_values": dict(extra)} if extra else {"variation": []}
    with se.make(name, num_envs=3, device="cpu") as env:
        result = _run(env, name, seed, explicit)
        default = _run(env, name, seed, baseline)
        evidence = {"stable_env": name, "leaf": leaf, "seed": seed, "steps": STEPS}
        moved = (result.pop("frames") != default.pop("frames")).any(-1).reshape(3, -1).sum(1)
        assert (moved[1:] > 0).all(), moved
        evidence["explicit"] = result
        evidence["px_moved_vs_default_per_lane"] = moved.tolist()
        if leaf in SAMPLED:
            sampled = _run(env, name, seed + 1, {"variation": [leaf]})
            sampled.pop("frames")
            evidence["sampled"] = sampled
    for case in ("explicit", "sampled"):
        if case in evidence:
            assert evidence[case]["worst_frame_mse"] <= 1e-3, evidence[case]
    return evidence


#: Per-lane appearance applied mid-episode, then configured shape for the next reset.
APPEARANCE = {
    "agent.color": np.asarray([[10, 200, 30], [250, 250, 0], [0, 0, 0]], np.uint8),
    "background.color": np.asarray([[30, 30, 30], [255, 255, 255], [200, 180, 220]], np.uint8),
    "rendering.render_goal": np.asarray([0, 1, 1]),
    "goal.position": np.asarray([[256.0, 256.0], [120.0, 380.0], [400.0, 90.0]]),
    "goal.angle": np.asarray([0.0, 1.25, -2.0]),
}
CONFIGURED = {"block.shape": np.asarray([3, 5, 7])}


def set_variation_gate(name: str = "PushT", seed: int = 41) -> dict:
    """``set_variation`` mid-episode: appearance now, everything at the next reset, both as SWM draws it.

    SWM has no ``set_variation``; its leaf ``set_value`` is what moves the running frame, and
    ``set_init_value`` is what a later reset restores. SWM draws the goal pose cached at reset, so
    its frame for a moved goal is the one drawn with ``goal_pose`` set to that pose.
    """
    envs = _swm_lanes(name, seed, {"variation": []})
    actions = _actions(name, seed)
    with se.make(name, num_envs=3, device="cpu") as env:
        try:
            env.reset(seed=seed, options={"variation": []})
            for action in actions[:3]:
                env.step(action, frames=None)
                for e, a in zip(envs, action, strict=True):
                    e.step(a)
            before = env.render().cpu().numpy()
            env.set_variation({**APPEARANCE, **CONFIGURED})
            for lane, e in enumerate(envs):
                for path, values in {**APPEARANCE, **CONFIGURED}.items():
                    group, leaf = path.split(".")
                    node = e.variation_space[group][leaf]
                    node.set_init_value(_swm_value(values[lane]))
                    if path in APPEARANCE:
                        node.set_value(_swm_value(values[lane]))
                goal = e.variation_space["goal"]
                e.goal_pose = np.concatenate([goal["position"].value, [goal["angle"].value]])
            now = env.render().cpu().numpy()
            immediate, px = _frames(now, envs)
            assert px == 0 and (now != before).any(-1).reshape(3, -1).sum(1).all(), px
            saved = env.get_state()
            for action in actions[3:]:
                env.step(action, frames=None)
            env.set_state(saved)
            restored, px_restored = _frames(env.render().cpu().numpy(), envs)
            assert px_restored == 0
            moments = saved["current", "body", "block", "moment"].cpu().numpy()
            for e in envs:
                e.close()
            envs = []
            for lane in range(3):
                e = _swm(name)()
                for path, values in {**APPEARANCE, **CONFIGURED}.items():
                    group, leaf = path.split(".")
                    e.variation_space[group][leaf].set_init_value(_swm_value(values[lane]))
                e.reset(seed=seed + 1 + lane, options={"variation": []})
                envs.append(e)
            state = env.reset(seed=seed + 1, options={"variation": []})
            _compare_reset(state, envs)
            assert (state["current", "body", "block", "moment"].cpu().numpy() != moments).all()
            after_reset, px_reset = _frames(env.render().cpu().numpy(), envs)
            assert px_reset == 0
        finally:
            for e in envs:
                e.close()
    return {
        "stable_env": name,
        "immediate_leaves": sorted(APPEARANCE),
        "next_reset_leaves": sorted(CONFIGURED),
        "immediate_frame_mse": immediate,
        "restored_frame_mse": restored,
        "next_reset_frame_mse": after_reset,
    }


def run_gate() -> dict:
    out = {name: [leaf_gate(name, leaf) for leaf in EXPLICIT] for name in CLASSES}
    out["set_variation"] = [set_variation_gate(name) for name in CLASSES]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    text = json.dumps(run_gate(), indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
