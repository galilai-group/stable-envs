"""Fetch reset variation leaves against pinned stable_worldmodel.

Each lane of one of our envs is reset with the same seed and options as its own
``swm/Fetch*-v3`` env. The full State must match exactly, the appearance our style
carries and the object mass must equal what the wrapper wrote into its MuJoCo model,
and a short rollout must stay exact. With ``frames=True`` the first frames are
compared as MSE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

import stable_envs as se
from tests.helpers.reference.fetch_compare import full_compare

EXPLICIT = {
    "background.color": np.asarray([[0.15, 0.62, 0.81], [0.93, 0.31, 0.12], [0.42, 0.42, 0.05]]),
    "rendering.transparent_arm": np.asarray([1, 0, 1]),
    "light.intensity": np.asarray([[0.35], [0.9], [0.62]]),
    "agent.start_position": np.asarray([[1.30, 0.70], [1.40, 0.85], [1.27, 0.62]]),
    "goal.start_position": np.asarray([[1.2, 0.7, 0.4247], [1.4, 0.8, 0.4247], [1.3, 0.65, 0.4247]]),
}
BLOCK = {
    "block.start_position": np.asarray([[1.2, 0.65], [1.35, 0.85], [1.42, 0.7]]),
    "block.angle": np.asarray([[0.3], [-1.2], [2.5]]),
    "block.mass": np.asarray([[0.5], [2.0], [12.0]]),
}


def leaves(scene) -> dict:
    return {**EXPLICIT, **(BLOCK if scene != "Reach" else {})}


def _swm(scene, size):
    import stable_worldmodel  # noqa: F401

    return gym.make(f"swm/Fetch{scene}-v3", render_mode="rgb_array", resolution=size, width=size, height=size)


def _wrapper(env):
    while not hasattr(env, "variation_space"):
        env = env.env
    return env


def _swm_appearance(env) -> dict:
    """What the wrapper wrote into its model, in our style units."""
    import mujoco

    w, m = _wrapper(env), env.unwrapped.model
    tex = w._skybox_tex_id
    adr, nc = int(m.tex_adr[tex]), int(m.tex_nchannel[tex])
    sky = m.tex_data[adr : adr + int(m.tex_width[tex]) * int(m.tex_height[tex]) * nc].reshape(-1, nc)[:, :3]
    assert (sky == sky[0]).all(), "swm fills the skybox flat"
    robot = [g for g in range(m.ngeom) if "robot0:" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    out = {
        "floor.color": m.geom_rgba[w._floor_geoms[0], :3].copy(),
        "table.color": m.geom_rgba[w._table_geoms[0], :3].copy(),
        "sky_texel": sky[0].copy(),
        "light.diffuse": m.light_diffuse[0, :1] / 0.8,
        "rendering.transparent_arm": float(np.all(m.geom_rgba[robot, 3] == 0.3)),
    }
    if w._object_geoms:
        out["object.color"] = m.geom_rgba[w._object_geoms[0], :3].copy()
    return out


def _check_appearance(env, refs) -> None:
    if env._has_object:
        mass = [ref.unwrapped.model.body_mass[_wrapper(ref)._object_body_id] for ref in refs]
        assert np.array_equal(env._simulator.object_mass, mass), (env._simulator.object_mass, mass)
    for lane, ref in enumerate(refs):
        want = _swm_appearance(ref)
        sky = env._style["sky.color"][lane].cpu().numpy().astype(np.float64)
        assert np.array_equal((sky * 255.0).astype(np.uint8), want.pop("sky_texel")), (lane, sky)
        for path, value in want.items():
            got = env._style[path][lane].cpu().numpy().reshape(-1)
            np.testing.assert_allclose(got, np.asarray(value, np.float64).reshape(-1), rtol=0, atol=1e-6, err_msg=path)


def _mse(ours, theirs) -> list[float]:
    d = (np.asarray(ours, np.float64) - np.asarray(theirs, np.float64)) / 255.0
    return (d**2).mean(axis=(1, 2, 3)).tolist()


def _case(scene, env, seed, our_options, lane_options, steps, frames, size) -> dict:
    refs = [_swm(scene, size) for _ in lane_options]
    try:
        for lane, (ref, options) in enumerate(zip(refs, lane_options)):
            ref.reset(seed=seed + lane, options=options)
        state = env.reset(seed=seed, options=our_options)
        full_compare(state.cpu(), refs)
        _check_appearance(env, refs)
        out = {"state_exact": True, "appearance_exact": True}
        if frames:
            ours = env.render().cpu().numpy()
            theirs = np.stack([np.asarray(ref.render()) for ref in refs])
            out["mse"] = _mse(ours, theirs)
            out["frames"] = (ours, theirs)
        actions = np.random.default_rng(seed).uniform(-1, 1, (steps, len(refs), 4)).astype(np.float32)
        for action in actions:
            state = env.step(action, frames=None)["state"]
            for ref, a in zip(refs, action):
                ref.step(a)
            full_compare(state.cpu(), refs)
        out["rollout_steps"] = steps
        return out
    finally:
        for ref in refs:
            ref.close()


def fetch_variation_gate(scene="Push", seed=31, steps=10, frames=False, size=480, device="cpu") -> dict:
    """Explicit per-lane values, then seeded sampling, then a plain reset back to the defaults."""
    lanes = 3
    explicit = leaves(scene)
    sampled = list(explicit) + ["table.color", "camera.angle_delta"] + (["object.color"] if scene != "Reach" else [])
    evidence = {"scene": scene, "seed": seed, "leaves": sorted(sampled)}
    with se.make(f"Fetch{scene}", num_envs=lanes, device=device) as env:
        per_lane = [
            {"variation": [], "variation_values": {k: v[i] for k, v in explicit.items()}} for i in range(lanes)
        ]
        evidence["explicit"] = _case(
            scene, env, seed, {"variation": [], "variation_values": explicit}, per_lane, steps, frames, size
        )
        assert env._style["rendering.transparent_arm"].reshape(-1).tolist() == [1.0, 0.0, 1.0]
        options = {"variation": sampled}
        evidence["sampled"] = _case(scene, env, seed + 7, options, [options] * lanes, steps, frames, size)
        drawn = env._style["rendering.transparent_arm"].reshape(-1).tolist()
        evidence["sampled"]["transparent_arm"] = drawn

        state = env.reset(seed=seed + 11)
        assert np.array_equal(env.get_style().cpu().numpy(), env.default_style().reshape(1, -1).repeat(lanes, 0)), (
            "a plain reset restores every variation default"
        )
        refs = [_swm(scene, size) for _ in range(lanes)]
        try:
            for lane, ref in enumerate(refs):
                ref.reset(seed=seed + 11 + lane, options={"variation": []})
            full_compare(state.cpu(), refs)
        finally:
            for ref in refs:
                ref.close()
    return evidence


def run_gate(frames=False, device="cpu") -> dict:
    out = {}
    for scene in ("Reach", "Push", "Slide", "PickAndPlace"):
        ev = fetch_variation_gate(scene, frames=frames, device=device)
        for case in ("explicit", "sampled"):
            ev[case].pop("frames", None)
        out[scene] = ev
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--frames", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    text = json.dumps(run_gate(frames=args.frames, device=args.device), indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
