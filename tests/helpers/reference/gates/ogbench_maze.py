"""OGBMaze gates against the official ogbench env and pinned swm; emit evidence only."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from tests.helpers.reference.ogbench_maze_semantics import OGBMazeReference
from tests.helpers.reference.saved import SWM_PIN
from tests.helpers.reference.semantics import (
    assert_lossy_projection_rejected,
    check_reset_matches_reference,
    check_state_roundtrip,
    check_step_matches_reference,
    check_two_histories_same_state,
    compare_state,
)


def _np_state_equal(first, second):
    return first[0] == second[0] and np.array_equal(first[1], second[1]) and first[2:] == second[2:]


def _plain_step(env, action):
    return env.step(action, states=None, frames=None).exclude("valid")


def _xy(state):
    return state["root", "position"][:, :2].cpu().numpy()


def cell_gate(cell, *, batch=2, seeds=(0, 17), steps=50, source="ogbench"):
    """One registered maze cell against the official ogbench env with the same seeds and actions.

    Covers seeded reset, a random-action rollout (State, reward, flags, observation, info),
    a round trip through ``get_state``/``set_state`` into the same and a fresh env, and for
    teleport mazes a step from every inbound teleport, which draws the outbound one from the
    legacy global RNG that State carries.
    """
    cls = se.registry.REGISTRY[cell]
    evidence = {"cell": cell, "source": source, "batch": batch, "cases": []}
    ref = OGBMazeReference(batch, loco=cls.LOCO, maze=cls.MAZE, source=source)
    try:
        with se.make(cell, num_envs=batch, device="cpu") as env:
            outside = np.random.get_state()
            for seed in seeds:
                check_reset_matches_reference(env, ref, seed=seed)
                actions = np.random.default_rng(seed + 200).uniform(-1, 1, (steps, batch, cls.action_dim))
                actions = actions.astype(np.float32)
                rollout = check_step_matches_reference(env, ref, actions)
                snapshot = env.get_state()
                with se.make(cell, num_envs=batch, device="cpu", seed=seed + 991) as fresh:
                    fresh.reset(seed=seed + 991)
                    check_state_roundtrip(env, actions=actions[:3])
                    history = check_two_histories_same_state(env, fresh, snapshot, -actions[:4])
                    fresh.set_state(snapshot)
                    compare_state(fresh.get_state(), snapshot)
                assert _np_state_equal(outside, np.random.get_state())
                evidence["cases"].append(
                    dict(seed=seed, reset=True, steps=len(rollout), roundtrip=True, fresh_env_steps=len(history))
                )
            if cls.MAZE == "teleport":
                inbound = ref.envs[0].env._teleport_info["teleport_in_xys"]
                outbound = np.asarray(ref.envs[0].env._teleport_info["teleport_out_xys"])
                for k, xy in enumerate(inbound):
                    check_reset_matches_reference(env, ref, seed=40 + k)
                    state = env.get_state()
                    state["root", "position"][:, :2] = torch.as_tensor(xy, dtype=torch.float64)
                    env.set_state(state)
                    ref.set_state(state)
                    zero = np.zeros((batch, cls.action_dim), np.float32)
                    got = check_step_matches_reference(env, ref, zero[None])
                    landed = _xy(env.get_state())
                    nearest = np.linalg.norm(landed[:, None] - outbound[None], axis=-1).min(1)
                    assert (nearest < 1.5).all(), (landed, outbound)
                    evidence["cases"].append(dict(teleport_in=list(xy), landed=landed.tolist(), steps=len(got)))
    finally:
        ref.close()
    return evidence


def run_gate(device="cpu"):
    import stable_worldmodel

    source = Path(stable_worldmodel.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert commit == SWM_PIN, (commit, SWM_PIN)
    assert not subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"], text=True)
    evidence = {
        "historical_id": "swm/OGBMaze-v0",
        "candidate_class": "PointMazeLargeNavigate",
        "historical_defaults": {
            "loco_env_type": "point",
            "maze_env_type": "maze",
            "maze_type": "large",
            "ob_type": "states",
            "width": 224,
            "height": 224,
        },
        "swm_commit": commit,
        "swm_source": str(source),
        "device": device,
        "execution_mode": "DELEGATED_CPU",
        "platform": platform.platform(),
        "versions": {
            k: importlib.metadata.version(k)
            for k in ("torch", "numpy", "gymnasium", "ogbench", "mujoco", "tensordict", "stable-worldmodel")
        },
        "scope_limit": "Exact empty-kwargs default semantics only; nondefault reset options/tasks and the full SWM appearance mapping are not covered.",
        "cases": [],
    }
    for batch in (1, 3):
        ref = OGBMazeReference(batch)
        with (
            se.make("PointMazeLargeNavigate", num_envs=batch, device=device, **EXACT) as env,
            se.make("PointMazeLargeNavigate", num_envs=batch, device=device, **EXACT) as other,
        ):
            try:
                outside = np.random.get_state()
                for seed in (0, 17, 42):
                    check_reset_matches_reference(env, ref, seed=seed)
                    assert _np_state_equal(outside, np.random.get_state())
                    actions = np.random.default_rng(seed + 200).uniform(-1, 1, (32, batch, 2)).astype(np.float32)
                    check_state_roundtrip(env, actions=actions[:4])
                    compare_state(env.get_state(), ref.get_state())
                    steps = check_step_matches_reference(env, ref, actions)
                    other.reset(seed=seed + 991)
                    for action in -actions[:9]:
                        other.step(action, frames=None)
                    snapshot = env.get_state()
                    frozen_snapshot = snapshot.clone()
                    env.step(actions[0], frames=None)
                    compare_state(snapshot, frozen_snapshot)
                    history = check_two_histories_same_state(env, other, snapshot, actions[:8])
                    # The goal is State: the adapter keeps it in its host columns and nowhere else.
                    np.testing.assert_array_equal(env._simulator._goal, snapshot["task", "goal_pos"].numpy())
                    frame = env.render(snapshot)
                    assert torch.equal(frame, env._render(env._pack_render_state(snapshot)))
                    before = env.get_state()
                    env.render(snapshot)
                    compare_state(env.get_state(), before)
                    assert_lossy_projection_rejected(env, snapshot)
                    assert _np_state_equal(outside, np.random.get_state())
                    evidence["cases"].append(
                        dict(
                            batch=batch,
                            seed=seed,
                            reset=True,
                            roundtrip=True,
                            one_step=True,
                            steps=len(steps),
                            two_history_steps=len(history),
                            full_state_render=True,
                            target_geom_restored=True,
                            legacy_rng_lane_isolated=True,
                            max_state_error=max(max(x["state"].values()) for x in steps),
                            reward_done_observation_info_error=0.0,
                        )
                    )
                # OGBench samples five warm-up actions while making a discarded goal observation, so
                # different action-space RNG histories must give the same continuation once the env
                # and global RNG State is restored.
                check_reset_matches_reference(env, ref, seed=31)
                snapshot = ref.get_state()
                other.reset(seed=777)
                for action in np.random.default_rng(777).uniform(-1, 1, (7, batch, 2)).astype(np.float32):
                    other.step(action, frames=None)
                env.set_state(snapshot)
                ref.set_state(snapshot)
                other.set_state(snapshot)
                got = env.reset(seed=None)
                expected = ref.reset(seed=None)
                alternate = other.reset(seed=None)
                compare_state(got, expected)
                compare_state(got, alternate)
                assert _np_state_equal(outside, np.random.get_state())
                evidence["cases"].append(
                    dict(
                        batch=batch,
                        unseeded_reset_replay=True,
                        warmup_action_rng_not_continuation_state=True,
                        env_and_legacy_rng_restored=True,
                    )
                )
                # The SWM registration exposes appearance leaves but no physical ones. A persistent
                # nondefault appearance still renders from the full State and leaves the physics untouched.
                env.set_variation({"agent.color": [0.2, 0.8, 0.4]})
                varied_state = env.get_state()
                varied_before = varied_state.clone()
                assert torch.equal(env.render(varied_state), env._render(env._pack_render_state(varied_state)))
                compare_state(env.get_state(), varied_before)
                compare_state(varied_state, varied_before)
                evidence["cases"].append(
                    dict(
                        batch=batch,
                        persistent_visual_variation="agent.color=[0.2,0.8,0.4]",
                        full_state_render=True,
                        simulator_unmutated=True,
                    )
                )
                # Positive reward and the default post-step success/termination boundary.
                check_reset_matches_reference(env, ref, seed=18)
                success_state = ref.get_state()
                success_state["root", "position"] = success_state["task", "goal_pos"][:, :2].clone()
                success_state["root", "velocity"].zero_()
                env.set_state(success_state)
                ref.set_state(success_state)
                got, expected = (
                    _plain_step(env, np.zeros((batch, 2), np.float32)),
                    ref.step(np.zeros((batch, 2), np.float32)),
                )
                compare_state(got, expected)
                assert (got["reward"] == 1).all() and got["success"].bool().all()
                assert got["terminated"].all() and not got["truncated"].any()
                # Termination is a returned boundary, not an implicit reset or a step prohibition.
                got, expected = (
                    _plain_step(env, np.full((batch, 2), 0.25, np.float32)),
                    ref.step(np.full((batch, 2), 0.25, np.float32)),
                )
                compare_state(got, expected)
                evidence["cases"].append(
                    dict(
                        batch=batch,
                        positive_reward=1.0,
                        success=1.0,
                        terminated=True,
                        truncated=False,
                        post_termination_step_matches=True,
                    )
                )

                # The raw OGBench env has no TimeLimit: parity continues past 1000 steps without truncation.
                check_reset_matches_reference(env, ref, seed=5)
                zeros = np.zeros((batch, 2), np.float32)
                for _ in range(1001):
                    got, expected = _plain_step(env, zeros), ref.step(zeros)
                    compare_state(got, expected)
                    assert not got["terminated"].any() and not got["truncated"].any()
                elapsed = env.get_state()["simulator", "time"]
                torch.testing.assert_close(elapsed, torch.full_like(elapsed, 100.1), atol=1e-10, rtol=0)
                evidence["cases"].append(
                    dict(
                        batch=batch,
                        no_registered_horizon_steps=1001,
                        terminated=False,
                        truncated=False,
                        time=float(elapsed[0, 0]),
                    )
                )
            finally:
                ref.close()
    return evidence


#: Per-lane explicit values (three lanes) for each swm/OGBMaze-v0 appearance leaf.
#: The gate compares State bit for bit, so CUDA envs step on the exact CPU physics, not MuJoCo Warp.
EXACT = {"simulation_backend": "native"}

APPEARANCE_CASES = {
    "floor.color": [
        [[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]],
        [[0.5, 0.5, 0.5], [0.2, 0.7, 0.2]],
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    ],
    "light.intensity": [[0.0], [0.5], [0.25]],
    "agent.color": [[0.1, 0.9, 0.1], [0.9, 0.1, 0.9], [0.0, 0.0, 0.0]],
    "wall.color": [[0.9, 0.2, 0.2], [0.0, 0.0, 0.0], [0.3, 0.6, 0.9]],
}
STATE_CASES = {"default": {}, **{k: {k: v} for k, v in APPEARANCE_CASES.items()}, "all": APPEARANCE_CASES}
#: Two lanes each: a swm pixels env holds two GL contexts once its recompile rebuilds the renderer.
PIXEL_CASES = {
    "gradient": {"floor.pixel_encoding": [1, 1]},
    "checker_colour": {"floor.pixel_encoding": [0, 0], "floor.color": APPEARANCE_CASES["floor.color"][:2]},
    "light": {"floor.pixel_encoding": [1, 1], "light.intensity": APPEARANCE_CASES["light.intensity"][1:]},
    "all": {"floor.pixel_encoding": [0, 1], **{k: v[:2] for k, v in APPEARANCE_CASES.items()}},
}
#: Worst-frame normalized MSE accepted between our first frame and swm's.
FRAME_MSE_MAX = 1e-3
_OURS = {"wall.color": "walls.color"}
#: Every swm env and renderer a gate builds stays referenced: releasing a GL context corrupts the
#: other live contexts in the process.
_LIVE = []


def _ours(values, lanes):
    return {_OURS.get(k, k): np.asarray(v, np.float32).reshape(lanes, -1) for k, v in values.items()}


def _mse(ours, theirs):
    from stable_envs.testing.metrics import normalized_mse

    errors = normalized_mse(np.asarray(ours), np.asarray(theirs))
    return dict(mean=float(errors.mean()), worst=float(errors.max()), least=float(errors.min()))


def _check_frames(case, frames, shift):
    """Within the bar, and every varied lane moved swm's frame off our default by more than our residual."""
    assert frames["worst"] <= FRAME_MSE_MAX, (case, frames)
    assert shift is None or shift["least"] > frames["worst"], (case, frames, shift)


def appearance_gate(case, cell="PointMazeLargeNavigate", *, seed=23, steps=20, device="cpu"):
    """One swm/OGBMaze-v0 appearance case against swm in the state scene, three lanes.

    swm recompiles the MJCF, so the same seed and per-lane ``variation_values`` must give the exact
    reset State and rollout, and the first frame must agree. swm's states-mode renderer is built on
    first render and never rebuilt, so every case uses fresh swm envs.
    """
    cls = se.registry.REGISTRY[cell]
    values = STATE_CASES[case]
    options = {"variation_values": values} if values else None
    with se.make(cell, num_envs=3, device=device, **EXACT) as env:
        env.reset(seed=seed)
        default = env.render().cpu().numpy()
        ref = OGBMazeReference(3, loco=cls.LOCO, maze=cls.MAZE, source="swm", options=options)
        _LIVE.append(ref)
        expected = ref.reset(seed=seed)
        theirs = np.stack([np.asarray(w.render()) for w in ref.envs])
        got = env.reset(seed=seed, options={"variation_values": _ours(values, 3)} if values else None)
        compare_state(got, expected)
        frames = _mse(env.render().cpu().numpy(), theirs)
        actions = np.random.default_rng(seed).uniform(-1, 1, (steps, 3, cls.action_dim)).astype(np.float32)
        rollout = check_step_matches_reference(env, ref, actions)
    shift = _mse(theirs, default) if values else None
    _check_frames(case, frames, shift)
    return dict(
        cell=cell,
        case=case,
        seed=seed,
        state_exact=True,
        rollout_steps=len(rollout),
        first_frame_mse=frames,
        swm_shift_mse=shift,
    )


def pixel_appearance_gate(cell, case, *, seed=23, device="cpu"):
    """One appearance case in a pixel scene against swm's ``ob_type='pixels'`` env at its qpos.

    swm's own default reset applies ``floor.pixel_encoding=0`` (its init value), so every case
    passes the encoding explicitly.
    """
    from stable_worldmodel.envs.ogbench.maze_env import MazeEnv

    cls = se.registry.REGISTRY[cell]
    values = PIXEL_CASES[case]
    theirs, qpos = [], []
    for lane in range(2):
        swm = MazeEnv(loco_env_type=cls.LOCO, maze_type=cls.MAZE, ob_type="pixels", width=224, height=224)
        _LIVE.extend([swm, swm.env.custom_renderer])
        lane_values = {k: np.asarray(v)[lane] for k, v in values.items()}
        lane_values["floor.pixel_encoding"] = int(lane_values["floor.pixel_encoding"])
        outside = np.random.get_state()
        np.random.seed(seed + lane)  # ogbench draws the task from the global RNG
        try:
            swm.reset(seed=seed + lane, options={"variation_values": lane_values})
        finally:
            np.random.set_state(outside)
        _LIVE.append(swm.env.custom_renderer)
        qpos.append(swm.env.data.qpos.copy())
        theirs.append(np.asarray(swm.render()))
    theirs = np.stack(theirs)
    rows = torch.as_tensor(np.stack(qpos), dtype=torch.float32)
    with se.make(cell, num_envs=2, device=device, **EXACT) as env:
        default = env._render(rows).cpu().numpy()
        env.reset(seed=seed, options={"variation_values": _ours(values, 2)})
        frames = _mse(env._render(rows).cpu().numpy(), theirs)
    shift = None if case == "gradient" else _mse(theirs, default)
    _check_frames(case, frames, shift)
    return dict(cell=cell, case=case, seed=seed, first_frame_mse=frames, swm_shift_mse=shift)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_gate(args.device)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
