"""Run the bounded historical-default OGBMaze recovery gate; emit evidence only."""

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
from tests.helpers.reference.semantics import (
    check_reset_matches_reference,
    check_state_roundtrip,
    check_step_matches_reference,
    check_two_histories_same_state,
    compare_state,
)

SWM_PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"


def _np_state_equal(first, second):
    return first[0] == second[0] and np.array_equal(first[1], second[1]) and first[2:] == second[2:]


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
        "scope_limit": "Exact empty-kwargs default semantics only; nondefault reset options/tasks and complete historical appearance mapping are uncertified.",
        "cases": [],
    }
    for batch in (1, 3):
        ref = OGBMazeReference(batch)
        with (
            se.make("PointMazeLargeNavigate", num_envs=batch, device=device) as env,
            se.make("PointMazeLargeNavigate", num_envs=batch, device=device) as other,
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
                        other.step(action)
                    snapshot = env.get_state()
                    frozen_snapshot = snapshot.clone()
                    env.step(actions[0])
                    compare_state(snapshot, frozen_snapshot)
                    history = check_two_histories_same_state(env, other, snapshot, actions[:8])
                    # Goal and target geom are both restored; OGBench's vector goal setter updates
                    # only the first two model coordinates, so the complete target is explicit State.
                    for simulator, row in zip(env._simulator.envs, snapshot.cpu().unbind(0), strict=True):
                        np.testing.assert_array_equal(
                            simulator.model.geom("target").pos, row["task", "goal_pos"].numpy()
                        )
                        np.testing.assert_array_equal(simulator.cur_goal_xy, row["task", "goal_pos"][:2].numpy())
                    frame = env.render(snapshot)
                    assert torch.equal(frame, env._render(env._pack_render_state(snapshot)))
                    before = env.get_state()
                    env.render(snapshot)
                    compare_state(env.get_state(), before)
                    for bad in (
                        env._pack_render_state(snapshot),
                        env._unpack_render_state(env._pack_render_state(snapshot)),
                    ):
                        try:
                            env.set_state(bad)
                        except (TypeError, ValueError):
                            pass
                        else:
                            raise AssertionError("Lossy renderer projection was accepted")
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
                # OGBench samples five warm-up actions while making a discarded goal
                # observation. Different action-space RNG histories must therefore still produce
                # the same continuation after the complete env/global RNG snapshot is restored.
                check_reset_matches_reference(env, ref, seed=31)
                snapshot = ref.get_state()
                other.reset(seed=777)
                for action in np.random.default_rng(777).uniform(-1, 1, (7, batch, 2)).astype(np.float32):
                    other.step(action)
                for first, second in zip(env._simulator.envs, other._simulator.envs, strict=True):
                    assert (
                        first.action_space.np_random.bit_generator.state
                        != second.action_space.np_random.bit_generator.state
                    )
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
                # The historical registration exposes visual knobs but no physical default knob.
                # Prove that a persistent nondefault appearance still consumes the full State via
                # the preserved private qpos projection and cannot mutate delegated physics.
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
                got, expected = env.step(np.zeros((batch, 2), np.float32)), ref.step(np.zeros((batch, 2), np.float32))
                compare_state(got, expected)
                assert (got["reward"] == 1).all() and got["success"].bool().all()
                assert got["terminated"].all() and not got["truncated"].any()
                # Termination is a returned boundary, not an implicit reset or a step prohibition.
                got, expected = (
                    env.step(np.full((batch, 2), 0.25, np.float32)),
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

                # The wrapped raw OGBench env has no TimeLimit. Prove continued parity past the
                # conventional 1000-step boundary rather than inventing a truncation lifecycle.
                check_reset_matches_reference(env, ref, seed=5)
                zeros = np.zeros((batch, 2), np.float32)
                for _ in range(1001):
                    got, expected = env.step(zeros), ref.step(zeros)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_gate(args.device)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
