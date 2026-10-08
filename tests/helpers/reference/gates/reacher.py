"""Run the bounded Reacher recovery gate; emit evidence, never benchmark claims."""

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
from tests.helpers.reference.dmc_semantics import ReacherReference
from tests.helpers.reference.semantics import (
    check_reset_matches_reference,
    check_state_roundtrip,
    check_step_matches_reference,
    check_two_histories_same_state,
    compare_state,
)

SWM_PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"


def run_gate(device="cpu"):
    import stable_worldmodel

    source = Path(stable_worldmodel.__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert commit == SWM_PIN, (commit, SWM_PIN)
    assert not subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--"], text=True)
    evidence = {
        "swm_commit": commit,
        "device": device,
        "execution_mode": "DELEGATED_CPU",
        "platform": platform.platform(),
        "versions": {
            k: importlib.metadata.version(k)
            for k in ("torch", "numpy", "dm-control", "mujoco", "tensordict", "stable-worldmodel")
        },
        "cases": [],
    }
    for batch in (1, 3):
        ref = ReacherReference(batch)
        with (
            se.make("DMCReacherHard", num_envs=batch, device=device) as env,
            se.make("DMCReacherHard", num_envs=batch, device=device) as other,
        ):
            try:
                for seed in (0, 17, 42):
                    check_reset_matches_reference(env, ref, seed=seed)
                    saved = env.get_state()
                    actions = np.random.default_rng(seed + 200).uniform(-1, 1, (32, batch, 2))
                    check_state_roundtrip(env, actions=actions[:4])
                    compare_state(env.get_state(), ref.get_state())
                    # First action is the explicit one-step check; following actions cover accumulation.
                    steps = check_step_matches_reference(env, ref, actions)
                    assert (saved == other.reset(seed=seed)).all()
                    other.reset(seed=seed + 991)
                    for action in -actions[:9]:
                        other.step(action, frames=None)
                    snapshot = env.get_state()
                    history = check_two_histories_same_state(env, other, snapshot, actions[:8])
                    frame = env.render(snapshot)
                    assert torch.equal(frame, env._render(env._pack_render_state(snapshot)))
                    # Rendering never advances or mutates the authoritative simulator.
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
                            max_state_error=max(max(x["state"].values()) for x in steps),
                            reward_done_error=0.0,
                        )
                    )
                # Positive reward is checked, not just a trajectory of zero rewards.
                env.reset(seed=18)
                ref.reset(seed=18)
                reward_state = ref.get_state()
                for i, wrapper in enumerate(ref.envs):
                    reward_state["task", "target_pos"][i, :2] = torch.as_tensor(wrapper.info["finger_pos"])
                env.set_state(reward_state)
                ref.set_state(reward_state)
                got, expected = env.step(np.zeros((batch, 2)), frames=None), ref.step(np.zeros((batch, 2)))
                compare_state(got, expected)
                assert (got["reward"] == 2).all()
                evidence["cases"].append(dict(batch=batch, positive_reward=2.0, terminated=False, truncated=False))
                # An underlying LAST does not truncate SWM's hard task. The next substep resets,
                # consumes RNG and changes target geometry. Replay must preserve this exact behavior.
                check_reset_matches_reference(env, ref, seed=5)
                actions = np.zeros((502, batch, 2))
                check_step_matches_reference(env, ref, actions[:499])
                near_limit = env.get_state()
                target_before = near_limit["task", "target_pos"].clone()
                check_step_matches_reference(env, ref, actions[499:])
                assert not torch.equal(target_before, env.get_state()["task", "target_pos"])
                other.reset(seed=77)
                check_two_histories_same_state(env, other, near_limit, actions[:4])
                evidence["cases"].append(
                    dict(
                        batch=batch,
                        time_limit_public_steps=502,
                        target_changes_on_implicit_reset=True,
                        lifecycle_rng_replay=True,
                    )
                )
            finally:
                ref.close()
    # Shared physical Variation must reach the delegated model, then match SWM reset and steps.
    ref = ReacherReference(
        1,
        options={
            "variation_values": {
                "agent.arm_density": np.array([650], np.float32),
                "agent.finger_density": np.array([1350], np.float32),
            }
        },
    )
    with se.make("DMCReacherHard", num_envs=1, device=device) as env:
        try:
            env.set_variation({"agent.arm_density": [650.0], "agent.finger_density": [1350.0]})
            check_reset_matches_reference(env, ref, seed=11)
            actions = np.random.default_rng(11).uniform(-1, 1, (32, 1, 2))
            check_step_matches_reference(env, ref, actions)
            evidence["physical_variation"] = dict(arm_density=650, finger_density=1350, steps=32, max_error=0.0)
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
