"""Semantic gate for DMC Acrobot, Pendulum and Cartpole swing-up against pinned SWM."""

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
from tests.helpers.reference.dmc_semantics import DMCReference as DMCFirstFamilyReference
from tests.helpers.reference.saved import SWM_PIN
from tests.helpers.reference.semantics import (
    assert_lossy_projection_rejected,
    check_reset_matches_reference,
    check_state_roundtrip,
    check_step_matches_reference,
    check_two_histories_same_state,
    compare_state,
)

CASES = {
    "acrobot": "DMCAcrobotBalance",
    "pendulum": "DMCPendulumSwingup",
    "cartpole": "DMCCartpoleSwingup",
}


def _positive_reward_state(state):
    state = state.clone()
    state["qpos"].zero_()
    state["qvel"].zero_()
    state["time"].zero_()
    state["warmstart"].zero_()
    state["lifecycle", "step_count"].zero_()
    state["lifecycle", "reset_next_step"].zero_()
    state["task", "cumulative_reward"].zero_()
    return state


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
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "dm-control", "mujoco", "tensordict", "stable-worldmodel")
        },
        "mappings": CASES,
        "excluded_mapping": {"DMCCartpoleBalance": "the SWM default is sparse swing-up, not dense balance"},
        "cases": [],
    }
    for domain, env_name in CASES.items():
        for batch in (1, 3):
            ref = DMCFirstFamilyReference(domain, batch)
            with (
                se.make(env_name, num_envs=batch, device=device) as env,
                se.make(env_name, num_envs=batch, device=device) as other,
            ):
                try:
                    for seed in (0, 17, 42):
                        check_reset_matches_reference(env, ref, seed=seed)
                        saved = env.get_state()
                        live = env.get_state()
                        saved["qpos"].add_(1)
                        saved["rng", "keys"].add_(1)
                        compare_state(env.get_state(), live)
                        actions = np.random.default_rng(seed + 200).uniform(-1, 1, (32, batch, 1))
                        check_state_roundtrip(env, actions=actions[:4])
                        compare_state(env.get_state(), ref.get_state())
                        steps = check_step_matches_reference(env, ref, actions)

                        other.reset(seed=seed + 991)
                        for action in -actions[:9]:
                            other.step(action, states=None, frames=None).exclude("valid")
                        snapshot = env.get_state()
                        history = check_two_histories_same_state(env, other, snapshot, actions[:8])

                        frame = env.render(snapshot)
                        assert torch.equal(frame, env._render(env._pack_render_state(snapshot)))
                        before = env.get_state()
                        env.render(snapshot)
                        compare_state(env.get_state(), before)
                        assert_lossy_projection_rejected(env, snapshot)
                        evidence["cases"].append(
                            {
                                "domain": domain,
                                "stable_env": env_name,
                                "batch": batch,
                                "seed": seed,
                                "reset": True,
                                "independent_snapshot": True,
                                "roundtrip": True,
                                "one_step": True,
                                "steps": len(steps),
                                "two_history_steps": len(history),
                                "full_state_render": True,
                                "max_state_error": max(max(step["state"].values()) for step in steps),
                                "reward_done_error": 0.0,
                            }
                        )

                    # All three sparse/default tasks have a known reward-1 configuration.
                    check_reset_matches_reference(env, ref, seed=18)
                    reward_state = _positive_reward_state(ref.get_state())
                    env.set_state(reward_state)
                    ref.set_state(reward_state)
                    got, expected = (
                        env.step(np.zeros((batch, 1)), states=None, frames=None).exclude("valid"),
                        ref.step(np.zeros((batch, 1))),
                    )
                    compare_state(got, expected)
                    assert (got["reward"] == 2).all()
                    evidence["cases"].append(
                        {
                            "domain": domain,
                            "batch": batch,
                            "positive_reward": 2.0,
                            "terminated": False,
                            "truncated": False,
                        }
                    )

                    # Derive the boundary from the pinned DMC step limit and repeat=2.
                    check_reset_matches_reference(env, ref, seed=5)
                    public_at_last = ref.step_limit // 2
                    assert ref.step_limit % 2 == 0
                    actions = np.zeros((public_at_last + 2, batch, 1))
                    check_step_matches_reference(env, ref, actions[: public_at_last - 1])
                    near_limit = env.get_state()
                    rng_before = near_limit.select("rng").clone()
                    check_step_matches_reference(env, ref, actions[public_at_last - 1 :])
                    after = env.get_state()
                    assert any(not torch.equal(rng_before[key], after[key]) for key in rng_before.keys(True, True))
                    other.reset(seed=77)
                    check_two_histories_same_state(env, other, near_limit, actions[:4])
                    evidence["cases"].append(
                        {
                            "domain": domain,
                            "batch": batch,
                            "dmc_step_limit": ref.step_limit,
                            "action_repeat": 2,
                            "boundary_public_steps": len(actions),
                            "implicit_reset_rng_changes": True,
                            "lifecycle_rng_replay": True,
                        }
                    )
                finally:
                    ref.close()

        # Rebuilding a float32 style row for a visual-only change must retain
        # the exact float64 nominal physics constants.
        ref = DMCFirstFamilyReference(domain, 1)
        with se.make(env_name, num_envs=1, device=device) as env:
            try:
                env.set_variation({"light.diffuse_scale": [1.1]})
                assert np.array_equal(
                    env._simulator.lanes.envs[0].physics.model.opt.gravity,
                    np.array([0.0, 0.0, -9.81], np.float64),
                )
                check_reset_matches_reference(env, ref, seed=23)
                actions = np.random.default_rng(23).uniform(-1, 1, (8, 1, 1))
                check_step_matches_reference(env, ref, actions)
                evidence.setdefault("visual_only_variation", {})[domain] = {
                    "leaf": "light.diffuse_scale",
                    "nominal_gravity_exact": True,
                    "steps": 8,
                    "max_error": 0.0,
                }
            finally:
                ref.close()

        options = {
            "variation_values": {
                "gravity.x": np.array([1.25], np.float64),
                "gravity.y": np.array([-0.75], np.float64),
                "gravity.z": np.array([-7.5], np.float64),
            }
        }
        ref = DMCFirstFamilyReference(domain, 1, options=options)
        with se.make(env_name, num_envs=1, device=device) as env:
            try:
                env.set_variation({"gravity.x": [1.25], "gravity.y": [-0.75], "gravity.z": [-7.5]})
                check_reset_matches_reference(env, ref, seed=11)
                actions = np.random.default_rng(11).uniform(-1, 1, (32, 1, 1))
                check_step_matches_reference(env, ref, actions)
                evidence.setdefault("physical_variation", {})[domain] = {
                    "gravity": [1.25, -0.75, -7.5],
                    "steps": 32,
                    "max_error": 0.0,
                }
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
