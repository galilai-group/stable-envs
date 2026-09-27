"""Measure whether the current public PushT interfaces admit the required pairing.

This never installs oracle poses into the candidate or implements replay.
"""

import argparse
import hashlib
import inspect
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel.envs  # noqa: F401

import stable_envs as se

from .common import array_hash, catalog, profile, state_hash, unavailable, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for registration in ["swm/PushT-v1", "swm/PushT-Discrete-v1"]:
        row = catalog()[registration]
        env = se.make("PushT", device="cpu", size=224)
        source = gym.make(registration, render_mode="rgb_array", resolution=224)
        try:
            records = []
            for seed in [0, 17, 42]:
                env.reset(seed=seed)
                source.reset(seed=seed)
                raw = source.unwrapped
                ours = env._pack_render_state(env.get_state()).cpu().numpy()[0]
                original = np.asarray([*raw.agent.position, *raw.block.position, raw.block.angle], np.float64)
                records.append(
                    dict(
                        seed=seed,
                        actual_action_prefix=[],
                        candidate_state_hash=state_hash(env.get_state()),
                        candidate_pose=ours.tolist(),
                        oracle_pose=original.tolist(),
                        candidate_pose_hash=array_hash(ours),
                        oracle_pose_hash=array_hash(original),
                        exact_numeric_pose_equal=bool(np.array_equal(ours, original)),
                    )
                )
            assert all(not r["exact_numeric_pose_equal"] for r in records)
            reason = (
                "The current candidate maps to gym_pusht-style absolute-target PushT, not the pinned SWM relative-action reset/step path. "
                "All three identical-seed default resets differ numerically before any action. No accepted exhaustive SWM State-equivalence witness exists; "
                "installing source poses would not establish matching hidden contact/RNG/task continuation."
            )
            if "Discrete" in registration:
                reason += " SWM additionally uses a 16x16 PolarQuantizer MultiDiscrete action API; the current candidate has only continuous absolute targets."
            paths = [Path(inspect.getfile(type(env))), Path(inspect.getfile(type(raw)))]
            value = dict(
                row,
                measurement_status="NONCOMPARABLE",
                profile=profile(),
                records=records,
                source_action_space=repr(source.action_space),
                candidate_action_spec=repr(env.action_spec),
                same_resolution=224,
                source_relative=raw.relative,
                no_pose_teleport=True,
                no_new_replay=True,
                cells={b: unavailable(reason, "NONCOMPARABLE") for b in ["cpu", "gpu", "original"]},
                source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            )
            write(args.output / (registration.split("/")[1] + ".json"), value)
        finally:
            env.close()
            source.close()


if __name__ == "__main__":
    main()
