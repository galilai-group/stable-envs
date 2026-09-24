"""Isolate reset entropy's effect on the following physical transition."""

import copy
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from probe import OUT, CubeEnv, SceneEnv, differences, encode, engine


def main():
    result = {}
    for name, cls in [("Cube", CubeEnv), ("Scene", SceneEnv)]:
        rows = []
        for seed in [0, 17, 42]:
            a, b = cls(), cls()
            try:
                # Diagnostic conditioning of persistent generators, not public seed parity.
                a.np_random = np.random.default_rng(seed)
                b.np_random = np.random.default_rng(seed)
                legacy = np.random.RandomState(seed).get_state()
                for env in [a, b]:
                    np.random.set_state(copy.deepcopy(legacy))
                    # Source action-space seeding is deliberately NOT patched.
                    env.reset(seed=seed, options={"variation": [], "task_id": 2})
                initial_left, initial_right = engine(a), engine(b)
                row = {
                    "seed": seed,
                    "initial_diff": differences(initial_left, initial_right),
                    "initial_left": initial_left,
                    "initial_right": initial_right,
                    "trace": [],
                }
                for i in range(4):
                    action = np.zeros(5, np.float64)
                    ra, rb = a.step(action), b.step(action)
                    row["trace"].append(
                        {
                            "step": i + 1,
                            "action": action,
                            "engine_diff": differences(engine(a), engine(b)),
                            "left": engine(a),
                            "right": engine(b),
                            "reward_done_success_left": [*ra[1:4], ra[4]["success"]],
                            "reward_done_success_right": [*rb[1:4], rb[4]["success"]],
                        }
                    )
                rows.append(row)
                print(
                    name,
                    seed,
                    "post-reset next-step physics differences",
                    [x["engine_diff"] for x in row["trace"]],
                    flush=True,
                )
            finally:
                a.close()
                b.close()
        result[name] = rows
    result["raw_registration_specs"] = {
        name: {
            "max_episode_steps": gym.spec(name).max_episode_steps,
            "entry_point": gym.spec(name).entry_point,
            "kwargs": gym.spec(name).kwargs,
        }
        for name in ["swm/OGBCube-v0", "swm/OGBScene-v0"]
    }
    result["mujoco_version"] = mujoco.mj_versionString()
    (OUT / "reset_effects.json").write_text(json.dumps(encode(result), indent=2) + "\n")

    # Supplemental dependency inventory: distribution metadata/lock hashes plus
    # runtime source/native library bytes for semantic dependencies and SWM World.
    records = {}
    for package in ["numpy", "scipy", "dm-control", "dm-env", "dm-tree", "lxml", "labmaze"]:
        dist = metadata.distribution(package)
        for item in dist.files or []:
            path = Path(dist.locate_file(item))
            if path.is_file() and path.suffix in {".py", ".so", ".dylib", ".xml"}:
                records[package + "/" + str(item)] = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    world = Path("/Users/samib/stable-worldmodel/stable_worldmodel/world/world.py")
    records["swm/world/world.py"] = {"path": str(world), "sha256": hashlib.sha256(world.read_bytes()).hexdigest()}
    (OUT / "dependency_hashes.json").write_text(json.dumps(records, indent=2) + "\n")
    print("OGB_RESET_EFFECTS_RECORDED_NO_PROMOTION", flush=True)


if __name__ == "__main__":
    main()
