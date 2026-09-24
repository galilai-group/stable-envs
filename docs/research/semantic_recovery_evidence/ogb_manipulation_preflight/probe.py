"""Read-only source preflight. Never installs a replacement public adapter.

Run with MUJOCO_GL=disable, SDL_VIDEODRIVER=dummy and the pinned OGB profile.
Expected failures are recorded, not relabelled semantic acceptance.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import traceback
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import ogbench
import stable_worldmodel
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
from stable_worldmodel.envs.ogbench.scene_env import SceneEnv

from stable_envs.envs.ogbench.manipulation import _ogbench_make

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
PARTS = (
    "TIME",
    "QPOS",
    "QVEL",
    "ACT",
    "HISTORY",
    "WARMSTART",
    "CTRL",
    "QFRC_APPLIED",
    "XFRC_APPLIED",
    "EQ_ACTIVE",
    "MOCAP_POS",
    "MOCAP_QUAT",
    "USERDATA",
    "PLUGIN",
)
PYTHON_FIELDS = (
    "cur_task_id",
    "cur_task_info",
    "_reward_task_id",
    "_success",
    "_target_task",
    "_target_block",
    "_target_button",
    "_target_button_states",
    "_target_drawer_pos",
    "_target_window_pos",
    "_cur_button_states",
    "_prev_button_states",
    "_cur_goal_ob",
    "_cur_goal_rendered",
    "_render_goal",
    "_prev_qpos",
    "_prev_qvel",
    "_prev_ob_info",
    "_reset_next_step",
    "_dirty",
    "_never_compiled",
    "_np_random_seed",
)
CACHE_FIELDS = ("site_xpos", "site_xmat", "cfrc_ext")
MODEL_FIELDS = ("dof_damping", "site_pos", "geom_rgba", "mat_rgba")


def encode(value):
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def engine(env):
    result = {}
    for part in PARTS:
        flag = getattr(mujoco.mjtState, "mjSTATE_" + part)
        value = np.empty(mujoco.mj_stateSize(env.model, flag), dtype=np.float64)
        mujoco.mj_getState(env.model, env.data, value, flag)
        result[part] = value
    return result


def snapshot(env):
    return {
        "engine": engine(env),
        "cache": {k: getattr(env.data, k).copy() for k in CACHE_FIELDS},
        "model": {k: getattr(env.model, k).copy() for k in MODEL_FIELDS},
        "python": {k: copy.deepcopy(getattr(env, k)) for k in PYTHON_FIELDS if hasattr(env, k)},
        "rng": copy.deepcopy(env.np_random.bit_generator.state),
        "global_rng": copy.deepcopy(np.random.get_state()),
    }


def restore(env, state):
    # Bounded trusted-source diagnostic, NOT a production input validator.
    for k, v in state["python"].items():
        setattr(env, k, copy.deepcopy(v))
    for k, v in state["model"].items():
        getattr(env.model, k)[:] = v
    for _ in range(2):
        for part, value in state["engine"].items():
            mujoco.mj_setState(env.model, env.data, value, getattr(mujoco.mjtState, "mjSTATE_" + part))
        if _ == 0:
            mujoco.mj_forward(env.model, env.data)
    for k, v in state["cache"].items():
        getattr(env.data, k)[:] = v
    env.np_random.bit_generator.state = copy.deepcopy(state["rng"])
    np.random.set_state(copy.deepcopy(state["global_rng"]))


def differences(left, right, prefix=""):
    result = {}
    if isinstance(left, dict):
        for key in left:
            result.update(differences(left[key], right[key], prefix + "/" + str(key)))
    elif isinstance(left, (tuple, list)):
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            result.update(differences(a, b, prefix + "/" + str(i)))
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            result[prefix] = {
                "count": int(np.count_nonzero(left != right)),
                "max_abs": float(np.max(np.abs(left.astype(float) - np.asarray(right).astype(float)))),
            }
    elif left != right:
        result[prefix] = {"left": encode(left), "right": encode(right)}
    return result


def reset_pair(cls, seed, task, condition_persistent_rng=False):
    a, b = cls(), cls()
    try:
        if condition_persistent_rng:
            # An explicitly labelled diagnostic; does NOT repair public seed handling.
            a.np_random = np.random.default_rng(seed)
            b.np_random = np.random.default_rng(seed)
        legacy = np.random.RandomState(seed).get_state()
        np.random.set_state(legacy)
        a.reset(seed=seed, options={"variation": [], "task_id": task})
        np.random.set_state(legacy)
        b.reset(seed=seed, options={"variation": [], "task_id": task})
        sa, sb = snapshot(a), snapshot(b)
        return {
            "seed": seed,
            "task_id": task,
            "condition_persistent_rng": condition_persistent_rng,
            "diff": differences(sa, sb),
            "left": sa,
            "right": sb,
            "fresh_action_space_objects": a.action_space is not a.action_space,
        }
    finally:
        a.close()
        b.close()


def contacts(env):
    return sorted(
        {
            (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)) or str(c.geom1))
            + " / "
            + (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)) or str(c.geom2))
            for c in env.data.contact
        }
    )


def continuation(cls, seed):
    a, b = cls(), cls()
    try:
        a.reset(seed=seed, options={"variation": [], "task_id": 2})
        b.reset(seed=seed + 991, options={"variation": [], "task_id": 4})
        contact_history = []
        for i in range(16):
            # Source controllers only: descend beside the cube, then close the gripper.
            target = a.data.joint("object_joint_0").qpos[:3].copy()
            target[2] = 0.035
            action = np.zeros(5, np.float64)
            action[:3] = np.clip((target - a.data.site_xpos[a._pinch_site_id]) / 0.05, -1, 1)
            action[3] = 0.2 if i % 2 else -0.2
            action[4] = 1.0 if i >= 9 else -1.0
            a.step(action)
            b.step(np.array([-0.4, 0.5, 0.7, -0.8, -0.5], np.float32))
            contact_history.append(contacts(a))
        saved = snapshot(a)
        different_before = differences(saved, snapshot(b))
        restore(b, saved)
        roundtrip = differences(saved, snapshot(b))
        trace = []
        for i in range(8):
            action = np.array([0.2, -0.3, -0.1, 0.35, 0.4], np.float32 if i % 2 else np.float64)
            np.random.set_state(saved["global_rng"])
            ra = a.step(action)
            np.random.set_state(saved["global_rng"])
            rb = b.step(action)
            trace.append(
                {
                    "step": i + 1,
                    "action": action,
                    "state_diff": differences(snapshot(a), snapshot(b)),
                    "output_diff": differences(ra, rb),
                    "contacts": contacts(a),
                }
            )
        # Same saved RNG and engine/task state; an explicit unseeded-reset witness.
        restore(a, saved)
        a.reset(options={"variation": [], "task_id": 2})
        reset_left = snapshot(a)
        restore(b, saved)
        b.reset(options={"variation": [], "task_id": 2})
        reset_right = snapshot(b)
        return {
            "seed": seed,
            "different_before": different_before,
            "roundtrip_diff": roundtrip,
            "contact_history": contact_history,
            "trace": trace,
            "saved": saved,
            "unseeded_reset_diff": differences(reset_left, reset_right),
            "unseeded_reset_left": reset_left,
            "unseeded_reset_right": reset_right,
        }
    finally:
        a.close()
        b.close()


def provenance():
    packages = [
        "ogbench",
        "mujoco",
        "gymnasium",
        "numpy",
        "dm-control",
        "dm-env",
        "dm-tree",
        "scipy",
        "lxml",
        "labmaze",
        "torch",
        "tensordict",
        "pygame",
        "pymunk",
        "opencv-python",
        "shapely",
    ]
    versions = {p: metadata.version(p) for p in packages}
    roots = {
        "swm": Path(stable_worldmodel.__file__).resolve().parent,
        "ogbench": Path(ogbench.__file__).resolve().parent,
        "gymnasium": Path(gym.__file__).resolve().parent,
        "mujoco": Path(mujoco.__file__).resolve().parent,
    }
    files = {}
    for label, root in roots.items():
        if label == "swm":
            paths = [
                root / "envs/ogbench/cube_env.py",
                root / "envs/ogbench/scene_env.py",
                root / "envs/ogbench/__init__.py",
                root / "envs/__init__.py",
                root / "envs/utils.py",
                root / "spaces.py",
                root / "world.py",
                root.parent / "uv.lock",
            ]
        else:
            paths = sorted(root.rglob("*"))
        for path in paths:
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[label + "/" + str(path.relative_to(root.parent if label == "swm" else root))] = {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "files": files,
        "swm_sha": subprocess.check_output(
            ["git", "-C", str(roots["swm"].parent), "rev-parse", "HEAD"], text=True
        ).strip(),
        "integration_mask": int(mujoco.mjtState.mjSTATE_INTEGRATION),
        "component_union": int(np.bitwise_or.reduce([int(getattr(mujoco.mjtState, "mjSTATE_" + p)) for p in PARTS])),
    }


def main():
    results = {"claim": "source preflight only; no promotion", "families": {}}
    (OUT / "provenance.json").write_text(json.dumps(provenance(), indent=2) + "\n")
    for name, cls in [("Cube", CubeEnv), ("Scene", SceneEnv)]:
        family = {"reset_pairs": [], "conditioned_reset_pairs": [], "continuations": []}
        results["families"][name] = family
        for seed in [0, 17, 42]:
            for task in range(1, 6):
                row = reset_pair(cls, seed, task)
                family["reset_pairs"].append(row)
                print(name, "reset", seed, task, "different fields", len(row["diff"]), flush=True)
            family["conditioned_reset_pairs"].append(reset_pair(cls, seed, 2, True))
            row = continuation(cls, seed)
            family["continuations"].append(row)
            print(
                name,
                "continuation",
                seed,
                "roundtrip",
                len(row["roundtrip_diff"]),
                "step differences",
                [len(r["state_diff"]) for r in row["trace"]],
                "reset differences",
                len(row["unseeded_reset_diff"]),
                flush=True,
            )
        try:
            ref = _ogbench_make(name, {})
            ref.close()
            family["frozen_constructor"] = {"status": "PASS", "rendering_tested": False}
        except Exception:
            family["frozen_constructor"] = {"status": "FAIL", "traceback": traceback.format_exc()}
        (OUT / "results.json").write_text(json.dumps(encode(results), indent=2) + "\n")
    print("OGB_MANIPULATION_PREFLIGHT_COMPLETE_NO_PROMOTION", flush=True)


if __name__ == "__main__":
    main()
