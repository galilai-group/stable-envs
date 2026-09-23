"""Bounded Phase A investigation, not a production serializer or acceptance gate.

Run from the integration checkout with pinned SWM on PYTHONPATH. The source
continuation is never restored. Private CFFI helpers are diagnostic only.
"""

import ast
import copy
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pymunk
from pymunk._chipmunk_cffi import lib
from stable_worldmodel.envs.pusht.env import PushT

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "checkpoint_c/pusht/07_agent_block_component_ablation.py"
# Reuse preserved diagnostic helpers without executing the old experiment.
module = ast.parse(OLD.read_text())
helpers = {"np": np, "pymunk": pymunk}
exec(
    compile(
        ast.Module(body=[n for n in module.body if isinstance(n, ast.FunctionDef)], type_ignores=[]), str(OLD), "exec"
    ),
    helpers,
)

body_snapshot = helpers["body_snapshot"]
restore_bodies = helpers["restore_bodies"]
arbiter_map = helpers["arbiter_map"]
copy_arbiter = helpers["copy_arbiter"]
vector = helpers["vector"]

FIELDS = ("position", "velocity", "angle", "angular_velocity", "force", "torque", "center_of_gravity")
ACTIONS = [
    np.zeros(2, np.float32),
    np.array([0.173456789, -0.081234567], np.float64),
    np.array([-0.07, 0.11], np.float32),
    np.array([0.03, -0.02], np.float32),
]
SAFE = {
    "variation": [],
    "state": np.array([100, 100, 250, 250, 0, 0, 0], np.float64),
    "goal_state": np.array([200, 200, 300, 300, 0, 0, 0], np.float64),
}


def plain(x):
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list, np.ndarray, pymunk.Vec2d)):
        return [plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def spaces(space):
    result = {"rng": copy.deepcopy(space._np_random.bit_generator.state) if space._np_random is not None else None}
    if hasattr(space, "spaces"):
        result["children"] = {k: spaces(v) for k, v in space.spaces.items()}
    if hasattr(space, "_value"):
        result["value"] = plain(space._value)
        result["init_value"] = plain(space._init_value)
    return result


def candidate(env):
    return plain(
        {
            "body": {
                name: {**{k: getattr(body, k) for k in FIELDS}, "sleeping": body.is_sleeping}
                for name, body in [("agent", env.agent), ("block", env.block)]
            },
            "goal_state": env.goal_state.copy(),
            "goal_pose": env.goal_pose.copy(),
            "rng": env.rng.bit_generator.state,
            "gym_rng": env.np_random.bit_generator.state,
            "variation": spaces(env.variation_space),
            "latest_action": env.latest_action,
            "contact_points": env.n_contact_points,
            "space": {
                "stamp": lib.cpSpaceGetTimestamp(env.space._space),
                "dt": lib.cpSpaceGetCurrentTimeStep(env.space._space),
                "shape_id_counter": lib.cpSpaceGetShapeIDCounter(env.space._space),
            },
        }
    )


def restore(source, target, mode):
    snapshot = body_snapshot(source)
    restore_bodies(target, snapshot)
    target.goal_state = source.goal_state.copy()
    target.goal_pose = source.goal_pose.copy()
    target._goal = source._goal.copy()  # source-owned learner image; not a proposed State field
    target.rng = copy.deepcopy(source.rng)
    target._np_random = copy.deepcopy(source.np_random)
    target.variation_space = copy.deepcopy(source.variation_space)
    target.latest_action = copy.deepcopy(source.latest_action)
    if mode == "detect_then_numeric_cache":
        target.space.step(0.01)
        restore_bodies(target, snapshot)
        sa, ta = arbiter_map(source), arbiter_map(target)
        assert sa.keys() == ta.keys(), (sa.keys(), ta.keys())
        for k in sa:
            copy_arbiter(sa[k], ta[k], "full")
    target.n_contact_points = source.n_contact_points
    for name in ("Timestamp", "CurrentTimeStep", "ShapeIDCounter"):
        getattr(lib, "cpSpaceSet" + name)(target.space._space, getattr(lib, "cpSpaceGet" + name)(source.space._space))


def prepare(case):
    e = PushT()
    options = copy.deepcopy(SAFE)
    if case == "wall":
        options["state"][2:4] = [400, 400]
    if case == "agent_block":
        options["state"][:2] = [250, 220]
    e.reset(seed=0, options=None if case == "multiple_contact" else options)
    history = []
    if case in ("agent_block", "multiple_contact"):
        for _ in range(30):
            action = ((np.array(e.block.position) - np.array(e.agent.position)) / 100).astype(np.float64)
            out = e.step(action)
            history.append(action.tolist())
            if out[4]["n_contacts"]:
                break
        assert out[4]["n_contacts"]
    return e, history


def contacts(e):
    membership = []
    e.block.each_arbiter(lambda a: membership.append(1))
    return {
        "cached": len(e.space._get_arbiters()),
        "block_active": len(membership),
        "pairs": [repr(k) for k in arbiter_map(e)],
    }


def cache_fields(e):
    rows = {}
    for key, a in arbiter_map(e).items():
        scalar = ("e", "u", "swapped", "stamp", "state", "processCollision", "count")
        row = {k: getattr(a, k) for k in scalar}
        for k in ("surface_vr", "n"):
            v = getattr(a, k)
            row[k] = [v.x, v.y]
        row["contacts"] = []
        for i in range(a.count):
            c = a.contacts[i]
            cr = {k: getattr(c, k) for k in ("nMass", "tMass", "bounce", "jnAcc", "jtAcc", "jBias", "bias", "hash")}
            for k in ("r1", "r2"):
                v = getattr(c, k)
                cr[k] = [v.x, v.y]
            row["contacts"].append(cr)
        rows[repr(key)] = row
    return rows


def differences(a, b, path=""):
    if isinstance(a, dict):
        return [v for k in a for v in differences(a[k], b[k], path + "/" + k)]
    return [] if a == b else [path]


def trial(case, history, mode):
    source, actions = prepare(case)
    if history == "same":
        target, _ = prepare(case)
    else:
        target, _ = prepare("free")
        target.step(np.array([-0.3, 0.2], np.float32))
        target.step(np.array([0.2, -0.1], np.float64))
    restore(source, target, mode)
    expected, actual = candidate(source), candidate(target)
    result = {
        "case": case,
        "target_history": history,
        "mode": mode,
        "preparation_actions": actions,
        "before_source": expected,
        "before_target": actual,
        "before_differences": differences(expected, actual),
        "contacts_source": contacts(source),
        "contacts_target": contacts(target),
        "numeric_cache_source": cache_fields(source),
        "numeric_cache_target": cache_fields(target),
        "numeric_cache_exact": cache_fields(source) == cache_fields(target),
        "continuation": [],
    }
    if mode == "detect_then_numeric_cache":
        assert result["numeric_cache_exact"]
    for action in ACTIONS:
        a, b = source.step(action.copy()), target.step(action.copy())
        sa, sb = candidate(source), candidate(target)
        va, vb = vector(a), vector(b)
        result["continuation"].append(
            {
                "action": action.tolist(),
                "dtype": str(action.dtype),
                "source_state": sa,
                "target_state": sb,
                "state_differences": differences(sa, sb),
                "source_output": va.tolist(),
                "target_output": vb.tolist(),
                "output_exact": bool(np.array_equal(va, vb)),
                "output_max_abs_error": float(np.max(np.abs(va - vb))),
            }
        )
    source.close()
    target.close()
    return result


if __name__ == "__main__":
    result = {
        "profile": {
            k: importlib.metadata.version(k)
            for k in ("pymunk", "numpy", "gymnasium", "pygame", "opencv-python", "cffi", "shapely")
        },
        "chipmunk": pymunk.chipmunk_version,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "helper_sha256": hashlib.sha256(OLD.read_bytes()).hexdigest(),
        "trials": [],
    }
    for case in ("free", "agent_block", "wall", "multiple_contact"):
        for history, mode in [
            ("same", "body_task_rng"),
            ("different", "body_task_rng"),
            ("different", "detect_then_numeric_cache"),
        ]:
            row = trial(case, history, mode)
            result["trials"].append(row)
            print(
                case,
                history,
                mode,
                "roundtrip",
                not row["before_differences"],
                "exact",
                [x["output_exact"] and not x["state_differences"] for x in row["continuation"]],
                "max_output_errors",
                [x["output_max_abs_error"] for x in row["continuation"]],
                flush=True,
            )
    # A terminal investigation marker does not mean promotion passed.
    assert any(not c["output_exact"] for r in result["trials"] if r["case"] != "free" for c in r["continuation"])
    result["status"] = "STOP_CONTACT_RESTORATION_FAILED"
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("PUSHT_PHASE_A_INVESTIGATION_COMPLETE_CONTACT_GATE_FAILED")
