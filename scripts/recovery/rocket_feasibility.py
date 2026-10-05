"""Private source-only Rocket snapshot feasibility probe; no production adapter."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from tests.helpers.reference.rocket_semantics import RocketLandingEnv, capture, differences, objects


def restore(env, state, mode, state_id=None, filename=None):
    if mode == "memory":
        env.env.restoreState(stateId=state_id)
    elif mode == "file":
        env.env.restoreState(fileName=str(filename))
    elif mode == "numeric":
        for i in range(env.env.getNumBodies()):
            body = env.env.getBodyUniqueId(i)
            prefix = f"physics/body_{i}/"
            env.env.resetBasePositionAndOrientation(body, state[prefix + "position"], state[prefix + "quaternion"])
            env.env.resetBaseVelocity(body, state[prefix + "linear_velocity"], state[prefix + "angular_velocity"])
            for link in range(-1, env.env.getNumJoints(body)):
                mass = state[prefix + f"link_{link}/mass_inertia"]
                env.env.changeDynamics(body, link, mass=mass[0], localInertiaDiagonal=mass[1:])
                if link >= 0 and env.env.getJointInfo(body, link)[2] != env.env.JOINT_FIXED:
                    q, v = state[prefix + f"link_{link}/joint_position_velocity"]
                    env.env.resetJointState(body, link, q, v)
    else:
        raise ValueError(mode)
    # Both official routes omit dynamic link mass/inertia: explicitly restore these too.
    if mode in ["memory", "file"]:
        for i in range(env.env.getNumBodies()):
            body = env.env.getBodyUniqueId(i)
            for link in range(-1, env.env.getNumJoints(body)):
                mass = state[f"physics/body_{i}/link_{link}/mass_inertia"]
                env.env.changeDynamics(body, link, mass=mass[0], localInertiaDiagonal=mass[1:])
    # Official snapshots also need Python-side actuator, reward, lifecycle and RNG state.
    for prefix, (obj, names) in objects(env).items():
        for name in names:
            v = state[prefix + "/" + name]
            setattr(obj, name, v.copy() if v.ndim else v.item())
    for name in ["out_of_bounds", "fatal_collision", "env_complete"]:
        env.info[name] = bool(state["info/" + name])
    env.info["goal"] = env.current_goal
    rng = copy.deepcopy(env.np_random.bit_generator.state)
    for name in ["state", "inc"]:
        rng["state"][name] = sum(int(word) << (32 * i) for i, word in enumerate(state["rng/" + name]))
    rng["has_uint32"], rng["uinteger"] = map(int, state["rng/cached"])
    env.np_random.bit_generator.state = rng


def trace(env, actions, raw=False):
    snapshots = []
    for action in actions:
        if raw:
            env.env.set_setpoint(0, action.copy())
            env.env.step()
            env.compute_state()
            env.reward = 0.0
            env.compute_term_trunc_reward()
        else:
            env.step(action.copy())
        snapshots.append(capture(env))
    return {k: np.stack([s[k] for s in snapshots]) for k in snapshots[0]}


def probe(env, label, out, actions, raw=False):
    snapshot = capture(env)
    sid = env.env.saveState()
    filename = out / (label + ".bullet")
    env.env.saveBullet(str(filename))
    np.savez_compressed(out / (label + "-snapshot.npz"), **snapshot)
    reference = trace(env, actions, raw)
    np.savez_compressed(out / (label + "-reference.npz"), **reference)
    result = {
        "continuation_actions": np.asarray(actions).tolist(),
        "raw_engine_continuation_after_public_terminal": raw,
        "initial_contacts": int(snapshot["aviary/contact_array"].sum()),
        "initial_height": float(snapshot["physics/body_1/position"][2]),
        "routes": {},
    }
    for mode in ["memory", "file", "numeric"]:
        # Substantially different engine history, including post-terminal contact settling.
        env.env.set_setpoint(0, np.array([0.8, -0.6, 0.3, 1.0, 0.8, 0.2, -0.2]))
        history_contacts = []
        for _ in range(35):
            env.env.step()
            history_contacts.append(int(env.env.contact_array.sum()))
        restore(env, snapshot, mode, sid, filename)
        immediate = capture(env)
        np.savez_compressed(out / (label + "-" + mode + "-immediate.npz"), **immediate)
        replay = trace(env, actions, raw)
        np.savez_compressed(out / (label + "-" + mode + ".npz"), **replay)
        result["routes"][mode] = {
            "different_history_steps": 35,
            "different_history_contacts": history_contacts,
            "immediate_differences": differences(snapshot, immediate),
            "trajectory_differences": differences(reference, replay),
        }
    env.env.removeState(sid)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    zero = np.zeros(7)
    burn = np.array([0.15, -0.1, 0.2, 1.0, 0.5, 0.1, -0.1])
    results = {}
    env = RocketLandingEnv()
    try:
        for seed in [0, 17, 42]:
            env.reset(seed=seed)
            results[f"free_flight_{seed}"] = probe(env, f"free_flight_{seed}", args.output, [zero] * 8)
        env.reset(seed=17)
        for _ in range(12):
            env.step(burn)
        results["thrust_rotation"] = probe(env, "thrust_rotation", args.output, [burn, zero] * 4)
        env.reset(seed=17)
        actions = []
        while not (env.termination or env.truncation) and env.lin_pos[2] > 6.0:
            env.step(zero)
            actions.append(zero.copy())
        results["near_contact"] = probe(env, "near_contact", args.output, [zero] * 12)
        # Re-run the actual reset-to-crash trajectory; all diagnostics originate in dynamics.
        env.reset(seed=17)
        while not (env.termination or env.truncation):
            env.step(zero)
        results["crash"] = probe(env, "crash", args.output, [zero] * 8, raw=True)
        env.env.set_setpoint(0, zero.copy())
        for _ in range(300):
            env.env.step()
        env.compute_state()
        results["resting_contact"] = probe(env, "resting_contact", args.output, [zero] * 8, raw=True)
        schema = [
            {"key": k.split("/"), "shape": [1, *v.shape], "dtype": str(v.dtype)} for k, v in capture(env).items()
        ]
        (args.output / "candidate-schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    finally:
        env.close()
    report = {
        "status": "FEASIBILITY_DIAGNOSTIC_ONLY",
        "cases": results,
        "elapsed_seconds": time.perf_counter() - start,
        "numeric_exact": all(not r["routes"]["numeric"]["trajectory_differences"] for r in results.values()),
        "note": "Official snapshot handles/files are diagnostic controls only, never proposed public State. No acceptance or production implementation.",
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {mode: list(v["trajectory_differences"])[:5] for mode, v in case["routes"].items()}
                for name, case in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
