"""Private source-only Rocket snapshot feasibility probe; no production adapter."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from stable_worldmodel.envs.rocket_landing.pyflyt_rocketlanding import RocketLandingEnv

ENV_FIELDS = "start_pos start_orn step_count termination truncation state action reward throttle_state ignition_state previous_throttle_state previous_ignition_state previous_render_ignition_state ignition_color landing_pad_contact ang_vel ang_pos lin_vel lin_pos ground_lin_vel previous_ang_vel previous_lin_vel previous_lin_pos previous_ground_lin_vel current_goal".split()
AVIARY_FIELDS = "physics_steps aviary_steps elapsed_time contact_array".split()
DRONE_FIELDS = "mode setpoint cmd state aux_state".split()


def objects(env):
    d = env.env.drones[0]
    result = {
        "wrapper": (env, ENV_FIELDS),
        "aviary": (env.env, AVIARY_FIELDS),
        "drone": (d, DRONE_FIELDS),
        "bodies": (d.bodies, ["local_body_velocities"]),
        "boosters": (d.boosters, ["ratio_fuel_remaining", "throttle", "ignition_state"]),
        "gimbal": (d.booster_gimbal, ["gimbal_state", "rotation1", "rotation2"]),
    }
    for i, surface in enumerate(d.lifting_surfaces.surfaces):
        result[f"fin_{i}"] = (surface, ["local_surface_velocity", "actuation"])
    return result


def capture(env):
    result = {}
    for prefix, (obj, names) in objects(env).items():
        for name in names:
            result[prefix + "/" + name] = np.asarray(getattr(obj, name)).copy()
    for name in ["out_of_bounds", "fatal_collision", "env_complete"]:
        result["info/" + name] = np.asarray(env.info[name]).copy()
    rng = env.np_random.bit_generator.state
    assert rng["bit_generator"] == "PCG64"
    for name in ["state", "inc"]:
        value = rng["state"][name]
        result["rng/" + name] = np.array([(value >> (32 * i)) & 0xFFFFFFFF for i in range(4)], np.int64)
    result["rng/cached"] = np.array([rng["has_uint32"], rng["uinteger"]], np.int64)
    assert all(obj.np_random is env.np_random for obj in [env.env, env.env.drones[0], env.env.drones[0].boosters])
    for i in range(env.env.getNumBodies()):
        body = env.env.getBodyUniqueId(i)
        pos, orn = env.env.getBasePositionAndOrientation(body)
        vel, ang = env.env.getBaseVelocity(body)
        result[f"physics/body_{i}/position"] = np.array(pos)
        result[f"physics/body_{i}/quaternion"] = np.array(orn)
        result[f"physics/body_{i}/linear_velocity"] = np.array(vel)
        result[f"physics/body_{i}/angular_velocity"] = np.array(ang)
        for link in range(-1, env.env.getNumJoints(body)):
            info = env.env.getDynamicsInfo(body, link)
            result[f"physics/body_{i}/link_{link}/mass_inertia"] = np.array([info[0], *info[2]])
            if link >= 0:
                j = env.env.getJointState(body, link)
                result[f"physics/body_{i}/link_{link}/joint_position_velocity"] = np.array(j[:2])
    d = env.env.drones[0]
    links = env.env.getLinkStates(d.Id, list(range(env.env.getNumJoints(d.Id))), computeLinkVelocity=1)
    result["diagnostic/cached_links"] = np.array([np.concatenate([np.asarray(v) for v in link]) for link in links])
    return result


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


def differences(a, b):
    result = {}
    assert a.keys() == b.keys()
    for k in a:
        if not np.array_equal(a[k], b[k]):
            result[k] = {
                "count": int(np.count_nonzero(a[k] != b[k])),
                "max_abs": float(np.max(np.abs(a[k].astype(float) - b[k].astype(float)))),
            }
    return result


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
