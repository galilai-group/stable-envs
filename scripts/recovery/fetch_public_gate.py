"""Bounded public Fetch CPU acceptance; pinned independent SWM oracle."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import stable_worldmodel
import torch

from scripts.recovery.fetch_pickandplace_adapter_probe import compare_state, exact_array, oracle
from stable_envs.envs.fetch import FetchPickAndPlace

TASK = "PickAndPlace"
SUFFIX = ""
ID = "swm/FetchPickAndPlace-v3"

PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"


def compare_output(out, expected):
    # SWM's observations and info["state"] are deliberately outside this gate.
    assert set(out.keys()) == {
        "state",
        "reward",
        "success",
        "terminated",
        "truncated",
        "states",
        "valid",
        "state_valid",
    }
    for key, index in (("reward", 1), ("terminated", 2), ("truncated", 3)):
        exact_array(out[key].cpu().numpy(), np.asarray([r[index] for r in expected]), key)
    exact_array(out["success"].cpu().numpy(), np.asarray([r[4]["is_success"] for r in expected]), "success")


def full_compare(state, refs):
    compare_state(state, oracle(refs))
    for lane, e in enumerate(refs):
        exact_array(state["time"][lane].numpy(), np.array([e.unwrapped.data.time]), "time")
        exact_array(
            state["task", "initial_gripper_xpos"][lane].numpy(), e.unwrapped.initial_gripper_xpos, "reset_anchor"
        )
        # compare_state independently extracts all older fields; also enforce
        # their complete declared tensor shapes/dtypes at the public boundary.
    from stable_envs._adapters.fetch.adapter import state_fields

    FIELDS = state_fields(refs[0].unwrapped.model.nq, refs[0].unwrapped.model.nv)

    assert set(state.keys(True, True)) == set(FIELDS)
    for key, (width, dtype) in FIELDS.items():
        assert state[key].shape == (len(refs), width) and state[key].dtype == dtype, key


def equal_state(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for k in a.keys(True, True):
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and torch.equal(a[k], b[k]), k


def case(seed, batch, dtype):
    refs = [gym.make(ID) for _ in range(batch)]
    a = FetchPickAndPlace(batch, device="cpu", seed=seed)
    b = FetchPickAndPlace(batch, device="cpu", seed=seed + 999)
    rng = np.random.default_rng(20000 + seed)
    actions = rng.uniform(-2, 2, (40, batch, 4)).astype(dtype)
    actions[0] = [-1, 1, -1, 1]
    actions[1] = [0.1, -0.2, 0.3, -0.4]
    try:
        for i, e in enumerate(refs):
            e.reset(seed=seed + i, options={"variation": []})
        full_compare(a.reset(seed=seed), refs)
        for i, e in enumerate(refs):
            u = a._simulator.envs[i].unwrapped
            exact_array(u.data.mocap_pos, e.unwrapped.data.mocap_pos, "reset_mocap_pos")
            exact_array(u.data.mocap_quat, e.unwrapped.data.mocap_quat, "reset_mocap_quat")
        snapshot = a.get_state()
        mutated = a.get_state()
        mutated["qpos"][0, 0] += 10
        equal_state(a.get_state(), snapshot)
        # Normal sequential steps must not call set_state.
        original = a._simulator.set_state

        def forbidden(*args, **kw):
            raise AssertionError("normal step restored simulator")

        a._simulator.set_state = forbidden
        for action in actions[:32]:
            expected = [e.step(action[i]) for i, e in enumerate(refs)]
            out = a.step(action, frames=None)
            compare_output(out, expected)
            full_compare(out["state"], refs)
            exact_array(out["success"].numpy(), np.array([r[4]["is_success"] for r in expected]), "success")
        a._simulator.set_state = original
        for action in actions[5:16]:
            b.step(action, frames=None)
        snapshot = a.get_state()
        b.set_state(snapshot)
        equal_state(b.get_state(), snapshot)
        for action in actions[32:]:
            expected = [e.step(action[i]) for i, e in enumerate(refs)]
            out = b.step(action, frames=None)
            compare_output(out, expected)
            full_compare(out["state"], refs)
        for e in refs:
            e.reset(options={"variation": []})
        full_compare(b.reset(), refs)
        for i, e in enumerate(refs):
            e.reset(seed=seed + i, options={"variation": []})
        full_compare(b.reset(seed=seed), refs)
        for bad in (torch.zeros(batch, b.state_dim), b._unpack_render_state(torch.zeros(batch, b.state_dim))):
            try:
                b.set_state(bad)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError("lossy restore accepted")
        try:
            b.render()
        except NotImplementedError as e:
            assert "CPU_BACKEND_MISSING" in str(e)
        else:
            raise AssertionError("CPU full rendering unexpectedly supported")
        return dict(
            seed=seed,
            batch=batch,
            dtype=np.dtype(dtype).name,
            reset_exact=True,
            mocap_reset_exact=True,
            steps=32,
            restored_steps=8,
            different_history=11,
            all_values_shapes_dtypes_exact=True,
            independent_snapshots=True,
            roundtrip=True,
            rng_replay=True,
            repeated_seed=True,
            no_sequential_restore=True,
            compact_restore_refused=True,
            cpu_renderer="CPU_BACKEND_MISSING",
        )
    finally:
        a.close()
        b.close()
        for e in refs:
            e.close()


def boundary():
    if TASK == "Reach":
        return reach_boundary()
    rows = []
    for distance in (0.0, np.nextafter(0.05, 0.0), 0.05, np.nextafter(0.05, np.inf), 0.2):
        e = gym.make(ID)
        a = FetchPickAndPlace(device="cpu", seed=17)
        try:
            e.reset(seed=17, options={"variation": []})
            u = e.unwrapped
            # Free object at x=0 removes cancellation when constructing exactly .05.
            u.data.qpos[15:18] = [0.0, 0.0, 0.7]
            u.data.qvel[:] = 0
            mujoco.mj_forward(u.model, u.data)
            # An independent upstream probe determines post-action achieved_goal.
            probe_env = gym.make(ID)
            probe = probe_env.unwrapped
            probe.reset(seed=17)
            probe.data.qpos[:] = u.data.qpos
            probe.data.qvel[:] = u.data.qvel
            mujoco.mj_forward(probe.model, probe.data)
            probe.data.qacc_warmstart[:] = u.data.qacc_warmstart
            i = u.model.body("robot0:gripper_link").id
            probe.data.xpos[i] = u.data.xpos[i]
            probe.data.xquat[i] = u.data.xquat[i]
            expected_achieved = probe.step(np.zeros(4))[0]["achieved_goal"]
            probe.close()
            assert expected_achieved[0] == 0.0
            u.goal = expected_achieved.copy()
            u.goal[0] = distance
            # Build the candidate public snapshot from independent source arrays.
            state = a.get_state()
            for key in ("qpos", "qvel"):
                state[key][0] = torch.from_numpy(getattr(u.data, key).copy())
            state["time"][0, 0] = u.data.time
            state["warmstart"][0] = torch.from_numpy(u.data.qacc_warmstart.copy())
            state["controller", "position"][0] = torch.from_numpy(u.data.xpos[i].copy())
            state["controller", "quaternion"][0] = torch.from_numpy(u.data.xquat[i].copy())
            state["goal"][0] = torch.from_numpy(u.goal.copy())
            a.set_state(state)
            r = e.step(np.zeros(4))
            out = a.step(np.zeros((1, 4)), frames=None)
            compare_output(out, [r])
            full_compare(out["state"], [e])
            got = float(np.linalg.norm(r[4]["proprio"][3:6] - u.goal))
            assert got == distance, (got, distance)
            assert float(r[1]) == (-distance if SUFFIX == "Dense" else -1.0 if distance > 0.05 else 0.0)
            assert float(r[4]["is_success"]) == float(distance < 0.05)
            assert r[2] is False
            rows.append(
                dict(
                    distance_hex=float(distance).hex(),
                    reward=float(r[1]),
                    success=float(r[4]["is_success"]),
                    terminated=False,
                    exact=True,
                )
            )
        finally:
            a.close()
            e.close()
    return rows


def reach_boundary():
    # Reach uses the gripper site, without a free object that can be placed at x=0.
    # Exercise the exact upstream functions at adjacent float64 distances, and
    # separately exercise public step on the achievable near-threshold states.
    rows = []
    e = gym.make(ID)
    a = FetchPickAndPlace(seed=17)
    try:
        for distance in (0.0, np.nextafter(0.05, 0.0), 0.05, np.nextafter(0.05, np.inf), 0.2):
            achieved, goal = np.zeros(3), np.array([distance, 0.0, 0.0])
            for u in (e.unwrapped, a._simulator.envs[0].unwrapped):
                assert float(u.compute_reward(achieved, goal, {})) == (
                    -distance if SUFFIX == "Dense" else -float(distance > 0.05)
                )
                assert float(u._is_success(achieved, goal)) == float(distance < 0.05)
            e.reset(seed=17, options={"variation": []})
            a.reset(seed=17)
            u = e.unwrapped
            u.goal = u._get_obs()["achieved_goal"] + np.array([distance, 0.0, 0.0])
            state = a.get_state()
            state["goal"][0] = torch.from_numpy(u.goal.copy())
            a.set_state(state)
            r = e.step(np.zeros(4))
            out = a.step(np.zeros((1, 4)), frames=None)
            compare_output(out, [r])
            full_compare(out["state"], [e])
            got = float(np.linalg.norm(r[4]["proprio"][:3] - u.goal))
            assert float(r[1]) == (-got if SUFFIX == "Dense" else -float(got > 0.05))
            rows.append(
                dict(
                    distance_hex=float(distance).hex(),
                    direct_rule_exact=True,
                    public_step_distance_hex=got.hex(),
                    public_step_exact=True,
                )
            )
        return rows
    finally:
        a.close()
        e.close()


def lifecycle():
    e = gym.make(ID)
    a = FetchPickAndPlace(seed=17)
    b = FetchPickAndPlace(seed=999)
    try:
        e.reset(seed=17, options={"variation": []})
        rows = []
        for step in range(1, 52):
            if step == 50:
                b.set_state(a.get_state())
            r = e.step(np.zeros(4))
            out = a.step(np.zeros((1, 4)), frames=None)
            compare_output(out, [r])
            full_compare(out["state"], [e])
            if step >= 50:
                replay = b.step(np.zeros((1, 4)), frames=None)
                compare_output(replay, [r])
                full_compare(replay["state"], [e])
            if step in (49, 50, 51):
                rows.append(dict(step=step, terminated=bool(r[2]), truncated=bool(r[3]), exact=True))
        return rows
    finally:
        e.close()
        a.close()
        b.close()


def variation():
    results = []
    configurations = [
        ("mass", {"block.mass": [8.0]}),
        (
            "start_goal",
            {
                "agent.start_position": [1.32, 0.72],
                "block.start_position": [1.3, 0.7],
                "block.angle": [0.4],
                "goal.start_position": [1.25, 0.8, 0.4247],
            },
        ),
        ("visual", {"object.color": [0.2, 0.8, 0.3]}),
    ]
    if TASK == "Reach":
        configurations = [
            ("start_goal", {"agent.start_position": [1.32, 0.72], "goal.start_position": [1.25, 0.8, 0.4247]}),
            ("visual", {"table.color": [0.2, 0.8, 0.3]}),
        ]
    for name, values in configurations:
        e = gym.make(ID)
        d = gym.make(ID)
        a = FetchPickAndPlace(seed=17)
        b = FetchPickAndPlace(seed=999)
        options = {"variation": [], "variation_values": {k: np.asarray(v, np.float64) for k, v in values.items()}}
        divergence = 0.0
        try:
            e.reset(seed=17, options=options)
            d.reset(seed=17, options={"variation": []})
            full_compare(a.reset(seed=17, options=options), [e])
            b.reset(seed=999, options=options)
            actions = np.random.default_rng(41).uniform(-1, 1, (40, 1, 4))
            for step, action in enumerate(actions):
                if step == 32:
                    for other in actions[:11]:
                        b.step(other, frames=None)
                    b.set_state(a.get_state())
                    equal_state(a.get_state(), b.get_state())
                r = e.step(action[0])
                base = d.step(action[0])
                out = (a if step < 32 else b).step(action, frames=None)
                compare_output(out, [r])
                full_compare(out["state"], [e])
                divergence = max(divergence, float(np.max(np.abs(r[4]["proprio"] - base[4]["proprio"]))))
            if name == "mass":
                assert divergence > 0
            if name == "visual":
                assert divergence == 0
            # Restore with persistent physical Variation, then unseeded reset.
            e.reset(options=options)
            full_compare(b.reset(options=options), [e])
            # A reset without explicit overrides restores initial mass but keeps
            # the historically mutated reset anchor. Restore into b beforehand.
            anchor_snapshot = b.get_state()
            b.reset(seed=999, options={"variation_values": {"agent.start_position": np.array([1.4, 0.8])}})
            b.set_state(anchor_snapshot)
            equal_state(b.get_state(), anchor_snapshot)
            e.reset(seed=42, options={"variation": []})
            full_compare(b.reset(seed=42), [e])
            assert TASK == "Reach" or (
                b._simulator.envs[0].unwrapped.model.body_mass[b._simulator._object[0]]
                == e.unwrapped.model.body_mass[e.unwrapped.model.body("object0").id]
            )
            results.append(
                dict(
                    kind=name,
                    values=values,
                    steps=32,
                    restored_steps=8,
                    exact=True,
                    default_divergence=divergence,
                    rng_reset_exact=True,
                    cleared_override_reset_exact=True,
                )
            )
        finally:
            a.close()
            b.close()
            e.close()
            d.close()
    # Persistent set_variation corresponds to SWM init_value, unlike a one-reset override.
    if TASK != "Reach":
        e = gym.make(ID, init_value={"block.mass": np.array([8.0])})
        a = FetchPickAndPlace(seed=17)
        try:
            a.set_variation({"block.mass": [8.0]})
            for seed in (17, 42):
                e.reset(seed=seed, options={"variation": []})
                full_compare(a.reset(seed=seed), [e])
                for action in np.random.default_rng(seed).uniform(-1, 1, (8, 1, 4)):
                    r = e.step(action[0])
                    out = a.step(action, frames=None)
                    compare_output(out, [r])
                    full_compare(out["state"], [e])
            results.append(
                dict(kind="persistent_mass", values={"block.mass": [8.0]}, resets=2, steps_per_reset=8, exact=True)
            )
        finally:
            e.close()
            a.close()
    samples = []
    for seed in (0, 17, 42):
        e = gym.make(ID)
        a = FetchPickAndPlace(seed=seed)
        try:
            visual_key = "table.color" if TASK == "Reach" else "object.color"
            options = {"variation": [visual_key]}
            e.reset(seed=seed, options=options)
            full_compare(a.reset(seed=seed, options=options), [e])
            w = e
            while not hasattr(w, "variation_space"):
                w = w.env
            color = w.variation_space[visual_key.split(".")[0]]["color"].value
            layout = {k: (o, n) for k, o, n in a.variation_space.flat_layout()}
            off, _ = layout[visual_key]
            exact_array(a.get_style()[0, off : off + 3].numpy(), color.astype(np.float32), "sampled_color")
            samples.append(dict(seed=seed, color=color.tolist(), simulator_rng_independent=True, exact=True))
        finally:
            e.close()
            a.close()
    return dict(
        cases=results,
        seeded_visual_rng=samples,
        unseeded_visual_rng="Fresh external entropy on every source variation_space.seed(None); no replay claim. Simulator replay certified with fixed Variation.",
    )


def main(output):
    root = Path(stable_worldmodel.__file__).resolve().parents[1]
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == PIN
    assert not subprocess.check_output(
        ["git", "-C", str(root), "status", "--short", "--untracked-files=no"], text=True
    )
    versions = {
        k: importlib.metadata.version(k)
        for k in ("numpy", "gymnasium", "gymnasium-robotics", "mujoco", "torch", "tensordict")
    }
    assert versions["mujoco"] == "3.8.1" and versions["gymnasium-robotics"] == "1.4.2"
    result = dict(
        status="PASS",
        task=TASK,
        historical_id=ID,
        observation_integration="DEFERRED",
        swm_commit=PIN,
        profile=dict(python=platform.python_version(), versions=versions),
        cases=[case(s, b, d) for s in (0, 17, 42) for b in (1, 3) for d in (np.float32, np.float64)],
        threshold=boundary(),
        lifecycle=lifecycle(),
        variation=variation(),
        execution_mode="DELEGATED_CPU",
        cpu_renderer="CPU_BACKEND_MISSING",
        renderer_gate="SEPARATE_OSCAR_PROFILE_REQUIRED",
    )
    # Canonical named tensors bridge the CPU authority and GPU renderer profiles.
    e = FetchPickAndPlace(num_envs=3, seed=17)
    try:
        for i in range(8):
            e.step(np.random.default_rng(i).uniform(-1, 1, (3, 4)), frames=None)
        state = e.get_state()
        fixture = {
            ".".join(k) if isinstance(k, tuple) else k: dict(
                dtype=str(state[k].dtype).removeprefix("torch."), values=state[k].tolist()
            )
            for k in state.keys(True, True)
        }
        fp = output.with_name(f"canonical-{TASK}.json")
        fp.write_text(json.dumps(fixture, indent=2) + "\n")
        result["canonical_state_sha256"] = hashlib.sha256(fp.read_bytes()).hexdigest()
    finally:
        e.close()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("FETCH_PUBLIC_CPU_GATE_PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", choices=("PickAndPlace", "Reach", "Push", "Slide"), default="PickAndPlace")
    p.add_argument("--suffix", choices=("", "Dense", "Dict"), default="")
    args = p.parse_args()
    TASK, SUFFIX = args.task, args.suffix
    ID = f"swm/Fetch{TASK}{SUFFIX}-v3"
    import stable_envs.envs.fetch as fetch

    FetchPickAndPlace = getattr(fetch, f"Fetch{TASK}{SUFFIX}")
    main(args.output)
