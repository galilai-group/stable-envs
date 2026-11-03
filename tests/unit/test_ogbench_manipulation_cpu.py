"""CPU contract and controlled SWM-oracle gates; renderer acceptance is separate."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("ogbench")
import mujoco
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv

from stable_envs.envs.ogbench import OGBCube, OGBScene
from stable_envs.envs.ogbench.simulation.compatibility import CompatibleCube, CompatibleScene
from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
from tests.helpers.reference.ogbench_manipulation_semantics import assert_oracle, load_oracle, numpy_equal
from tests.helpers.reference.semantics import assert_same_state

ROOT = Path(__file__).resolve().parents[2]
CLASSES = {"Cube": OGBCube, "Scene": OGBScene}


@pytest.fixture(scope="module")
def oracle():
    pytest.importorskip("stable_worldmodel")
    return load_oracle()


@pytest.mark.swm
@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("seed", [0, 17, 42])
@pytest.mark.parametrize("task", [None, 1, 2, 3, 4, 5, "custom"])
def test_controlled_oracle_reset_and_action_parity(family, seed, task, oracle):
    cls = CompatibleCube if family == "Cube" else CompatibleScene
    raw_cls = oracle.OwnedCube if family == "Cube" else oracle.OwnedScene
    env, raw = cls(seed=seed), raw_cls(entropy_seed=seed)
    try:
        options = {} if task is None else {"task_id": task}
        if task == "custom":
            payload = copy.deepcopy(raw.task_infos[2])
            if family == "Cube":
                payload["goal_xyzs"][0, 0] += 0.01
            else:
                payload["goal"]["block_xyzs"][0, 0] += 0.01
            options = {"task_info": payload}
        if task == 3:
            options["variation"] = ["cube.size", "agent.ee_start_position"]
        elif task == 4:
            options.update(
                variation=[],
                variation_values={
                    "cube.size": [0.025],
                    "cube.color": [[0.1, 0.8, 0.4]],
                    "agent.ee_start_position": np.array([0.42, 0.03, 0.29]),
                },
            )
        elif task == 5:
            options.update(variation=["floor.color", "agent.color", "camera.angle_delta", "light.intensity"])
        env.reset(seed=seed, options=options)
        raw.reset(seed=seed, options=options)
        assert_oracle(env, raw, oracle)
        # Unclipped out-of-bounds actions exercise upstream affine scaling/workspace clipping.
        actions = np.concatenate(
            [np.eye(5), -np.eye(5), np.array([[2.0, -3.0, 0.6, 4.0, -2.0], [0.3, -0.7, -0.8, 0.9, 1.0]])]
        )
        for i, action in enumerate(actions):
            action = action.astype(np.float32 if i % 2 else np.float64)
            a, b = env.step(action), raw.step(action)
            assert not oracle.differences(a, b)
            assert_oracle(env, raw, oracle)
    finally:
        env.close()
        raw.close()


@pytest.mark.swm
@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("reward_task", [0, 1, 5])
def test_reward_routes_delegate_without_task_draw(family, reward_task, oracle):
    cls = CompatibleCube if family == "Cube" else CompatibleScene
    raw_cls = oracle.OwnedCube if family == "Cube" else oracle.OwnedScene
    a, b = cls(seed=42, reward_task_id=reward_task), raw_cls(entropy_seed=42, reward_task_id=reward_task)
    try:
        before = a.task_rng.get_state()
        for options in ({"task_id": 3}, {"task_info": copy.deepcopy(a.task_infos[3])}):
            a.reset(seed=42, options=options)
            b.reset(seed=42, options=options)
            assert_oracle(a, b, oracle)
            numpy_equal(before, a.task_rng.get_state())
            assert a.cur_task_id == (2 if reward_task == 0 else reward_task)
            assert a.compute_reward() == b.compute_reward()
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("family", CLASSES)
def test_state_roundtrip_reset_replay_and_atomic_validation(family):
    outside = np.random.get_state()
    with CLASSES[family](num_envs=2, seed=None) as a, CLASSES[family](num_envs=2, seed=19) as b:
        saved = a.get_state()
        immutable = saved.clone()
        b.set_state(saved)
        assert_same_state(saved, b.get_state())
        for i in range(6):
            action = torch.full((2, 5), 0.2 if i % 2 else -0.4, dtype=torch.float64 if i % 2 else torch.float32)
            assert_same_state(a.step(action, frames=None), b.step(action, frames=None))
        assert_same_state(saved, immutable)
        a.set_state(saved)
        b.set_state(saved)
        for seed in (None, 0, 17, 42, None):
            assert_same_state(a.reset(seed), b.reset(seed))
        anchor = a.get_state()
        # Every leaf's presence and dtype is checked, including the final lane, before mutation.
        for key in anchor.keys(True, True):
            bad = anchor.clone()
            del bad[key]
            with pytest.raises(ValueError):
                a.set_state(bad)
            bad = anchor.clone()
            bad[key] = bad[key].to(torch.float32 if bad[key].dtype != torch.float32 else torch.float64)
            with pytest.raises(ValueError):
                a.set_state(bad)
        bads = []
        bad = anchor.clone()
        bad["extra"] = torch.zeros(2, 1)
        bads.append(bad)
        for key, value in [
            (("physics", "time"), -1.0),
            (("task", "id"), 6),
            (("rng", "action", "increment"), 0),
            (("rng", "environment", "has_uint32"), 2),
            (("rng", "action", "uinteger"), 2**32),
            (("rng", "task", "position"), 625),
            (("rng", "task", "key"), 2**32),
            (("rng", "task", "has_gauss"), 2),
            (("compatibility", "signature"), 0),
            (("variation", "current", "cube.size"), 0.3),
            (("lifecycle", "reset_next_step"), True),
        ]:
            bad = anchor.clone()
            bad[key][-1] = value
            bads.append(bad)
        bad = anchor.clone()
        bad["physics", "qvel"][-1, 0] = float("nan")
        bads.append(bad)
        bad = anchor.clone()
        bad["physics", "qpos"] = bad["physics", "qpos"][:, :-1]
        bads.append(bad)
        if family == "Scene":
            bad = anchor.clone()
            bad["scene", "cur_button_states"][-1, 1] = 2
            bads.append(bad)
        for bad in bads:
            with pytest.raises((ValueError, TypeError)):
                a.set_state(bad)
            assert_same_state(anchor, a.get_state())
        packed = a._pack_render_state(anchor)
        with pytest.raises(TypeError):
            a.set_state(packed)
        with pytest.raises(ValueError):
            a.set_state(a._unpack_render_state(packed))
        assert torch.equal(packed, a._pack_render_state(a._unpack_render_state(packed)))
        assert a.state_spec[("physics", "qpos")].shape == tuple(anchor["physics", "qpos"].shape[1:])
        numpy_equal(outside, np.random.get_state())


@pytest.mark.parametrize("family", CLASSES)
def test_variation_anchors_model_restore_and_rejections(family):
    with CLASSES[family](seed=17) as a, CLASSES[family](seed=42) as b:
        original = a.get_state()
        a.set_variation({"cube.size": np.array([0.025]), "agent.ee_start_position": np.array([0.43, 0.04, 0.29])})
        pending = a.get_state()
        start = ("variation", "current", "agent.ee_start_position")
        assert torch.equal(original[start], pending[start])
        assert (pending["variation", "current", "cube.size"] == 0.025).all()
        b.set_state(pending)
        assert_same_state(a.reset(), b.reset())
        saved = a.get_state()
        a.set_variation({"cube.size": np.array([0.015])})
        a.reset(42)
        a.set_state(saved)
        assert_same_state(saved, a.get_state())
        assert_same_state(a.reset(), b.reset())
        before = a.get_state()
        for options in (
            {"mode": "data_collection"},
            {"render_goal": True},
            {"state": np.zeros(41)},
            {"task_id": 0},
            {"variation": ["unknown"]},
            {"variation_values": {"cube.size": [0.2]}},
        ):
            with pytest.raises(ValueError):
                a.reset(options=options)
            assert_same_state(before, a.get_state())
        with pytest.raises(TypeError):
            CLASSES[family](mode="data_collection")
        with pytest.raises(TypeError):
            CLASSES[family](multiview=True)
        assert a.sample_states(2, rng=np.random.default_rng(0)).shape == (2, a.state_dim)
        payload = copy.deepcopy(a._simulator.envs[0].task_infos[0])
        if family == "Cube":
            payload["init_xyzs"] = payload["init_xyzs"].tolist()
        else:
            payload["init"]["block_xyzs"] = payload["init"]["block_xyzs"].tolist()
        with pytest.raises(ValueError, match="NumPy arrays"):
            a.reset(options={"task_info": payload})
        assert_same_state(before, a.get_state())


@pytest.mark.parametrize("family", CLASSES)
def test_variation_values_take_one_row_per_lane(family):
    with CLASSES[family](num_envs=2, seed=3) as env:
        sizes = np.array([[0.012], [0.028]])
        state = env.reset(3, {"variation_values": {"cube.size": sizes, "cube.goal_yaw": np.array([0.5])}})
        assert np.array_equal(state["model", "compiled", "cube.size"].numpy(), sizes)
        assert (state["variation", "current", "cube.goal_yaw"] == 0.5).all()
        with pytest.raises(ValueError):
            env.reset(3, {"variation_values": {"cube.size": np.array([[0.012], [0.02], [0.028]])}})
        assert_same_state(state, env.get_state())


@pytest.mark.parametrize("cls", [CompatibleCube, CompatibleScene])
def test_metadata_reads_and_checker_construction_are_controlled(cls):
    a, b = cls(seed=42), cls(seed=42)
    try:
        assert a._model is b._model is None
        box = a.action_space
        for _ in range(5):
            assert a.action_space is box
        wa, wb = gym.wrappers.PassiveEnvChecker(a), gym.wrappers.PassiveEnvChecker(b)
        assert wa.unwrapped is a and wb.unwrapped is b
        assert a._model is not None and b._model is not None
        for _ in range(5):
            _ = wa.action_space
            _ = wa.observation_space
        assert np.array_equal(a.data.qpos, b.data.qpos)
        for e in (a, b):
            e.reset()
        assert np.array_equal(a.data.qpos, b.data.qpos)
        assert a.np_random.bit_generator.state == b.np_random.bit_generator.state
    finally:
        a.close()
        b.close()


def test_interleaved_threaded_isolation_and_global_numpy_nonmutation():
    outside = np.random.get_state()
    environments = [
        OGBManipulationAdapter(f, seed=s) for f, s in [("Cube", 17), ("Scene", 17), ("Cube", 42), ("Scene", 42)]
    ]
    initial = [e.get_state() for e in environments]
    try:
        serial = []
        for e in environments:
            for i in range(3):
                e.reset()
                e.step(np.full((1, 5), i * 0.1))
            serial.append(e.get_state())
        for e, s in zip(environments, initial):
            e.set_state(s)
        for i in range(3):
            for actor in reversed(range(4)):
                before = [e.get_state() for e in environments]
                environments[actor].reset()
                environments[actor].step(np.full((1, 5), i * 0.1))
                for other, e in enumerate(environments):
                    if other != actor:
                        assert_same_state(before[other], e.get_state())
        for e, s, expected in zip(environments, initial, serial):
            assert_same_state(e.get_state(), expected)
            e.set_state(s)
        barrier = Barrier(4)

        def worker(e):
            for i in range(3):
                barrier.wait()
                e.reset()
                e.step(np.full((1, 5), i * 0.1))
            return e.get_state()

        with ThreadPoolExecutor(max_workers=4) as pool:
            threaded = list(pool.map(worker, environments))
        for expected, got in zip(serial, threaded):
            assert_same_state(expected, got)
        numpy_equal(outside, np.random.get_state())
    finally:
        for e in environments:
            e.close()


@pytest.mark.parametrize("family", CLASSES)
def test_step_delegation_substeps_and_no_implicit_restore(family):
    with CLASSES[family]() as env:
        raw = env._simulator.envs[0]
        assert raw.step.__func__ is ManipSpaceEnv.step
        assert raw.set_control.__func__ is ManipSpaceEnv.set_control
        records = []
        original = mujoco.mj_step

        def observe(*args, **kwargs):
            records.append(kwargs.get("nstep", 1))
            return original(*args, **kwargs)

        before = raw.data.time
        with (
            patch.object(mujoco, "mj_step", observe),
            patch.object(env._simulator, "set_state", side_effect=AssertionError("implicit restore")),
        ):
            env.step(np.array([[2.0, -3.0, 0.2, 0.8, -0.7]], np.float64), frames=None)
        assert records == [25] and raw.data.time == pytest.approx(before + 0.05)
        assert np.array_equal(raw.unnormalize_action(np.ones(5)), np.array([0.05, 0.05, 0.05, 0.3, 1.0]))


@pytest.mark.swm
@pytest.mark.parametrize("interaction", ["cube", "button", "drawer", "window"])
def test_contact_histories_oracle_and_independent_restore(interaction, oracle):
    family = "Cube" if interaction == "cube" else "Scene"
    raw_cls = oracle.OwnedCube if family == "Cube" else oracle.OwnedScene
    raw = raw_cls(entropy_seed=17)
    try:
        with CLASSES[family](seed=17) as a, CLASSES[family](seed=42) as b:
            a.reset(17, {"task_id": 1, "variation": []})
            raw.reset(seed=17, options={"task_id": 1, "variation": []})
            for _ in range(4):
                b.step(np.array([[0.4, -0.3, 0.2, 0.6, -0.8]]), frames=None)
            u = a._simulator.envs[0]
            if interaction == "cube":
                target = u.data.joint("object_joint_0").qpos[:3].copy()
                target[2] = 0.035
            elif interaction == "button":
                target = u.data.site_xpos[u._button_site_ids[0]].copy()
                target[2] = 0.06
            else:
                target = u.data.site_xpos[getattr(u, "_" + interaction + "_site_id")].copy()
            contacts, positions, buttons = set(), [], []
            saved = frozen = None
            for i in range(50):
                goal = target.copy()
                if i < 12:
                    goal[2] += 0.09
                if i >= 32:
                    goal[{"drawer": 1, "window": 0, "cube": 2}.get(interaction, 2)] += {
                        "drawer": -0.12,
                        "window": 0.18,
                        "cube": 0.15,
                    }.get(interaction, 0.0)
                action = np.zeros(5, dtype=np.float64 if i % 2 else np.float32)
                action[:3] = np.clip((goal - u.data.site_xpos[u._pinch_site_id]) / 0.05, -1, 1)
                action[4] = 1.0 if interaction == "button" or i >= 24 else -1.0
                if i == 20:
                    saved = a.get_state()
                    frozen = saved.clone()
                    b.set_state(saved)
                    assert_same_state(saved, b.get_state())
                out = a.step(action[None], frames=None)
                raw.step(action)
                assert_oracle(u, raw, oracle)
                if i >= 20:
                    assert_same_state(out, b.step(action[None], frames=None))
                for contact in u.data.contact:
                    pair = tuple(
                        u.model.body(int(u.model.geom_bodyid[g])).name for g in (contact.geom1, contact.geom2)
                    )
                    if any("ur5e/" in body for body in pair) and any(
                        "ur5e/" not in body and body != "world" for body in pair
                    ):
                        contacts.add(" / ".join(pair))
                if interaction == "cube":
                    positions.append(float(u.data.joint("object_joint_0").qpos[2]))
                else:
                    positions.append(
                        float(
                            u.data.joint(
                                "buttonbox_joint_0" if interaction == "button" else interaction + "_slide"
                            ).qpos[0]
                        )
                    )
                    buttons.append(u._cur_button_states.copy().tolist())
            assert_same_state(saved, frozen)
            assert contacts
            if interaction == "cube":
                assert max(positions) > 0.12  # Actual grasp and lift, not free-space approach.
            elif interaction == "button":
                assert min(positions) <= -0.02 and buttons[0] != buttons[-1]
            else:
                assert max(positions) - min(positions) > 0.005
            # Same-instance replay after a different controller/contact history.
            a.set_state(saved)
            b.set_state(saved)
            for i in range(8):
                action = np.array([[0.2, -0.3, -0.1, 0.35, 0.4]], np.float32 if i % 2 else np.float64)
                assert_same_state(a.step(action, frames=None), b.step(action, frames=None))
    finally:
        raw.close()


@pytest.mark.parametrize("family", CLASSES)
def test_solver_and_cached_pose_fields_are_behaviorally_relevant(family):
    with CLASSES[family](seed=17) as a, CLASSES[family](seed=17) as b:
        a.reset(17, {"task_id": 1 if family == "Cube" else 2, "variation": []})
        u = a._simulator.envs[0]
        fixed_target = u.data.joint("object_joint_0").qpos[:3].copy()
        fixed_target[2] = 0.125
        for i in range(10 if family == "Cube" else 20):
            target = fixed_target.copy() if family == "Cube" else u.data.joint("object_joint_0").qpos[:3].copy()
            if family == "Scene":
                target[2] = 0.035
            act = np.zeros((1, 5))
            act[0, :3] = np.clip((target - u.data.site_xpos[u._pinch_site_id]) / 0.05, -1, 1)
            act[0, 4] = 1 if family == "Scene" and i >= 9 else -1
            a.step(act, frames=None)
        saved = a.get_state()
        b.set_state(saved)
        b._simulator.envs[0].data.site_xpos[u._pinch_site_id, 0] += 0.001
        action = np.array([[0.2, -0.3, -0.1, 0.35, 0.4]])
        if family == "Cube":
            action[0, :3] = np.clip((fixed_target - u.data.site_xpos[u._pinch_site_id]) / 0.05, -1, 1)
            action[0, 3:] = [0.0, -1.0]
        baseline = a.step(action, frames=None)
        changed = b.step(action, frames=None)
        assert not torch.equal(baseline["state", "physics", "qpos"], changed["state", "physics", "qpos"])
        a.set_state(saved)
        b.set_state(saved)
        if family == "Cube":
            b._simulator.envs[0].data.qacc_warmstart[:] = 0
        else:
            b._simulator.envs[0].data.qacc_warmstart[:] += 1e6
        baseline, changed = a.step(action, frames=None), b.step(action, frames=None)
        assert not torch.equal(baseline["state", "physics", "qpos"], changed["state", "physics", "qpos"])


@pytest.mark.parametrize("family", CLASSES)
def test_success_boundaries_and_raw_lifecycle(family):
    cls = CompatibleCube if family == "Cube" else CompatibleScene
    env = cls(seed=17)
    try:
        env.reset(seed=17, options={"task_id": 1, "variation": []})
        for distance, expected in [(np.nextafter(0.04, 0), True), (0.04, True), (np.nextafter(0.04, np.inf), False)]:
            env.data.mocap_pos[0] = 0.0
            env.data.joint("object_joint_0").qpos[:3] = [distance, 0, 0]
            if family == "Scene":
                env._cur_button_states[:] = env._target_button_states
                env.data.joint("drawer_slide").qpos[0] = env._target_drawer_pos
                env.data.joint("window_slide").qpos[0] = env._target_window_pos
            assert (
                bool(env._compute_successes()[0] if family == "Cube" else env._compute_successes()[0][0]) == expected
            )
        env._success = True
        assert env.compute_reward() == 1.0 and env.terminate_episode()
        env._success = False
        assert env.compute_reward() == 0.0 and not env.terminate_episode()
        assert not env.truncate_episode()
        # No raw wrapper/counter adds World's separate 100-step limit.
        env.reset(seed=17, options={"task_id": 1, "variation": []})
        for _ in range(101):
            assert env.step(np.zeros(5))[3] is False
        assert not hasattr(env, "_elapsed_steps")
    finally:
        env.close()


@pytest.mark.parametrize("family", CLASSES)
def test_seed_metadata_is_not_needed_after_arbitrary_width_seed(family):
    with CLASSES[family](seed=2**300 + 17) as a, CLASSES[family](seed=0) as b:
        state = a.get_state()
        b.set_state(state)
        assert_same_state(a.reset(), b.reset())
        assert_same_state(a.step(np.zeros((1, 5)), frames=None), b.step(np.zeros((1, 5)), frames=None))


@pytest.mark.swm
@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("size", [np.array([0.01]), np.array([0.025], np.float32), [0.025], [0.03]])
def test_size_dtype_model_reconstruction_and_oracle(family, size, oracle):
    raw_cls = oracle.OwnedCube if family == "Cube" else oracle.OwnedScene
    raw = raw_cls(entropy_seed=17)
    options = {"task_id": 2, "variation": [], "variation_values": {"cube.size": size}}
    try:
        with CLASSES[family](seed=17) as a, CLASSES[family](seed=42) as b:
            a.reset(17, options)
            raw.reset(seed=17, options=options)
            assert_oracle(a._simulator.envs[0], raw, oracle)
            b.set_state(a.get_state())
            assert_same_state(a.get_state(), b.get_state())
            assert_same_state(a.step(np.ones((1, 5)) * 0.1, frames=None), b.step(np.ones((1, 5)) * 0.1, frames=None))
            assert_same_state(a.reset(), b.reset())
    finally:
        raw.close()


@pytest.mark.swm
@pytest.mark.parametrize("family", CLASSES)
def test_owned_variation_child_streams_and_no_recompilation_branch(family, oracle):
    cls = CompatibleCube if family == "Cube" else CompatibleScene
    raw_cls = oracle.OwnedCube if family == "Cube" else oracle.OwnedScene
    a, b = cls(seed=17), raw_cls(entropy_seed=17)
    try:
        for i, options in enumerate(
            [
                {"variation": []},
                {"variation": [], "variation_values": {"agent.color": np.array([0.2, 0.4, 0.6])}},
                {"variation": [], "variation_values": {"agent.color": np.array([0.3, 0.5, 0.7])}},
                {},
            ]
        ):
            options["task_id"] = 2
            old_a, old_b = a._model, b._model
            a.reset(options=options)
            b.reset(options=options)
            assert_oracle(a, b, oracle)
            assert (a._model is old_a) == (b._model is old_b)
            for name, node in oracle.spaces(a.variation_space):
                counterpart = dict(oracle.spaces(b.variation_space))[name]
                assert node.np_random.bit_generator.state == counterpart.np_random.bit_generator.state
    finally:
        a.close()
        b.close()


@pytest.mark.swm
@pytest.mark.parametrize("family", CLASSES)
@pytest.mark.parametrize("seed", [0, 17, 42])
def test_identical_genuine_box_settling_actions(family, seed, oracle):
    cls = CompatibleCube if family == "Cube" else CompatibleScene
    raw_cls = oracle.OwnedCube if family == "Cube" else oracle.OwnedScene
    a, b = cls(seed=seed), raw_cls(entropy_seed=seed)
    try:
        boxes = [a.action_space, b.action_space]
        tape = [[], []]
        original = gym.spaces.Box.sample

        def observe(box, *args, **kwargs):
            result = original(box, *args, **kwargs)
            for i, owned in enumerate(boxes):
                if box is owned:
                    tape[i].append(result.copy())
            return result

        with patch.object(gym.spaces.Box, "sample", observe):
            a.reset(seed=seed)
            b.reset(seed=seed)
        assert gym.spaces.Box.sample is original
        assert len(tape[0]) == len(tape[1]) == 2
        assert all(np.array_equal(x, y) for x, y in zip(*tape))
        assert_oracle(a, b, oracle)
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("family", CLASSES)
def test_restore_failure_cannot_partially_commit_a_batch(family):
    cls = CompatibleCube if family == "Cube" else CompatibleScene
    with CLASSES[family](num_envs=2, seed=17) as a:
        saved = a.get_state()
        for field, value in [(("controller", "site_xmat"), 0.0), (("task", "reward_id"), 5)]:
            bad = saved.clone()
            bad[field][1] = value
            with pytest.raises(ValueError):
                a.set_state(bad)
            assert_same_state(saved, a.get_state())
        bad = saved.clone()
        bad["empty_extra"] = TensorDict({}, [2])
        with pytest.raises(ValueError):
            a.set_state(bad)
        assert_same_state(saved, a.get_state())
        old = list(a._simulator.envs)
        calls = []
        inherited = cls.compile_model_and_data

        def fail_second(env):
            calls.append(env)
            if len(calls) == 2:
                raise RuntimeError("test compilation failure")
            return inherited(env)

        with patch.object(cls, "compile_model_and_data", fail_second):
            with pytest.raises(RuntimeError, match="test compilation failure"):
                a.set_state(saved)
        assert a._simulator.envs == old
        assert_same_state(saved, a.get_state())


def test_scene_drawer_window_and_button_success_boundaries():
    env = CompatibleScene(seed=17)
    try:
        env.reset(seed=17, options={"task_id": 1, "variation": []})
        env._target_drawer_pos = env._target_window_pos = 0.0
        for distance, expected in [(np.nextafter(0.04, 0), True), (0.04, True), (np.nextafter(0.04, np.inf), False)]:
            env.data.joint("drawer_slide").qpos[0] = -distance
            env.data.joint("window_slide").qpos[0] = distance
            successes = env._compute_successes()
            assert bool(successes[2]) == expected and bool(successes[3]) == expected
        for bits in ([0, 0], [0, 1], [1, 0], [1, 1]):
            env._cur_button_states = np.array(bits)
            env._target_button_states = np.array([1, 0])
            assert env._compute_successes()[1] == [bits[0] == 1, bits[1] == 0]
    finally:
        env.close()
