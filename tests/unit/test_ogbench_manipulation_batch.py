"""Cube/Scene on native ``mujoco.batch``: equal to OGBench's per-lane step bit for bit, and the backend routing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stable_envs import get_field, schema_of

pytest.importorskip("ogbench")
pytest.importorskip("mujoco.batch")

from stable_envs.envs.ogbench.simulation import execution
from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
from stable_envs.envs.ogbench.simulation.manipulation_batch import OGBManipulationBatchAdapter
from stable_envs.envs.ogbench.simulation.manipulation_control import BatchController
from tests.helpers.reference.semantics import assert_same_state

FAMILIES = ("Cube", "Scene")


def _actions(rng, t, lanes):
    """Smooth sweeps, with pushes toward the table that reach the Scene buttons and full-scale jumps."""
    action = np.clip(0.9 * np.sin(t / 9 + np.arange(5)[None] + np.arange(lanes)[:, None]), -1, 1)
    action = action + rng.normal(0, 0.3, action.shape)
    if t % 10 < 4:
        action[:, 2] = -1.0
    if t % 13 == 0:
        action[: lanes // 2] = rng.choice([-1.0, 1.0], (lanes // 2, 5))
    if t % 17 == 0:
        action[-1] = 0.0
    return np.clip(action, -1, 1).astype(np.float32 if t % 2 else np.float64)


def _same_outputs(a, b):
    for key in ("reward", "success", "terminated", "truncated"):
        assert torch.equal(a[key], b[key]), key


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("reward_task_id", [None, 2])
@pytest.mark.parametrize("lanes", [1, 6])
def test_batch_step_equals_ogbench_lanes_bit_for_bit(family, reward_task_id, lanes):
    rng = np.random.default_rng(3)
    ref = OGBManipulationAdapter(family, lanes, seed=11, reward_task_id=reward_task_id)
    bat = OGBManipulationBatchAdapter(family, lanes, seed=11, reward_task_id=reward_task_id, threads=3)
    try:
        assert_same_state(bat.get_state(), ref.get_state())
        for t in range(120):
            action = _actions(rng, t, lanes)
            active = None if t % 7 or t == 0 else torch.from_numpy(rng.random(lanes) < 0.6)
            want, got = ref.step(action, active=active), bat.step(action, active=active)
            assert_same_state(got["state"], want["state"])
            _same_outputs(got, want)
            if t % 50 == 49:
                assert_same_state(bat.reset(), ref.reset())
    finally:
        ref.close()
        bat.close()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("lanes", [1, 4])
def test_batch_restore_and_variation_continue_bit_for_bit(family, lanes):
    rng = np.random.default_rng(8)
    ref = OGBManipulationAdapter(family, lanes, seed=2)
    bat = OGBManipulationBatchAdapter(family, lanes, seed=2, threads=2)
    try:
        for t in range(15):
            action = _actions(rng, t, lanes)
            ref.step(action)
            bat.step(action)
        saved = bat.get_state()
        sizes = np.array([0.015, 0.02, 0.025, 0.03])[:lanes, None]
        for adapter in (ref, bat):
            adapter.configure([{"cube.size": size} for size in sizes])
        assert_same_state(bat.get_state(), ref.get_state())
        for t in range(15, 40):
            action = _actions(rng, t, lanes)
            assert_same_state(bat.step(action)["state"], ref.step(action)["state"])
        for adapter in (ref, bat):
            adapter.set_state(saved)
        assert_same_state(bat.get_state(), ref.get_state())
        for t in range(40, 60):
            action = _actions(rng, t, lanes)
            assert_same_state(bat.step(action)["state"], ref.step(action)["state"])
    finally:
        ref.close()
        bat.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_threaded_physics_equals_one_thread(family):
    lanes, rng = 8, np.random.default_rng(6)
    serial = OGBManipulationBatchAdapter(family, lanes, seed=7, threads=1)
    threaded = OGBManipulationBatchAdapter(family, lanes, seed=7, threads=4)
    try:
        assert serial.physics.b.nthread == 1 and threaded.physics.b.nthread == 4
        for t in range(40):
            action = _actions(rng, t, lanes)
            assert_same_state(threaded.step(action)["state"], serial.step(action)["state"])
    finally:
        serial.close()
        threaded.close()


@pytest.mark.parametrize("lanes", [1, 2])
def test_scene_cube_repair_equals_ogbench(lanes):
    ref = OGBManipulationAdapter("Scene", lanes, seed=4)
    bat = OGBManipulationBatchAdapter("Scene", lanes, seed=4)
    try:
        for _ in range(2):
            # The last lane's cube leaves the workspace; post_step puts it back from the lane's generator.
            ref.envs[-1].data.qpos[bat._cube] = 5.0
            bat._qpos[-1, bat._cube] = 5.0
            action = np.zeros((lanes, 5))
            assert_same_state(bat.step(action)["state"], ref.step(action)["state"])
        assert_same_state(bat.reset(), ref.reset())
    finally:
        ref.close()
        bat.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_batch_with_worker_lanes_equals_ogbench_lanes(family):
    """Worker processes reset, restore and vary the lanes of large batches; the batch steps them."""
    from stable_envs.envs.ogbench.simulation.manipulation_process import OGBManipulationProcessAdapter

    lanes, rng = 4, np.random.default_rng(4)
    ref = OGBManipulationAdapter(family, lanes, seed=9)
    workers = OGBManipulationProcessAdapter(family, lanes, "cpu", 9, None, processes=2)
    bat = OGBManipulationBatchAdapter(family, lanes, seed=9, threads=2, lanes=workers)
    try:
        assert_same_state(bat.get_state(), ref.get_state())
        for t in range(60):
            action = _actions(rng, t, lanes)
            assert_same_state(bat.step(action)["state"], ref.step(action)["state"])
            if t == 29:
                assert_same_state(bat.reset(), ref.reset())
                for adapter in (ref, bat):
                    adapter.configure([{"cube.size": np.array([size])} for size in (0.015, 0.02, 0.025, 0.03)])
                assert_same_state(bat.get_state(), ref.get_state())
    finally:
        ref.close()
        bat.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_state_has_one_leaf_per_section_and_dtype(family):
    batch = OGBManipulationBatchAdapter(family, 2, seed=5)
    try:
        state = batch.step(np.zeros((2, 5)))["state"]
        schema = batch.schema
        assert schema_of(state) == schema and schema.name == "OGB" + family
        assert [k if isinstance(k, str) else "/".join(k) for k in state.keys(True, True)] == list(schema.order)
        assert all(
            key == f"{key.split('/')[0]}/{str(g.dtype).removeprefix('torch.')}" for key, g in schema.groups.items()
        )
        assert set(schema.names) == set(batch.fields)
        # The step writes its fields into the live groups through views cut once.
        for name in ("output/prev_qpos", "controller/site_xpos", "task/success"):
            assert np.shares_memory(schema.get(batch._live, name), batch._live[schema.slices[name].group])
            assert np.array_equal(get_field(state, name).numpy(), schema.get(batch._live, name))
        assert torch.equal(get_field(state, "physics.qpos"), torch.from_numpy(batch._qpos))
        assert np.array_equal(get_field(state, "variation/current/cube/size").numpy(), [[0.02]] * 2)
        before = state.clone()
        batch.step(np.ones((2, 5)))
        assert (state == before).all()
    finally:
        batch.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_batch_controller_writes_ogbench_controls(family):
    from stable_envs.envs.ogbench.simulation.compatibility import CompatibleCube, CompatibleScene, LanePool

    cls = CompatibleCube if family == "Cube" else CompatibleScene
    lanes, rng, pool = 5, np.random.default_rng(1), LanePool()
    envs = [cls(seed=i, pool=pool) for i in range(lanes)]
    for i, env in enumerate(envs):
        env.reset(seed=i)
    controller = BatchController(envs[0], lanes, 2)
    pinch = envs[0]._pinch_site_id
    try:
        for t in range(60):
            action = _actions(rng, t, lanes)
            qpos = np.stack([env.data.qpos for env in envs])
            ctrl = np.stack([env.data.ctrl for env in envs])
            pos = np.stack([env.data.site_xpos[pinch] for env in envs])
            mat = np.stack([env.data.site_xmat[pinch] for env in envs])
            controller.control(action, qpos, pos, mat, ctrl, np.arange(lanes))
            for i, env in enumerate(envs):
                env.transition(action[i])
                assert np.array_equal(ctrl[i], env.data.ctrl), (t, i)
    finally:
        controller.close()
        for env in envs:
            env.close()


@pytest.mark.parametrize(
    "batch,device,backend,switch,expected",
    [
        (1, "cpu", "auto", 1, "native"),
        (4096, "cpu", None, 1, "native"),
        (1, "cuda", "auto", 1, "mjwarp"),
        (8, "cuda:1", None, 1, "mjwarp"),
        (255, "cuda", "auto", 256, "native"),
        (256, "cuda", "auto", 256, "mjwarp"),
        (4096, "cpu", "auto", 256, "native"),
        (64, "cuda", "auto", 64, "mjwarp"),
        (4096, "cuda", "native", 256, "native"),
        (1, "cuda", "mjwarp", 256, "mjwarp"),
    ],
)
def test_manipulation_routing(monkeypatch, batch, device, backend, switch, expected):
    from stable_envs import tuning
    from stable_envs.envs.ogbench.simulation import manipulation_batch, manipulation_warp

    monkeypatch.setattr(manipulation_batch, "OGBManipulationBatchAdapter", lambda *a: ("native", a))
    monkeypatch.setattr(manipulation_warp, "OGBManipulationWarpAdapter", lambda *a: ("mjwarp", a))
    with tuning.override(large_backend_min_envs=switch):
        kind, args = execution.make_manipulation_adapter("Cube", batch, device, 17, 1, backend)
    assert kind == expected and args == ("Cube", batch, device, 17, 1)


def test_manipulation_routing_rejects_bad_options():
    with pytest.raises(ValueError, match="requires device='cuda'"):
        execution.make_manipulation_adapter("Cube", 8, "cpu", 0, None, "mjwarp")
    with pytest.raises(ValueError, match="simulation_backend"):
        execution.make_manipulation_adapter("Cube", 8, "cpu", 0, None, "mjx")
    with pytest.raises(ValueError, match="simulation_backend"):
        execution.make_manipulation_adapter("Cube", 8, "cuda", 0, None, "default")
