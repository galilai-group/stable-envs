"""Cube/Scene on native ``mujoco.batch``: equal to OGBench's per-environment step bit for bit, and the backend routing."""

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


def _actions(rng, t, instances):
    """Smooth sweeps, with pushes toward the table that reach the Scene buttons and full-scale jumps."""
    action = np.clip(0.9 * np.sin(t / 9 + np.arange(5)[None] + np.arange(instances)[:, None]), -1, 1)
    action = action + rng.normal(0, 0.3, action.shape)
    if t % 10 < 4:
        action[:, 2] = -1.0
    if t % 13 == 0:
        action[: instances // 2] = rng.choice([-1.0, 1.0], (instances // 2, 5))
    if t % 17 == 0:
        action[-1] = 0.0
    return np.clip(action, -1, 1).astype(np.float32 if t % 2 else np.float64)


def _same_outputs(a, b):
    for key in ("reward", "success", "terminated", "truncated"):
        assert torch.equal(a[key], b[key]), key


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("reward_task_id", [None, 2])
@pytest.mark.parametrize("instances", [1, 6])
def test_batch_step_equals_ogbench_environments_bit_for_bit(family, reward_task_id, instances):
    rng = np.random.default_rng(3)
    ref = OGBManipulationAdapter(family, instances, seed=11, reward_task_id=reward_task_id)
    bat = OGBManipulationBatchAdapter(family, instances, seed=11, reward_task_id=reward_task_id, threads=3)
    try:
        assert_same_state(bat.get_state(), ref.get_state())
        for t in range(120):
            action = _actions(rng, t, instances)
            active = None if t % 7 or t == 0 else torch.from_numpy(rng.random(instances) < 0.6)
            want, got = ref.step(action, active=active), bat.step(action, active=active)
            assert_same_state(got["state"], want["state"])
            _same_outputs(got, want)
            if t % 50 == 49:
                assert_same_state(bat.reset(), ref.reset())
    finally:
        ref.close()
        bat.close()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("instances", [1, 4])
def test_batch_restore_and_variation_continue_bit_for_bit(family, instances):
    rng = np.random.default_rng(8)
    ref = OGBManipulationAdapter(family, instances, seed=2)
    bat = OGBManipulationBatchAdapter(family, instances, seed=2, threads=2)
    try:
        for t in range(15):
            action = _actions(rng, t, instances)
            ref.step(action)
            bat.step(action)
        saved = bat.get_state()
        sizes = np.array([0.015, 0.02, 0.025, 0.03])[:instances, None]
        for adapter in (ref, bat):
            adapter.configure([{"cube.size": size} for size in sizes])
        assert_same_state(bat.get_state(), ref.get_state())
        for t in range(15, 40):
            action = _actions(rng, t, instances)
            assert_same_state(bat.step(action)["state"], ref.step(action)["state"])
        for adapter in (ref, bat):
            adapter.set_state(saved)
        assert_same_state(bat.get_state(), ref.get_state())
        for t in range(40, 60):
            action = _actions(rng, t, instances)
            assert_same_state(bat.step(action)["state"], ref.step(action)["state"])
    finally:
        ref.close()
        bat.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_threaded_physics_equals_one_thread(family):
    instances, rng = 8, np.random.default_rng(6)
    serial = OGBManipulationBatchAdapter(family, instances, seed=7, threads=1)
    threaded = OGBManipulationBatchAdapter(family, instances, seed=7, threads=4)
    try:
        assert serial.physics.b.nthread == 1 and threaded.physics.b.nthread == 4
        for t in range(40):
            action = _actions(rng, t, instances)
            assert_same_state(threaded.step(action)["state"], serial.step(action)["state"])
    finally:
        serial.close()
        threaded.close()


@pytest.mark.parametrize("instances", [1, 2])
def test_scene_cube_repair_equals_ogbench(instances):
    ref = OGBManipulationAdapter("Scene", instances, seed=4)
    bat = OGBManipulationBatchAdapter("Scene", instances, seed=4)
    try:
        for _ in range(2):
            # The last environment's cube leaves the workspace; post_step puts it back from the environment's generator.
            ref.envs[-1].data.qpos[bat._cube] = 5.0
            bat._qpos[-1, bat._cube] = 5.0
            action = np.zeros((instances, 5))
            assert_same_state(bat.step(action)["state"], ref.step(action)["state"])
        assert_same_state(bat.reset(), ref.reset())
    finally:
        ref.close()
        bat.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_batch_with_worker_environments_equals_ogbench_environments(family):
    """Worker processes reset, restore and vary the environments of large batches; the batch steps them."""
    from stable_envs.envs.ogbench.simulation.manipulation_process import OGBManipulationProcessAdapter

    instances, rng = 4, np.random.default_rng(4)
    ref = OGBManipulationAdapter(family, instances, seed=9)
    workers = OGBManipulationProcessAdapter(family, instances, "cpu", 9, None, processes=2)
    bat = OGBManipulationBatchAdapter(family, instances, seed=9, threads=2, instances=workers)
    try:
        assert_same_state(bat.get_state(), ref.get_state())
        for t in range(60):
            action = _actions(rng, t, instances)
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
    from stable_envs.envs.ogbench.simulation.compatibility import CompatibleCube, CompatibleScene, InstancePool

    cls = CompatibleCube if family == "Cube" else CompatibleScene
    instances, rng, pool = 5, np.random.default_rng(1), InstancePool()
    envs = [cls(seed=i, pool=pool) for i in range(instances)]
    for i, env in enumerate(envs):
        env.reset(seed=i)
    controller = BatchController(envs[0], instances, 2)
    pinch = envs[0]._pinch_site_id
    try:
        for t in range(60):
            action = _actions(rng, t, instances)
            qpos = np.stack([env.data.qpos for env in envs])
            ctrl = np.stack([env.data.ctrl for env in envs])
            pos = np.stack([env.data.site_xpos[pinch] for env in envs])
            mat = np.stack([env.data.site_xmat[pinch] for env in envs])
            controller.control(action, qpos, pos, mat, ctrl, np.arange(instances))
            for i, env in enumerate(envs):
                env.transition(action[i])
                assert np.array_equal(ctrl[i], env.data.ctrl), (t, i)
    finally:
        controller.close()
        for env in envs:
            env.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_controller_site_jacobian_equals_mujoco(family):
    import mujoco

    from stable_envs.envs.ogbench.simulation.compatibility import CompatibleCube, CompatibleScene, InstancePool

    cls = CompatibleCube if family == "Cube" else CompatibleScene
    env = cls(seed=0, pool=InstancePool())
    env.reset(seed=0)
    instances = 6
    controller = BatchController(env, instances, 2)
    try:
        if controller._jacobian is None:
            pytest.skip("Numba is not installed")
        b, rng = controller._ik, np.random.default_rng(4)
        controller._ik_qpos[:] = rng.uniform(-np.pi, np.pi, controller._ik_qpos.shape)
        b.forward(None, mujoco.mjtStage.mjSTAGE_POS)
        site_xpos = controller._ik_site_xpos[:, controller._site]
        for rows in (np.arange(instances), np.array([1, 4])):
            jacp, jacr = b.jac(controller._site_body, site_xpos[rows], None if len(rows) == instances else rows)
            ours = controller._jacobian(
                controller._ik_cdof, controller._ik_subtree_com, site_xpos, rows, controller._chain, controller._root
            )
            assert np.array_equal(ours, np.concatenate([jacp, jacr], axis=1))
    finally:
        controller.close()
        env.close()


@pytest.mark.parametrize(
    "batch,device,backend,expected",
    [
        (1, "cpu", "auto", "native"),
        (4096, "cpu", None, "native"),
        (1, "cuda", "auto", "native"),
        (8, "cuda:1", None, "native"),
        (256, "cuda", "auto", "native"),
        (4096, "cuda", "native", "native"),
        (1, "cuda", "mjwarp", "mjwarp"),
    ],
)
def test_manipulation_routing(monkeypatch, batch, device, backend, expected):
    from stable_envs.envs.ogbench.simulation import manipulation_batch, manipulation_warp

    monkeypatch.setattr(manipulation_batch, "OGBManipulationBatchAdapter", lambda *a: ("native", a))
    monkeypatch.setattr(manipulation_warp, "OGBManipulationWarpAdapter", lambda *a: ("mjwarp", a))
    kind, args = execution.make_manipulation_adapter("Cube", batch, device, 17, 1, backend)
    assert kind == expected and args == ("Cube", batch, device, 17, 1)


def test_manipulation_routing_rejects_bad_options():
    with pytest.raises(ValueError, match="requires device='cuda'"):
        execution.make_manipulation_adapter("Cube", 8, "cpu", 0, None, "mjwarp")
    with pytest.raises(ValueError, match="simulation_backend"):
        execution.make_manipulation_adapter("Cube", 8, "cpu", 0, None, "mjx")
    with pytest.raises(ValueError, match="simulation_backend"):
        execution.make_manipulation_adapter("Cube", 8, "cuda", 0, None, "default")
