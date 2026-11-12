import numpy as np
import pytest

FIELDS = ("qpos", "qvel", "act", "qacc", "qacc_warmstart", "xpos", "xquat", "sensordata", "qfrc_bias")


@pytest.mark.family("dmc")
@pytest.mark.parametrize("seed", range(6))
def test_quadruped_height_search_leaves_the_data_of_dm_control(seed):
    pytest.importorskip("dm_control")
    from dm_control.suite import quadruped

    from stable_envs.envs.dmcontrol.simulation.aligned import _make_env, _non_contacting_height

    physics = _make_env("quadruped", seed).physics
    reference = physics.copy(share_model=False)
    orientation = np.random.default_rng(seed).normal(size=4)
    orientation /= np.linalg.norm(orientation)
    flags = int(physics.model.opt.disableflags)
    quadruped._find_non_contacting_height(reference, orientation)
    _non_contacting_height(physics, orientation)
    ours, theirs = physics.data.ptr, reference.data.ptr
    assert int(physics.model.opt.disableflags) == flags
    assert ours.ncon == theirs.ncon == 0
    for name in FIELDS:
        assert np.array_equal(getattr(ours, name), getattr(theirs, name)), name
    assert [w.number for w in ours.warning] == [w.number for w in theirs.warning]


@pytest.mark.family("dmc")
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("domain", ["cheetah", "quadruped"])
def test_batch_reset_leaves_the_data_of_dm_control(domain, seed):
    pytest.importorskip("dm_control")
    from stable_envs.envs.dmcontrol.simulation.factory import make_batch, make_local

    batch, local = make_batch(domain, 4, threads=2), make_local(domain, 4)
    try:
        batch.reset(seed=seed)
        local.reset(seed=seed)
        for ours, theirs in zip(batch.lanes.envs, local.envs, strict=True):
            a, b = ours.physics.data.ptr, theirs.physics.data.ptr
            for name in (*FIELDS, "time", "ctrl"):
                assert np.array_equal(getattr(a, name), getattr(b, name)), name
            assert [w.number for w in a.warning] == [w.number for w in b.warning]
            assert str(ours.task.random.get_state()) == str(theirs.task.random.get_state())
            assert ours._env._step_count == theirs._env._step_count == 0
    finally:
        batch.close()
        local.close()


@pytest.mark.family("dmc")
@pytest.mark.parametrize("seed", [None, 4])
@pytest.mark.parametrize("domain", ["cheetah", "quadruped"])
def test_batch_reset_state_holds_the_task_rng_of_dm_control(domain, seed):
    pytest.importorskip("dm_control")
    import torch

    from stable_envs.envs.dmcontrol.simulation.factory import make_batch, make_local

    batch, local = make_batch(domain, 3, threads=2), make_local(domain, 3)
    try:
        for _ in range(2):
            ours, theirs = batch.reset(seed=seed), local.reset(seed=seed)
            for key in ("keys", "position", "has_gauss", "cached_gaussian"):
                assert torch.equal(ours["rng", key], theirs["rng", key]), key
            for key in ("qpos", "qvel", "time", "warmstart"):
                assert torch.equal(ours[key], theirs[key]), key
    finally:
        batch.close()
        local.close()
