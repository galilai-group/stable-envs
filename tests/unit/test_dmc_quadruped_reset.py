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
def test_cheetah_batch_reset_leaves_the_data_of_dm_control(seed):
    pytest.importorskip("dm_control")
    from stable_envs.envs.dmcontrol.simulation.factory import make_batch, make_local

    batch, local = make_batch("cheetah", 4, threads=2), make_local("cheetah", 4)
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
