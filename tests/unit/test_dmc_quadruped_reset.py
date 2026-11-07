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
