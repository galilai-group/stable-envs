"""Pooled native resets preserve independent lane state and physics variation."""

from contextlib import closing

import numpy as np
import pytest
import torch

mjw = pytest.importorskip("mujoco_warp")
pytest.importorskip("dm_control")

from stable_envs.envs.dmcontrol.simulation.factory import FACTORIES, make_local
from stable_envs.envs.dmcontrol.simulation.mjwarp_reset import WarpResetPool, warp_batch_fields
from stable_envs.envs.dmcontrol.simulation.physics import LEAVES
from tests.helpers.reference.semantics import assert_same_state


def pool(key):
    return WarpResetPool(key, 3, 91000, *warp_batch_fields())


def assert_models_equal(pooled, native):
    for lane, env in enumerate(native.envs):
        model = env.physics.model.ptr
        for name, column in pooled.models.columns.items():
            np.testing.assert_array_equal(column[lane], getattr(model, name), err_msg=name)
        for name, column in pooled.models.options.items():
            np.testing.assert_array_equal(column[lane], getattr(model.opt, name), err_msg=name)
        for name, column in pooled.models.statistics.items():
            np.testing.assert_array_equal(column[lane], getattr(model.stat, name), err_msg=name)
        ids = pooled.models.fixed_indices
        np.testing.assert_array_equal(pooled.models.fixed_positions[lane], env.physics.data.geom_xpos[ids])
        np.testing.assert_array_equal(
            pooled.models.fixed_rotations[lane], env.physics.data.geom_xmat[ids].reshape(-1, 3, 3)
        )


@pytest.mark.parametrize("key", sorted(FACTORIES))
def test_pool_matches_initial_seeded_unseeded_and_subset_resets(key):
    with closing(pool(key)) as pooled, closing(make_local(key, 3, "cpu", 91000)) as native:
        assert len(pooled.envs) == 1
        assert_same_state(pooled.get_state(), native.get_state())
        assert_models_equal(pooled, native)
        for seed in (None, 91020, None):
            assert_same_state(pooled.reset(seed), native.reset(seed))
            assert_models_equal(pooled, native)
        saved = native.get_state()
        saved["rng", "has_gauss"].fill_(True)
        saved["rng", "cached_gaussian"][:, 0] = torch.tensor([0.125, -0.375, 0.75], dtype=torch.float64)
        saved["task", "cumulative_reward"][:, 0] = torch.tensor([11.0, 17.0, 23.0], dtype=torch.float64)
        pooled.set_state(saved)
        native.set_state(saved)
        assert_same_state(pooled.get_state(), native.get_state())
        held = pooled.get_state()
        # Deliberately visit lanes out of order, leaving the middle lane alone.
        pooled.commit(pooled.prepare_reset(lanes=[2, 0]))
        for lane in (2, 0):
            native.envs[lane].reset()
        assert_same_state(pooled.get_state(), native.get_state())
        assert_same_state(held, saved)
        assert_models_equal(pooled, native)
        assert_same_state(pooled.reset(None), native.reset(None))


@pytest.mark.parametrize("key", sorted(LEAVES))
def test_pool_preserves_per_lane_variation_and_restored_models(key):
    with closing(pool(key)) as pooled, closing(make_local(key, 3, "cpu", 91000)) as native:
        original = native.get_state()
        values = {leaf.path: [[leaf.low], [leaf.default], [leaf.high]] for leaf in LEAVES[key]}
        native.configure_physics(values)
        pooled.configure_physics(values)
        assert_models_equal(pooled, native)
        assert_same_state(pooled.reset(None), native.reset(None))
        assert_models_equal(pooled, native)
        pooled.set_state(original)
        native.set_state(original)
        assert_models_equal(pooled, native)
        assert_same_state(pooled.reset(None), native.reset(None))


def test_invalid_later_row_keeps_installed_state_models_and_rng():
    with (
        closing(pool("cartpole-balance")) as pooled,
        closing(make_local("cartpole-balance", 3, "cpu", 91000)) as native,
    ):
        before = pooled.get_state()
        bad = before.clone()
        bad["physics", "agent", "cart_mass"][:, 0] = torch.tensor([1.25, 1.0, -1.0])
        with pytest.raises(ValueError):
            pooled.set_state(bad)
        assert_same_state(pooled.get_state(), before)
        assert_models_equal(pooled, native)
        # The scratch owner may have been partially modified; the next reset
        # must reinstall a valid lane and consume exactly its saved RNG stream.
        assert_same_state(pooled.reset(None), native.reset(None))


def _sharded_pool(key, threads, monkeypatch, lanes=5):
    from stable_envs.envs.dmcontrol.simulation import mjwarp_reset

    monkeypatch.setattr(mjwarp_reset, "batch_threads", lambda count: min(threads, count))
    return WarpResetPool(key, lanes, 91000, *warp_batch_fields())


@pytest.mark.parametrize("key", ["cheetah", "quadruped"])
def test_threaded_domains_reset_shards_on_one_owner_per_thread(key, monkeypatch):
    with closing(_sharded_pool(key, 1, monkeypatch)) as single, closing(_sharded_pool(key, 3, monkeypatch)) as sharded:
        assert len(single._owners) == 1 and single._pool is None
        assert len(sharded._owners) == 3 and sharded._pool is not None
        assert len({id(owner.envs[0]) for owner in sharded._owners}) == 3
        assert_same_state(sharded.get_state(), single.get_state())
        for seed in (None, 91020, None):
            assert_same_state(sharded.reset(seed), single.reset(seed))
        # A subset smaller than the owner count leaves some owners idle.
        for lanes in ([4, 1], [0], [3, 0, 2, 1, 4]):
            sharded.commit(sharded.prepare_reset(lanes=lanes))
            single.commit(single.prepare_reset(lanes=lanes))
            assert_same_state(sharded.get_state(), single.get_state())
            for name, column in sharded.models.columns.items():
                np.testing.assert_array_equal(column, single.models.columns[name], err_msg=name)


def test_other_domains_reset_on_one_owner(monkeypatch):
    with closing(_sharded_pool("walker", 8, monkeypatch)) as pooled:
        assert len(pooled._owners) == 1 and pooled._pool is None
