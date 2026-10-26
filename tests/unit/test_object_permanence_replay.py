"""Replay checkpoints describe a trajectory cursor, without physics engines."""

import subprocess
import sys

import pytest
import torch

from stable_envs.envs.object_permanence.replay import G34Replay


def test_replay_restores_after_unrelated_history_and_owns_snapshots():
    with G34Replay() as source, G34Replay() as other:
        checkpoint = source.advance(44)
        expected = source.poses(checkpoint)
        other.advance(110)
        assert other.set_state(checkpoint) is None
        assert torch.equal(other.poses(other.get_state()), expected)
        assert (source.advance(7) == other.advance(7)).all()
        checkpoint["replay", "frame"].zero_()
        expected.zero_()
        assert int(other.get_state()["replay", "frame"][0, 0]) == 51
        assert other.poses(other.get_state())[0, 0] != 0
        assert int(other.advance(1000)["replay", "frame"][0, 0]) == 119
        assert int(other.reset()["replay", "frame"][0, 0]) == 0
    with pytest.raises(RuntimeError):
        other.get_state()


def test_replay_rejects_wrong_trajectory_or_invalid_cursor_before_mutation():
    with G34Replay() as replay:
        saved = replay.advance(44)
        rows = replay.poses(replay.state_at(torch.arange(120)))
        rows[0, 0] += 0.1
        with G34Replay(rows) as different:
            with pytest.raises(ValueError, match="different trajectory"):
                different.set_state(saved)
        for bad in (-1, 120):
            broken = saved.clone()
            broken["replay", "frame"].fill_(bad)
            with pytest.raises(ValueError, match="outside"):
                replay.set_state(broken)
            assert (replay.get_state() == saved).all()
        with pytest.raises(ValueError):
            replay.set_state(rows)
        with pytest.raises(ValueError):
            replay.advance(-1)


def test_replay_and_scene_import_when_mujoco_is_unavailable():
    code = """
import sys
sys.modules['mujoco'] = None
sys.modules['nvdiffrast'] = None
from stable_envs.envs.object_permanence import G34Replay
from stable_envs.envs.object_permanence.rendering.g34_nvdiff import scene_arrays
with G34Replay() as replay:
    assert replay.poses(replay.advance(44)).shape == (1, 7)
    assert scene_arrays()[4].shape == (5, 4)
assert sys.modules['mujoco'] is None
assert sys.modules['nvdiffrast'] is None
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
