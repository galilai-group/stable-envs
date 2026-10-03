"""Synthetic benchmark-buffer checks; not source-image or GPU acceptance."""

import numpy as np
import pytest

from scripts.benchmarks.online_stepping.fetch_parallel_source import FetchSource, ParallelFetchSource


class World:
    def __init__(self, lane, done):
        self.lane, self.done = lane, done
        self.steps = self.renders = 0
        self.buffer = np.zeros((2, 2, 3), np.uint8)

    def step(self, action):
        self.steps += 1
        self.buffer.fill(20 * self.lane + self.steps)
        return None, np.float32(self.steps), False, self.steps >= self.done, {'is_success': np.float32(0)}

    def render(self):
        self.renders += 1
        return self.buffer


def actor(registration='swm/FetchReach-v3', done=(2, 99)):
    ref = FetchSource.__new__(FetchSource)
    ref.registration, ref.batch, ref.render_size = registration, 2, 2
    ref.envs = [World(i, limit) for i, limit in enumerate(done)]
    return ref


@pytest.mark.parametrize('indices', [None, (), (3,), (3, 1, 0), (0, 1, 2, 3)])
@pytest.mark.parametrize('done', [(2, 99), (1, 1)])
def test_owned_selected_frames_order_and_terminal_holds(indices, done):
    ref = actor(done=done)
    result = ref.step(np.zeros((2, 4, 4)), indices)
    valid = np.arange(4)[None, :] < np.array(done)[:, None]
    np.testing.assert_array_equal(result['valid'], valid)
    assert result['reward'].dtype == np.float32
    assert all(result['reward'][lane, t] == (t + 1 if valid[lane, t] else 0) for lane in range(2) for t in range(4))
    if indices is None:
        assert 'frames' not in result and 'frame_valid' not in result
    else:
        assert result['frames'].shape == (2, len(indices), 2, 2, 3)
        assert result['frames'].dtype == np.uint8
        np.testing.assert_array_equal(result['frame_valid'], valid[:, list(indices)])
        for lane in range(2):
            for k, t in enumerate(indices):
                np.testing.assert_array_equal(result['frames'][lane, k], np.full((2, 2, 3), lane * 20 + min(t + 1, done[lane])))
    assert [e.renders for e in ref.envs] == [len({min(t + 1, limit) for t in indices or ()}) for limit in done]
    held = {key: value.copy() for key, value in result.items()}
    ref.step(np.zeros((2, 1, 4)), (0,))
    assert [e.steps for e in ref.envs] == [min(4, limit) + 1 for limit in done]
    for key in held:
        np.testing.assert_array_equal(result[key], held[key])


@pytest.mark.parametrize('suffix', ['', 'Dense', 'Dict'])
def test_pickandplace_frames_fail_before_physics_or_worker_construction(suffix, monkeypatch):
    name = f'swm/FetchPickAndPlace{suffix}-v3'
    ref = actor(name)
    with pytest.raises(ValueError, match='changes continuation State'):
        ref.step(np.zeros((2, 4, 4)), (3,))
    assert all(e.steps == e.renders == 0 for e in ref.envs)
    def forbidden(*args, **kwargs):
        raise AssertionError('Must reject before constructing source worlds')
    monkeypatch.setattr(FetchSource, '__init__', forbidden)
    with pytest.raises(ValueError, match='changes continuation State'):
        ParallelFetchSource(name, '/unused', 2, 2, 4, frame_indices=(3,), render_size=64)


@pytest.mark.parametrize('indices', [None, ()])
def test_pickandplace_no_frame_path_remains_available(indices):
    ref = actor('swm/FetchPickAndPlace-v3')
    result = ref.step(np.zeros((2, 4, 4)), indices)
    assert result['valid'].sum() == 6 and all(e.renders == 0 for e in ref.envs)


@pytest.mark.parametrize('indices', [(0, 0), (-1,), (4,), (True,), (1.5,)])
def test_invalid_frame_selection_fails_before_mutation(indices):
    ref = actor()
    with pytest.raises(ValueError, match='Frame indices'):
        ref.step(np.zeros((2, 4, 4)), indices)
    assert all(e.steps == 0 for e in ref.envs)


@pytest.mark.parametrize('value', [np.zeros((2, 2, 3), np.float32), np.zeros((3, 3, 3), np.uint8)])
def test_no_hidden_cast_or_resize(value):
    ref = actor()
    ref.envs[0].buffer = value
    with pytest.raises(ValueError, match='frame shape or dtype'):
        ref.step(np.zeros((2, 1, 4)), (0,))


def test_gate_rejects_one_bad_frame_despite_low_average():
    import torch

    from scripts.benchmarks.online_stepping.fetch_source_frame_gate import compare

    source = torch.zeros((4, 8, 2, 2, 3), dtype=torch.uint8)
    candidate = source.clone()
    candidate[0, 0] = 20
    assert ((candidate.double() / 255) ** 2).mean() < .001
    with pytest.raises(AssertionError):
        compare({'frames': candidate}, {'frames': source}, library=True)
    candidate[0, 0] = 8
    errors = compare({'frames': candidate}, {'frames': source}, library=True)
    assert len(errors) == 32 and max(errors) == pytest.approx((8 / 255) ** 2)
    # Parallel source images must be exact; the library tolerance does not apply.
    with pytest.raises(AssertionError):
        compare({'frames': candidate}, {'frames': source})


@pytest.mark.parametrize('defect', ['dtype', 'shape', 'reward', 'missing'])
def test_image_gate_rejects_wrong_contract(defect):
    import torch

    from scripts.benchmarks.online_stepping.fetch_source_frame_gate import compare

    expected = {'frames': torch.zeros((2, 1, 2, 2, 3), dtype=torch.uint8), 'reward': torch.zeros(2, 1)}
    actual = {k: v.clone() for k, v in expected.items()}
    if defect == 'dtype':
        actual['frames'] = actual['frames'].float()
    elif defect == 'shape':
        actual['frames'] = actual['frames'][:, 0]
    elif defect == 'reward':
        actual['reward'][0] = .00001
    else:
        actual.pop('reward')
    with pytest.raises(AssertionError):
        compare(actual, expected, library=True)


@pytest.mark.parametrize('horizon, count', [(1, 0), (1, 1), (8, 0), (8, 4)])
def test_normalize_tasks_preserves_selection_axes(horizon, count):
    import torch
    from tensordict import TensorDict

    from scripts.benchmarks.online_stepping.fetch_source_transport_gate import tasks

    result = TensorDict({
        'reward': torch.zeros((2,) if horizon == 1 else (2, horizon)),
        'valid': torch.ones(2, horizon, dtype=torch.bool),
        'frames': torch.zeros(2, count, 2, 2, 3, dtype=torch.uint8),
        'frame_valid': torch.ones(2, count, dtype=torch.bool),
        'state': TensorDict({'x': torch.zeros(2, 3)}, batch_size=[2]),
    }, batch_size=[2])
    normalized = tasks(result, horizon)
    assert set(normalized) == {'reward', 'valid', 'frames', 'frame_valid'}
    assert normalized['reward'].shape == normalized['valid'].shape == (2, horizon)
    assert normalized['frames'].shape == (2, count, 2, 2, 3)
    assert normalized['frame_valid'].shape == (2, count)


@pytest.mark.parametrize('extra', [[], ['--source-root', '/unused'],
    ['--source-root', '/unused', '--source-manifest', '/unused', '--source-mode', 'ours'],
    ['--source-root', '/unused', '--source-manifest', '/unused', '--batches', '64'],
    ['--source-root', '/unused', '--source-manifest', '/unused', '--gates-only']])
def test_coordinator_rejects_incompatible_options_before_launch(extra, monkeypatch, tmp_path):
    import sys

    from scripts.benchmarks.online_stepping import dmc_state_export_oscar as coordinator

    def forbidden(*args, **kwargs):
        raise AssertionError('Must validate before launching any child')
    monkeypatch.setattr(coordinator.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(sys, 'argv', ['gate', '--benchmark', 'fetch_source_frame_gate',
                                     '--domain', 'swm/FetchReach-v3', '--output', str(tmp_path / 'out'), *extra])
    with pytest.raises(SystemExit) as error:
        coordinator.main()
    assert error.value.code == 2 and not (tmp_path / 'out').exists()
