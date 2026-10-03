"""Synthetic source-buffer contracts; these do not prove GL parity or speed."""

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.benchmarks.online_stepping.ogbench_parallel_source import OGBSource


class FakeWorld:
    def __init__(self, lane, done_at, truncated=False):
        self.lane, self.done_at, self.truncated = lane, done_at, truncated
        self.steps = self.renders = self.closes = 0
        self.buffer = np.zeros((2, 2, 3), np.uint8)
        self.cameras = []
        self.env = SimpleNamespace(custom_renderer=None)
        self._renderer = None

    def step(self, action):
        self.steps += 1
        # Reuse and mutate the renderer's buffer even on unrendered steps.
        self.buffer.fill(self.lane * 20 + self.steps)
        done = self.steps >= self.done_at
        info = {key: np.full(2, self.steps, np.float64)
                for key in ('xy', 'prev_qpos', 'prev_qvel', 'qpos', 'qvel')}
        info['success'] = done
        return np.full(2, self.steps, np.float64), float(self.steps), done and not self.truncated, done and self.truncated, info

    def render(self, **kwargs):
        self.renders += 1
        self.cameras.append(kwargs)
        return self.buffer

    def close(self):
        self.closes += 1


def source(family, done=(2, 99), truncated=False):
    result = OGBSource.__new__(OGBSource)
    result.family, result.batch, result.render_size = family, 2, 2
    result.envs = [FakeWorld(i, n, truncated) for i, n in enumerate(done)]
    result.reference = SimpleNamespace(_in_legacy_rng=lambda i, fn, action: fn(action))
    return result


def actions(family, horizon=4):
    return np.zeros((2, horizon, 2 if family == 'Maze' else 5), np.float64)


@pytest.mark.parametrize('family', ['Maze', 'Cube', 'Scene'])
@pytest.mark.parametrize('truncated', [False, True])
@pytest.mark.parametrize('indices', [(0, 1, 2, 3), (3, 1, 0), (3,), () , None])
def test_selected_frames_hold_terminal_endpoint_and_own_outputs(family, truncated, indices):
    actor = source(family, truncated=truncated)
    result = actor.step(actions(family), indices)
    np.testing.assert_array_equal(result['valid'], [[True, True, False, False], [True] * 4])
    np.testing.assert_array_equal(result['reward'], [[1., 2., 0., 0.], [1., 2., 3., 4.]])
    if indices is None:
        assert 'frames' not in result and 'frame_valid' not in result
    else:
        assert result['frames'].shape == (2, len(indices), 2, 2, 3)
        assert result['frames'].dtype == np.uint8
        np.testing.assert_array_equal(result['frame_valid'], result['valid'][:, list(indices)])
        for k, t in enumerate(indices):
            np.testing.assert_array_equal(result['frames'][0, k], np.full((2, 2, 3), min(t + 1, 2)))
            np.testing.assert_array_equal(result['frames'][1, k], np.full((2, 2, 3), 21 + t))
    selected = indices or ()
    assert [e.renders for e in actor.envs] == [len({min(t + 1, 2) for t in selected}), len(selected)]
    assert [e.steps for e in actor.envs] == [2, 4]
    for env in actor.envs:
        assert all(c == ({} if family == 'Maze' else {'camera': 'front_pixels'}) for c in env.cameras)
    held = {key: value.copy() for key, value in result.items()}
    # New calls preserve source behavior; they do not invent resets or freeze
    # lanes indefinitely because an earlier chunk terminated.
    actor.step(actions(family, 1), (0,))
    assert [e.steps for e in actor.envs] == [3, 5]
    for key in held:
        np.testing.assert_array_equal(result[key], held[key])


@pytest.mark.parametrize('family', ['Maze', 'Cube', 'Scene'])
def test_all_lanes_stop_but_requested_terminal_frames_are_returned(family):
    actor = source(family, done=(1, 1))
    result = actor.step(actions(family), (3, 2, 1, 0))
    assert [e.steps for e in actor.envs] == [1, 1]
    assert [e.renders for e in actor.envs] == [1, 1]
    np.testing.assert_array_equal(result['frame_valid'], [[False, False, False, True]] * 2)
    for lane in range(2):
        np.testing.assert_array_equal(result['frames'][lane], np.full((4, 2, 2, 3), lane * 20 + 1))


@pytest.mark.parametrize('indices', [(0, 0), (-1,), (4,), (True,), (1.0,)])
def test_invalid_frame_selection_rejected_before_advancing(indices):
    actor = source('Cube')
    with pytest.raises(ValueError, match='Frame indices'):
        actor.step(actions('Cube'), indices)
    assert all(e.steps == e.renders == 0 for e in actor.envs)


@pytest.mark.parametrize('buffer', [np.zeros((2, 2, 3), np.float32), np.zeros((3, 3, 3), np.uint8)])
def test_source_image_is_never_silently_resized_or_cast(buffer):
    actor = source('Scene')
    actor.envs[0].buffer = buffer
    with pytest.raises(ValueError, match='image shape/dtype'):
        actor.step(actions('Scene'), (0,))


@pytest.mark.parametrize('family', ['Maze', 'Cube', 'Scene'])
def test_benchmark_teardown_closes_dedicated_renderer_once(family):
    actor = source(family)
    envs = list(actor.envs)
    renderers = []
    for env in envs:
        renderer = FakeWorld(0, 1)
        target, field = (env.env, 'custom_renderer') if family == 'Maze' else (env, '_renderer')
        setattr(target, field, renderer)
        renderers.append((target, field, renderer))
    actor.close()
    actor.close()
    assert actor.envs == [] and actor.reference is None
    assert all(e.closes == 1 for e in envs)
    assert all(getattr(target, field) is None and renderer.closes == 1 for target, field, renderer in renderers)


@pytest.mark.parametrize('horizon', [1, 8])
@pytest.mark.parametrize('count', [0, 1])
def test_public_single_action_keeps_frame_and_validity_time_axes(horizon, count):
    import torch
    from tensordict import TensorDict

    from scripts.benchmarks.online_stepping.ogbench_source_frame_gate import public_tasks

    result = TensorDict({
        'reward': torch.zeros((2,) if horizon == 1 else (2, horizon)),
        'valid': torch.ones((2, horizon), dtype=torch.bool),
        'state_valid': torch.ones((2, horizon), dtype=torch.bool),
        'frame_valid': torch.ones((2, count), dtype=torch.bool),
        'frames': torch.zeros((2, count, 2, 2, 3), dtype=torch.uint8),
        'state': TensorDict({'x': torch.ones((2, 1))}, [2]),
    }, [2])
    tasks = public_tasks(result, horizon)
    assert set(tasks) == {'reward', 'valid', 'frame_valid', 'frames'}
    assert tasks['reward'].shape == (2, horizon)
    assert tasks['frames'].shape == (2, count, 2, 2, 3)
    assert tasks['frame_valid'].shape == (2, count)


def test_gate_rejects_source_transport_drift_and_excess_library_mse():
    from scripts.benchmarks.online_stepping.ogbench_source_frame_gate import compare

    expected = {'frames': np.zeros((2, 1, 2, 2, 3), np.uint8)}
    small = {'frames': np.ones((2, 1, 2, 2, 3), np.uint8)}
    large = {'frames': np.full((2, 1, 2, 2, 3), 255, np.uint8)}
    assert compare(small, expected, library=True) == pytest.approx([1 / 255 ** 2] * 2, rel=1e-15, abs=0)
    with pytest.raises(AssertionError):
        compare(small, expected)
    with pytest.raises(AssertionError):
        compare(large, expected, library=True)
    with pytest.raises(AssertionError):
        compare({}, expected, library=True)


@pytest.mark.parametrize('extra', [[], ['--source-root', '/not-used'],
    ['--source-root', '/not-used', '--source-manifest', '/not-used', '--source-mode', 'ours'],
    ['--source-root', '/not-used', '--source-manifest', '/not-used', '--batches', '64']])
def test_coordinator_rejects_incomplete_or_misapplied_gate_args(monkeypatch, tmp_path, extra):
    import sys

    from scripts.benchmarks.online_stepping.dmc_state_export_oscar import main

    output = tmp_path / 'must-not-create'
    monkeypatch.setattr(sys, 'argv', ['gate', '--benchmark', 'ogbench_source_frame_gate',
                                    '--domain', 'Cube', '--output', str(output), *extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize('family', ['Maze', 'Cube', 'Scene'])
@pytest.mark.parametrize('mutates', [False, True])
def test_render_audit_rejects_continuation_mutation(family, mutates):
    actor = source(family)
    actor._audit_render, actor.render_audit_count = True, 0
    actor.snapshot = lambda: [dict(steps=np.asarray(env.steps)) for env in actor.envs]
    if mutates:
        original = actor.envs[0].render
        def render(**kwargs):
            actor.envs[0].steps += 1
            return original(**kwargs)
        actor.envs[0].render = render
        with pytest.raises(AssertionError):
            actor.step(actions(family, 1), (0,))
        assert actor.render_audit_count == 0
    else:
        actor.step(actions(family, 1), (0,))
        assert actor.render_audit_count == 2
