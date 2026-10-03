"""CPU B4/P2 Fetch source transport/dispatcher prerequisite; no GL or timing."""

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from scripts.benchmarks.online_stepping.fetch_parallel_source import FetchSource, ParallelFetchSource, import_source
from stable_envs._adapters.fetch_pickandplace import FetchAdapter
from stable_envs._stepping import step as dispatch_step


def fixture(task, reward):
    adapter = FetchAdapter(4, seed=17, task=task, reward_type=reward)
    env = SimpleNamespace(num_envs=4, device=torch.device('cpu'), has_dynamics=True,
        execution_mode='DELEGATED_CPU', action_spec=SimpleNamespace(shape=(4,), dtype=torch.float64),
        _simulator=adapter, _check_open=lambda _: adapter._check_open(),
        _step=adapter.step, _advance_without_snapshot=adapter.advance,
        get_state=adapter.get_state, set_state=adapter.set_state, close=adapter.close)
    env.step = partial(dispatch_step, env)
    return env


def tasks(result, horizon):
    return {key: value.unsqueeze(1) if horizon == 1 and key != 'valid' else value
            for key, value in result.exclude('state', 'states', 'state_valid').items()}


def compare(actual, expected):
    assert set(actual) == set(expected)
    for key in actual:
        torch.testing.assert_close(torch.as_tensor(actual[key]).cpu(), torch.as_tensor(expected[key]).cpu(), rtol=0, atol=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import_source(args.source_root)
    from scripts.benchmarks.online_stepping.fetch_source_render_lifecycle import IDS, PIN, differences
    from scripts.recovery.fetch_public_gate import equal_state, full_compare

    manifest = json.loads(args.source_manifest.read_text())
    assert manifest['commit'] == PIN
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((args.source_root / name).read_bytes()).hexdigest() == digest, name
    assert importlib.metadata.version('mujoco') == '3.11.0'
    torch.set_num_threads(1)
    files = [Path(__file__).relative_to(Path.cwd()),
             Path('scripts/benchmarks/online_stepping/fetch_parallel_source.py'),
             Path('scripts/benchmarks/online_stepping/fetch_source_render_lifecycle.py'),
             Path('scripts/benchmarks/online_stepping/classic_parallel_source.py'),
             Path('scripts/recovery/fetch_public_gate.py'), Path('scripts/recovery/fetch_pickandplace_adapter_probe.py'),
             Path('stable_envs/_stepping.py'), Path('stable_envs/_adapters/fetch_pickandplace.py'),
             Path('stable_envs/_adapters/fetch_observations.py')]
    report = dict(status='RUNNING', host=platform.node(), parent_pid=os.getpid(), source_pin=PIN,
        source_files=manifest['files'], candidate_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        versions={n: importlib.metadata.version(n) for n in
                  ('numpy', 'mujoco', 'gymnasium', 'gymnasium-robotics', 'torch', 'tensordict')},
        scope='CPU B4/P2 actual source shared task arrays; render-free adapter/dispatcher fixture',
        no_rendering=True, no_timing=True, state_rpc='untimed pickled independent source snapshots',
        runtime_files={}, cases=[], worker_closures=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    try:
        with patch.object(MujocoRenderer, 'render', side_effect=AssertionError('No GL in CPU transport gate')), \
                patch('mujoco.Renderer', side_effect=AssertionError('No GL in CPU transport gate')):
            for registration in IDS:
                task = registration.removeprefix('swm/Fetch').removesuffix('-v3').removesuffix('Dense').removesuffix('Dict')
                reward = 'dense' if 'Dense' in registration else 'sparse'
                for horizon in (1, 8):
                    with contextlib.closing(FetchSource(registration, args.source_root, 4)) as serial, \
                            ParallelFetchSource(registration, args.source_root, 4, 2, horizon) as parallel, \
                            contextlib.closing(fixture(task, reward)) as env:
                        assert parallel.vector.shared_memory and parallel.vector.copy
                        assert parallel.vector.autoreset_mode.name == 'DISABLED'
                        source_initial, public_initial = serial.snapshot(), env.get_state()
                        report['runtime_files'].update(parallel.runtime_files)
                        full_compare(public_initial, serial.envs)
                        for limits in ([50] * 4, [1, 3, 2, 50]):
                            initial = copy.deepcopy(source_initial)
                            for row, maximum in zip(initial, limits, strict=True):
                                row['max_episode_steps'] = np.asarray(maximum)
                            for dtype in (np.float32, np.float64):
                                serial.restore(initial)
                                parallel.restore(initial)
                                env._simulator._set_episode_limits(np.asarray(limits))
                                env.set_state(public_initial)
                                assert not differences(initial, serial.snapshot())
                                assert not differences(initial, parallel.snapshot())
                                full_compare(env.get_state(), serial.envs)
                                actions = np.random.default_rng(31).uniform(-1, 1, (4, horizon, 4)).astype(dtype)
                                expected = serial.step(actions)
                                actual = parallel.step(actions)
                                result = env.step(actions[:, 0] if horizon == 1 else actions, states=slice(None), frames=None)
                                compare(actual, expected)
                                compare(tasks(result, horizon), expected)
                                assert not differences(parallel.snapshot(), serial.snapshot())
                                full_compare(result['state'], serial.envs)
                                valid = torch.arange(horizon)[None, :] < torch.tensor(limits)[:, None]
                                assert torch.equal(actual['valid'], valid) and torch.equal(result['state_valid'], valid)
                                for lane, maximum in enumerate(limits):
                                    for t in range(maximum, horizon):
                                        equal_state(result['states'][lane, maximum - 1], result['states'][lane, t])
                                held_tasks = {key: value.clone() for key, value in actual.items()}
                                held_public = result.clone()
                                endpoint = copy.deepcopy(serial.snapshot())
                                for owner in (serial, parallel):
                                    owner.step(-actions)
                                continued = env.step(-actions[:, 0] if horizon == 1 else -actions, frames=None, states=None)
                                assert not differences(serial.snapshot(), parallel.snapshot())
                                full_compare(continued['state'], serial.envs)
                                compare(actual, held_tasks)
                                equal_state(result, held_public)
                                moved = any(not np.array_equal(old['qpos'], new['qpos'])
                                            for old, new in zip(endpoint, serial.snapshot(), strict=True))
                                assert moved
                                serial.restore(initial)
                                parallel.restore(initial)
                                compare(serial.step(actions), held_tasks)
                                compare(parallel.step(actions), held_tasks)
                                assert not differences(endpoint, serial.snapshot())
                                assert not differences(endpoint, parallel.snapshot())
                                # Change the source RNG history, restore it,
                                # then consume it through actual unseeded reset.
                                for owner in (serial, parallel):
                                    owner.reset_episode(seed=999)
                                    owner.restore(initial)
                                    owner.reset_episode()
                                env.set_state(public_initial)
                                full_compare(env._simulator.reset(), serial.envs)
                                assert not differences(serial.snapshot(), parallel.snapshot())
                                inspection = list(parallel.vector.call('inspection'))
                                assert len({row['pid'] for row in inspection}) == 2
                                assert all(row['pid'] != os.getpid() and row['viewers_absent'] and row['torch_threads'] == 1
                                           and row['start'] == i * 2 and row['size'] == 2 for i, row in enumerate(inspection))
                                report['cases'].append(dict(registration=registration, batch=4, workers=2, horizon=horizon,
                                    action_dtype=np.dtype(dtype).name, limits=limits, valid_steps=int(valid.sum()),
                                    source_tasks_snapshot='PASS', source_adapter_full_state='PASS',
                                    shared_output_ownership='PASS', restored_replay='PASS', unseeded_reset_rng='PASS',
                                    termination_padding_and_state_hold='PASS', continuation_without_autoreset='PASS', worker_info=inspection))
                                save()
                        print(registration, horizon, 'four source transport/dispatcher cases PASS', flush=True)
                    closure = [dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive()) for p in parallel.vector.processes]
                    assert all(row['exitcode'] == 0 and not row['alive'] for row in closure)
                    report['worker_closures'].append(dict(registration=registration, horizon=horizon, workers=closure))
                    save()
        assert len(report['cases']) == 96 and len(report['worker_closures']) == 24
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
