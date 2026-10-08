"""Oscar-only B2/P1/P2 source/public frame gate; no timing or scaling claims.

Run each family/device in a fresh interpreter. CPU requires software GL, CUDA
requires NVIDIA GL. Includes real source goal-reaching transitions, full owned
State, selected-frame ordering, and source-worker image transport. Never run
public MuJoCo rendering on the local Mac.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from benchmarks.online.classic_source_pipeline import image_errors
from benchmarks.online.common import PIN, equal_state
from benchmarks.online.dmc_source_pipeline import require_gl
from benchmarks.online.ogbench_parallel_source import OGBSource, ParallelOGBSource
from benchmarks.online.ogbench_source_terminal_gate import NAMES, reached_goal
from benchmarks.online.ogbench_source_transport_gate import equal
from tests.helpers.reference.ogbench_manipulation_semantics import assert_oracle


def public_tasks(result, horizon):
    tasks = {}
    for key, value in result.exclude('state', 'states', 'state_valid').items(True, True):
        name = '.'.join(key) if isinstance(key, tuple) else key
        # History/validity retain a time dimension even for a single action.
        tasks[name] = value.unsqueeze(1) if horizon == 1 and name not in ('valid', 'frames', 'frame_valid') else value
    return tasks


def compare(actual, expected, *, library=False):
    assert set(actual) == set(expected), (set(actual), set(expected))
    errors = []
    for key in expected:
        left = torch.as_tensor(actual[key]).cpu()
        right = torch.as_tensor(expected[key]).cpu()
        if key == 'frames' and library:
            errors = image_errors(left, right)
            assert all(error <= .001 for error in errors), errors
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', '--domain', dest='family', choices=NAMES, required=True)
    p.add_argument('--device', '--devices', dest='device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--verify-gl', action='store_true')
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if platform.system() != 'Linux' or not os.environ.get('SLURM_JOB_ID') or not args.verify_gl:
        p.error('Run in an Oscar compute allocation with --verify-gl; real public/source constructors open GL')
    if args.output.exists():
        raise FileExistsError(args.output)
    family, device = args.family, args.device
    torch.set_num_threads(1)
    if device == 'cuda':
        assert '3090' in torch.cuda.get_device_name()
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest['commit'] == PIN
    for rel, digest in manifest['files'].items():
        assert hashlib.sha256((args.source_root / rel).read_bytes()).hexdigest() == digest, rel
    paths = [Path(__file__).relative_to(Path.cwd()),
             *Path('benchmarks/online').glob('ogbench*source*.py'),
             Path('benchmarks/online/classic_parallel_source.py'),
             Path('benchmarks/online/classic_source_pipeline.py'),
             Path('benchmarks/online/dmc_source_pipeline.py'),
             Path('benchmarks/online/common.py'),
             Path('tests/helpers/reference/ogbench_maze_semantics.py'),
             Path('tests/helpers/reference/ogbench_manipulation_semantics.py'),
             Path('tests/helpers/reference/ogbench_manipulation_source.py'),
             *Path('stable_envs').rglob('*.py')]
    report = dict(status='RUNNING', family=family, device=device, size=64, batch=2, workers=2,
        host=platform.node(), job=os.environ['SLURM_JOB_ID'], pid=os.getpid(),
        affinity=sorted(os.sched_getaffinity(0)), source_pin=PIN, source_files=manifest['files'],
        gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
        versions={n: importlib.metadata.version(n) for n in
                  ('ogbench', 'mujoco', 'dm-control', 'torch', 'numpy', 'gymnasium', 'tensordict')},
        candidate_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        runtime_files={}, construction='PUBLIC', no_timing=True, source_workers=[1, 2],
        source_gl_in_workers_only=True, independent_no_render_physics=True,
        scope='B2 source/public selected images and terminal lifecycle; not production P8/scaling acceptance',
        fixture='seed17+lane; move goal to current configuration in zero/one/both lanes, without source patches',
        cases=[], worker_closures=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    try:
        for horizon in (1, 8):
            for selection, frame_slice in (('none', None), ('empty', slice(9, 12)),
                    ('final', slice(-1, None)), ('all', slice(None)), ('reverse_sparse', slice(None, None, -2))):
                indices = None if frame_slice is None else tuple(range(horizon))[frame_slice]
                with contextlib.closing(OGBSource(family, args.source_root, 2, render_size=64)) as physics, \
                        contextlib.closing(se.make(NAMES[family], num_envs=2, device=device, seed=17,
                                                  size=64, backend='auto' if family == 'Maze' else 'opengl')) as env, \
                        ParallelOGBSource(family, args.source_root, 2, 1, horizon,
                                          frame_indices=indices, render_size=64) as serial, \
                        ParallelOGBSource(family, args.source_root, 2, 2, horizon,
                                          frame_indices=indices, render_size=64) as workers:
                    for owner in (serial, workers):
                        owner.enable_render_audit()
                    before_gl = [owner.snapshot() for owner in (serial, workers)]
                    source_gl, worker_gl = serial.renderer_info()[0], list(workers.renderer_info())
                    for before, owner in zip(before_gl, (serial, workers), strict=True):
                        equal(before, owner.snapshot())
                    require_gl(source_gl, device)
                    assert len(worker_gl) == 2
                    for identity in worker_gl:
                        require_gl(identity, device)
                    for owner in (serial, workers):
                        report['runtime_files'].update(owner.runtime_files)
                    original_source, original_public = physics.snapshot(), env.get_state()

                    def witness():
                        if family == 'Maze':
                            equal_state(env.get_state().cpu(), physics.reference.get_state())
                        else:
                            for candidate, raw in zip(env._simulator.envs, physics.envs, strict=True):
                                assert_oracle(candidate, raw, physics.module)

                    for reached in (0, 1, 2):
                        physics.restore(original_source)
                        env.set_state(original_public)
                        for lane in range(reached):
                            reached_goal(physics.envs[lane].env if family == 'Maze' else physics.envs[lane], family)
                            reached_goal(env._simulator.envs[lane], family)
                        witness()
                        initial_source, initial_public = physics.snapshot(), env.get_state()
                        env.set_state(initial_public)
                        equal_state(initial_public, env.get_state())
                        for owner in (serial, workers):
                            owner.restore(initial_source)
                            equal(initial_source, owner.snapshot())
                        # Goal fixtures use neutral controls. The nonterminal
                        # fixture also exercises actual motion and frame changes.
                        actions = (np.zeros((2, horizon, workers.action_dim), np.float64) if reached else
                                   np.random.default_rng(31).uniform(-.5, .5, (2, horizon, workers.action_dim)))
                        physics_tasks = physics.step(actions)
                        expected = serial.step(actions)
                        actual = workers.step(actions, device)
                        result = env.step(actions[:, 0] if horizon == 1 else actions,
                                          states=slice(None), frames=frame_slice)
                        compare(actual, expected)
                        compare({k: v for k, v in expected.items() if k not in ('frames', 'frame_valid')}, physics_tasks)
                        errors = compare(public_tasks(result, horizon), expected, library=True)
                        assert len(errors) == 2 * len(indices or ())
                        if indices is not None:
                            assert result['frames'].device.type == actual['frames'].device.type == device
                        for owner in (serial, workers):
                            equal(physics.snapshot(), owner.snapshot())
                        equal_state(result['state'], env.get_state())
                        valid = torch.as_tensor(expected['valid'], device=device)
                        assert torch.equal(result['state_valid'], valid)
                        if reached:
                            assert bool(torch.as_tensor(expected['terminated'][:reached, 0]).all())
                            assert not bool(valid[:reached, 1:].any())
                            for lane in range(reached):
                                for t in range(1, horizon):
                                    equal_state(result['states'][lane, 0], result['states'][lane, t])
                        else:
                            assert bool(valid.all())
                        held_tasks = {key: value.clone() for key, value in actual.items()}
                        held_public = result.clone()
                        continuation = np.full_like(actions, .01)
                        next_physics = physics.step(continuation)
                        next_expected = serial.step(continuation)
                        next_actual = workers.step(continuation, device)
                        next_public = env.step(continuation[:, 0] if horizon == 1 else continuation,
                                               states=None, frames=frame_slice)
                        compare(next_actual, next_expected)
                        compare({k: v for k, v in next_expected.items() if k not in ('frames', 'frame_valid')}, next_physics)
                        next_errors = compare(public_tasks(next_public, horizon), next_expected, library=True)
                        compare(actual, held_tasks)
                        # TensorDict leaves include the complete owned State,
                        # state history, frames and validity, not just tasks.
                        equal_state(result, held_public)
                        for owner in (serial, workers):
                            equal(physics.snapshot(), owner.snapshot())
                        # This witness reads action_space and increments an
                        # instrumentation counter; keep it after snapshot checks.
                        witness()
                        assert bool(next_actual['valid'][:, 0].all())
                        inspections = [list(owner.vector.call('inspection')) for owner in (serial, workers)]
                        assert all(row['render_audit_count'] > 0 for group in inspections for row in group)
                        for raw in physics.envs:
                            target = raw.env if family == 'Maze' else raw
                            assert getattr(target, 'custom_renderer' if family == 'Maze' else '_renderer', None) is None
                        all_errors = errors + next_errors
                        report['cases'].append(dict(horizon=horizon, selection=selection, reached_lanes=reached,
                            source_gl=source_gl, worker_gl=worker_gl, frame_indices=indices,
                            valid_steps=int(valid.sum()), frames_checked=len(all_errors),
                            mse_mean=statistics.mean(all_errors) if all_errors else None,
                            mse_sd=statistics.stdev(all_errors) if len(all_errors) > 1 else None,
                            mse_max=max(all_errors) if all_errors else None,
                            source_worker_exact='PASS', source_engine_rng_witness='PASS',
                            source_render_nonmutation='PASS', source_inspection=inspections,
                            public_state_ownership='PASS', frame_order_validity_and_lifecycle='PASS',
                            initial_state_roundtrip='PASS'))
                        save()
                        print(family, device, horizon, selection, reached, 'PASS', flush=True)
                for owner in (serial, workers):
                    closure = [dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive()) for p in owner.vector.processes]
                    assert all(row['exitcode'] == 0 and not row['alive'] for row in closure)
                    report['worker_closures'].append(dict(horizon=horizon, selection=selection, workers=closure))
                    save()
        assert len(report['cases']) == 30 and len(report['worker_closures']) == 20
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
