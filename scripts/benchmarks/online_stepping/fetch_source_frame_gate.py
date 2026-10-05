"""Oscar-only source/public Fetch images; correctness only, never throughput.

Serial (P1) and parallel (P2) source renderers live in separate owner processes.
The parent keeps an independent no-render source physics witness and the library
renderer. Original internal 480px raster/resize behavior is preserved. PickAndPlace
is excluded from this rendered protocol because source rendering changes dynamics.
"""

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
from pathlib import Path

import mujoco
import numpy as np
import torch

import stable_envs as se
from scripts.benchmarks.online_stepping.classic_source_pipeline import image_errors
from scripts.benchmarks.online_stepping.dmc_source_pipeline import require_gl
from scripts.benchmarks.online_stepping.fetch_parallel_source import FetchSource, ParallelFetchSource, import_source
from scripts.benchmarks.online_stepping.fetch_source_transport_gate import tasks

REGISTRATIONS = [f'swm/Fetch{task}{suffix}-v3' for task in ('Reach', 'Push', 'Slide') for suffix in ('', 'Dense', 'Dict')]


def compare(actual, expected, *, library=False):
    assert set(actual) == set(expected)
    errors = []
    for key in actual:
        a, b = torch.as_tensor(actual[key]).cpu(), torch.as_tensor(expected[key]).cpu()
        if key == 'frames' and library:
            errors = image_errors(a, b)
            assert all(value <= .001 for value in errors), errors
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', '--domain', dest='registration', choices=REGISTRATIONS, required=True)
    p.add_argument('--device', '--devices', dest='device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--verify-gl', action='store_true')
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if platform.system() != 'Linux' or not os.environ.get('SLURM_JOB_ID') or not args.verify_gl:
        p.error('Requires an Oscar compute allocation and --verify-gl; no local rendering')
    if args.output.exists():
        raise FileExistsError(args.output)
    import_source(args.source_root)
    from scripts.benchmarks.online_stepping.fetch_source_render_lifecycle import PIN, differences
    from scripts.recovery.fetch_public_gate import equal_state, full_compare

    assert mujoco.__version__ == '3.11.0'
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest['commit'] == PIN
    for rel, digest in manifest['files'].items():
        assert hashlib.sha256((args.source_root / rel).read_bytes()).hexdigest() == digest, rel
    device, registration = args.device, args.registration
    torch.set_num_threads(1)
    if device == 'cuda':
        assert '3090' in torch.cuda.get_device_name()
    files = [Path(__file__).relative_to(Path.cwd()),
             *Path('scripts/benchmarks/online_stepping').glob('fetch*source*.py'),
             Path('scripts/benchmarks/online_stepping/classic_parallel_source.py'),
             Path('scripts/benchmarks/online_stepping/classic_source_pipeline.py'),
             Path('scripts/benchmarks/online_stepping/dmc_source_pipeline.py'),
             Path('tests/helpers/reference/fetch_semantics.py'), Path('scripts/recovery/fetch_public_gate.py'), Path('scripts/recovery/fetch_pickandplace_adapter_probe.py'),
             *Path('stable_envs').rglob('*.py')]
    report = dict(status='RUNNING', registration=registration, device=device, batch=4, source_workers=[1, 2],
        host=platform.node(), job=os.environ['SLURM_JOB_ID'], pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0)),
        gpu=torch.cuda.get_device_name() if device == 'cuda' else None,
        source_pin=PIN, source_files=manifest['files'], source_internal_size=480, output_size=64,
        versions={n: importlib.metadata.version(n) for n in
                  ('numpy', 'mujoco', 'gymnasium', 'gymnasium-robotics', 'torch', 'tensordict')},
        candidate_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, runtime_files={},
        no_timing=True, source_gl_in_workers_only=True, construction='PUBLIC_B4_LOCAL_ADAPTER',
        scope='B4 P1/P2 source images/transport and public local-adapter correctness; not scaling or P8 acceptance',
        cases=[], worker_closures=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    try:
        from OpenGL import GL

        # Complete this probe before any library cache creates a renderer.
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
        with mujoco.Renderer(model, height=32, width=32) as probe:
            probe._gl_context.make_current()
            vendor, renderer = GL.glGetString(GL.GL_VENDOR).decode(), GL.glGetString(GL.GL_RENDERER).decode()
            identity = dict(vendor=vendor, renderer=renderer,
                            software=any(tag in renderer.lower() for tag in ('llvmpipe', 'softpipe', 'swrast')))
            require_gl(identity, device)
            report['parent_gl'] = identity
        for horizon in (1, 8):
            for selection, frame_slice in (('none', None), ('empty', slice(20, 30)), ('final', slice(-1, None)),
                                           ('all', slice(None)), ('reverse_sparse', slice(None, None, -2))):
                indices = None if frame_slice is None else tuple(range(horizon))[frame_slice]
                with contextlib.closing(FetchSource(registration, args.source_root, 4)) as physics, \
                        ParallelFetchSource(registration, args.source_root, 4, 1, horizon,
                                            frame_indices=indices, render_size=64) as serial, \
                        ParallelFetchSource(registration, args.source_root, 4, 2, horizon,
                                            frame_indices=indices, render_size=64) as parallel, \
                        contextlib.closing(se.make(registration.removeprefix('swm/').removesuffix('-v3'),
                                                  num_envs=4, seed=17, size=64, device=device, backend='opengl')) as env:
                    source_initial, public_initial = physics.snapshot(), env.get_state()
                    identities = []
                    for owner in (serial, parallel):
                        owner.enable_render_audit()
                        before = owner.snapshot()
                        current = list(owner.renderer_info())
                        assert not differences(before, owner.snapshot())
                        for info in current:
                            require_gl(info, device)
                            assert info['internal_width'] == info['internal_height'] == 480 and info['output_size'] == 64
                        identities.append(current)
                        report['runtime_files'].update(owner.runtime_files)
                    full_compare(public_initial.cpu(), physics.envs)
                    assert not differences(source_initial, serial.snapshot())
                    for limits in ([50] * 4, [1, 3, 2, 50]):
                        initial = copy.deepcopy(source_initial)
                        for row, maximum in zip(initial, limits, strict=True):
                            row['max_episode_steps'] = np.asarray(maximum)
                        for owner in (physics, serial, parallel):
                            owner.restore(initial)
                        env._simulator._set_episode_limits(np.asarray(limits))
                        env.set_state(public_initial)
                        equal_state(env.get_state(), public_initial)
                        actions = np.random.default_rng(31).uniform(-1, 1, (4, horizon, 4))
                        raw_tasks = physics.step(actions)
                        expected, actual = serial.step(actions, device), parallel.step(actions, device)
                        result = env.step(actions[:, 0] if horizon == 1 else actions, states=slice(None), frames=frame_slice)
                        compare(actual, expected)
                        compare({k: v for k, v in expected.items() if k not in ('frames', 'frame_valid')}, raw_tasks)
                        errors = compare(tasks(result, horizon), expected, library=True)
                        assert len(errors) == 4 * len(indices or ())
                        if indices is not None:
                            assert result['frames'].device.type == actual['frames'].device.type == device
                        for owner in (serial, parallel):
                            assert not differences(physics.snapshot(), owner.snapshot())
                        full_compare(result['state'].cpu(), physics.envs)
                        valid = torch.arange(horizon, device=device)[None, :] < torch.tensor(limits, device=device)[:, None]
                        assert torch.equal(result['valid'], valid) and torch.equal(result['state_valid'], valid)
                        for lane, maximum in enumerate(limits):
                            for t in range(maximum, horizon):
                                equal_state(result['states'][lane, maximum - 1], result['states'][lane, t])
                        held_tasks, held_public = {k: v.clone() for k, v in actual.items()}, result.clone()
                        raw_following = physics.step(-actions)
                        following = serial.step(-actions, device)
                        compare(parallel.step(-actions, device), following)
                        compare({k: v for k, v in following.items() if k not in ('frames', 'frame_valid')}, raw_following)
                        next_public = env.step(-actions[:, 0] if horizon == 1 else -actions, states=None, frames=frame_slice)
                        errors += compare(tasks(next_public, horizon), following, library=True)
                        full_compare(next_public['state'].cpu(), physics.envs)
                        for owner in (serial, parallel):
                            assert not differences(physics.snapshot(), owner.snapshot())
                        compare(actual, held_tasks)
                        equal_state(result, held_public)
                        inspection = [list(owner.vector.call('inspection')) for owner in (serial, parallel)]
                        assert all(row['render_audit_count'] > 0 for group in inspection for row in group)
                        assert all(not ref.unwrapped.mujoco_renderer._viewers for ref in physics.envs)
                        report['cases'].append(dict(horizon=horizon, selection=selection, frame_indices=indices, limits=limits,
                            source_gl=identities, source_inspection=inspection, valid_steps=int(valid.sum()),
                            frames_checked=len(errors), mse_mean=statistics.mean(errors) if errors else None,
                            mse_sd=statistics.stdev(errors) if len(errors) > 1 else None,
                            mse_max=max(errors) if errors else None,
                            source_parallel_exact='PASS', source_render_nonmutation='PASS', source_full_state_tasks='PASS',
                            output_ownership='PASS', terminal_padding_state_hold='PASS'))
                        save()
                        print(registration, device, horizon, selection, limits, 'PASS', flush=True)
                for owner in (serial, parallel):
                    closed = [dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive()) for p in owner.vector.processes]
                    assert all(row['exitcode'] == 0 and not row['alive'] for row in closed)
                    report['worker_closures'].append(dict(horizon=horizon, selection=selection, workers=closed))
                    save()
        assert len(report['cases']) == 20 and len(report['worker_closures']) == 20
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
