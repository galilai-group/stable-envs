"""Reject inadmissible benchmark evidence; these are not GPU/fidelity results."""

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.benchmarks.online_stepping.fetch_source_pipeline import (
    copy_state_to_device,
    require_image_gate,
    source_state,
)


def test_source_state_projection_preserves_owned_rng_and_controller_fields():
    row = dict(qpos=np.arange(22, dtype=np.float64), qvel=np.arange(21, dtype=np.float64),
        time=np.asarray(.08), qacc_warmstart=np.arange(21, dtype=np.float64), goal=np.arange(3, dtype=np.float64),
        initial_gripper_xpos=np.arange(3, dtype=np.float64), controller_position=np.arange(3, dtype=np.float64),
        controller_quaternion=np.arange(4, dtype=np.float64), elapsed=np.asarray(4, dtype=np.int64),
        rng={'bit_generator': 'PCG64', 'state': {'state': 2**127 + 2**64 + 7, 'inc': 2**96 + 9},
             'has_uint32': 1, 'uinteger': 2**32 - 1})
    state = source_state([row, copy.deepcopy(row)])
    assert state.batch_size == (2,)
    assert state['rng', 'state_words'][0].tolist() == [7, 0, 1, 2**31]
    assert state['rng', 'increment_words'][0].tolist() == [9, 0, 0, 1]
    assert state['rng', 'uinteger'][0].item() == 2**32 - 1
    assert state['lifecycle', 'elapsed_steps'].dtype == torch.int64
    row['qpos'][:] = 0
    assert state['qpos'][0, 1].item() == 1
    copied = copy_state_to_device(state, 'cpu')
    copied['qpos'][0] = 0
    assert state['qpos'][0, 1].item() == 1


def fake_gate():
    software = dict(vendor='Mesa', renderer='llvmpipe', software=True)
    manifest = {'commit': 'synthetic', 'files': {'source.py': 'synthetic'}}
    files = ['scripts/benchmarks/online_stepping/fetch_source_frame_gate.py',
             'scripts/benchmarks/online_stepping/fetch_parallel_source.py']
    gate = dict(status='PASS', registration='swm/FetchReach-v3', device='cpu', source_pin=manifest['commit'],
        source_files=manifest['files'], batch=4, source_workers=[1, 2], source_internal_size=480, output_size=64,
        candidate_files={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        versions={n: importlib.metadata.version(n) for n in
                  ('numpy', 'mujoco', 'gymnasium', 'gymnasium-robotics', 'torch', 'tensordict')},
        parent_gl=software, cases=[], worker_closures=[])
    for horizon in (1, 8):
        for selection in ('none', 'empty', 'final', 'all', 'reverse_sparse'):
            count = (0 if selection in ('none', 'empty') else 1 if selection == 'final' else
                     horizon if selection == 'all' else len(range(horizon)[::-2]))
            for limits in ([50] * 4, [1, 3, 2, 50]):
                gate['cases'].append(dict(horizon=horizon, selection=selection, limits=limits,
                    source_parallel_exact='PASS', source_render_nonmutation='PASS', source_full_state_tasks='PASS',
                    output_ownership='PASS', terminal_padding_state_hold='PASS', frames_checked=8*count,
                    mse_mean=0 if count else None, mse_max=0 if count else None, mse_sd=0 if count else None,
                    source_gl=[[software], [software, software]]))
            for workers in (1, 2):
                gate['worker_closures'].append(dict(horizon=horizon, selection=selection,
                    workers=[dict(exitcode=0, alive=False) for _ in range(workers)]))
    return gate, manifest


@pytest.mark.parametrize('defect', [None, 'sibling', 'device', 'source', 'code', 'versions', 'missing', 'duplicate',
    'mse', 'nan_sd', 'count', 'nonmutation', 'gl', 'workers', 'closure', 'missing_closure'])
def test_rendered_timing_requires_complete_current_independent_gate(defect, tmp_path):
    gate, manifest = fake_gate()
    if defect == 'sibling':
        gate['registration'] = 'swm/FetchPush-v3'
    elif defect == 'device':
        gate['device'] = 'cuda'
    elif defect == 'source':
        gate['source_pin'] = 'wrong'
    elif defect == 'code':
        gate['candidate_files'][next(iter(gate['candidate_files']))] = '0' * 64
    elif defect == 'versions':
        gate['versions']['mujoco'] = 'wrong'
    elif defect == 'missing':
        gate['cases'].pop()
    elif defect == 'duplicate':
        gate['cases'][-1] = gate['cases'][0]
    elif defect == 'mse':
        gate['cases'][4]['mse_max'] = .0011
    elif defect == 'nan_sd':
        gate['cases'][4]['mse_sd'] = float('nan')
    elif defect == 'count':
        gate['cases'][4]['frames_checked'] = 1
    elif defect == 'nonmutation':
        gate['cases'][4]['source_render_nonmutation'] = 'FAIL'
    elif defect == 'gl':
        gate['parent_gl']['software'] = False
    elif defect == 'workers':
        gate['cases'][4]['source_gl'] = [gate['parent_gl']]
    elif defect == 'closure':
        gate['worker_closures'][0]['workers'][0]['alive'] = True
    elif defect == 'missing_closure':
        gate['worker_closures'][-1] = gate['worker_closures'][0]
    path = tmp_path / 'gate.json'
    path.write_text(json.dumps(gate))
    if defect is None:
        assert require_image_gate(path, 'swm/FetchReach-v3', 'cpu', manifest) == hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        with pytest.raises(AssertionError):
            require_image_gate(path, 'swm/FetchReach-v3', 'cpu', manifest)


@pytest.mark.parametrize('registration, fail_gate, calls', [('swm/FetchReach-v3', False, 4),
    ('swm/FetchReachDict-v3', True, 1), ('swm/FetchPickAndPlaceDense-v3', False, 3)])
def test_coordinator_runs_registration_gate_before_any_rendered_timing(registration, fail_gate, calls, monkeypatch, tmp_path):
    import subprocess
    import sys

    from scripts.benchmarks.online_stepping import fetch_source_oscar

    launched = []
    def launch(command, *, check):
        assert check
        launched.append(command)
        if fail_gate:
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(fetch_source_oscar.subprocess, 'run', launch)
    monkeypatch.setattr(sys, 'argv', ['driver', '--registration', registration, '--source-root', str(tmp_path),
        '--source-manifest', str(tmp_path / 'manifest.json'), '--output', str(tmp_path / 'results')])
    if fail_gate:
        with pytest.raises(subprocess.CalledProcessError):
            fetch_source_oscar.main()
    else:
        fetch_source_oscar.main()
    assert len(launched) == calls
    assert all(command[command.index('--domain') + 1] == registration for command in launched)
    if 'PickAndPlace' in registration:
        assert all('--image-gates' not in command for command in launched)
    else:
        assert launched[0][launched[0].index('--benchmark') + 1] == 'fetch_source_frame_gate'
        assert all('--image-gates' in command for command in launched[1:])


@pytest.mark.parametrize('defect', [None, 'host', 'job', 'version', 'duplicate_pid', 'state', 'task',
    'missing_cell', 'missing_component', 'zero_time', 'missing_sample', 'rendered_claim'])
def test_summary_joins_only_complete_same_device_source_comparisons(defect, tmp_path):
    from scripts.benchmarks.online_stepping.summarize_fetch_source import summarize

    source = tmp_path/'source'
    source.mkdir()
    files = ['stable_worldmodel/envs/gymnasium_robotics/fetch.py', 'a', 'b', 'c', 'd']
    for f in files:
        path = source/f
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic source input')
    manifest = dict(commit='176d54532b3c0a251a4f509b74f2fdacc143a63d',
                    files={f: hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files})
    manifest_path = tmp_path/'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    candidate_files = ['scripts/benchmarks/online_stepping/fetch_source_pipeline.py',
        'scripts/benchmarks/online_stepping/fetch_parallel_source.py', 'stable_envs/_stepping.py',
        'stable_envs/_adapters/fetch_process.py', 'stable_envs/envs/fetch_opengl.py']
    reports = {}
    for i, mode in enumerate(('source_serial', 'source_parallel', 'ours')):
        for j, device in enumerate(('cpu', 'cuda')):
            report = dict(status='PASS', gate_only=False, registration='swm/FetchPickAndPlace-v3', mode=mode,
                device=device, job='synthetic', host='synthetic', affinity=list(range(8)), pid=100+i*2+j,
                parent_torch_threads=8 if device=='cpu' else 1, source_threads_per_worker=1, output_size=64,
                backend='opengl', gpu=None if device=='cpu' else 'RTX 3090', source_gpu_renderer_available=True,
                source_physics_device='cpu', source_state_export_in_online_timing=False, ours_state_history=False,
                source_readback_and_output_transfer_in_timing=True, ours_mandatory_final_state=True,
                source_observations_computed=True, fresh_strategy_process=True, oracle_closed_before_population=True,
                rendered_protocol=False, source_internal_size=None, image_gate_sha256=None,
                source_pin=manifest['commit'], source_files=manifest['files'],
                versions={n: importlib.metadata.version(n) for n in
                          ('numpy', 'mujoco', 'gymnasium', 'gymnasium-robotics', 'torch', 'tensordict')},
                candidate_files={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in candidate_files},
                source_runtime_files={'/synthetic/'+files[0]: manifest['files'][files[0]]},
                action_policy='synthetic actions', initial_state_policy='synthetic states', cells=[], components=[])
            for horizon in (1, 8):
                report['cells'].append(dict(batch=64, horizon=horizon, selection='none',
                    physics_workers=1 if mode=='source_serial' else 8, exact_state_task='PASS', output_ownership='PASS',
                    valid_steps=64*horizon, frames_checked=0, seconds=[{'source_serial':4., 'source_parallel':1., 'ours':2.}[mode]]*6,
                    initial_state_sha256='a'*64, endpoint_state_sha256='b'*64, source_tasks_sha256='c'*64,
                    source_gl=[], actor_gl=[], mse_mean=None, mse_sd=None, mse_max=None))
            if mode=='ours':
                report['components'] = [dict(batch=64, operation=op, seconds=[1.]*6) for op in
                    ('task_no_state_with_ipc_transfer', 'get_state', 'state_cpu_to_device')]
            reports[mode, device] = report
    bad = reports['ours', 'cuda']
    if defect in ('host', 'job'):
        bad[defect] = 'different'
    elif defect == 'version':
        bad['versions']['numpy'] = 'different'
    elif defect == 'duplicate_pid':
        bad['pid'] = reports['source_serial', 'cpu']['pid']
    elif defect == 'state':
        bad['cells'][0]['endpoint_state_sha256'] = 'd'*64
    elif defect == 'task':
        bad['cells'][0]['source_tasks_sha256'] = 'd'*64
    elif defect == 'missing_cell':
        bad['cells'].pop()
    elif defect == 'missing_component':
        bad['components'].pop()
    elif defect == 'zero_time':
        bad['cells'][0]['seconds'][0] = 0
    elif defect == 'missing_sample':
        bad['cells'][0]['seconds'].pop()
    elif defect == 'rendered_claim':
        bad['cells'][0]['frames_checked'] = 64
    results = tmp_path/'results'
    for (mode, device), report in reports.items():
        path = results/mode/f'{device}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report))
    if defect is None:
        rows, components = summarize(results, Path.cwd(), source, manifest_path, [64])
        assert len(rows)==4 and len(components)==6
        assert all(row['speedup_vs_best_same_device_source']==.5 for row in rows)
        assert all(row['ours_fps_mean'] is None and row['mse_mean'] is None for row in rows)
    else:
        with pytest.raises(AssertionError):
            summarize(results, Path.cwd(), source, manifest_path, [64])
