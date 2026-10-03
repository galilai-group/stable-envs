"""Reconcile this fixed evidence checkpoint; never rerun timings or promote rows."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument('--evidence', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
root = a.evidence
matrix = Path('docs/research/semantic_recovery_matrix.json')
entries = json.loads(matrix.read_text())['rows']
assert len(entries) == 42 and Counter(r['state_status'] for r in entries) == {'VERIFIED': 39, 'INCOMPLETE': 3}
classic = {'acrobot': 'swm/AcrobotControl-v1', 'cartpole': 'swm/CartPoleControl-v1',
           'pendulum': 'swm/PendulumControl-v1', 'mountain_car': 'swm/MountainCarControl-v0',
           'mountain_car_continuous': 'swm/MountainCarContinuousControl-v0'}
navigation = {'two_room': 'swm/TwoRoom-v1', 'piecewise': 'swm/Piecewise-v0'}
inputs = {str(matrix): hashlib.sha256(matrix.read_bytes()).hexdigest()}
measured = {}
for directory, names, batch in [('classic-verified', classic, '256'), ('navigation-verified', navigation, '64')]:
    for domain, registration in names.items():
        path = root / directory / (domain + '.csv')
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        cells = [r for r in csv.DictReader(path.open()) if r['batch'] == batch and r['horizon'] == '8' and r['selection'] == 'all']
        assert {r['output_device'] for r in cells} == {'cpu', 'cuda'} and len(cells) == 2
        measured[registration] = {r['output_device']: r for r in cells}
rows = []
for entry in entries:
    name, family = entry['historical_id'], entry['family']
    if family == 'craftax':
        status, job, note = 'DEFERRED', '', 'Craftax excluded from this goal'
    elif entry['state_status'] != 'VERIFIED':
        status, job, note = 'CORRECTNESS_BLOCKED', '', 'Independent semantic acceptance required'
    elif name in measured:
        status, job, note = 'MEASURED_QUALIFIED', measured[name]['cpu']['job'], 'Bounded online source comparison; not universal acceptance'
    elif family == 'fetch' and 'PickAndPlace' in name:
        path = root / 'pick-verified' / name.removeprefix('swm/') / 'online.csv'
        data = list(csv.DictReader(path.open()))
        assert len(data) == 4 and all(r['frames'] == 'none' and r['mse_mean'] == r['ours_fps_mean'] == '' for r in data)
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        status, job, note = 'MEASURED_NO_FRAMES_ONLY', data[0]['job'], 'Source render changes continuation; stepping CSV available, no rendered comparison'
    elif family == 'fetch':
        path = root / 'fetch/results' / name.removeprefix('swm/') / 'image-gates/cpu.json'
        report = json.loads(path.read_text())
        assert report['status'] == 'FAIL' and report['registration'] == name
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        status, job, note = 'IMAGE_GATE_FAILED', report['job'], 'CPU nonempty-frame MSE exceeds 0.001; no timing or promotion'
    elif family == 'ogbench' and 'Maze' in name:
        status, job, note = 'SOURCE_IMAGE_EQUIVALENCE_FAILED', '6976168', 'CPU 30 cases passed; CUDA source P1/P2 differ by one level in 72 channels'
    elif family == 'ogbench':
        status, job, note = 'SOURCE_IMAGE_GATE_RUNNING', '6976161', 'No source timing claimed; Cube/Scene validator control is separate'
    elif family == 'navigation':
        status, job, note = 'SOURCE_COMPARISON_RUNNING', '6976163', 'SimplePointMaze remains uncollected'
    elif any(task in name for task in ('Acrobot', 'Cartpole', 'Pendulum')):
        status, job, note = 'SOURCE_COMPARISON_RUNNING', '6976162', 'DMC first three source sweeps uncollected'
    else:
        status, job, note = 'SOURCE_COMPARISON_MISSING', '', 'Completed internal controls do not substitute for source timing'
    row = dict(registration=name, semantic_status=entry['state_status'], online_source_status=status, job=job,
               batch='', horizon='', frames='', cpu_mse_mean='', cpu_mse_sd='', gpu_mse_mean='', gpu_mse_sd='',
               source_cpu_serial_fps_mean='', source_cpu_serial_fps_sd='', source_cpu_parallel_fps_mean='',
               source_cpu_parallel_fps_sd='', ours_cpu_fps_mean='', ours_cpu_fps_sd='', ours_cpu_vs_best_source='',
               source_native_gpu_fps_mean='', source_native_gpu_fps_sd='',
               source_cpu_cuda_delivery_fps_mean='', source_cpu_cuda_delivery_fps_sd='',
               ours_gpu_fps_mean='', ours_gpu_fps_sd='', ours_gpu_vs_best_cpu_cuda_delivery='', note=note)
    if name in measured:
        cpu, gpu = measured[name]['cpu'], measured[name]['cuda']
        assert cpu['host'] == gpu['host'] and cpu['job'] == gpu['job']
        assert cpu['source_gpu_supported'] == gpu['source_gpu_supported'] == 'False'
        row.update(batch=cpu['batch'], horizon=8, frames='all')
        for device, data in [('cpu', cpu), ('gpu', gpu)]:
            for suffix in ('mean', 'sd'):
                row[f'{device}_mse_{suffix}'] = data[f'mse_{suffix}']
                row[f'ours_{device}_fps_{suffix}'] = data[f'ours_fps_{suffix}']
        for mode in ('serial', 'parallel'):
            for suffix in ('mean', 'sd'):
                row[f'source_cpu_{mode}_fps_{suffix}'] = cpu[f'source_{mode}_fps_{suffix}']
        for suffix in ('mean', 'sd'):
            row[f'source_cpu_cuda_delivery_fps_{suffix}'] = gpu[f'source_parallel_fps_{suffix}']
        row['ours_cpu_vs_best_source'] = cpu['ours_vs_best_source']
        row['ours_gpu_vs_best_cpu_cuda_delivery'] = gpu['ours_vs_best_source']
    rows.append(row)
a.output.mkdir(parents=True)
with (a.output / 'coverage.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
(a.output / 'inputs.json').write_text(json.dumps(dict(input_sha256=inputs, counts=dict(Counter(r['online_source_status'] for r in rows)),
    scope='Fixed online checkpoint; raw evidence validated separately; no offline-render or internal-control speedups'), indent=2) + '\n')
print(Counter(r['online_source_status'] for r in rows))
