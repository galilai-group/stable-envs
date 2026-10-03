"""Reproduce the fixed eleven-registration online coverage checkpoint."""
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
root = Path(__file__).parent
previous = root.parent / 'dmc_cartpole_source_completed_20261003/coverage/coverage.csv'
rows = list(csv.DictReader(previous.open()))
assert len(rows) == len({r['registration'] for r in rows}) == 42
names = {'acrobot': 'swm/AcrobotDMControl-v0', 'pendulum': 'swm/PendulumDMControl-v0',
         'simple_point_maze': 'swm/SimplePointMaze-v0'}
for domain, registration in names.items():
    native_gpu = domain != 'simple_point_maze'
    path = root / 'tables' / (domain + '/comparison.csv' if native_gpu else domain + '.csv')
    cells = [r for r in csv.DictReader(path.open()) if r['batch'] == '64' and r['horizon'] == '8' and r['selection'] == 'all']
    assert len(cells) == 2
    cells = {r['device' if native_gpu else 'output_device']: r for r in cells}
    cpu, gpu = cells['cpu'], cells['cuda']
    assert cpu['job'] == gpu['job'] and cpu['host'] == gpu['host']
    row = next(r for r in rows if r['registration'] == registration)
    row.update(online_source_status='MEASURED_QUALIFIED', job=cpu['job'], batch='64', horizon='8', frames='all',
               note='Bounded complete online comparison; all cells and SDs retained in evidence; no universal performance claim',
               gpu_comparison_kind='SAME_DEVICE_SOURCE_GPU' if native_gpu else 'SOURCE_CPU_PLUS_CUDA_DELIVERY_NO_SOURCE_GPU')
    for device, data in [('cpu', cpu), ('gpu', gpu)]:
        for suffix in ('mean', 'sd'):
            row[f'{device}_mse_{suffix}'] = data[f'mse_{suffix}']
            row[f'ours_{device}_fps_{suffix}'] = data[f'ours_fps_{suffix}']
    for mode in ('serial', 'parallel'):
        for suffix in ('mean', 'sd'):
            row[f'source_cpu_{mode}_fps_{suffix}'] = cpu[f'source_{mode}_fps_{suffix}']
            if native_gpu:
                row[f'source_native_gpu_{mode}_fps_{suffix}'] = gpu[f'source_{mode}_fps_{suffix}']
    best = max(('serial', 'parallel'), key=lambda m: float(gpu[f'source_{m}_fps_mean']))
    for suffix in ('mean', 'sd'):
        target = f'source_native_gpu_fps_{suffix}' if native_gpu else f'source_cpu_cuda_delivery_fps_{suffix}'
        row[target] = gpu[f'source_{best}_fps_{suffix}']
    row['ours_cpu_vs_best_source'] = cpu['ours_vs_best_source']
    row['ours_gpu_vs_best_native_source' if native_gpu else 'ours_gpu_vs_best_cpu_cuda_delivery'] = gpu['ours_vs_best_source']
assert sum(r['online_source_status'] == 'MEASURED_QUALIFIED' for r in rows) == 11
with a.output.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
