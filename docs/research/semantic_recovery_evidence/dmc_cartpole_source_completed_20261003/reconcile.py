"""Add the terminal-verified DMC Cartpole result to the previous coverage snapshot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument('--evidence', type=Path, required=True)
p.add_argument('--previous', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
rows = list(csv.DictReader(a.previous.open()))
assert len(rows) == len({r['registration'] for r in rows}) == 42
cells = [r for r in csv.DictReader((a.evidence / 'tables/comparison.csv').open())
         if r['batch'] == '64' and r['horizon'] == '8' and r['selection'] == 'all']
assert len(cells) == 2
cpu, gpu = ({r['device']: r for r in cells}[k] for k in ('cpu', 'cuda'))
assert cpu['host'] == gpu['host'] and cpu['job'] == gpu['job'] == '6976162'
for r in rows:
    r.update(source_native_gpu_serial_fps_mean='', source_native_gpu_serial_fps_sd='',
             source_native_gpu_parallel_fps_mean='', source_native_gpu_parallel_fps_sd='',
             ours_gpu_vs_best_native_source='', gpu_comparison_kind='')
    if r['ours_gpu_fps_mean']:
        r['gpu_comparison_kind'] = 'SOURCE_CPU_PLUS_CUDA_DELIVERY_NO_SOURCE_GPU'
    if r['registration'] != 'swm/CartpoleDMControl-v0':
        continue
    r.update(online_source_status='MEASURED_QUALIFIED', job=cpu['job'], batch=64, horizon=8, frames='all',
             gpu_comparison_kind='SAME_DEVICE_SOURCE_GPU',
             note='Bounded B64/T1,T8 no/final/all frames; six samples; same allocation; no broad-state or universal speedup claim')
    for device, data in [('cpu', cpu), ('gpu', gpu)]:
        for suffix in ('mean', 'sd'):
            r[f'{device}_mse_{suffix}'] = data[f'mse_{suffix}']
            r[f'ours_{device}_fps_{suffix}'] = data[f'ours_fps_{suffix}']
    for mode in ('serial', 'parallel'):
        for suffix in ('mean', 'sd'):
            r[f'source_cpu_{mode}_fps_{suffix}'] = cpu[f'source_{mode}_fps_{suffix}']
            r[f'source_native_gpu_{mode}_fps_{suffix}'] = gpu[f'source_{mode}_fps_{suffix}']
    best = max(('serial', 'parallel'), key=lambda m: float(gpu[f'source_{m}_fps_mean']))
    for suffix in ('mean', 'sd'):
        r[f'source_native_gpu_fps_{suffix}'] = gpu[f'source_{best}_fps_{suffix}']
    r['ours_cpu_vs_best_source'] = cpu['ours_vs_best_source']
    r['ours_gpu_vs_best_native_source'] = gpu['ours_vs_best_source']
a.output.mkdir(parents=True)
with (a.output / 'coverage.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
paths = [a.previous, a.evidence / 'tables/comparison.csv', a.evidence / 'terminal.txt']
(a.output / 'inputs.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2) + '\n')
