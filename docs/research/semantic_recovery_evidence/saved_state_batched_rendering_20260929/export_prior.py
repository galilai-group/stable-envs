import csv, json, hashlib
from pathlib import Path
root = Path('docs/research/semantic_recovery_evidence')
src = root / 'renderer_profiles_20260928'
out = root / 'saved_state_batched_rendering_20260929'
envs = list(csv.DictReader((src / 'per_environment_42.csv').open()))
timings = list(csv.DictReader((src / 'timings.csv').open()))
rows = []
for env in envs:
    r = {'env': env['env'], 'measurement_date': '2026-09-28', 'scope': env['scope']}
    for device, label in [('cpu', 'cpu'), ('cuda', 'gpu')]:
        r.update({f'{label}_backend': env[f'{label}_backend'], f'{label}_hardware': env[f'{label}_hardware'], f'{label}_status': env[f'{label}_status']})
        for batch in [64, 256]:
            matches = [t for t in timings if t['registration'] == env['env'] and t['device'] == device and t['backend'] == env[f'{label}_backend'] and t['stage'] == 'prepared_batch_render' and t['batch'] == str(batch)]
            assert len(matches) <= 1
            prefix = f'{label}_b{batch}'
            for field in ['mean_ms_per_frame', 'sd_ms_per_frame', 'frames_per_second', 'samples', 'evidence']:
                r[f'{prefix}_{field}'] = ''
            if matches:
                t = matches[0]
                r.update({f'{prefix}_mean_ms_per_frame': float(t['mean_ms']) / batch, f'{prefix}_sd_ms_per_frame': float(t['sd_ms']) / batch, f'{prefix}_frames_per_second': 1000 * batch / float(t['mean_ms']), f'{prefix}_samples': t['count'], f'{prefix}_evidence': '../renderer_profiles_20260928/' + t['result']})
    rows.append(r)
assert len(rows) == 42
with (out / 'saved_state_rendering_42_prior.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(out / 'prior-provenance.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [src / 'timings.csv', src / 'per_environment_42.csv']}, indent=2) + '\n')
print('Exported 42 rows; CPU and GPU batches 64/256; input table hashes recorded.')
