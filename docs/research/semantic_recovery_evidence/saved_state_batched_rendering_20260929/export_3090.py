import csv, hashlib, json
from pathlib import Path
root = Path('docs/research/semantic_recovery_evidence/saved_state_batched_rendering_20260929')
plan = json.loads(Path('tmp/renderer-saturation-20260929/plan.json').read_text())
rows, sweep, hashes = [], [], {}
for task in plan['tasks']:
    p = root / 'raw-3090' / str(task['index']) / 'result.json'
    r = json.loads(p.read_text()) if p.exists() else {}
    if p.exists():
        hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    row = dict(env=task['registration'], status=r.get('status', 'PENDING'), backend=task['backend'], gpu=r.get('profile', {}).get('gpu', ''), host=r.get('profile', {}).get('host', ''), independent_states=r.get('independent_states', ''), selected_batch=r.get('selected_batch', ''), mean_ms_per_frame='', sd_ms_per_frame='', frames_per_second='', samples='', stop_reason=r.get('stop_reason', ''), scope=task['scope'], evidence=str(p.relative_to(root)) if p.exists() else '')
    if r.get('status') == 'COMPLETE':
        assert r['nonmutation_passed'] and '3090' in row['gpu']
        t = r['confirmation']
        row.update({k:t[k] for k in ['mean_ms_per_frame', 'sd_ms_per_frame', 'frames_per_second']})
        row['samples'] = t['count']
    for b in [64,128,256]:
        for key in ['mean_ms_per_frame','sd_ms_per_frame','frames_per_second']:
            row[f'b{b}_{key}'] = ''
    for sample in r.get('rows', []):
        if sample['status'] != 'COMPLETE': continue
        t = sample['timing']; b = sample['batch']
        sweep.append(dict(env=task['registration'], batch=b, backend=task['backend'], gpu=row['gpu'], **{k:t[k] for k in ['count','mean_ms','sd_ms','mean_ms_per_frame','sd_ms_per_frame','frames_per_second']}, max_batch_difference_uint8=sample['max_batch_difference_uint8']))
        if b in [64,128,256]:
            for key in ['mean_ms_per_frame','sd_ms_per_frame','frames_per_second']:
                row[f'b{b}_{key}'] = t[key]
    rows.append(row)
for filename, data in [('saved_state_rendering_42_rtx3090.csv',rows),('rtx3090_batch_sweep.csv',sweep)]:
    with (root / filename).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
(root / 'rtx3090-result-sha256.json').write_text(json.dumps(hashes, indent=2)+'\n')
from collections import Counter
print(dict(Counter(r['status'] for r in rows)))
