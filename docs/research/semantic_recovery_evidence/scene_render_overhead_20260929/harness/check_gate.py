"""Evaluate downloaded paired evidence without changing or filling measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(__doc__)
p.add_argument('root', type=Path)
a = p.parse_args()
status, rows = {}, []
for index in [30, 70, 72]:
    path = a.root / ('candidate' if index == 30 else 'cpu-candidate') / str(index) / 'result.json'
    if not path.exists():
        status[str(index)] = 'PENDING'
        continue
    new = json.loads(path.read_text())
    if new['status'] != 'COMPLETE':
        status[str(index)] = new['status']
        continue
    assert new['state_nonmutation_passed']
    assert {m['batch'] for m in new['measurements']} == {64, 128, 256}
    assert max(m['paired_mse']['maximum'] for m in new['measurements']) <= .001
    oldpath = a.root / 'baseline' / str(index) / 'result.json'
    old = json.loads(oldpath.read_text()) if oldpath.exists() else None
    if index == 30:
        assert old and old['status'] == 'COMPLETE' and old['state_nonmutation_passed']
        assert new['profile']['gpu'] == old['profile']['gpu'] == 'NVIDIA GeForce RTX 3090'
        assert new['profile']['host'] == old['profile']['host']
        assert new['frames'] == old['frames'] and new['corpus_sha256'] == old['corpus_sha256']
    for m in new['measurements']:
        pixel_max, pixel_mse, oldfps = '', '', ''
        if old:
            filename = f"ours_b{m['batch']}_{m['output_device']}.npy"
            left = np.load(path.parent / filename, allow_pickle=False)
            right = np.load(oldpath.parent / filename, allow_pickle=False)
            assert left.shape == right.shape
            delta = left.astype(np.float64) - right.astype(np.float64)
            pixel_max = int(np.abs(delta).max())
            pixel_mse = float(np.mean((delta / 255) ** 2))
            assert pixel_max <= 1, (filename, pixel_max, pixel_mse)
            prior = next(v for v in old['measurements'] if (v['batch'], v['output_device']) == (m['batch'], m['output_device']))
            oldfps = prior['ours']['frames_per_second']
        rows.append(dict(env=new['registration'], device=new['device'], batch=m['batch'], output_device=m['output_device'], host=new['profile']['host'], old_fps=oldfps, ours_fps=m['ours']['frames_per_second'], source_fps=m['source']['frames_per_second'], speedup=m['speedup'], mse_mean=m['paired_mse']['mean'], mse_sd=m['paired_mse']['sd'], ours_ms_mean=m['ours']['mean_ms_per_frame'], ours_ms_sd=m['ours']['sd_ms_per_frame'], baseline_candidate_max_uint8=pixel_max, baseline_candidate_mse=pixel_mse, result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    status[str(index)] = 'PASS'
if rows:
    with (a.root / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
(a.root / 'gate.json').write_text(json.dumps(status, indent=2)+'\n')
print(json.dumps(status))
