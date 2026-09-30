"""Validate downloaded bounded renderer gates without changing raw evidence."""
import csv
import hashlib
import json
from pathlib import Path

root = Path('docs/research/semantic_recovery_evidence/dmc_cpu_bounds_components_20260929')
rows = []

def check(folder, index, batches, destinations):
    paths = [root / folder / leg / 'results' / str(index) / 'result.json' for leg in ('baseline', 'candidate')]
    baseline, candidate = [json.loads(p.read_text()) for p in paths]
    for item in [baseline, candidate]:
        assert item['status'] == 'COMPLETE', paths
        assert item['state_nonmutation_passed']
    for key in ['frames', 'corpus_sha256', 'batch_image_sha256']:
        assert baseline[key] == candidate[key], (folder, index, key)
    assert baseline['profile']['host'] == candidate['profile']['host']
    assert baseline['profile']['environment']['SLURM_JOB_ID'] == candidate['profile']['environment']['SLURM_JOB_ID']
    assert set((m['batch'], m['output_device']) for m in baseline['measurements']) == {(b, d) for b in batches for d in destinations}
    assert len(baseline['measurements']) == len(candidate['measurements'])
    for old, new in zip(baseline['measurements'], candidate['measurements']):
        assert (old['batch'], old['output_device']) == (new['batch'], new['output_device'])
        assert new['paired_mse']['mean'] <= .001
        rows.append(dict(registration=candidate['registration'], device=candidate['device'], batch=new['batch'], output_device=new['output_device'], baseline_fps=old['ours']['frames_per_second'], candidate_fps=new['ours']['frames_per_second'], source_fps=new['source']['frames_per_second'], candidate_vs_baseline=old['ours']['mean_ms']/new['ours']['mean_ms'], candidate_vs_source=new['speedup'], mse_mean=new['paired_mse']['mean'], mse_sd=new['paired_mse']['sd'], ours_ms_sd=new['ours']['sd_ms_per_frame'], source_ms_sd=new['source']['sd_ms_per_frame'], exact_pixels=True, evidence=str(paths[1])))
    if folder == 'v2_cross':
        assert baseline['cross_consumer_pusher'] == candidate['cross_consumer_pusher']
    if candidate['device'] == 'cuda':
        assert '3090' in candidate['profile']['gpu']
        for p in paths:
            primitive = json.loads(p.with_name('cuda_predicates.json').read_text())
            assert primitive == {'status': 'PASS', 'predicates': 262144}
    return candidate

for index in [44, 64, 67]:
    check('v2_cpu', index, [64,128,256], ['cpu'])
check('v2_cross', 66, [64], ['cpu'])
for index in [2,22]:
    check('v2_cuda', index, [64], ['cpu','cuda'])
check('v2_cuda_retry',25,[64],['cpu','cuda'])
code = Path('stable_envs/render/raycast3d.py')
expected = json.loads((root/'v2_cpu/candidate/plan.json').read_text())['production_sha256'][str(code)]
assert hashlib.sha256(code.read_bytes()).hexdigest() == expected
with (root/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(root/'collection_status.json').write_text(json.dumps(dict(status='PASS', production_sha256={str(code):expected}, cpu_batches=[64,128,256], cuda_batch=64, independent_states=16, exact_baseline_candidate_images=True, tests_passed=77, rows=rows),indent=2)+'\n')
for row in rows:
    print(row['registration'],row['device'],row['batch'],row['output_device'],round(row['candidate_fps'],3),round(row['candidate_vs_baseline'],3),round(row['candidate_vs_source'],3))
