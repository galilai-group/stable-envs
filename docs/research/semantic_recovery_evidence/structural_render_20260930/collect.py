"""Summarize isolated structural diagnostics; never changes the accepted table."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROWS = []
for campaign in ('structural-render-20260930-v2', 'structural-render-20260930-a5000-v3'):
    for path in sorted((ROOT / 'runs' / campaign / 'results').glob('*/result.json')):
        result = json.loads(path.read_text())
        assert result['status'] == 'COMPLETE', (path, result['status'])
        assert result['state_nonmutation_passed']
        for probe in sorted(path.parent.glob('structural_b*.json')):
            row = json.loads(probe.read_text())
            assert row['status'] == 'PASS'
            times = row['timings']
            for variant, quality in row['variants'].items():
                data = dict(registration=result['registration'], device=result['device'],
                            hardware=str(result['profile']['gpu'] or result['profile']['cpu']),
                            batch=row['batch'], output_device=row['destination'], variant=variant,
                            independent_states=result['independent_states'],
                            candidate_vs_baseline=times['baseline']['mean_ms']/times[variant]['mean_ms'],
                            candidate_vs_source=times['source']['mean_ms']/times[variant]['mean_ms'],
                            mse_vs_baseline=quality['mean'], max_frame_mse_vs_baseline=quality['maximum'],
                            mse_vs_source=quality['source_mse']['mean'],
                            max_frame_mse_vs_source=quality['source_mse']['maximum'],
                            baseline_image_exact=quality['exact'], evidence=str(probe.relative_to(ROOT)))
                for label, key in [('baseline','baseline'), ('candidate',variant), ('source','source')]:
                    timing=times[key]
                    data[label+'_fps']=timing['fps']
                    data[label+'_fps_sd']=statistics.stdev(1000*row['batch']/t for t in timing['samples_ms'])
                    data[label+'_batch_ms']=timing['mean_ms']
                    data[label+'_batch_ms_sd']=timing['sd_ms']
                ROWS.append(data)
assert len(ROWS)==32, len(ROWS)  # 3 CPU rows * 4 batches; Scene 5 batches * 2 outputs * 2 atlas capacities.
with (ROOT/'comparison.csv').open('w',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(ROWS[0]))
    writer.writeheader(); writer.writerows(ROWS)
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='sha256.json'}
(ROOT/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'{len(ROWS)} diagnostic rows; all selected runs COMPLETE and nonmutating')
