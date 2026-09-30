"""Summarize completed residual profiles without modifying their raw records."""
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
result = json.loads((root / 'results/66/result.json').read_text())
summary = dict(status=result['status'], registration=result['registration'], profile=result['profile'], batches=[])
for measurement in result['measurements']:
    batch = measurement['batch']
    d = json.loads((root / f'results/66/profile_b{batch}/diagnosis.json').read_text())
    assert d['status'] == 'COMPLETE'
    sections = d['sections']
    elapsed = d['section_elapsed_ms']
    buckets = {
        'pose_preparation': sections['pose_preparation']['inclusive_ms'],
        'bounds': sections['bounds']['inclusive_ms'],
        'intersections': sections['intersect']['inclusive_ms'],
        'texture_bilinear': sections['texture_bilinear']['inclusive_ms'],
        'normalize': sections['normalize']['inclusive_ms'],
        'cast_reduction_and_shadow_packing': sum(sections[k]['exclusive_ms'] for k in ['primary_cast', 'shadow_cast', 'shadow_flatten']),
        'remaining_chunk_work': sections['chunk']['exclusive_ms'],
    }
    summary['batches'].append(dict(batch=batch,paired=measurement,
        cpu_utilization={label:{k:statistics.mean(s[k] for s in d[label+'_unprofiled'])
                               for k in ['cpu_one_core_percent','cpu_allocation_percent','major_faults']}
                         for label in ['ours','source']},
        section_elapsed_ms=elapsed, buckets={k:{'ms':v,'percent':100*v/elapsed} for k,v in buckets.items()},
        status=d['status']))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
