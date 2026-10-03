"""Local mechanism probe; not a replacement for same-host Oscar acceptance."""
import argparse
import cProfile
import hashlib
import json
import platform
import pstats
import time
from pathlib import Path

import torch

from scripts.benchmarks.online_stepping.classic_snapshot import actions_for
from scripts.benchmarks.online_stepping.classic_snapshot_sustained import digest
from tests.helpers.reference.classic_snapshot import make

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode', choices=['baseline', 'candidate'], required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
torch.set_num_threads(1)
root = Path.cwd()
manifest = json.loads((root.parent / 'source_manifest.json').read_text())
for rel, expected in manifest['files'].items():
    assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == expected, rel
report = dict(mode=args.mode, host=platform.node(), machine=platform.machine(),
              torch=torch.__version__, batch=1024, horizon=8, states='all', frames='final',
              scope='Local CPU cProfile mechanism probe, not Oscar performance acceptance',
              source_manifest_sha256=hashlib.sha256((root.parent / 'source_manifest.json').read_bytes()).hexdigest(),
              driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seconds=[])
with make('mountain_car_continuous', 1024, 'cpu', legacy=args.mode == 'baseline') as env:
    state = env.get_state()
    acts = actions_for('mountain_car_continuous', 1024)
    expected = env.step(acts, states=slice(None), frames=slice(-1, None))
    assert bool(expected['valid'].all())
    report['output_sha256'] = digest(expected)
    del expected
    for _ in range(3):
        env.set_state(state)
        start = time.perf_counter()
        out = env.step(acts, states=slice(None), frames=slice(-1, None))
        report['seconds'].append(time.perf_counter() - start)
        assert digest(out) == report['output_sha256']
        del out
    prof = cProfile.Profile()
    for _ in range(2):
        env.set_state(state)
        prof.enable()
        out = env.step(acts, states=slice(None), frames=slice(-1, None))
        prof.disable()
        assert digest(out) == report['output_sha256']
        del out
    report['profile_calls'] = 2
    report['functions'] = [dict(file=f, line=line, function=name, primitive_calls=v[0],
                                 calls=v[1], self_seconds=v[2], cumulative_seconds=v[3])
                           for (f, line, name), v in pstats.Stats(prof).stats.items()]
report['status'] = 'PASS'
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(args.mode, report['seconds'])
