import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

base, run = map(Path, sys.argv[1:])
manifest = json.loads((base / 'manifest.json').read_text())
for name, digest in manifest['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
import torch
assert torch.cuda.is_available(), 'Required CUDA device unavailable'
import matplotlib
identity = dict(commit=manifest['commit'], host=platform.node(), python=platform.python_version(),
                gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, job=os.environ['SLURM_JOB_ID'],
                versions={name:md.version(name) for name in ('torch','tensordict','numpy','gymnasium','matplotlib')},
                verified_files=len(manifest['files']))
(run/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
command = [sys.executable, '-m', 'stable_envs.testing.supervisor', '--rundir', str(run/'supervisor'),
           '--json', str(run/'summary.json'), '--max-segments', '1', '--', '-q', '-rs',
           '--junitxml='+str(run/'tests.xml'), *manifest['required_nodes']]
status = subprocess.run(command).returncode
(run/'pytest-exit.txt').write_text(str(status)+'\n')
assert status == 0, f'Supervisor exited {status}'
report = json.loads((run/'summary.json').read_text())
assert report['counts']['passed'] == len(manifest['required_nodes']), report['counts']
assert all(value == 0 for name,value in report['counts'].items() if name != 'passed'), report['counts']
events = [json.loads(line) for path in (run/'supervisor').glob('events.*.jsonl') for line in path.read_text().splitlines()]
actual = [row['nodeid'] for row in events if row.get('kind') == 'end' and row.get('outcome') == 'passed']
assert len(actual) == len(set(actual)) and set(actual) == set(manifest['required_nodes']), actual
(run/'verification.json').write_text(json.dumps(dict(status='PASS',required_nodes=manifest['required_nodes'],
    passed_nodes=actual, scope='Five CPU and five CUDA disk restoration/continuation/capture checks, five exact old/new corpus controls; no source pixel or performance certification'),indent=2)+'\n')
print('TESTING_CORPUS_CUDA_PASS', flush=True)
