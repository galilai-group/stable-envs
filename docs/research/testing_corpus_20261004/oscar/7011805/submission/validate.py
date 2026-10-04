import hashlib
import importlib.metadata as md
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from stable_envs.testing.corpus import load_trajectory, write_trajectory
from stable_envs.testing.metrics import leaf_difference
from stable_envs.testing.provenance import pinned_source_identity
from tests.helpers.reference.bridges import NativeBridge, load_source
from tests.helpers.reference.profiles import PINS, PROFILES
from tests.helpers.reference.saved import SWM_PIN

base,run=map(Path,sys.argv[1:])
manifest=json.loads((base/'manifest.json').read_text())
for name,digest in manifest['files'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
source_root='/oscar/scratch/sboughan/se-semantic-recovery/swm-reference'
identity=pinned_source_identity(source_root,SWM_PIN,PINS['native'])
load_source(source_root,'native')
report={'candidate_commit':manifest['commit'],'source':identity,'host':platform.node(),
        'gpu':torch.cuda.get_device_name(0),'versions':{n:md.version(n) for n in ['torch','numpy','gymnasium','tensordict']},
        'comparisons':{}}
for case in PROFILES['native']:
    saved=load_trajectory(Path('tests/fixtures/semantics/shared_native')/case.domain,registration=case.registration)
    bridge=NativeBridge(case,3)
    try:
        bridge.reset(17)
        initial=bridge.source.get_state()
        outputs=[bridge.step(a.copy()) for a in saved.arrays['actions']]
        values={'actions':saved.arrays['actions']}
        for key in initial.keys(True,True):
            name=key if isinstance(key,str) else '/'.join(key)
            values['state/'+name]=np.stack([initial[key].numpy(),*[row['state'][key].numpy() for row in outputs]])
        for key in ['reward','terminated','truncated','observation','info_state']:
            values[key]=np.stack([row[key].numpy() for row in outputs])
        report['comparisons'][case.registration]={key:leaf_difference(v,saved.arrays[key]) for key,v in values.items()}
        write_trajectory(run/'linux-source'/case.domain,registration=case.registration,seed=17,source=identity,
                         producer={'commit':manifest['commit'],'files':manifest['files'],
                                   'diagnostic_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},arrays=values)
    finally:
        bridge.close()
(run/'source-roundoff.json').write_text(json.dumps(report,indent=2)+'\n')
files=['tests/contracts/test_'+part+'.py' for part in ['state','simulation','task','online']]
command=[sys.executable,'-m','pytest','-q','-rs',*files,'--contract-profile','native',
         '--contract-source',source_root,'--junitxml='+str(run/'source-tests.xml')]
import pytest
status=int(pytest.main(command[3:]))
(run/'pytest-exit.txt').write_text(str(status)+'\n')
assert status==0,status
print('SOURCE_ROUNDOFF_AUDIT_PASS',flush=True)
