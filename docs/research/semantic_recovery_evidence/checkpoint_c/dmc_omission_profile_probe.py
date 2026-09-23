"""Read-only Oscar CPU-profile diagnostic; never constitutes acceptance."""
import importlib.metadata
import json
import platform
import stable_envs as se
from scripts.recovery.dmc_aligned_gate import CASES, _omission_experiments
from tests.helpers.reference.dmc_aligned_semantics import DMCAlignedReference
print('DIAGNOSTIC_ONLY', json.dumps({'platform': platform.platform(), 'versions': {name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'mujoco', 'dm-control']}}), flush=True)
for domain, name in CASES.items():
    reference = DMCAlignedReference(domain, 1)
    try:
        with se.make(name, num_envs=1, device='cpu') as env:
            print('DOMAIN', domain, flush=True)
            print(json.dumps(_omission_experiments(domain, env, reference)), flush=True)
    finally:
        reference.close()
