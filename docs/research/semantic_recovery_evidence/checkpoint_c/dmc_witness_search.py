"""Bounded deterministic warmstart witness search; diagnostic, not acceptance."""
import json
import importlib.metadata
import numpy as np
import stable_envs as se
from scripts.recovery.dmc_aligned_gate import CASES, PREFIXES, _max_error
from tests.helpers.reference.dmc_aligned_semantics import DMCAlignedReference
from tests.helpers.reference.semantics import compare_state, check_step_matches_reference
print('DIAGNOSTIC_ONLY', json.dumps({name: importlib.metadata.version(name) for name in ['torch','numpy','mujoco','dm-control']}), flush=True)
results={}
for domain, name in CASES.items():
    reference=DMCAlignedReference(domain,1)
    try:
        with se.make(name,num_envs=1,device='cpu') as env:
            actions=np.random.default_rng(200).uniform(-1,1,(96,1,reference.action_dim))
            trials=[]
            prefixes=list(dict.fromkeys([PREFIXES[domain],1,2,3,4,6,7,8,10,12,16,20,24,32,48,64]))
            for prefix in prefixes:
                compare_state(env.reset(seed=0),reference.reset(seed=0))
                check_step_matches_reference(env,reference,actions[:prefix])
                saved=env.get_state()
                data=env._simulator.envs[0].physics.data
                ncon,nefc=int(data.ncon),int(data.nefc)
                for action in -actions[prefix:prefix+9]: env.step(action)
                wrong=saved.clone();wrong['warmstart']=env.get_state()['warmstart']
                env.set_state(wrong)
                continuation=actions[prefix:prefix+8]
                bad=[env.step(action) for action in continuation]
                expected=[reference.step(action) for action in continuation]
                without=max(_max_error(a,b,exclude=(('state','warmstart'),)) for a,b in zip(bad,expected,strict=True))
                env.set_state(saved)
                good=[env.step(action) for action in continuation]
                restored=max(_max_error(a,b) for a,b in zip(good,expected,strict=True))
                row=dict(domain=domain,seed=0,action_seed=200,prefix=prefix,ncon=ncon,nefc=nefc,error_without=without,error_with=restored)
                print('WITNESS_TRIAL',json.dumps(row),flush=True);trials.append(row)
                assert restored==0,('Restored continuation differs',row)
                if without>1e-10: break
            results[domain]=trials
    finally: reference.close()
print('WITNESS_SEARCH_COMPLETE',json.dumps(results),flush=True)
