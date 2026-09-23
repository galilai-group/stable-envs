"""Exact adapter-only Fetch probe; renderer/public-environment gates outstanding."""
import importlib.metadata
import json
import platform
import sys
import numpy as np
import torch
import gymnasium as gym
import stable_worldmodel
from stable_envs._adapters.fetch_pickandplace import FetchPickAndPlaceAdapter


def oracle(envs):
    result=[]
    for e in envs:
        u=e.unwrapped
        while not isinstance(e,gym.wrappers.TimeLimit):e=e.env
        r=u.np_random.bit_generator.state
        i=u.model.body('robot0:gripper_link').id
        result.append({'qpos':u.data.qpos.copy(),'qvel':u.data.qvel.copy(),
            'warmstart':u.data.qacc_warmstart.copy(),'position':u.data.xpos[i].copy(),
            'quaternion':u.data.xquat[i].copy(),'goal':u.goal.copy(),'elapsed':e._elapsed_steps,'rng':r})
    return result


def compare_state(state,expected):
    for lane,r in enumerate(expected):
        row=state[lane].cpu()
        for f in ('qpos','qvel','warmstart','goal'):
            assert np.array_equal(row[f].numpy(),r[f]),(lane,f,np.max(abs(row[f].numpy()-r[f])))
        for f in ('position','quaternion'):
            assert np.array_equal(row['controller',f].numpy(),r[f]),(lane,f)
        assert row['lifecycle','elapsed_steps'].item()==r['elapsed']
        for field,source in [('state_words','state'),('increment_words','inc')]:
            value=sum(int(x) << (32*i) for i,x in enumerate(row['rng',field]))
            assert value==r['rng']['state'][source]
        for f in ('has_uint32','uinteger'):assert row['rng',f].item()==r['rng'][f]


def compare_output(out,refs):
    for lane,r in enumerate(refs):
        for key,index in [('observation',0),('reward',1),('terminated',2),('truncated',3)]:
            assert np.array_equal(out[key][lane].cpu().numpy(),np.asarray(r[index])),(lane,key)
        for f in ('proprio','state','goal_state','is_success'):
            assert np.array_equal(out['info',f][lane].cpu().numpy(),r[4][f]),(lane,'info',f)


def case(seed,batch,dtype):
    refs=[gym.make('swm/FetchPickAndPlace-v3') for _ in range(batch)]
    a=FetchPickAndPlaceAdapter(batch,seed=seed);b=FetchPickAndPlaceAdapter(batch,seed=seed+999)
    rng=np.random.default_rng(20000+seed)
    actions=rng.uniform(-2,2,(40,batch,4)).astype(dtype)
    actions[0,:,:]=[-1,1,-1,1];actions[1,:,:]=[.1,-.2,.3,-.4]
    try:
        for i,e in enumerate(refs):e.reset(seed=seed+i,options={'variation':[]})
        compare_state(a.get_state(),oracle(refs))
        independent=a.get_state();a_state=a.get_state();independent['qpos'][0,0]+=10
        assert torch.equal(a.get_state()['qpos'],a_state['qpos'])
        for action in actions[:32]:
            r=[e.step(action[i]) for i,e in enumerate(refs)];out=a.step(torch.from_numpy(action))
            compare_output(out,r);compare_state(out['state'],oracle(refs))
        for action in actions[5:16]:b.step(action)
        state=a.get_state();b.set_state(state)
        for key in state.keys(True,True):assert torch.equal(b.get_state()[key],state[key]),key
        for action in actions[32:]:
            r=[e.step(action[i]) for i,e in enumerate(refs)];out=b.step(action)
            compare_output(out,r);compare_state(out['state'],oracle(refs))
        # Full restoration also restores future unseeded simulator reset RNG.
        for e in refs:e.reset(options={'variation':[]})
        compare_state(b.reset(),oracle(refs))
        for i,e in enumerate(refs):e.reset(seed=seed+i,options={'variation':[]})
        compare_state(b.reset(seed=seed),oracle(refs))
        return {'seed':seed,'batch':batch,'dtype':np.dtype(dtype).name,'reset_exact':True,
                'trajectory_steps':32,'different_history_steps':11,'restored_continuation_steps':8,
                'independent_snapshots':True,'roundtrip_exact':True,'outputs_exact':True,
                'unseeded_reset_rng_replay':True,'repeated_seeded_reset':True}
    finally:
        a.close();b.close()
        for e in refs:e.close()


def lifecycle():
    e=gym.make('swm/FetchPickAndPlace-v3');a=FetchPickAndPlaceAdapter(1,seed=17);b=FetchPickAndPlaceAdapter(1,seed=999)
    try:
        e.reset(seed=17,options={'variation':[]});rows=[]
        for step in range(1,52):
            if step==50:b.set_state(a.get_state())
            r=e.step(np.zeros(4,np.float64));out=a.step(np.zeros((1,4),np.float64));compare_output(out,[r]);compare_state(a.get_state(),oracle([e]))
            if step>=50:
                restored=b.step(np.zeros((1,4)));compare_output(restored,[r]);compare_state(b.get_state(),oracle([e]))
            if step in (49,50,51):rows.append({'step':step,'terminated':bool(r[2]),'truncated':bool(r[3]),'exact':True})
        try:a.set_state(torch.zeros((1,25)))
        except TypeError:pass
        else:raise AssertionError('compact row restoration accepted')
        return {'steps':rows,'compact_restore_refused':True}
    finally:a.close();b.close();e.close()

if __name__=='__main__':
    result={'status':'ADAPTER_ONLY_NOT_PUBLIC_ENVIRONMENT_ACCEPTANCE','profile':{'python':platform.python_version(),'versions':{k:importlib.metadata.version(k) for k in ('numpy','gymnasium','gymnasium-robotics','mujoco')}},
       'cases':[case(s,b,d) for s in (0,17,42) for b in (1,3) for d in (np.float32,np.float64)],
       'lifecycle':lifecycle(),
       'outstanding':['public environment integration','reward threshold candidate gate','Variation gate','full/private render equality','render non-mutation','frozen renderer preservation']}
    print(json.dumps(result,indent=2))
