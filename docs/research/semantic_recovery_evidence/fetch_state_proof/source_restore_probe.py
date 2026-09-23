"""Explicit Fetch source restoration experiments; this is NOT a candidate gate.

No mj_getState/mj_setState, package patches, SWM patches or tolerance are used.
The renderer is not called. All env steps use the public historical SWM action.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np
import stable_worldmodel
from gymnasium_robotics.envs.fetch import fetch_env
from gymnasium_robotics.utils import mujoco_utils

PIN = '176d54532b3c0a251a4f509b74f2fdacc143a63d'
DATA_FIELDS = ('qpos', 'qvel', 'qacc_warmstart', 'mocap_pos', 'mocap_quat', 'ctrl', 'qfrc_applied', 'xfrc_applied')
GROUPS = {
    'qpos': ('qpos',), 'qvel': ('qvel',), 'warmstart': ('qacc_warmstart',),
    'mocap': ('mocap_pos', 'mocap_quat'), 'ctrl': ('ctrl',),
    'applied_forces': ('qfrc_applied', 'xfrc_applied'),
    'controller_position': ('controller_position',),
    'controller_quaternion': ('controller_quaternion',),
    'goal': ('goal',), 'time': ('time',), 'elapsed': ('elapsed',), 'rng': ('rng',),
    'controller_overwritten_inputs': ('mocap_pos', 'mocap_quat', 'ctrl', 'qfrc_applied', 'xfrc_applied'),
}

def make(seed):
    env = gym.make('swm/FetchPickAndPlace-v3')
    env.reset(seed=seed, options={'variation': []})
    return env

def limit(env):
    while not isinstance(env, gym.wrappers.TimeLimit): env = env.env
    return env

def body(env):
    return mujoco.mj_name2id(env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, 'robot0:gripper_link')

def snapshot(env):
    u = env.unwrapped; i = body(env)
    return dict({f: getattr(u.data, f).copy() for f in DATA_FIELDS},
        time=np.asarray(u.data.time), goal=u.goal.copy(), elapsed=np.asarray(limit(env)._elapsed_steps),
        rng=copy.deepcopy(u.np_random.bit_generator.state),
        controller_position=u.data.xpos[i].copy(), controller_quaternion=u.data.xquat[i].copy())

def restore(env, state, omit=()):
    """Rebuild derived MuJoCo values, then restore warmstart and controller cache.

    Omitted fields retain the target simulator's different-history values, except
    an omitted controller cache stays at mj_forward's reconstructed value.
    """
    u = env.unwrapped; i = body(env)
    old = snapshot(env)
    for f in DATA_FIELDS:
        if f not in omit: getattr(u.data, f)[:] = state[f]
    if 'time' not in omit: u.data.time = state['time']
    if 'goal' not in omit: u.goal = state['goal'].copy()
    if 'elapsed' not in omit: limit(env)._elapsed_steps = int(state['elapsed'])
    if 'rng' not in omit: u.np_random.bit_generator.state = copy.deepcopy(state['rng'])
    mujoco.mj_forward(u.model, u.data)
    # mj_forward changes warmstart: retain the old history for an omission.
    u.data.qacc_warmstart[:] = old['qacc_warmstart'] if 'qacc_warmstart' in omit else state['qacc_warmstart']
    if 'controller_position' not in omit: u.data.xpos[i] = state['controller_position']
    if 'controller_quaternion' not in omit: u.data.xquat[i] = state['controller_quaternion']

def numeric(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for k,v in value.items(): result.update(numeric(v, prefix + '/' + str(k)))
        return result
    if isinstance(value, tuple):
        result = {}
        for k,v in enumerate(value): result.update(numeric(v, prefix + '/' + str(k)))
        return result
    a = np.asarray(value)
    return {prefix:a} if a.dtype.kind in 'biuf' or isinstance(value, int) else {}

def errors(a, b, excluded=()):
    aa, bb = numeric(a), numeric(b)
    assert aa.keys() == bb.keys()
    out = {}
    for key in aa:
        if any(key == '/' + f or key.startswith('/' + f + '/') for f in excluded): continue
        if not np.array_equal(aa[key],bb[key]):
            out[key] = float(np.max(np.abs(aa[key].astype(np.float64)-bb[key].astype(np.float64))))
    return out

def case(seed, dtype, group=None, prefix=8, continuation=8):
    a,b = make(seed),make(seed+1000)
    rng = np.random.default_rng(10000+seed)
    actions = rng.uniform(-1,1,(prefix+11+continuation,4)).astype(dtype)
    try:
        for action in actions[:prefix]: a.step(action)
        for action in actions[prefix:prefix+11]: b.step(action)
        state, different = snapshot(a), snapshot(b)
        forces_zero = all(np.count_nonzero(s[f]) == 0 for s in (state,different) for f in ('qfrc_applied','xfrc_applied'))
        omitted = GROUPS[group] if group else ()
        restore(b,state,omitted)
        traces=[]
        for step,action in enumerate(actions[prefix+11:prefix+11+continuation],1):
            oa,ob=a.step(action),b.step(action)
            traces.append({'step':step,'state_errors_excluding_omitted':errors(snapshot(a),snapshot(b),omitted),
                           'output_errors':errors(oa,ob)})
        return {'seed':seed,'dtype':np.dtype(dtype).name,'prefix_steps':prefix,'different_history_steps':11,
                'applied_forces_zero_in_both_histories':forces_zero, 'omitted_group':group,'different_history_fields':list(errors(state,different)),
                'continuation':traces,'exact':all(not t['state_errors_excluding_omitted'] and not t['output_errors'] for t in traces)}
    finally:a.close();b.close()

def rng_case(seed, omit):
    a,b=make(seed),make(seed+1000)
    try:
        state=snapshot(a);restore(b,state,('rng',) if omit else ())
        oa=a.reset(options={'variation':[]});ob=b.reset(options={'variation':[]})
        return {'seed':seed,'omitted_rng':omit,'reset_output_errors':errors(oa,ob),'reset_state_errors':errors(snapshot(a),snapshot(b),('rng',) if omit else ())}
    finally:a.close();b.close()

def elapsed_case(omit):
    a,b=make(17),make(42)
    try:
        for _ in range(49):a.step(np.zeros(4,np.float64))
        for _ in range(11):b.step(np.ones(4,np.float64)*.1)
        restore(b,snapshot(a),('elapsed',) if omit else ())
        out=[]
        for step in (50,51):
            oa,ob=a.step(np.zeros(4)),b.step(np.zeros(4))
            out.append({'source_step':step,'source_terminated':bool(oa[2]),'source_truncated':bool(oa[3]),
                        'restored_terminated':bool(ob[2]),'restored_truncated':bool(ob[3]),'output_errors':errors(oa,ob)})
        return {'omitted_elapsed':omit,'steps':out}
    finally:a.close();b.close()

if __name__ == '__main__':
    root=Path(stable_worldmodel.__file__).resolve().parents[1]
    pin=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert pin==PIN
    result={'status':'SOURCE_RESTORATION_EXPERIMENT_NOT_CANDIDATE_GATE','swm_commit':pin,
      'historical_id':'swm/FetchPickAndPlace-v3','upstream_id':'FetchPickAndPlace-v4',
      'profile':{'python':platform.python_version(),'versions':{k:importlib.metadata.version(k) for k in ['numpy','gymnasium','gymnasium-robotics','mujoco','torch','tensordict']}},
      'source_sha256':{inspect.getfile(x):hashlib.sha256(Path(inspect.getfile(x)).read_bytes()).hexdigest() for x in [fetch_env,mujoco_utils]},
      'full_restoration':[case(s,d,continuation=32) for s in (0,17,42) for d in (np.float32,np.float64)],
      'omissions':[case(s,np.float64,g) for g in GROUPS for s in (0,17,42)],
      'rng_reset':[rng_case(s,omit) for s in (0,17,42) for omit in (False,True)],
      'elapsed_boundary':[elapsed_case(omit) for omit in (False,True)],
      'unproved':['minimal final schema','batch adapter','candidate parity','physical/visual Variation','full render projection','frozen renderer preservation']}
    assert all(x['exact'] for x in result['full_restoration'])
    for group in ('qpos','qvel','warmstart','controller_position','controller_quaternion','goal'):
        assert all(any(t['output_errors'] for t in x['continuation']) for x in result['omissions'] if x['omitted_group'] == group)
    assert all(x['exact'] for x in result['omissions'] if x['omitted_group'] == 'controller_overwritten_inputs')
    assert all(not x['reset_output_errors'] and not x['reset_state_errors'] for x in result['rng_reset'] if not x['omitted_rng'])
    assert all(not s['output_errors'] for x in result['elapsed_boundary'] if not x['omitted_elapsed'] for s in x['steps'])
    print(json.dumps(result,indent=2))
