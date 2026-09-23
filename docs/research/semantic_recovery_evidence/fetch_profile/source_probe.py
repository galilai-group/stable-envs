"""Bounded pinned-source Fetch audit; no candidate restoration/acceptance claim."""
import hashlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import gymnasium as gym
import gymnasium_robotics
import stable_worldmodel
from gymnasium_robotics.envs.fetch import fetch_env, pick_and_place
from gymnasium_robotics.envs import robot_env
from gymnasium_robotics.utils import mujoco_utils

pin=subprocess.check_output(['git','-C',str(Path(stable_worldmodel.__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()
assert pin=='176d54532b3c0a251a4f509b74f2fdacc143a63d'
name='swm/FetchPickAndPlace-v3'
a,b=gym.make(name),gym.make(name)
try:
 obs,info=a.reset(seed=17);b.reset(seed=17)
 u=a.unwrapped
 chain=[];w=a
 while isinstance(w,gym.Wrapper):
  chain.append({'class':type(w).__name__,'max_episode_steps':getattr(w,'_max_episode_steps',None)})
  w=w.env
 limits=[];w=a
 while isinstance(w,gym.Wrapper):
  if isinstance(w,gym.wrappers.TimeLimit):limits.append(w)
  w=w.env
 assert len(limits)==1
 result={'status':'SOURCE_AUDIT_ONLY_NOT_CANDIDATE_GATE','swm_commit':pin,'historical_id':name,'registration_kwargs':gym.spec(name).kwargs,'upstream_class':type(u).__name__,'wrapper_chain':chain,'profile':{'platform':platform.platform(),'versions':{k:importlib.metadata.version(k) for k in ['gymnasium-robotics','gymnasium','mujoco','numpy','torch','tensordict']}},'model':{k:int(getattr(u.model,k)) for k in ['nq','nv','nu','na','nmocap','nuserdata','npluginstate']},'task':{k:getattr(u,k) for k in ['has_object','block_gripper','n_substeps','reward_type','distance_threshold','obj_range','target_range','target_in_the_air']},'dt':u.dt,'action_shape':list(u.action_space.shape),'action_dtype':str(u.action_space.dtype),'observation_shape':list(obs.shape),'observation_dtype':str(obs.dtype),'proprio_shape':list(info['proprio'].shape),'proprio_dtype':str(info['proprio'].dtype),'rng_class':type(u.np_random.bit_generator).__name__,'state_candidates':{k:list(np.shape(getattr(u.data,k))) for k in ['qpos','qvel','mocap_pos','mocap_quat','qacc_warmstart','act','ctrl','qfrc_applied','xfrc_applied']},'source_files_sha256':{inspect.getfile(m):hashlib.sha256(Path(inspect.getfile(m)).read_bytes()).hexdigest() for m in [fetch_env,pick_and_place,robot_env,mujoco_utils]}}
 action=np.array([2.,-2.,.3,.75],np.float64)
 left,right=a.step(action),b.step(np.clip(action,-1,1))
 assert np.array_equal(left[0],right[0]) and left[1:4]==right[1:4]
 for field in result['state_candidates']:
  assert np.array_equal(getattr(u.data,field),getattr(b.unwrapped.data,field)),field
 result['out_of_range_probe']={'action':action.tolist(),'clipped_equivalence_exact':True,'compared_data_fields':list(result['state_candidates'])}
 boundary=[]
 for dist in [np.nextafter(.05,0),.05,np.nextafter(.05,np.inf)]:
  achieved=np.array([dist,0.,0.]);goal=np.zeros(3)
  boundary.append({'distance':float(dist),'reward':float(u.compute_reward(achieved,goal,{})),'success':float(u._is_success(achieved,goal))})
 result['reward_success_boundary']=boundary
 a.reset(seed=17);steps=[]
 for step in range(1,52):
  out=a.step(np.zeros(4,np.float64))
  if step in [49,50,51]:steps.append({'public_step':step,'elapsed_steps':limits[0]._elapsed_steps,'terminated':bool(out[2]),'truncated':bool(out[3]),'time':float(u.data.time)})
 result['timelimit_probe']=steps
 result['unproved']=['candidate State sufficiency','omission necessity','different-history restoration','candidate reset/step/reward parity','candidate Variation','candidate renderer preservation']
 print(json.dumps(result,indent=2))
finally:
 a.close();b.close()
