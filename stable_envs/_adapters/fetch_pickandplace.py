"""Private delegated semantics for the sparse historical FetchPickAndPlace only.

The source-proof report establishes warmstart and the gripper controller cache.
No SWM code or opaque MuJoCo state serialization is used at runtime.
"""
from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

FIELDS = {
    'qpos': (22, torch.float64),
    'qvel': (21, torch.float64),
    'warmstart': (21, torch.float64),
    ('controller', 'position'): (3, torch.float64),
    ('controller', 'quaternion'): (4, torch.float64),
    'goal': (3, torch.float64),
    ('lifecycle', 'elapsed_steps'): (1, torch.int64),
    ('rng', 'state_words'): (4, torch.int64),
    ('rng', 'increment_words'): (4, torch.int64),
    ('rng', 'has_uint32'): (1, torch.int64),
    ('rng', 'uinteger'): (1, torch.int64),
}


def _words(integer):
    return [(int(integer) >> (32*i)) & 0xffffffff for i in range(4)]


def _integer(words):
    return sum(int(word) << (32*i) for i, word in enumerate(words))


class FetchPickAndPlaceAdapter:
    execution_mode = 'DELEGATED_CPU'

    def __init__(self, num_envs, device='cpu', seed=0):
        import gymnasium as gym
        import mujoco
        from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv

        self.num_envs, self.device = int(num_envs), torch.device(device)
        self._mujoco = mujoco
        self.envs = [gym.wrappers.TimeLimit(MujocoFetchPickAndPlaceEnv(reward_type='sparse'), 50)
                     for _ in range(self.num_envs)]
        self._gripper = [mujoco.mj_name2id(e.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY,
                                          'robot0:gripper_link') for e in self.envs]
        self._object = [mujoco.mj_name2id(e.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY,
                                         'object0') for e in self.envs]
        self._mass = np.array([e.unwrapped.model.body_mass[i] for e,i in zip(self.envs,self._object)])
        self._closed = False
        self.reset(seed=seed)

    def _check_open(self):
        if self._closed: raise RuntimeError('FetchPickAndPlaceAdapter is closed')

    def configure_object_mass(self, values):
        self._mass = np.asarray(values,np.float64).reshape(self.num_envs).copy()
        # Historical Variation values are applied after the next upstream reset.

    def reset(self, seed=None):
        self._check_open()
        for lane,e in enumerate(self.envs):
            e.reset(seed=None if seed is None else int(seed)+lane)
            # Historical SWM applies the current mass after upstream reset.
            e.unwrapped.model.body_mass[self._object[lane]] = self._mass[lane]
        return self.get_state()

    def get_state(self):
        self._check_open()
        rows=[]
        for e,i in zip(self.envs,self._gripper):
            u=e.unwrapped; r=u.np_random.bit_generator.state
            if r['bit_generator'] != 'PCG64': raise ValueError('Fetch requires PCG64 RNG')
            values={
                'qpos':u.data.qpos.copy(), 'qvel':u.data.qvel.copy(),
                'warmstart':u.data.qacc_warmstart.copy(),
                ('controller','position'):u.data.xpos[i].copy(),
                ('controller','quaternion'):u.data.xquat[i].copy(),
                'goal':u.goal.copy(), ('lifecycle','elapsed_steps'):[e._elapsed_steps],
                ('rng','state_words'):_words(r['state']['state']),
                ('rng','increment_words'):_words(r['state']['inc']),
                ('rng','has_uint32'):[r['has_uint32']], ('rng','uinteger'):[r['uinteger']],
            }
            row=TensorDict({},batch_size=[],device=self.device)
            for key,(_,dtype) in FIELDS.items():
                row[key]=torch.as_tensor(values[key],dtype=dtype,device=self.device).clone()
            rows.append(row)
        return torch.stack(rows)

    def set_state(self,state):
        self._check_open()
        if not isinstance(state,TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError('Fetch requires full TensorDict State with batch [num_envs]')
        keys=state.keys(True,True)
        for key,(width,dtype) in FIELDS.items():
            if key not in keys: raise ValueError(f'Full State missing {key}')
            value=state[key]
            if value.shape != (self.num_envs,width) or value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError(f'Invalid State field {key}')
        for key in (('rng','state_words'),('rng','increment_words'),('rng','uinteger')):
            if ((state[key]<0)|(state[key]>0xffffffff)).any(): raise ValueError(f'Invalid PCG64 field {key}')
        if ((state['rng','has_uint32']<0)|(state['rng','has_uint32']>1)).any():
            raise ValueError('Invalid PCG64 has_uint32')
        if (state['lifecycle','elapsed_steps']<0).any(): raise ValueError('Negative elapsed_steps')
        host=state.cpu()
        for lane,(e,i) in enumerate(zip(self.envs,self._gripper)):
            row,u=host[lane],e.unwrapped
            u.data.qpos[:]=row['qpos'].numpy();u.data.qvel[:]=row['qvel'].numpy()
            u.goal=row['goal'].numpy().copy()
            # Controller overwrites mocap and ctrl before every public step.
            # Applied-force arrays remain zero under this registration's API.
            u.data.qfrc_applied[:]=0;u.data.xfrc_applied[:]=0
            self._mujoco.mj_forward(u.model,u.data)
            u.data.qacc_warmstart[:]=row['warmstart'].numpy()
            u.data.xpos[i]=row['controller','position'].numpy()
            u.data.xquat[i]=row['controller','quaternion'].numpy()
            e._elapsed_steps=row['lifecycle','elapsed_steps'].item()
            u.np_random.bit_generator.state={
                'bit_generator':'PCG64',
                'state':{'state':_integer(row['rng','state_words'].tolist()),
                         'inc':_integer(row['rng','increment_words'].tolist())},
                'has_uint32':row['rng','has_uint32'].item(),'uinteger':row['rng','uinteger'].item(),
            }

    def step(self,actions):
        self._check_open()
        if torch.is_tensor(actions): actions=actions.detach().cpu().numpy()
        actions=np.asarray(actions)
        if actions.shape != (self.num_envs,4): raise ValueError('Fetch action must have shape (num_envs,4)')
        # Preserve public dtype: upstream multiplication before concatenation is observable.
        outputs=[e.step(actions[lane]) for lane,e in enumerate(self.envs)]
        observations=[np.concatenate([o['observation'],o['desired_goal']]).astype(np.float32)
                      for o,*_ in outputs]
        observation=torch.as_tensor(np.stack(observations),device=self.device)
        info=TensorDict({
            'proprio':torch.as_tensor(np.stack([x[0]['observation'] for x in outputs]),device=self.device),
            'state':observation.clone(),
            'goal_state':torch.as_tensor(np.stack([x[0]['desired_goal'] for x in outputs]),device=self.device),
            'is_success':torch.as_tensor(np.array([x[4]['is_success'] for x in outputs]),device=self.device),
        },[self.num_envs],device=self.device)
        return TensorDict({
            'state':self.get_state(), 'observation':observation,
            'reward':torch.as_tensor(np.array([x[1] for x in outputs]),device=self.device),
            'terminated':torch.as_tensor(np.array([x[2] for x in outputs]),device=self.device),
            'truncated':torch.as_tensor(np.array([x[3] for x in outputs]),device=self.device),
            'success':info['is_success'].clone(),'info':info,
        },[self.num_envs],device=self.device)

    def close(self):
        if not self._closed:
            for e in self.envs:e.close()
            self._closed=True
