"""Observe the first failing upstream joint assignment without modifying it."""
import importlib.metadata
import json
import numpy as np
import mujoco
import gymnasium as gym
import gymnasium_robotics
from gymnasium_robotics.utils import mujoco_utils
print('PROFILE',json.dumps({n:importlib.metadata.version(n) for n in ['gymnasium-robotics','gymnasium','mujoco','numpy']}),flush=True)
original=mujoco_utils.set_joint_qpos
def diagnose(model,data,name,value):
    i=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
    joint_type=model.jnt_type[i]
    print('JOINT',json.dumps({'name':name,'id':i,'type':int(joint_type),'python_type':str(type(joint_type)),'enums':{n:int(getattr(mujoco.mjtJoint,n)) for n in ['mjJNT_FREE','mjJNT_BALL','mjJNT_HINGE','mjJNT_SLIDE']},'numpy_scalar_equality':{n:bool(joint_type==getattr(mujoco.mjtJoint,n)) for n in ['mjJNT_FREE','mjJNT_BALL','mjJNT_HINGE','mjJNT_SLIDE']},'integer_equality':{n:bool(int(joint_type)==int(getattr(mujoco.mjtJoint,n))) for n in ['mjJNT_FREE','mjJNT_BALL','mjJNT_HINGE','mjJNT_SLIDE']},'model_nq':model.nq,'model_nv':model.nv,'model_nu':model.nu,'model_na':model.na,'model_nmocap':model.nmocap}),flush=True)
    return original(model,data,name,value)
mujoco_utils.set_joint_qpos=diagnose
env=gym.make('FetchPickAndPlace-v4')
env.close()
