import numpy as np
from stable_worldmodel.envs.pusht.env import PushT

def chase(u): return ((np.array(u.block.position)-np.array(u.agent.position))/100).astype(np.float64)
def vec(o):
 obs,r,t,tr,info=o
 return np.r_[obs['state'],r,float(t),float(tr),float(info['n_contacts'])]

def prep(seed=0):
 a,b=PushT(),PushT();a.reset(seed=seed);b.reset(seed=seed)
 for i in range(4):
  x=chase(a); y=chase(b); assert np.array_equal(x,y); oa=a.step(x);ob=b.step(y);assert np.array_equal(vec(oa),vec(ob))
 return a,b
act=np.array([0.173456789,-0.081234567],np.float64)
for field,mut in [('none',lambda u:None),
 ('agent_velocity',lambda u:setattr(u.agent,'velocity',(0,0))),
 ('block_velocity',lambda u:setattr(u.block,'velocity',(0,0))),
 ('block_angular_velocity',lambda u:setattr(u.block,'angular_velocity',0)),
 ('agent_angle',lambda u:setattr(u.agent,'angle',1.0)),
 ('agent_angular_velocity',lambda u:setattr(u.agent,'angular_velocity',2.0)),
]:
 a,b=prep(); before=(tuple(a.agent.velocity),tuple(a.block.velocity),a.block.angular_velocity)
 mut(b); va,vb=vec(a.step(act)),vec(b.step(act)); print(field,'before',before,'max',np.max(np.abs(va-vb)),'diff',va-vb)
 a.close();b.close()
