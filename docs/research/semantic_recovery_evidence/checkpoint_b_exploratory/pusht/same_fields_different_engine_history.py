import numpy as np
from stable_worldmodel.envs.pusht.env import PushT
optsA={'variation':[],'state':np.array([100,100,400,400,0,0,0],np.float64),'goal_state':np.array([200,200,300,300,0,0,0],np.float64)}
optsB={'variation':[],'state':np.array([100,100,250,250,0,0,0],np.float64),'goal_state':np.array([200,200,300,300,0,0,0],np.float64)}
a,b=PushT(),PushT();oa,_=a.reset(seed=0,options=optsA);b.reset(seed=0,options=optsB)
# copy every public body field but not private space/arbiter cache
for x,y in [(a.agent,b.agent),(a.block,b.block)]:
 y.angle=x.angle;y.position=tuple(x.position);y.velocity=tuple(x.velocity);y.angular_velocity=x.angular_velocity;y.force=tuple(x.force);y.torque=x.torque
 b.space.reindex_shapes_for_body(y)
b.goal_state=a.goal_state.copy();b._goal=a._goal.copy()
act=np.zeros(2,np.float32);xa=a.step(act);xb=b.step(act)
print('same accessible pre',oa['state'].tolist())
print('A post',xa[0]['state'].tolist(),'contacts',xa[4]['n_contacts'])
print('B post',xb[0]['state'].tolist(),'contacts',xb[4]['n_contacts'])
print('delta', (xa[0]['state']-xb[0]['state']).tolist(), 'reward_delta',float(xa[1]-xb[1]))
a.close();b.close()
