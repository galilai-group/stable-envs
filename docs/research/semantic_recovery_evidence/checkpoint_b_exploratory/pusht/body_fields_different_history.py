import numpy as np
from stable_worldmodel.envs.pusht.env import PushT

def body_state(u):
    return dict(
      agent_p=np.array(u.agent.position), agent_v=np.array(u.agent.velocity), agent_a=float(u.agent.angle), agent_w=float(u.agent.angular_velocity),
      block_p=np.array(u.block.position), block_v=np.array(u.block.velocity), block_a=float(u.block.angle), block_w=float(u.block.angular_velocity),
      block_f=np.array(u.block.force), block_t=float(u.block.torque),
      goal=np.array(u.goal_state,copy=True), goal_image=np.array(u._goal,copy=True),
    )

def restore(u,s):
    u.agent.position=tuple(s['agent_p']);u.agent.velocity=tuple(s['agent_v']);u.agent.angle=s['agent_a'];u.agent.angular_velocity=s['agent_w']
    u.block.angle=s['block_a'];u.block.position=tuple(s['block_p']);u.block.velocity=tuple(s['block_v']);u.block.angular_velocity=s['block_w'];u.block.force=tuple(s['block_f']);u.block.torque=s['block_t']
    u.goal_state=s['goal'].copy();u._goal=s['goal_image'].copy()
    u.space.reindex_shapes_for_body(u.agent);u.space.reindex_shapes_for_body(u.block)
    u.agent.activate();u.block.activate()

def act_to_block(u):
    return ((np.array(u.block.position)-np.array(u.agent.position))/100).astype(np.float64)

def outvec(out):
    obs,r,t,tr,info=out
    return np.r_[obs['state'],r,float(t),float(tr),float(info['n_contacts'])]

for seed in range(30):
    a=PushT();b=PushT();a.reset(seed=seed);b.reset(seed=seed+1000)
    contacts=[]
    for i in range(30):
      o=a.step(act_to_block(a)); contacts.append(o[4]['n_contacts'])
      if o[4]['n_contacts']:
        break
    if not contacts[-1]:
      a.close();b.close();continue
    snap=body_state(a)
    # establish a very different contact history in b
    for _ in range(10): b.step(act_to_block(b))
    restore(b,snap)
    action=np.array([0.173456789,-0.081234567],np.float64)
    oa=a.step(action);ob=b.step(action)
    va,vb=outvec(oa),outvec(ob)
    print('seed',seed,'contact_step',len(contacts),'prior_contacts',contacts[-1], 'max',np.max(np.abs(va-vb)),'diff',va-vb)
    print(' post block v',tuple(a.block.velocity),tuple(b.block.velocity),'w',a.block.angular_velocity,b.block.angular_velocity)
    a.close();b.close()
    if np.max(np.abs(va-vb))>0: break
