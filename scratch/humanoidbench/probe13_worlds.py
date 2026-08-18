import os, sys
os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np, torch
sys.path.insert(0,"/mnt/vast/home/randall.balestriero/code/stable-envs")
import stable_envs as se
from stable_envs.envs import humanoidbench as HB
task, robot = sys.argv[1], sys.argv[2]
name=f"HumanoidBench{'H1Hand' if robot=='h1hand' else 'H1'}{''.join(p.capitalize() for p in task.split('_'))}"
env=se.make(name,batch=1,device="cuda")
pool=env.sample_states(4)
c=HB._context(robot,task,env.size,HB.MJW_CHUNK)
sty=env.resolve_style(None,None,batch=HB.MJW_CHUNK)
q=np.tile(pool[0],(HB.MJW_CHUNK,1)).astype(np.float32)
env._upload(c, torch.as_tensor(q,device=env.device), sty, 0, HB.MJW_CHUNK)
torch.cuda.synchronize(); c.wp.capture_launch(c.graph); c.wp.synchronize()
f=c.bgra[...,[2,1,0]].contiguous().cpu().numpy()
d=np.abs(f.astype(np.int16)-f[0:1].astype(np.int16))
print(name,"identical qpos in all 64 worlds: max across worlds", int(d.max()),
      "| worlds differing:", int((d.reshape(64,-1).max(1)>0).sum()),
      "| worst px count", int((d.max(-1)>0).sum()))
