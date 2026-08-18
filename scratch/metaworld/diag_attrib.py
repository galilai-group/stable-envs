"""Attribute the residual to individual MOVER geoms, so parity_note names a mechanism, not a guess."""
import os, sys, json
os.environ.setdefault("MUJOCO_GL","egl")
sys.path.insert(0,"/mnt/vast/home/randall.balestriero/code/stable-envs")
import numpy as np, mujoco, torch
import stable_envs as se
from stable_envs.render import mjwarp as MW
from stable_envs.testing import conformance as C
import stable_envs.envs.metaworld as MWE

task=sys.argv[1]; size=int(sys.argv[2]) if len(sys.argv)>2 else 480; n=8
cls=se.REGISTRY[MWE._class_name(task)]
env=cls(batch=n,size=size,device="cuda")
b=MWE._build_bundle(task); m=b.model
states=env.sample_states(n,rng=np.random.default_rng(0))
official=cls.reference_render(C.reference_env(cls,size),states).astype(np.int16)
ours=env.render(states).cpu().numpy().astype(np.int16)
env.render(states)   # AGAIN: the first render's movers-off launch left c.seg holding the
                     # movers-OFF segmentation; the second render replays only the movers-on graph
c=MWE._context(task,size,n)
seg=c.seg[:n].cpu().numpy()
d=np.abs(official-ours).max(-1)
alpha=MW.effective_alpha(m,"geom")
rows=[]
for gid in np.unique(seg[seg>=0]):
    mask=(seg==gid)
    if mask.sum()==0: continue
    mid=int(m.geom_matid[gid])
    t=m.mat_texid[mid] if mid>=0 else -1
    tid=int(np.max(t)) if np.ndim(t) else int(t)
    rows.append({"geom":mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(gid)) or f"geom{gid}",
        "id":int(gid),"mover":bool(gid in set(int(x) for x in b.movers)),
        "type":int(m.geom_type[gid]),"alpha":round(float(alpha[gid]),3),
        "mat":(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_MATERIAL,mid) if mid>=0 else None),
        "tex_type":(int(m.tex_type[tid]) if tid>=0 else None),
        "px_frac":round(float(mask.mean()),5),"mean_err":round(float(d[mask].mean()),2),
        "err_share":round(float(d[mask].sum()/max(d.sum(),1)),4)})
rows.sort(key=lambda r:-r["err_share"])
print("RESULT "+json.dumps({"task":task,"total_mean":round(float(d.mean()),3),"top":rows[:10]}))
