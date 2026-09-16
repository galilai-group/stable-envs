"""Which classic-control physics leaves move OFFICIAL pixels at a fixed state?"""
import json,sys,numpy as np,gymnasium,stable_worldmodel  # noqa
from stable_envs.core.env import make
P={"swm/AcrobotControl-v1":"Acrobot","swm/CartPoleControl-v1":"CartPole",
   "swm/PendulumControl-v1":"Pendulum","swm/MountainCarControl-v0":"MountainCar",
   "swm/MountainCarContinuousControl-v0":"MountainCarContinuous"}
def holder(e):
    n=e
    while n is not None:
        if getattr(n,"variation_space",None) is not None: return n
        n=getattr(n,"env",None)
out={}
for sid,cell in P.items():
    env=make(cell,batch=1,device="cpu"); cls=type(env)
    states=env.sample_states(8,rng=np.random.default_rng(11))
    e=gymnasium.make(sid,render_mode="rgb_array"); h=holder(e); vs=h.variation_space; ref=h.env.unwrapped
    h.reset(seed=0)
    def frames():
        fr=[]
        for st in states:
            cls.reference_set_state(ref,st); fr.append(h.render().copy())
        return np.stack(fr)
    def setleaf(nm,v):
        n=vs
        for p in nm.split("."): n=n[p]
        n.set_value(np.asarray(v,n.dtype))
    for nm in vs.names():
        n=vs
        for p in nm.split("."): n=n[p]
        setleaf(nm,np.asarray(n.init_value))
    h._apply_physical_variations([]); base=frames()
    rec={}
    for nm in vs.names():
        if not nm.startswith("physics."): continue
        n=vs
        for p in nm.split("."): n=n[p]
        iv=np.asarray(n.init_value,np.float64); lo=np.asarray(n.low,np.float64).reshape(iv.shape); hi=np.asarray(n.high,np.float64).reshape(iv.shape)
        far=np.where(np.abs(iv-lo)>np.abs(iv-hi),lo,hi); pv=iv+0.9*(far-iv)
        setleaf(nm,pv); h._apply_physical_variations([nm])
        f=frames(); d=np.abs(f.astype(np.int32)-base.astype(np.int32))
        rec[nm]={"probe":pv.tolist(),"px":int((d.max(-1)>0).sum()),"max":int(d.max()),
                 "px_total":int(base.shape[0]*base.shape[1]*base.shape[2])}
        setleaf(nm,iv); h._apply_physical_variations([nm])
        print(sid,nm,rec[nm]["px"],file=sys.stderr)
    out[sid]=rec; e.close()
json.dump(out,open(sys.argv[1],"w"),indent=1)
