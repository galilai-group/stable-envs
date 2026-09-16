"""Does the leaf reach the COMPILED model at all? Hash every numeric MjModel array."""
import json, sys, hashlib, numpy as np, gymnasium, stable_worldmodel, mujoco  # noqa
IDS = ["swm/AcrobotDMControl-v0","swm/BallInCupDMControl-v0","swm/CartpoleDMControl-v0",
 "swm/CheetahDMControl-v0","swm/FingerDMControl-v0","swm/HopperDMControl-v0",
 "swm/HumanoidDMControl-v0","swm/ManipulatorDMControl-v0","swm/PendulumDMControl-v0",
 "swm/QuadrupedDMControl-v0","swm/ReacherDMControl-v0","swm/WalkerDMControl-v0"]
def holder(e):
    n=e
    while n is not None:
        if getattr(n,"variation_space",None) is not None: return n
        n=getattr(n,"env",None)
def node(vs,nm):
    n=vs
    for p in nm.split("."): n=n[p]
    return n
def probe(sp):
    iv=np.asarray(sp.init_value)
    if hasattr(sp,"n"): return int((int(iv)+1)%int(sp.n))
    lo=np.asarray(sp.low,np.float64).reshape(np.shape(iv)); hi=np.asarray(sp.high,np.float64).reshape(np.shape(iv))
    far=np.where(np.abs(iv-lo)>np.abs(iv-hi),lo,hi)
    return np.asarray(iv+0.9*(far-iv), sp.dtype)
ARRS = None
def digests(h):
    global ARRS
    m=h.env.physics.model
    m=m.ptr if hasattr(m,"ptr") else m
    if ARRS is None:
        ARRS=sorted(a for a in dir(m) if not a.startswith("_")
                    and isinstance(getattr(m,a,None), np.ndarray))
    out={}
    for a in ARRS:
        try: out[a]=hashlib.sha256(np.ascontiguousarray(np.asarray(getattr(m,a))).tobytes()).hexdigest()[:12]
        except Exception: pass
    out["__opt_gravity"]=hashlib.sha256(np.asarray(m.opt.gravity).tobytes()).hexdigest()[:12]
    out["__scalars"]=hashlib.sha256(str([int(getattr(m,k)) for k in ("nq","nv","njnt","ngeom","nsite","nbody","ntex")]).encode()).hexdigest()[:12]
    return out
res={}
for i in IDS:
    e=gymnasium.make(i); h=holder(e); vs=h.variation_space
    h.reset(seed=5); base=digests(h)
    rec={}
    for nm in vs.names():
        h.reset(seed=5, options={"variation_values":{nm:probe(node(vs,nm))}})
        d=digests(h)
        changed=sorted(k for k in base if d.get(k)!=base[k])
        rec[nm]={"model_arrays_changed":changed}
        h.reset(seed=5)  # restore
    res[i]=rec; print(i,"done",file=sys.stderr); e.close()
json.dump(res,open(sys.argv[1],"w"),indent=1)
