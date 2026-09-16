import json, sys, numpy as np, torch
import stable_envs as se
from stable_envs.core.env import REGISTRY, make
CELLS = ["Acrobot","CartPole","MountainCarContinuous","MountainCar","Pendulum",
 "DMCAcrobotSwingup","DMCBallInCupCatch","DMCCartpoleBalance","DMCCartpoleSwingup","DMCCheetahRun",
 "DMCFingerTurnHard","DMCHopperHop","DMCHumanoidStand","DMCManipulatorBringBall",
 "DMCPendulumSwingup","DMCQuadrupedWalk","DMCReacherHard","DMCWalkerWalk"]
def leaf(sp):
    d={"cls":type(sp).__name__}
    if hasattr(sp,"low"):
        d["low"]=np.asarray(sp.low).tolist(); d["high"]=np.asarray(sp.high).tolist()
        d["shape"]=list(sp.shape); d["dtype"]=str(sp.dtype)
    if hasattr(sp,"n"): d["n"]=int(sp.n)
    iv=getattr(sp,"init_value",None)
    d["init_value"]=np.asarray(iv).tolist() if iv is not None else None
    return d
out={}
for c in CELLS:
    try:
        e = make(c, batch=1, device="cpu")
        vs = e.make_variation_space()
        rec={"state_dim":int(e.state_dim),"state_names":list(e.state_names),
             "default_size":int(getattr(type(e),"default_size",0)),
             "reference_id":getattr(type(e),"reference_id",None),
             "flat_dim":int(vs.flat_dim),
             "flat_layout":[list(x) for x in vs.flat_layout()],
             "leaves":{}}
        for nm in vs.names():
            node=vs
            for p in nm.split("."): node=node[p]
            rec["leaves"][nm]=leaf(node)
        rec["has_step"]=hasattr(e,"step")
        rec["sim_state_names"]=list(getattr(e,"sim_state_names",()) or ())
        out[c]=rec
    except Exception as ex:
        import traceback; out[c]={"error":f"{type(ex).__name__}: {ex}", "tb":traceback.format_exc()[-800:]}
    print(c,"done",file=sys.stderr)
json.dump(out, open(sys.argv[1],"w"), indent=1)
