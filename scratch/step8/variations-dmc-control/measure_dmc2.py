"""Robustness pass: the 0-px leaves over 8 states, and the target leaves with the target VISIBLE."""
import json, sys, numpy as np, gymnasium, stable_worldmodel  # noqa
IDS = ["swm/AcrobotDMControl-v0","swm/BallInCupDMControl-v0","swm/CartpoleDMControl-v0",
 "swm/CheetahDMControl-v0","swm/FingerDMControl-v0","swm/HopperDMControl-v0",
 "swm/HumanoidDMControl-v0","swm/ManipulatorDMControl-v0","swm/PendulumDMControl-v0",
 "swm/QuadrupedDMControl-v0","swm/ReacherDMControl-v0","swm/WalkerDMControl-v0"]
SIZE = {"swm/ManipulatorDMControl-v0":224, "swm/QuadrupedDMControl-v0":224}
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
out={}
for i in IDS:
    sz=SIZE.get(i,84); e=gymnasium.make(i); h=holder(e); vs=h.variation_space; cam=h.camera_id
    # 8 states from real rollouts
    states=[]
    for s in range(8):
        h.reset(seed=100+s)
        for _ in range(s*3): h.step(np.zeros(h.action_space.shape, np.float32))
        states.append(np.concatenate([np.asarray(h.env.physics.data.qpos,np.float64).copy(),
                                      np.asarray(h.env.physics.data.qvel,np.float64).copy()]))
    has_rt = "rendering.render_target" in vs.names()
    def frames(extra=None, visible=False):
        base={}
        if visible and has_rt: base["rendering.render_target"]=1
        if extra: base.update(extra)
        fr=[]
        for k,st in enumerate(states):
            o={"state":st}
            if base: o["variation_values"]=dict(base)
            h.reset(seed=100+k, options=o)
            fr.append(h.render(width=sz,height=sz,camera_id=cam).copy())
        return np.stack(fr)
    rec={"size":sz,"n_states":8,"has_render_target":has_rt,"hidden":{},"visible":{}}
    for mode,vis in (("hidden",False),("visible",True)):
        if mode=="visible" and not has_rt: continue
        b=frames(visible=vis)
        for nm in vs.names():
            if nm=="rendering.render_target": continue
            f=frames({nm:probe(node(vs,nm))}, visible=vis)
            d=np.abs(f.astype(np.int32)-b.astype(np.int32))
            rec[mode][nm]={"px":int((d.max(-1)>0).sum()),"max":int(d.max()),
                           "per_state_px":[int((x.max(-1)>0).sum()) for x in d]}
        if has_rt:
            f=frames({"rendering.render_target":(0 if vis else 1)}, visible=vis)
            d=np.abs(f.astype(np.int32)-b.astype(np.int32))
            rec[mode]["rendering.render_target"]={"px":int((d.max(-1)>0).sum()),"max":int(d.max()),
                "per_state_px":[int((x.max(-1)>0).sum()) for x in d]}
    out[i]=rec; print(i,"done",file=sys.stderr); e.close()
json.dump(out,open(sys.argv[1],"w"),indent=1)
