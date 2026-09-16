"""Per-leaf live measurement on the SWM DMC wrappers.

For each leaf: reset at a FIXED state with the leaf at its default, render; reset at the same
state with the leaf at a probe value, render; report pixel diff AND a digest of the compiled
mujoco model's appearance/physics arrays, so 'changed nothing at all' is distinguishable from
'changed the model but not this frame'.
"""
import json, sys, hashlib, numpy as np, gymnasium, stable_worldmodel  # noqa
IDS = ["swm/AcrobotDMControl-v0","swm/BallInCupDMControl-v0","swm/CartpoleDMControl-v0",
 "swm/CheetahDMControl-v0","swm/FingerDMControl-v0","swm/HopperDMControl-v0",
 "swm/HumanoidDMControl-v0","swm/ManipulatorDMControl-v0","swm/PendulumDMControl-v0",
 "swm/QuadrupedDMControl-v0","swm/ReacherDMControl-v0","swm/WalkerDMControl-v0"]
SIZE = {"swm/ManipulatorDMControl-v0":224, "swm/QuadrupedDMControl-v0":224}

def holder(e):
    n = e
    while n is not None:
        if getattr(n, "variation_space", None) is not None: return n
        n = getattr(n, "env", None)
    return None

def node(vs, nm):
    n = vs
    for p in nm.split("."): n = n[p]
    return n

def probe_value(sp):
    lo, hi, iv = getattr(sp,"low",None), getattr(sp,"high",None), np.asarray(sp.init_value)
    if hasattr(sp, "n"):
        return int(1 - int(iv)) if int(sp.n)==2 else int((int(iv)+1) % int(sp.n))
    lo = np.asarray(lo, np.float64).reshape(np.shape(iv)); hi = np.asarray(hi, np.float64).reshape(np.shape(iv))
    far = np.where(np.abs(iv-lo) > np.abs(iv-hi), lo, hi)  # the farther endpoint
    v = iv + 0.9 * (far - iv)                              # 90% of the way, strictly interior
    return np.asarray(v, sp.dtype)

def model_sig(h):
    m = h.env.physics.model
    m = m.ptr if hasattr(m,"ptr") else m
    parts = []
    for a in ("geom_rgba","mat_rgba","geom_type","geom_size","light_diffuse","body_mass",
              "geom_friction","opt","site_rgba","site_type","site_size","tex_data","jnt_type"):
        if a == "opt":
            parts.append(np.asarray(m.opt.gravity).tobytes())
        else:
            try: parts.append(np.ascontiguousarray(np.asarray(getattr(m, a))).tobytes())
            except Exception: parts.append(b"")
    return {"nq": int(m.nq), "nv": int(m.nv),
            "sig": hashlib.sha256(b"|".join(parts)).hexdigest()[:16],
            "appearance": hashlib.sha256(b"|".join(parts[:5]+parts[7:12])).hexdigest()[:16]}

out = {}
for i in IDS:
    sz = SIZE.get(i, 84)
    e = gymnasium.make(i); h = holder(e); vs = h.variation_space
    cam = h.camera_id
    m0 = h.env.physics.model
    m0 = m0.ptr if hasattr(m0,"ptr") else m0
    nq, nv = int(m0.nq), int(m0.nv)
    rng = np.random.default_rng(0)
    h.reset(seed=3)
    st = np.concatenate([np.asarray(h.env.physics.data.qpos,np.float64).copy(),
                         np.asarray(h.env.physics.data.qvel,np.float64).copy()])
    def frame(vals=None):
        opts = {"state": st}
        if vals: opts["variation_values"] = vals
        h.reset(seed=3, options=opts)
        sg = model_sig(h)
        dd = h.env.physics.data
        sg["qpos"] = hashlib.sha256(np.ascontiguousarray(np.asarray(dd.qpos)).tobytes()).hexdigest()[:12]
        sg["xpos"] = hashlib.sha256(np.ascontiguousarray(np.asarray(dd.xpos)).tobytes()).hexdigest()[:12]
        sg["geom_xpos"] = hashlib.sha256(np.ascontiguousarray(np.asarray(dd.geom_xpos)).tobytes()).hexdigest()[:12]
        return h.render(width=sz, height=sz, camera_id=cam).copy(), sg
    base, bsig = frame()
    base2, bsig2 = frame()
    rec_noise = {"px": int((np.abs(base2.astype(np.int32)-base.astype(np.int32)).max(-1)>0).sum()),
                 "qpos_same": bsig2["qpos"]==bsig["qpos"], "sig_same": bsig2["sig"]==bsig["sig"]}
    rec = {"noise_control": rec_noise, "size": sz, "camera_id": cam, "nq": nq, "nv": nv, "base_sig": bsig, "leaves": {}}
    for nm in vs.names():
        sp = node(vs, nm)
        pv = probe_value(sp)
        try:
            f, sig = frame({nm: pv})
            d = np.abs(f.astype(np.int32)-base.astype(np.int32))
            rec["leaves"][nm] = {"probe": (pv.tolist() if isinstance(pv,np.ndarray) else pv),
                "default": np.asarray(sp.init_value).tolist(),
                "px_changed": int((d.max(-1)>0).sum()), "px_total": int(f.shape[0]*f.shape[1]),
                "max_abs": int(d.max()), "model_sig_changed": sig["sig"]!=bsig["sig"],
                "appearance_sig_changed": sig["appearance"]!=bsig["appearance"],
                "nq_changed": sig["nq"]!=bsig["nq"],
                "qpos_changed": sig["qpos"]!=bsig["qpos"],
                "geom_xpos_changed": sig["geom_xpos"]!=bsig["geom_xpos"]}
        except Exception as ex:
            rec["leaves"][nm] = {"error": f"{type(ex).__name__}: {ex}"}
        print(i, nm, rec["leaves"][nm].get("px_changed", rec["leaves"][nm].get("error")), file=sys.stderr)
    out[i] = rec
    e.close()
json.dump(out, open(sys.argv[1],"w"), indent=1)
