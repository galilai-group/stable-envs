import json, sys, numpy as np, gymnasium, stable_worldmodel  # noqa
IDS = ["swm/AcrobotControl-v1","swm/CartPoleControl-v1","swm/MountainCarContinuousControl-v0",
 "swm/MountainCarControl-v0","swm/PendulumControl-v1","swm/AcrobotDMControl-v0",
 "swm/BallInCupDMControl-v0","swm/CartpoleDMControl-v0","swm/CheetahDMControl-v0",
 "swm/FingerDMControl-v0","swm/HopperDMControl-v0","swm/HumanoidDMControl-v0",
 "swm/ManipulatorDMControl-v0","swm/PendulumDMControl-v0","swm/QuadrupedDMControl-v0",
 "swm/ReacherDMControl-v0","swm/WalkerDMControl-v0"]

def find_vs(e):
    n = e
    while n is not None:
        vs = getattr(n, "variation_space", None)
        if vs is not None: return n, vs
        n = getattr(n, "env", None)
    return None, None

def leaf(sp):
    import stable_worldmodel.spaces as S
    d = {"cls": type(sp).__name__}
    if hasattr(sp, "low"):
        d["low"] = np.asarray(sp.low).tolist(); d["high"] = np.asarray(sp.high).tolist()
        d["shape"] = list(sp.shape); d["dtype"] = str(sp.dtype)
    if hasattr(sp, "n"): d["n"] = int(sp.n)
    iv = getattr(sp, "init_value", None)
    d["init_value"] = np.asarray(iv).tolist() if iv is not None else None
    return d

out = {}
for i in IDS:
    try:
        e = gymnasium.make(i)
        holder, vs = find_vs(e)
        rec = {"holder": type(holder).__name__ if holder else None, "leaves": {}}
        for nm in vs.names():
            node = vs
            for p in nm.split("."): node = node[p]
            rec["leaves"][nm] = leaf(node)
        out[i] = rec
        e.close()
    except Exception as ex:
        out[i] = {"error": f"{type(ex).__name__}: {ex}"}
    print(i, "done", file=sys.stderr)
json.dump(out, open(sys.argv[1],"w"), indent=1)
