"""Local CPU digests of the frozen Step-4 corpora at the DEFAULT style, for the DMC parity cells."""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import stable_envs as se
from stable_envs.testing import conformance as C
from stable_envs.envs import dmcontrol as D

STATES = "examples/results/step4/states"
out = {}
for cname in sys.argv[1:] or list(D.DOMAINS):
    p = os.path.join(STATES, f"{cname}.npz")
    if not os.path.exists(p):
        out[cname] = "no corpus"; continue
    st = np.load(p)["states"]
    e = se.make(cname, batch=len(st), device="cpu")
    f = e.render(st).cpu().numpy()
    ds = e.default_style()
    f2 = e.render(st, style=np.tile(np.asarray(ds, np.float32), (len(st), 1)))
    out[cname] = {"digest_ours": C.frames_digest(f),
                  "digest_ours_default_style": C.frames_digest(f2.cpu().numpy()),
                  "style_dim": int(e.style_dim), "style_names": list(e.style_names),
                  "describe_digest_input": {k: e.describe().get(k) for k in
                       ("name","state_dim","state_names","size","style_dim","style_names",
                        "reference_id","fidelity_tier","variations","parity_interior_min")},
                  "mean_pixel": float(f.mean()), "n": int(len(st))}
print(json.dumps(out, indent=1, default=str))
