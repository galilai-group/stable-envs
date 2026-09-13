import os, sys, json
os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np, stable_envs as se
from stable_envs.testing import conformance as C

out = {}
for name in sys.argv[1:]:
    e = se.make(name, batch=16, device="cpu")
    rows = []
    for seed in range(4):                       # independent draws, to set a bound from the SPREAD
        rows.append(C.check_task_object_visibility(e, n=16, seed=seed))
    # and on the frozen Step-4 corpus
    st = np.load(f"examples/results/step4/states/{name}.npz")["states"]
    corpus = C.check_task_object_visibility(e, states=st)
    out[name] = {"draws": rows, "frozen_corpus": corpus}
print(json.dumps(out, indent=1))
