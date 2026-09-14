"""``check_task_object_visibility`` over independent draws AND over the frozen Step-4 corpus.

The numbers ``tests/gpu/test_conformance.py:TASK_OBJECT_PRESENCE_MARGIN`` is set from. Four draws of
16 sampled states plus the 64-state frozen corpus per cell, so the floor comes from the SPREAD rather
than from one evaluation -- the rule the rest of the suite's floors follow.

    PYTHONPATH=$PWD MUJOCO_GL=glfw python probe_guard.py --device cpu  DMCReacherHard ...
    PYTHONPATH=$PWD                  python probe_guard.py --device cuda DMCReacherHard ...
"""
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np

import stable_envs as se
from stable_envs.testing import conformance as C

ap = argparse.ArgumentParser()
ap.add_argument("envs", nargs="+")
ap.add_argument("--device", default="cpu")
ap.add_argument("--draws", type=int, default=4)
ap.add_argument("--states", default="examples/results/step4/states")
a = ap.parse_args()

out = {}
for name in a.envs:
    e = se.make(name, batch=16, device=a.device)
    rows = [C.check_task_object_visibility(e, n=16, seed=s) for s in range(a.draws)]
    st = np.load(os.path.join(a.states, f"{name}.npz"))["states"]
    out[name] = {"device": a.device, "draws": rows,
                 "frozen_corpus": C.check_task_object_visibility(e, states=st)}
print(json.dumps(out, indent=1))

print("\n== summary ==")
for k, v in out.items():
    ms = [r["objects"]["visibility.target"] for r in v["draws"]]
    ms.append(v["frozen_corpus"]["objects"]["visibility.target"])
    print(f"{k:20} margin {min(m['presence_margin'] for m in ms):.4f}-"
          f"{max(m['presence_margin'] for m in ms):.4f}  "
          f"declared_mae {min(m['presence_mean_abs_declared'] for m in ms):.2f}-"
          f"{max(m['presence_mean_abs_declared'] for m in ms):.2f}  "
          f"flipped_mae {min(m['presence_mean_abs_flipped'] for m in ms):.1f}-"
          f"{max(m['presence_mean_abs_flipped'] for m in ms):.1f}  "
          f"footprint_px {[m['footprint_px'] for m in ms]}  "
          f"per_row {[m['per_row_max_abs_diff'] for m in ms]}")
