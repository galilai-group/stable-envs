"""WHY the same qpos draws two different Meta-World frames, and what a sufficient state would be.

``probe_metaworld.py`` measured that 45 of 50 tasks render the same ``qpos`` differently in two
episodes. This asks where the extra information lives: which MODEL arrays a ``reset`` mutates, and
whether restoring those arrays alongside ``qpos`` is enough to reproduce the frame — which is the
difference between "the state is bigger than qpos" and "the frame is not a function of any
fixed-width state".
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scratch.metaworld.probe_metaworld import make                          # noqa: E402

ARRAYS = ("body_pos", "body_quat", "site_pos", "site_quat", "geom_pos", "geom_quat", "geom_size",
          "geom_rgba", "site_rgba", "mat_rgba", "light_pos", "cam_pos", "geom_matid", "site_matid")
TASKS = ["reach-v3", "pick-place-v3", "door-open-v3", "button-press-v3", "drawer-open-v3"]


def probe(task, size=128):
    env = make(task, size)
    u = env.unwrapped
    m, d = u.model, u.data

    env.reset(seed=1)
    snap1 = {a: np.array(getattr(m, a), copy=True) for a in ARRAYS}
    mocap1 = np.array(d.mocap_pos, copy=True), np.array(d.mocap_quat, copy=True)
    q1 = np.array(d.qpos, copy=True)
    ctrl1 = np.array(d.ctrl, copy=True)
    f1 = u.render().copy()

    env.reset(seed=2)
    snap2 = {a: np.array(getattr(m, a), copy=True) for a in ARRAYS}
    changed = {a: int((np.abs(snap1[a] - snap2[a]) > 0).any(-1).sum()) for a in ARRAYS
               if snap1[a].ndim == 2}
    changed.update({a: int((snap1[a] != snap2[a]).sum()) for a in ARRAYS if snap1[a].ndim == 1})
    changed = {k: v for k, v in changed.items() if v}

    # same qpos, episode 2's model -> different frame?
    d.qpos[:] = q1
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    f_q_only = u.render().copy()

    # qpos AND the mutated model arrays AND the mocap target restored -> same frame?
    for a in ARRAYS:
        getattr(m, a)[:] = snap1[a]
    d.mocap_pos[:], d.mocap_quat[:] = mocap1
    d.qpos[:] = q1
    d.qvel[:] = 0
    d.ctrl[:] = ctrl1
    mujoco.mj_forward(m, d)
    f_full = u.render().copy()

    def diff(x, y):
        return int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max())

    return {"task": task, "model_rows_changed_per_reset": changed,
            "qpos_only_maxdiff": diff(f_q_only, f1),
            "qpos_plus_model_maxdiff": diff(f_full, f1),
            "extra_state_floats": int(sum(snap1[a].shape[1] * v for a, v in changed.items()))}


if __name__ == "__main__":
    out = []
    for t in (sys.argv[1:] or TASKS):
        try:
            r = probe(t)
        except Exception as e:                                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"task": t, "error": f"{type(e).__name__}: {e}"}
        out.append(r)
        print(json.dumps(r), flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_model_mutation.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
