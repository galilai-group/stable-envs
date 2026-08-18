"""Parity of the GPU renderer against gymnasium's own, per env, plus throughput.

Reports the conformance metrics (all-pixel exact, interior exact, mean |diff|) and, separately, the
same metrics restricted to MOVER pixels — the only pixels the analytic shader is responsible for,
since the plate is the reference's own rasterization. Also sweeps SPECULAR_SCALE so the one fitted
constant in the module is checked rather than inherited.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import stable_envs as se                                                    # noqa: E402
from stable_envs.envs import mujoco_locomotion as ML                        # noqa: E402
from stable_envs.testing import conformance as C                            # noqa: E402

NAMES = list(ML.ENVS)


def mover_mask(cls, ref, states, size):
    """(N,H,W) bool: pixels the reference draws a MOVING geom or site on."""
    import mujoco
    u = ref.unwrapped
    m, d = u.model, u.data
    pr = dict(geom_group=np.array(m.geom_group, copy=True), site_group=np.array(m.site_group,
                                                                               copy=True))
    gmov = [i for i in range(m.ngeom)
            if int(pr["geom_group"][i]) in ML.VISIBLE_GROUPS and int(m.geom_bodyid[i]) != 0]
    smov = [i for i in range(m.nsite)
            if int(pr["site_group"][i]) in ML.VISIBLE_GROUPS and int(m.site_bodyid[i]) != 0]
    r = mujoco.Renderer(m, height=size, width=size)
    r.enable_segmentation_rendering()
    cam = u.mujoco_renderer.viewer.cam
    out = []
    for st in np.asarray(states, np.float64):
        u.set_state(st, np.zeros(m.nv))
        r.update_scene(d, camera=cam)
        s = r.render()
        objid, objtype = s[..., 0].astype(np.int64), s[..., 1].astype(np.int64)
        g = (objtype == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(objid, gmov)
        si = (objtype == int(mujoco.mjtObj.mjOBJ_SITE)) & np.isin(objid, smov)
        out.append(g | si)
    r.disable_segmentation_rendering()
    return np.stack(out)


def erode(mask):
    """One-pixel erosion, so coverage rules at the silhouette are not in the shading number."""
    import scipy.ndimage as ndi
    return np.stack([ndi.binary_erosion(x) for x in mask])


def run(name, size=None, n=32, spec_scales=(0.988,)):
    cls = se.REGISTRY[name]
    size = int(size or cls.default_size)
    env = cls(batch=n, size=size, device="cuda")
    ref = C.reference_env(cls, size)
    states = env.sample_states(n, rng=np.random.default_rng(0))
    official = cls.reference_render(ref, states)

    rec = {"env": name, "size": size, "n": n, "state_dim": env.state_dim,
           "state_names": list(env.state_names)}
    mm = mover_mask(cls, ref, states, size)
    rec["mover_frac"] = float(mm.mean())
    mi = erode(mm)
    rec["mover_interior_px"] = int(mi.sum())

    for scale in spec_scales:
        ML.SPECULAR_SCALE = float(scale)
        ours = env.render(states).cpu().numpy()
        r = C.pixel_agreement(ours, official)
        d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
        key = f"spec{scale}"
        rec[key] = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
        rec[key]["mover_exact"] = float((d[mm] == 0).mean()) if mm.any() else None
        rec[key]["mover_within1"] = float((d[mm] <= 1).mean()) if mm.any() else None
        rec[key]["mover_mean"] = float(d[mm].mean()) if mm.any() else None
        rec[key]["mover_interior_exact"] = float((d[mi] == 0).mean()) if mi.any() else None
        rec[key]["mover_interior_within1"] = float((d[mi] <= 1).mean()) if mi.any() else None
        rec[key]["mover_interior_mean"] = float(d[mi].mean()) if mi.any() else None
        rec[key]["nonmover_exact"] = float((d[~mm] == 0).mean())
        rec[key]["nonmover_max"] = int(d[~mm].max())
    ML.SPECULAR_SCALE = 0.988

    # throughput, ours vs the official renderer
    for B in (64, 256):
        st = env.sample_states(B, rng=np.random.default_rng(1))
        e2 = cls(batch=B, size=size, device="cuda")
        e2.render(st[:4])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e2.render(st)
        torch.cuda.synchronize()
        rec[f"fps_gpu_b{B}"] = round(B / (time.perf_counter() - t0), 1)
    t0 = time.perf_counter()
    cls.reference_render(ref, states[:16])
    rec["fps_reference"] = round(16 / (time.perf_counter() - t0), 1)
    return rec


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    size = None
    for a in sys.argv[1:]:
        if a.startswith("--size="):
            size = int(a.split("=")[1])
    scales = (1.0, 0.988, 0.97) if "--sweep" in sys.argv else (0.988,)
    out = []
    for nm in (args or NAMES):
        try:
            r = run(nm, size=size, spec_scales=scales)
        except Exception as e:                                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"env": nm, "error": f"{type(e).__name__}: {e}"}
        out.append(r)
        print(json.dumps(r), flush=True)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f"parity{'_' + str(size) if size else ''}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
