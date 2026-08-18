"""Where the composite disagrees with the official frame: on the MOVERS, or on the PLATE?

Interior agreement is 0.9992 on reach-v3 and ALL-pixel agreement is 0.802, so 20% of the frame differs
somewhere the interior mask does not look — which is either the ray-cast movers (mesh shading) or the
baked plate (a mover's shadow that the bake cannot contain, because hiding a geom hides its shadow).
The two have completely different fixes, so they are separated here before anything is engineered.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def run(task: str, size: int, n: int) -> dict:
    import mujoco
    import torch

    import stable_envs as se
    from stable_envs.envs.metaworld import HIDDEN_GROUP, _build_bundle, _class_name, _raster
    from stable_envs.testing import conformance as C

    cls = se.REGISTRY[_class_name(task)]
    env = cls(batch=8, size=size, device="cuda")
    b = _build_bundle(task)
    states = env.sample_states(n, rng=np.random.default_rng(0))
    ref = C.reference_env(cls, size)
    official = cls.reference_render(ref, states).astype(np.int16)
    ours = env.render(states).cpu().numpy().astype(np.int16)

    # our mover mask, per state, straight out of the segmentation buffer
    from stable_envs.envs.metaworld import _context
    c = _context(task, size, 8)
    masks = []
    for lo in range(0, n, 8):
        st = torch.as_tensor(states[lo:lo + 8], device="cuda")
        env.render(st)
        masks.append(torch.isin(c.seg[:len(st)],
                                torch.as_tensor(b.movers, device="cuda").to(c.seg.dtype)).cpu().numpy())
    mv = np.concatenate(masks)[:n]

    d = np.abs(official - ours).max(-1)
    rec = {"task": task, "size": size, "n": n,
           "mover_frac": float(mv.mean()),
           "diff_frac": float((d > 0).mean()),
           "on_movers": {"frac_of_frame": float(((d > 0) & mv).mean()),
                         "frac_of_movers": float(d[mv].astype(bool).mean()),
                         "mean": round(float(d[mv].mean()), 3),
                         "max": int(d[mv].max())},
           "on_plate": {"frac_of_frame": float(((d > 0) & ~mv).mean()),
                        "frac_of_plate": float(d[~mv].astype(bool).mean()),
                        "mean": round(float(d[~mv].mean()), 3),
                        "max": int(d[~mv].max())}}

    # dilate the mover mask by one pixel: a silhouette edge is a coverage disagreement between two
    # rasterizers and is not the same defect as a wrong colour in the middle of a flat surface.
    import scipy.ndimage as ndi
    near = np.stack([ndi.binary_dilation(f, iterations=1) for f in mv])
    far = (~near)
    rec["on_plate_away_from_any_silhouette"] = {
        "frac_of_frame": float(((d > 0) & far).mean()),
        "mean": round(float(d[far].mean()), 3), "max": int(d[far].max())}

    # THE SHADOW QUESTION. Render the official scene with the movers hidden -- that is exactly what the
    # plate bake produces -- and compare it against the official frame ON PLATE PIXELS. Whatever
    # differs there is a mover's contribution to a pixel our composite takes from the plate, and the
    # only mechanism that reaches a plate pixel is the shadow (and reflection) of a mover.
    m, d_ = b.model, b.data
    r = _raster(task, size, m)
    shadow = []
    for st in states[:n]:
        from stable_envs.envs.metaworld import _apply_state
        _apply_state(b, m, d_, np.asarray(st, np.float64))
        r.update_scene(d_, camera=b.cam)
        full = r.render().copy().astype(np.int16)
        g0 = np.array(m.geom_group, copy=True)
        s0 = np.array(m.site_group, copy=True)
        m.geom_group[b.movers] = HIDDEN_GROUP
        if len(b.mover_sites):
            m.site_group[b.mover_sites] = HIDDEN_GROUP
        mujoco.mj_forward(m, d_)
        r.update_scene(d_, camera=b.cam)
        plate = r.render().copy().astype(np.int16)
        m.geom_group[:] = g0
        m.site_group[:] = s0
        shadow.append(np.abs(full - plate).max(-1))
    sh = np.stack(shadow)
    plate_pix = ~mv
    rec["mover_contribution_to_plate_pixels"] = {
        "frac_of_frame": float(((sh > 0) & plate_pix).mean()),
        "mean_on_those": round(float(sh[(sh > 0) & plate_pix].mean()) if ((sh > 0) & plate_pix).any()
                               else 0.0, 3),
        "max": int(sh[plate_pix].max())}

    # and the same question asked of OUR disagreement: how much of the plate-pixel error is explained
    # by the mover-contribution mask?
    bad_plate = (d > 0) & plate_pix
    rec["plate_error_explained_by_mover_contribution"] = float(
        (bad_plate & (sh > 0)).sum() / max(int(bad_plate.sum()), 1))

    # per-channel histogram of the signed difference on plate pixels away from silhouettes
    sd = (official - ours)[far]
    vals, counts = np.unique(sd, return_counts=True)
    top = sorted(zip(counts.tolist(), vals.tolist()), reverse=True)[:8]
    rec["signed_diff_far_from_movers"] = [[int(v), round(cnt / sd.size, 5)] for cnt, v in top]
    return rec


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "reach-v3"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 480
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    print("RESULT " + json.dumps(run(task, size, n)), flush=True)
