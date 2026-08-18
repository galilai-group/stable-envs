"""A large flat NON-floor surface (the slide ramp, the stair treads, the pole room's walls) comes out
one level off, everywhere. Is that a one-sided quantisation, a specular term, or fast math?

probe5: ``slide`` scores floor interior 0.846 and BODIES interior 0.0018 over 33 % of the frame, at an
overall mean of 4.2/255 -- i.e. the ramp is not wrong, it is off by a level. The signed histogram tells
a quantisation apart from a shading error: ``{-1: ~100 %}`` is one float quantised two ways (the
signature ``gl_rounding`` was found by), anything two-sided is shading.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/humanoidbench")

from hb_common import (LOGS, MjwRenderer, agreement, interior_mask, jsave, make_env, mj_render,
                       mj_segment, renderer, resolved_model, sample_qpos, set_state)

SIZE = 256
NSTATE = 8


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "slide"
    robot = sys.argv[2] if len(sys.argv) > 2 else "h1"
    import mujoco
    res = {"task": task, "robot": robot}

    env, u = make_env(task, robot, SIZE)
    cam_name, path = u.camera_name, u.fullpath
    Q = sample_qpos(task, robot, NSTATE, seed=0, steps=60)
    env.close()

    m, _ = resolved_model(path, SIZE)
    d = mujoco.MjData(m)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    ref = mj_render(m, d, cam, SIZE, Q, tag="ref")
    seg = mj_segment(m, d, cam, SIZE, Q, tag="seg")
    isgeom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    floor = isgeom & (seg[..., 0] == 0)
    body = isgeom & ~floor
    im = interior_mask(ref)

    def hist(got, msk):
        s = (got.astype(np.int16) - ref.astype(np.int16))[..., 1][:, :-1, :-1]  # green channel
        mm = msk[:, :-1, :-1] & im
        if not mm.any():
            return {}
        v = np.clip(s[mm], -6, 6)
        u_, c = np.unique(v, return_counts=True)
        n = float(c.sum())
        return {str(int(a)): round(float(b) / n, 5) for a, b in zip(u_, c)}

    R = MjwRenderer(m, cam, SIZE, NSTATE)
    got = R.render(Q)
    res["L0"] = agreement(got, ref)
    res["signed_hist_bodies"] = hist(got, body)
    res["signed_hist_floor"] = hist(got, floor)
    del R

    # specular off on BOTH sides
    keep = np.array(m.mat_specular, copy=True)
    m.mat_specular[:] = 0.0
    ref_ns = mj_render(m, d, cam, SIZE, Q, tag="nospec")
    R = MjwRenderer(m, cam, SIZE, NSTATE, specular=False)
    g = R.render(Q)
    res["no_specular_both"] = agreement(g, ref_ns)
    del R
    m.mat_specular[:] = keep

    # the reference's own multisampling: is the residual coverage?
    keep_os = int(m.vis.quality.offsamples)
    m.vis.quality.offsamples = 1
    _RC = renderer.__globals__["_RCACHE"]
    for k in [k for k in _RC if k[0] == id(m)]:
        _RC.pop(k)
    ref1 = mj_render(m, d, cam, SIZE, Q, tag="os1")
    R = MjwRenderer(m, cam, SIZE, NSTATE)
    res["vs_offsamples1_reference"] = agreement(R.render(Q), ref1)
    del R
    m.vis.quality.offsamples = keep_os
    for k in [k for k in _RC if k[0] == id(m)]:
        _RC.pop(k)

    jsave(f"{LOGS}/probe9_{robot}_{task}.json", res)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
