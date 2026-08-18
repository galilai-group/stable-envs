"""Where exactly does MJWarp disagree with MuJoCo on the floor, and what is the mechanism?

The floor is 70 % of a locomotion frame and carries 99.7 % of the interior error. It is a PLANE (so
MJWarp derives UVs for it), its material is flat white with no reflectance, and its texture is a
DEGENERATE checker: 100x100 texels, 9,604 at 229 and 396 at 178, i.e. a uniform grey with a one-texel
``mark="edge"`` border, tiled 20x20 over a 40 m plane -- a 2 cm grid line every 2 m.

Candidate mechanisms, each with a different signature:
  fog          the model sets ``fogend=10``; if ``mjRND_FOG`` were on by default the far floor would
               be blended toward the fog colour and MJWarp does no fog at all.
  grid lines   MuJoCo filters the 2 cm line, MJWarp point-samples it -> disagreement concentrated on
               and beside the lines, and rising with distance.
  a level      the texture-modulate path quantising differently -> a uniform offset everywhere.
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
    task = sys.argv[1] if len(sys.argv) > 1 else "walk"
    robot = sys.argv[2] if len(sys.argv) > 2 else "h1hand"
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

    # ---- is any renderer flag on that MJWarp does not implement? ----------------------------
    r = renderer(m, SIZE, "ref")
    set_state(m, d, Q[0])
    r.update_scene(d, camera=int(cam))
    res["scene_flags"] = {n: int(r.scene.flags[int(getattr(mujoco.mjtRndFlag, n))])
                          for n in dir(mujoco.mjtRndFlag) if n.startswith("mjRND_")
                          and int(getattr(mujoco.mjtRndFlag, n)) < len(r.scene.flags)}

    def with_flag(name, value):
        out = []
        try:
            for q in Q:
                set_state(m, d, q)
                r.update_scene(d, camera=int(cam))
                r.scene.flags[int(getattr(mujoco.mjtRndFlag, name))] = value
                out.append(np.ascontiguousarray(r.render()))
        finally:
            r.update_scene(d, camera=int(cam))
            r.scene.flags[int(getattr(mujoco.mjtRndFlag, name))] = \
                res["scene_flags"][name]
        return np.stack(out)

    fog = with_flag("mjRND_FOG", 1)
    df = np.abs(fog.astype(np.int16) - ref.astype(np.int16)).max(-1)
    res["fog_would_change"] = dict(pixels=round(float((df > 0).mean()), 5), max=int(df.max()))

    # ---- MJWarp, and the spatial anatomy of the residual -------------------------------------
    R = MjwRenderer(m, cam, SIZE, NSTATE)
    got = R.render(Q)
    diff = np.abs(got.astype(np.int16) - ref.astype(np.int16)).max(-1)
    im = interior_mask(ref)
    fl = floor[:, :-1, :-1] & im
    di = diff[:, :-1, :-1]

    res["floor_error_hist"] = {str(k): int(v) for k, v in
                               zip(*np.unique(np.clip(di[fl], 0, 60), return_counts=True))}
    res["floor_signed_mean"] = round(float(
        (got.astype(np.int16) - ref.astype(np.int16))[..., 0][:, :-1, :-1][fl].mean()), 4)

    # is the reference a GRID LINE pixel? the mark is 178, the field 229, both then shaded, so a line
    # pixel is simply darker than the local mode
    lum = ref.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
    lum_i = lum[:, :-1, :-1]
    out = {}
    for tag, sel in (("all", fl),):
        vals = lum_i[sel]
        mode = float(np.median(vals))
        line = sel & (lum_i < mode - 6)
        field = sel & (lum_i >= mode - 6)
        out[tag] = dict(
            median_lum=round(mode, 2),
            line_share=round(float(line.sum() / max(sel.sum(), 1)), 5),
            line_exact=round(float((di[line] == 0).mean()), 6) if line.any() else None,
            line_mean=round(float(di[line].mean()), 4) if line.any() else None,
            field_exact=round(float((di[field] == 0).mean()), 6) if field.any() else None,
            field_mean=round(float(di[field].mean()), 4) if field.any() else None)
    res["line_vs_field"] = out

    # exactness by image row (distance proxy), 8 bands
    rows = np.where(fl.any((0, 2)))[0]
    if rows.size:
        edges = np.linspace(rows.min(), rows.max() + 1, 9).astype(int)
        by_row = []
        for a, b in zip(edges[:-1], edges[1:]):
            msk = np.zeros_like(fl)
            msk[:, a:b, :] = fl[:, a:b, :]
            if msk.any():
                by_row.append(dict(rows=[int(a), int(b)],
                                   n=int(msk.sum()),
                                   exact=round(float((di[msk] == 0).mean()), 5),
                                   mean=round(float(di[msk].mean()), 3),
                                   ref_lum_std=round(float(lum_i[msk].std()), 3),
                                   got_lum_std=round(float(
                                       (got.astype(np.float64) @ np.array([0.299, 0.587, 0.114]))
                                       [:, :-1, :-1][msk].std()), 3)))
        res["floor_by_row"] = by_row

    res["headline"] = agreement(got, ref)
    jsave(f"{LOGS}/probe6_{robot}_{task}.json", res)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
