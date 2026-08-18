"""Parity of the shippable pipeline against the OFFICIAL frame, on this repo's own definition.

The recorded RoboCasa/MJWarp parity (interior mean 4.96, 89.4 % within 8) was measured against a
hand-built ``mujoco.Renderer`` on a pinned scene. This re-scores it against the frame RoboCasa
itself produced -- ``env.sim.render(...)``, saved by ``dump_corpus.py`` at collection time -- on
four different kitchens, and scores INTERIOR pixels the way ``stable_envs.testing.conformance``
does: a pixel is interior when its 3x3 neighbourhood in the reference is one flat colour, so a
disagreement there is a wrong colour, shape or place rather than two anti-aliasing rules.

That threshold is the one that decides whether RoboCasa can be shipped: every env in the registry,
including the DM Control family landed this week, asserts ``parity_interior_min = 1.0``.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def interior_mask(ref: np.ndarray) -> np.ndarray:
    """True where the reference's 3x3 neighbourhood is one flat colour (the repo's definition)."""
    h, w, _ = ref.shape
    m = np.ones((h, w), bool)
    for dy, dx in itertools.product((-1, 0, 1), repeat=2):
        if dy == 0 and dx == 0:
            continue
        s = np.roll(np.roll(ref, dy, 0), dx, 1)
        m &= (s == ref).all(-1)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    return m


# The zero-alpha pre-pass lives in the package, not here: another agent landed
# ``stable_envs/render/mjwarp.py`` with it, and its version resolves the material's alpha only when
# the geom carries the compiler default rgba -- MuJoCo's own rule -- and covers SITES as well as
# geoms. Reimplementing it in a probe is how the two would drift apart.
# Load the package's mjwarp module by PATH: importing `stable_envs.render.mjwarp` normally would
# execute `stable_envs/__init__.py`, which pulls in gymnasium, and the MJWarp venv deliberately has
# neither gymnasium nor torch-side env deps. The module itself has no package-internal imports.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "se_mjwarp", "/mnt/vast/home/randall.balestriero/code/stable-envs/stable_envs/render/mjwarp.py")
_se_mjwarp = _ilu.module_from_spec(_spec)
import sys as _sys
_sys.modules["se_mjwarp"] = _se_mjwarp      # @dataclass resolves annotations via sys.modules
_spec.loader.exec_module(_se_mjwarp)
prepare_model, texture_gaps = _se_mjwarp.prepare_model, _se_mjwarp.texture_gaps
create_render_context = _se_mjwarp.create_render_context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(OUT, "corpus"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--out", default=os.path.join(OUT, "parity_now.json"))
    args = ap.parse_args()

    import mujoco
    import torch
    import warp as wp
    import mujoco_warp as mjw

    rec = {"res": args.res, "mujoco": mujoco.__version__,
           "mujoco_warp": getattr(mjw, "__version__", "?"), "kitchens": []}

    configs = [dict(use_shadows=s, enable_specular=sp, render_skybox=sk, use_ambient_lighting=True)
               for s, sp, sk in itertools.product((True, False), (True, False), (True, False))]

    for xml in sorted(glob.glob(os.path.join(args.corpus, "*", "ep000.xml"))):
        tag = os.path.basename(os.path.dirname(xml))
        ref = np.load(xml.replace(".xml", "_frame.npy"))
        if ref.shape[0] != args.res:
            continue
        mjm = mujoco.MjModel.from_xml_path(xml)
        prep = prepare_model(mjm, groups=(1,))
        n_hidden = len(prep.hidden_geoms)
        gaps = texture_gaps(mjm, groups=(1,))
        mjd = mujoco.MjData(mjm)
        mjd.qpos[:] = np.load(xml.replace(".xml", "_qpos.npy"))
        mujoco.mj_forward(mjm, mjd)
        cid = max(0, mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, args.cam))
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mjd, nworld=1)
        im = interior_mask(ref)
        K = {"tag": tag, "n_zero_alpha_geoms": n_hidden,
             "n_zero_alpha_sites": len(prep.hidden_sites),
             "n_texgen_gap_geoms": len(gaps),
             "interior_frac": float(im.mean()), "configs": []}
        print(f"\n== {tag}: {prep.summary()}, {len(gaps)} texgen-gap geoms, "
              f"interior {im.mean():.3f} ==", flush=True)
        for cfg in configs:
            rc = mjw.create_render_context(mjm, nworld=1, cam_res=(args.res, args.res),
                                           render_rgb=True, use_textures=True,
                                           enabled_geom_groups=[1],
                                           cam_active=[i == cid for i in range(mjm.ncam)], **cfg)
            mjw.kinematics(m, d); mjw.camlight(m, d)
            mjw.refit_bvh(m, d, rc); mjw.render(m, d, rc)
            wp.synchronize()
            bgra = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(1, args.res, args.res, 4)
            got = bgra[0, ..., [2, 1, 0]].cpu().numpy()
            sd = ref.astype(np.int16) - got.astype(np.int16)     # SIGNED: ref - mjwarp
            dd = np.abs(sd)
            mx = dd.max(-1)
            # MJWarp truncates float->uint8 where the GL pipeline rounds, so a purely-truncation
            # residual is `ref - got` in {0, +1} on every channel and NEVER negative. Measuring the
            # signed histogram says how much of the disagreement a shader-side round would remove --
            # without needing the fix, and without assuming it.
            trunc_ok = ((sd >= 0) & (sd <= 1)).all(-1)
            hist = {int(k): float(v) / sd.size for k, v in
                    zip(*np.unique(np.clip(sd, -4, 4), return_counts=True))}
            row = dict(cfg,
                       all_exact=float((mx == 0).mean()), all_mean=float(dd.mean()),
                       int_exact=float((mx[im] == 0).mean()), int_mean=float(dd[im].mean()),
                       int_within8=float((mx[im] <= 8).mean()),
                       int_within32=float((mx[im] <= 32).mean()),
                       int_trunc_explained=float(trunc_ok[im].mean()),
                       all_trunc_explained=float(trunc_ok.mean()),
                       signed_hist_clipped=hist)
            K["configs"].append(row)
            print("  ", json.dumps(row), flush=True)
            del rc
        best = max(K["configs"], key=lambda r: (r["int_trunc_explained"], r["int_within8"]))
        K["best"] = best
        print("  BEST int_exact=%.4f int_within8=%.4f int_mean=%.2f  "
              "TRUNCATION-EXPLAINED interior=%.4f all=%.4f"
              % (best["int_exact"], best["int_within8"], best["int_mean"],
                 best["int_trunc_explained"], best["all_trunc_explained"]), flush=True)
        print("  signed (ref-got) histogram, clipped +-4:", json.dumps(
            {k: round(v, 4) for k, v in sorted(best["signed_hist_clipped"].items())}), flush=True)
        rec["kitchens"].append(K)
        del m, d, mjm, mjd

    rec["worst_best_int_exact"] = min(k["best"]["int_exact"] for k in rec["kitchens"])
    rec["worst_best_int_within8"] = min(k["best"]["int_within8"] for k in rec["kitchens"])
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print("\nworst-kitchen best interior-exact %.4f, within-8 %.4f"
          % (rec["worst_best_int_exact"], rec["worst_best_int_within8"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
