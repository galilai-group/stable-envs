"""Is a texel substitution on the plate rig the same picture as letting swm recompile the scene?

THE QUESTION. ``stable_worldmodel`` honours DM Control's ``floor.color`` by writing the ``grid``
texture's ``rgb1``/``rgb2`` into the MJCF and RECOMPILING the whole model. This patch instead
substitutes the compiled texture's texels in place and re-uploads the texture to the plate
renderer's own GL context. The two are only the same operation if they produce the same bytes, and
that is a measurement, not an argument.

THE METHOD. Two PROCESSES, because ``dm_control``'s asset table and this module's bundle/scene
caches are process-global and a recompile inside one process would be served a cached model.

  ``--mode recompile`` patches ``dm_control.suite.common.ASSETS['./common/materials.xml']`` so the
  suite's own grid texture declares the probe colours, and then renders at the DEFAULT style. That
  is a full recompile: the model, the plate rig and every downstream buffer are built from a scene
  whose texture was compiled with those colours, exactly as swm's ``mark_dirty`` path does.

  ``--mode substitute`` leaves the assets alone and renders at ``variation={'floor.color': probe}``,
  which goes through ``_apply_floor`` -> ``_plate_texels.substitute``.

``markrgb`` is NOT a probe colour in either arm: the suite declares it equal to ``rgb2`` and swm's
recompile leaves it at its literal value, so the 1,196 edge-mark texels must stay put on both sides.
They are the strongest part of this test -- an implementation that took its masks by equality would
move them and fail here, and nowhere else.

Usage:
    python bake_vs_recompile.py --mode recompile  --out a.json
    python bake_vs_recompile.py --mode substitute --out b.json
    python bake_vs_recompile.py --compare a.json b.json --out verdict.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

PROBES = [[[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]],
          [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
          [[0.37, 0.61, 0.24], [0.05, 0.95, 0.5]]]
SIZE = 84
NSTATES = 8


def patch_assets(rgb1, rgb2):
    from dm_control.suite import common
    key = "./common/materials.xml"
    xml = common.ASSETS[key].decode()
    tex = re.search(r'<texture[^>]*name="grid"[^>]*/>', xml).group(0)
    new = re.sub(r'rgb1="[^"]+"', 'rgb1="%s"' % " ".join(repr(float(x)) for x in rgb1), tex)
    new = re.sub(r'rgb2="[^"]+"', 'rgb2="%s"' % " ".join(repr(float(x)) for x in rgb2), new)
    assert new != tex
    common.ASSETS[key] = xml.replace(tex, new).encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["recompile", "substitute"])
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--cells", default="DMCCartpoleBalance,DMCReacherHard,DMCPendulumSwingup")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        x, y = (json.load(open(p)) for p in a.compare)
        rows, ok = [], True
        for cell in sorted(set(x["cells"]) & set(y["cells"])):
            same_tex = x["cells"][cell]["tex_sha256"] == y["cells"][cell]["tex_sha256"]
            same_px = x["cells"][cell]["frames_sha256"] == y["cells"][cell]["frames_sha256"]
            ok &= same_tex and same_px
            rows.append({"cell": cell, "probe": x["probe"], "tex_identical": same_tex,
                         "frames_identical": same_px,
                         "recompile_frames_sha256": x["cells"][cell]["frames_sha256"],
                         "substitute_frames_sha256": y["cells"][cell]["frames_sha256"],
                         "px_total": x["cells"][cell]["px_total"]})
        json.dump({"verdict": "IDENTICAL" if ok else "DIFFERS", "rows": rows},
                  open(a.out, "w"), indent=1)
        print(json.dumps(rows, indent=1))
        print("VERDICT", "IDENTICAL" if ok else "DIFFERS")
        return

    os.environ.setdefault("MUJOCO_GL", "glfw")
    rgb1, rgb2 = PROBES[a.probe]
    if a.mode == "recompile":
        patch_assets(rgb1, rgb2)
    import numpy as np
    import mujoco
    import stable_envs as se

    out = {"mode": a.mode, "probe": [rgb1, rgb2], "size": SIZE, "states": NSTATES, "cells": {}}
    for cell in a.cells.split(","):
        E = se.REGISTRY[cell](batch=NSTATES, size=SIZE, device="cpu")
        st = E.sample_states(NSTATES, np.random.default_rng(0))
        var = None if a.mode == "recompile" else {"floor.color": np.asarray([rgb1, rgb2])}
        f = E.render(st, variation=var)
        # the bundle for the RENDERED size, not ``_model()``: this family keeps one model and
        # one Renderer per (domain, size) and several cells declare a default_size other than
        # the one measured here, so ``_model()`` would hash a model this render never touched.
        m = E._bundle(SIZE)[1]
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "grid")
        adr, nch = int(m.tex_adr[t]), int(m.tex_nchannel[t])
        n = int(m.tex_width[t]) * int(m.tex_height[t]) * nch
        tex = np.asarray(m.tex_data[adr:adr + n]).copy()
        out["cells"][cell] = {
            "tex_sha256": hashlib.sha256(tex.tobytes()).hexdigest(),
            "frames_sha256": hashlib.sha256(np.ascontiguousarray(f.numpy()).tobytes()).hexdigest(),
            "px_total": int(f.shape[0]) * SIZE * SIZE}
        print(cell, out["cells"][cell]["tex_sha256"][:16],
              out["cells"][cell]["frames_sha256"][:16], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


main()
