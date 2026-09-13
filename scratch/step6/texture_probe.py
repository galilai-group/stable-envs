"""Did the reacher floor texture stop rendering between MuJoCo versions?

The Step 5 corpus shows a 2x2 checkerboard floor. Nothing in the current stack draws one -- not
stable-envs, not stable_worldmodel's reacher, and not dm_control's own `suite.load`, whose frame
Step 4 compared against and found smooth. dm_control still DECLARES the texture
(`builtin="checker"`, rgb1 .1 .2 .3, rgb2 .2 .3 .4, texrepeat 1 1, texuniform true), so the
declaration and the render disagree.

If an older MuJoCo paints it and a newer one does not, the corpus is a faithful record of an older
renderer rather than a corrupted file, and every "our frames differ from the corpus" result is a
statement about the stack. Run this under two interpreters with different mujoco versions and
compare the checker contrast.

    python scratch/step6/texture_probe.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import numpy as np
    import mujoco
    from dm_control import suite

    env = suite.load("reacher", "hard", task_kwargs={"random": 0})
    env.reset()
    p = env.physics
    m = p.model.ptr if hasattr(p.model, "ptr") else p.model

    # Read what the model SAYS about the floor texture, then look at what it DRAWS.
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "grid")
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "grid")
    decl = {
        "texture_found": tid >= 0, "material_found": mid >= 0,
        "tex_type": int(m.tex_type[tid]) if tid >= 0 else None,
        "tex_height": int(m.tex_height[tid]) if tid >= 0 else None,
        "tex_width": int(m.tex_width[tid]) if tid >= 0 else None,
        "mat_texrepeat": m.mat_texrepeat[mid].tolist() if mid >= 0 else None,
        "mat_texuniform": int(m.mat_texuniform[mid]) if mid >= 0 else None,
        "mat_reflectance": float(m.mat_reflectance[mid]) if mid >= 0 else None,
    }
    # mat_texid moved shape across versions (per-role table in newer mujoco); record whatever it is.
    try:
        decl["mat_texid"] = np.asarray(m.mat_texid[mid]).reshape(-1).tolist() if mid >= 0 else None
    except Exception as e:
        decl["mat_texid"] = f"unavailable: {e}"

    frame = p.render(height=args.size, width=args.size, camera_id=0)
    f = frame.astype(np.int32)
    # The checker is a 2x2 split of the floor. Quadrant means differ strongly when it is painted and
    # barely at all when it is not; the frozen corpus shows ~17/255 between light and dark squares.
    h, w, _ = f.shape
    q = [f[:h // 2, :w // 2].mean(), f[:h // 2, w // 2:].mean(),
         f[h // 2:, :w // 2].mean(), f[h // 2:, w // 2:].mean()]
    res = {"mujoco": mujoco.__version__, "declaration": decl,
           "quadrant_means": [round(x, 2) for x in q],
           "quadrant_spread": round(max(q) - min(q), 2),
           "distinct_colours": int(len(np.unique(frame.reshape(-1, 3), axis=0)))}
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"mujoco{mujoco.__version__.replace('.', '')}"
    (args.out / f"texture_{tag}.json").write_text(json.dumps(res, indent=2))
    try:
        import imageio.v2 as imageio
        imageio.imwrite(args.out / f"texture_{tag}.png", frame)
    except Exception:
        pass
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
