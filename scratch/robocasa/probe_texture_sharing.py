"""Is "389 MB per scene" additive across scenes, or do kitchens share most of their textures?

The batching objection rests on texture memory, so it has to survive the obvious rebuttal: many
kitchens use the same wood, the same marble, the same steel. If the texture payloads are largely
shared, the union over N scenes grows far slower than 389N MB and the memory argument weakens.

This hashes the raw ``tex_data`` bytes of every texture in each of N resampled kitchens and reports
the union size as a function of N -- so the growth rate is measured rather than extrapolated from a
single scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="PickPlaceCounterToSink")
    ap.add_argument("--n-scenes", type=int, default=10)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from robocasa.utils.env_utils import create_env

    env = create_env(env_name=args.env_name, robots="PandaOmron",
                     camera_names=["robot0_agentview_left"],
                     camera_widths=128, camera_heights=128, seed=0, render_onscreen=False)

    seen_tex, seen_mesh = {}, {}
    rows = []
    for i in range(args.n_scenes):
        env.reset()
        m = env.sim.model._model
        # per-texture byte ranges
        per_scene_tex = 0
        for t in range(m.ntex):
            adr = int(m.tex_adr[t])
            nbytes = int(m.tex_width[t]) * int(m.tex_height[t]) * int(m.tex_nchannel[t]) \
                if hasattr(m, "tex_nchannel") else int(m.tex_width[t]) * int(m.tex_height[t]) * 3
            buf = m.tex_data[adr:adr + nbytes].tobytes()
            h = hashlib.blake2b(buf, digest_size=16).hexdigest()
            seen_tex[h] = nbytes
            per_scene_tex += nbytes
        per_scene_mesh = 0
        for k in range(m.nmesh):
            va, vn = int(m.mesh_vertadr[k]), int(m.mesh_vertnum[k])
            fa, fn = int(m.mesh_faceadr[k]), int(m.mesh_facenum[k])
            buf = m.mesh_vert[va:va + vn].tobytes() + m.mesh_face[fa:fa + fn].tobytes()
            h = hashlib.blake2b(buf, digest_size=16).hexdigest()
            seen_mesh[h] = len(buf)
            per_scene_mesh += len(buf)
        rows.append({
            "scene": i, "layout": int(env.layout_id), "style": int(env.style_id),
            "ntex": int(m.ntex), "nmesh": int(m.nmesh), "ngeom": int(m.ngeom), "nq": int(m.nq),
            "scene_tex_MB": per_scene_tex / 1e6,
            "scene_mesh_MB": per_scene_mesh / 1e6,
            "union_tex_MB": sum(seen_tex.values()) / 1e6,
            "union_tex_count": len(seen_tex),
            "union_mesh_MB": sum(seen_mesh.values()) / 1e6,
            "union_mesh_count": len(seen_mesh),
        })
        print(json.dumps(rows[-1]))

    mean_scene = float(np.mean([r["scene_tex_MB"] for r in rows]))
    out = {
        "rows": rows,
        "mean_per_scene_tex_MB": mean_scene,
        "union_tex_MB_after_N": rows[-1]["union_tex_MB"],
        "additive_would_be_MB": mean_scene * args.n_scenes,
        "sharing_factor": (mean_scene * args.n_scenes) / max(rows[-1]["union_tex_MB"], 1e-9),
        "marginal_tex_MB_per_new_scene": (rows[-1]["union_tex_MB"] - rows[len(rows)//2]["union_tex_MB"])
                                         / max(len(rows) - len(rows)//2, 1),
        "union_mesh_MB_after_N": rows[-1]["union_mesh_MB"],
    }
    print("\n" + json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    with open(os.path.join(args.out, "texture_sharing.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
