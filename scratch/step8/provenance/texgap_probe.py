"""Does this interpreter's mujoco_warp still refuse OGBScene's UV-less textured mesh?

Step-7 W5 (`examples/results/step7/ogb-manipulation/findings.md` S11) found that
``mujoco_warp``'s ``sample_texture`` MESH branch indexed ``mesh_texcoord`` with no guard, and that
OGBScene's two button geoms wear mesh ``stopbuttonrim`` with ``texcoordadr = -1, texcoordnum = 0``.
``stable_envs.render.mjwarp.texture_gaps`` reports that condition and the cell bakes the material
away (``bake_unsamplable_textures``) so the scene renders at all.

This probe reads the model and reports, per interpreter: the gap list, the two buttons' mesh
texcoord fields, and whether the shipped ``sample_texture`` source carries a negative-offset guard.
It renders nothing and constructs no stable-envs env, so it says the same thing under an
interpreter where the cell would crash.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import importlib.metadata as md
    out = {"versions": {p: (md.version(p) if _has(p) else None)
                        for p in ("mujoco", "mujoco-warp", "warp-lang")}}

    # The upstream guard, read off the shipped source rather than inferred from the version string.
    try:
        from mujoco_warp._src import render as mjw_render
        src = inspect.getsource(mjw_render.sample_texture.func if hasattr(mjw_render.sample_texture, "func")
                                else mjw_render.sample_texture)
    except Exception as e:                                                  # noqa: BLE001
        src = ""
        out["sample_texture_source_error"] = f"{type(e).__name__}: {e}"
    out["sample_texture_guards_missing_texcoords"] = ("texcoord_offset >= 0" in src
                                                      or "texcoordadr" in src)
    out["sample_texture_sha256"] = __import__("hashlib").sha256(src.encode()).hexdigest() if src else None

    # The model side: OGBScene's own mesh fields and what texture_gaps says about them.
    try:
        from stable_envs.envs.ogb_manip import _bundle
        from stable_envs.render.mjwarp import texture_gaps, unsamplable_texture_materials
        m = _bundle("Scene").model
        out["model_source"] = "stable_envs.envs.ogb_manip._bundle('Scene').model"
        out["texture_gaps"] = [str(g) for g in texture_gaps(m)]
        try:
            out["unsamplable_texture_materials"] = {
                str(k): [int(i) for i in v]
                for k, v in (unsamplable_texture_materials(m) or {}).items()}
        except Exception as e:                                              # noqa: BLE001
            out["unsamplable_texture_materials_error"] = f"{type(e).__name__}: {e}"
        uvless = [i for i in range(int(m.ngeom))
                  if int(m.geom_dataid[i]) >= 0 and int(m.mesh_texcoordadr[int(m.geom_dataid[i])]) < 0]
        out["geoms_on_uvless_mesh"] = uvless
        out["mesh_texcoordadr_of_those"] = [int(m.mesh_texcoordadr[int(m.geom_dataid[i])]) for i in uvless]
    except Exception as e:                                                  # noqa: BLE001
        out["model_probe_error"] = f"{type(e).__name__}: {e}"

    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))


def _has(pkg: str) -> bool:
    import importlib.metadata as md
    try:
        md.version(pkg)
        return True
    except Exception:                                                       # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
