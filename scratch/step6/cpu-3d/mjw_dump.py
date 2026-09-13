"""Dump the exact mujoco_warp sampling/shading source a CPU path would have to reproduce.

se-step5 only (it has mujoco_warp). Prints version + source of the texture, skybox, ray and
quantisation helpers to a text artifact, so the design note quotes the law rather than a memory of it.
"""
import inspect
import sys
from pathlib import Path

import mujoco_warp
from mujoco_warp._src import render as R
from mujoco_warp._src import render_util as RU

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
parts = [f"mujoco_warp {getattr(mujoco_warp, '__version__', '?')}\n"]
for mod, names in ((R, ("compute_ray", "sample_texture", "sample_skybox", "pack_rgba_to_uint32")),
                   (RU, ("create_warp_texture", "_convert_texture_data", "create_render_context"))):
    for n in names:
        f = getattr(mod, n, None)
        if f is None:
            parts.append(f"\n### {mod.__name__}.{n}: ABSENT\n"); continue
        try:
            src = inspect.getsource(f)
        except Exception as e:
            src = f"<unavailable: {e}>"
        parts.append(f"\n### {mod.__name__}.{n}\n{src}\n")
# the skybox face layout / lod facts from the context builder, by grep over the module source
src = inspect.getsource(RU)
keep = [l for l in src.splitlines() if any(k in l for k in ("skybox", "mip", "filter_mode", "Texture2D", "lod"))]
parts.append("\n### render_util lines mentioning skybox/mip/filter/Texture2D/lod\n" + "\n".join(keep) + "\n")
out.write_text("".join(parts))
print("wrote", out, len("".join(parts)), "chars")
