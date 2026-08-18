"""Depth and per-geom identity out of the same MJWarp megakernel, and what to feed a network.

The RGB-only corrector is per (scene, camera): it is worse than doing nothing on a camera it never
saw and it paints marble-like speckle onto plain cabinets in a kitchen it never saw. That failure has
one obvious cause. The information the network is missing -- *this surface is a textured granite box
that ``mujoco_warp`` drew as one flat colour* -- is not in the RGB frame. It has to be **inferred**
from image statistics, and image statistics are exactly what changes when the camera or the kitchen
changes. ``mujoco_warp`` will hand that information over directly, out of the same megakernel, for
0.4-2.3 % of the render (`probe_render_fps.py`).

Two per-pixel signals, and the difference between them is the whole experiment:

* **depth**, which is scene-independent by construction;
* **the geom id** the ray hit, which is an integer that means nothing on its own. What the network
  sees is a **row of a table indexed by that id**, and there are two ways to build that table:

  ``learned`` -- an ``nn.Embedding`` fitted with the rest of the network. Maximum capacity, and
  **structurally unable to transfer**: geom 742 in one kitchen is a different object in another, so
  the table is meaningless outside the scene it was fitted on.

  ``descriptor`` -- ``geom_features``, computed from whatever ``MjModel`` is loaded. Every column is
  a property MuJoCo already knows and MJWarp already failed to draw: the mean colour of the texture
  that should be on this surface, how textured that texture is, whether this is one of the primitives
  whose UVs ``sample_texture`` cannot derive, and the material's specular terms. A table built this
  way is **defined for a kitchen the corrector has never seen**, which is the only reason a
  cross-scene test is even askable.

Both are implemented so the comparison is one flag, and the honest question is not whether the
descriptor helps in distribution -- it is whether it closes the out-of-distribution gap.
"""
from __future__ import annotations

import numpy as np

#: Columns of ``geom_features``. Order is load-bearing: a checkpoint stores a table built with it.
FEATURE_NAMES = ("mat_r", "mat_g", "mat_b", "tex_std", "texgen_gap", "specular", "shininess",
                 "reflectance", "log_size")
N_FEATURES = len(FEATURE_NAMES)
#: mjtGeom codes whose UVs MJWarp can derive; everything else samples texel (0,0). Mirrors
#: ``stable_envs.render.mjwarp.TEXTURED_OK`` rather than re-deriving it.
TEXTURED_OK = (0, 7)


def _texture_stats(m, tex: int):
    """``(mean_rgb in [0,1], std over texels in [0,1])`` for one texture."""
    w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
    nc = int(m.tex_nchannel[tex]) if hasattr(m, "tex_nchannel") else 3
    adr = int(m.tex_adr[tex])
    a = np.asarray(m.tex_data[adr:adr + w * h * nc], np.float64).reshape(h, w, nc)
    rgb = a[..., :3] if nc >= 3 else np.repeat(a, 3, axis=-1)
    return rgb.mean((0, 1)) / 255.0, float(rgb.std((0, 1)).mean()) / 255.0


def geom_features(m, groups=(1,)) -> np.ndarray:
    """``(ngeom + 1, N_FEATURES)`` float32. Row 0 is the background; geom ``g`` is row ``g + 1``.

    Everything here is read off the compiled model, so the same function produces a valid table for a
    kitchen this repository has never compiled before. That is the point: a corrector that consumes
    this table has a chance of transferring, and a corrector that consumes a learned embedding of the
    same ids has none.

    ``texgen_gap`` is the column that carries the actual defect. It is 1.0 exactly on the geoms
    ``render/mjwarp.py`` documents as broken -- textured BOX / CYLINDER / SPHERE / CAPSULE, where
    ``sample_texture`` cannot derive UVs and the whole surface takes texel (0, 0) -- so the network is
    told *where the renderer lied* rather than having to work it out from the picture.
    """
    import mujoco
    rgb_role = int(getattr(mujoco.mjtTextureRole, "mjTEXROLE_RGB", 1))
    out = np.zeros((int(m.ngeom) + 1, N_FEATURES), np.float32)
    tex_cache: dict = {}
    for g in range(int(m.ngeom)):
        row = out[g + 1]
        mid = int(m.geom_matid[g])
        base = np.asarray(m.geom_rgba[g], np.float64)[:3]
        tex_std = 0.0
        gap = 0.0
        if mid >= 0:
            t = m.mat_texid[mid]
            tex = int(t[rgb_role]) if np.ndim(t) else int(t)
            mrgba = np.asarray(m.mat_rgba[mid], np.float64)[:3]
            # MuJoCo prefers the geom's own rgba unless it is the compiler default; mirror that rule
            # rather than always taking the material, which is the third gap in ``mjwarp.py``.
            if np.allclose(np.asarray(m.geom_rgba[g], np.float64), [0.5, 0.5, 0.5, 1.0]):
                base = mrgba
            if tex >= 0:
                if tex not in tex_cache:
                    tex_cache[tex] = _texture_stats(m, tex)
                mean, tex_std = tex_cache[tex]
                base = mean
                if int(m.geom_type[g]) not in TEXTURED_OK:
                    gap = 1.0
            row[5] = float(np.ravel(m.mat_specular[mid])[0])
            row[6] = float(np.ravel(m.mat_shininess[mid])[0])
            row[7] = float(np.ravel(m.mat_reflectance[mid])[0])
        row[0:3] = base
        row[3] = tex_std
        row[4] = gap
        row[8] = float(np.log1p(np.asarray(m.geom_size[g], np.float64).mean()))
    return out


# ------------------------------------------------------------------- reading the buffers out

def patch_context(mod, lod: int = 1, extra: dict | None = None):
    """Patch the env module's ``create_render_context`` to also write depth and segmentation.

    Returns the original so a caller can restore it. The context cache MUST be cleared, which
    ``probe_budget_frontier.ctx_cache`` does, or the next render silently reuses a context built
    without the buffers and every aux plane is zero -- which reads as "the aux channels did nothing".
    """
    from probe_budget_frontier import ctx_cache
    import stable_envs.render.mjwarp as MW
    original = MW.create_render_context
    ov = dict(enable_backface_culling=False, render_depth=True, render_seg=True)
    ov.update(extra or {})

    def wrap(m, **kw):
        kw.update(ov)
        want = max(int(kw.pop("tex_lod", 1)), int(lod))
        if want > 1:
            kw["tex_lod"] = want
        return original(m, **kw)

    mod.MW.create_render_context = wrap
    ctx_cache(mod).clear()
    return original


def read_aux(ctx, n: int, res: int, cam_index: int = 0, depth_scale: float = 8.0):
    """``(depth uint8 (n,res,res), geom id int32 (n,res,res))`` from a context that was just rendered.

    **``cam_index`` is the index into the ACTIVE camera list, not the model camera id**, and getting
    that wrong is not a wrong picture, it is an illegal memory access or a silent read of another
    camera's buffer. ``create_render_context`` does ``active_cam_indices = np.nonzero(cam_active)``
    and then sizes ``rgb_adr`` / ``depth_adr`` / ``seg_adr`` to ``len(active_cam_indices)``, so a
    context built for one camera has address arrays of length 1 whatever that camera's id is. This is
    the same convention ``robocasa._context`` already relies on when it reads ``rc.rgb_data`` as one
    flat buffer -- it works precisely because the active camera's address is 0. Default 0, and every
    caller in this module builds single-camera contexts.

    ``depth_scale`` is metres at full scale: ``get_depth`` divides and clamps to [0, 1], so a scale
    below the scene's extent silently saturates the far field. 8 m covers a RoboCasa kitchen and a
    HumanoidBench arena from their own cameras with room to spare; it is stored in the split metadata
    so a later reader cannot mistake the units.

    The segmentation is ``(objid, objtype)`` per pixel with ``(-1, -1)`` on background. Only the id is
    kept and it is shifted by one so that background is 0, which makes the table lookup a plain
    gather with no branch.
    """
    import warp as wp
    from mujoco_warp._src import render_util as ru
    import torch

    nworld = int(ctx.nworld)
    d = wp.zeros((nworld, res, res), dtype=float)
    ru.get_depth(ctx.rc, int(cam_index), float(depth_scale), d)
    s = wp.zeros((nworld, res, res), dtype=wp.vec2i)
    ru.get_segmentation(ctx.rc, int(cam_index), s)
    wp.synchronize()
    depth = (wp.to_torch(d)[:n] * 255.0).round_().clamp_(0, 255).to(torch.uint8)
    seg = wp.to_torch(s)[:n, :, :, 0].to(torch.int32) + 1        # background -1 -> 0
    return depth.contiguous(), seg.contiguous()
