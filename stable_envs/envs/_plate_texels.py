"""Per-value texel substitution on a baked plate: one texture edit, one upload, one cache key.

WHAT THIS IS FOR. Several ``stable_worldmodel`` appearance leaves are TEXTURE writes rather than
material writes -- DM Control's ``floor.color`` and OGBench's ``floor.color`` both rewrite the
builtin ``grid`` texture's ``rgb1``/``rgb2`` and let swm recompile the whole scene. A recompile is
not available to a cell that has baked a static plate, cached a render context and captured a CUDA
graph against one compiled model. It is also not necessary: the compiled texture of a two-colour
checker is a small set of distinct texel values, so the same bytes are reached by SUBSTITUTING those
texels in the renderer's own model and re-uploading the texture to its own GL context.

VERIFIED, not assumed, by Workstream C on ``swm/OGBCube-v0`` and ``swm/OGBScene-v0``: substituting
here and uploading gives a plate BYTE-IDENTICAL to letting swm recompile the whole model at the same
``floor.color`` -- ``tex_data`` 0 bytes differing, 0 of 50,176 pixels differing, both cells. This
module is that operation, lifted out of ``ogb_manip.py`` so that ``dmcontrol.py`` runs the same code
rather than a second copy of it that is free to drift.

THE THREE THINGS IT IS EASY TO GET WRONG, and which are therefore the whole of this module:

1. **The quantisation rule.** MuJoCo's texture compiler TRUNCATES ``int(255 * v)``; it does not
   round. 0.16 becomes 40 where rounding gives 41. ``quantise`` is that rule and the only place it
   is written down. (``stable_worldmodel``'s own Fetch skybox write is
   ``(bg * 255).astype(np.uint8)`` -- the same truncation, independently.)
2. **The upload.** ``mujoco.Renderer`` uploads textures to its GL context when it is BUILT. Writing
   ``tex_data`` afterwards changes the model and not the context, so without ``mjr_uploadTexture``
   the plate silently keeps the old texture and the leaf reads as inert.
3. **Which GL context is current.** ``mujoco.Renderer`` calls ``make_current()`` in exactly two
   places -- its constructor and ``render()`` -- and ``mjr_uploadTexture`` is neither. The GL calls
   it issues go to whatever context the THREAD holds, and GL texture names are per-context, so an
   upload made under the wrong context overwrites an unrelated texture object in that context. This
   is not theoretical: measured on gpu2264 before ``ogb_manip`` added the line, a second renderer
   built at a different resolution had its skybox overwritten by another rig's floor upload and two
   128 px plates differed by 219/255. ``substitute`` makes the renderer's context current itself, so
   a caller cannot forget.

WHAT IT DELIBERATELY DOES NOT DO. It does not own a cache, does not choose a texture, does not know
what a "floor" is and does not decide when to re-bake. The families keep their own plate caches and
their own keys; this module supplies the one key HELPER (``texel_key``) so that two requests which
compile to the same bytes share one bake, and the substitution itself.
"""
from __future__ import annotations

import numpy as np

__all__ = ["quantise", "texel_slots", "substitute", "texel_key"]


def quantise(v) -> np.ndarray:
    """A float colour as MuJoCo's own texture compiler stores it: ``int(255 * v)``, TRUNCATED.

    Not rounded, and the difference is visible: the DM Control grid's 0.1 becomes 25 where rounding
    gives 26, and 0.3 becomes 76 where rounding gives 77. Measured against the compiled ``tex_data``
    of all twelve ``DMControlSuite`` cells (25,51,76 / 51,76,102 for the declared 0.1,0.2,0.3 /
    0.2,0.3,0.4) and of both OGB cells; every channel truncates.
    """
    return np.clip((np.asarray(v, np.float64) * 255.0).astype(np.int64), 0, 255).astype(np.uint8)


def texel_slots(model, texture: str | int, default_colours, masks=None) -> dict | None:
    """Where a texture lives in ``tex_data``, and which of its texels carry each default colour.

    ``default_colours`` is ``(k, 3)`` -- the colours the UNMODIFIED texture is built from, in the
    order the leaf declares them. Returns ``None`` when the model has no such texture, which is the
    signal to a caller that the leaf is not offerable on this cell.

    THE MASKS ARE TAKEN ONCE, FROM THE DEFAULT TEXTURE, and that is a correctness argument rather
    than a caching one: if they were recomputed from the live texture a later write could not tell
    "the texels that used to be colour 1" from "the texels that a previous request happened to paint
    colour 1", so two requested colours colliding with each other would corrupt the masks
    permanently. Taking them from the pristine texture makes every substitution a function of the
    REQUEST alone.

    Texels matching neither default -- MuJoCo's ``markrgb`` grid lines, a gradient's interpolated
    band -- are deliberately left alone, because swm does not touch them either. The masks are
    therefore asserted to be non-empty AND their union is reported, so a caller can see how much of
    the texture the leaf actually reaches rather than assuming it reaches all of it.

    ``masks`` OVERRIDES the equality rule, and DM Control is why it has to exist. Its ``grid``
    texture declares ``markrgb=".2 .3 .4"``, which is EXACTLY its ``rgb2``: the 1,196 edge-mark
    texels and the 44,402 rgb2 texels are byte-identical in the compiled texture, and swm's
    recompile moves the second set and leaves the first at ``.2 .3 .4``. Equality alone cannot tell
    them apart, so a caller that knows the texture's construction supplies the masks and this
    function still owns the addressing, the assertions and the bookkeeping.
    """
    import mujoco
    t = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture)
         if isinstance(texture, str) else int(texture))
    if t < 0:
        return None
    adr = int(model.tex_adr[t])
    nch = int(model.tex_nchannel[t]) if hasattr(model, "tex_nchannel") else 3
    n = int(model.tex_height[t]) * int(model.tex_width[t]) * nch
    flat = np.asarray(model.tex_data[adr:adr + n]).reshape(-1, nch)
    cols = np.asarray(default_colours, np.float64).reshape(-1, 3)
    if masks is None:
        masks = [np.all(flat[:, :3] == quantise(c)[None, :], axis=1) for c in cols]
    else:
        masks = [np.asarray(mk, bool).reshape(-1) for mk in masks]
        assert len(masks) == len(cols), f"{len(masks)} masks for {len(cols)} colours"
        for i, mk in enumerate(masks):
            assert mk.shape[0] == flat.shape[0], (
                f"mask {i} is {mk.shape[0]} texels, texture {texture!r} is {flat.shape[0]}")
            bad = int((flat[mk, :3] != quantise(cols[i])[None, :]).any(1).sum())
            assert bad == 0, (
                f"mask {i} selects {bad} texels that do not hold default colour {i} "
                f"({list(cols[i])}); the masks and the compiled texture disagree, so a substitution "
                f"would paint texels swm does not paint")
    for i, mk in enumerate(masks):
        assert mk.any(), (
            f"texture {texture!r} carries none of default colour {i} ({list(cols[i])}), so the texel "
            f"mask would be empty and the leaf would silently do nothing")
    return {"texid": t, "adr": adr, "n": n, "nch": nch, "masks": tuple(masks),
            "default": cols.copy(), "covered": int(sum(int(mk.sum()) for mk in masks)),
            "texels": int(flat.shape[0])}


def substitute(model, renderer, slots: dict, colours) -> None:
    """Put ``colours`` into ``model``'s texture and onto ``renderer``'s own GL context.

    ``model`` must be the one ``renderer`` was built from. Written UNCONDITIONALLY, with no "it
    already looks like that" shortcut: a rig's texture and GL context outlive any one bake and are
    reused by every colour any instance asks for, so "the texture is whatever the last bake left" is
    the one state this must not have. A conditional upload would be correct only if a recorded value
    could never disagree with the context, and it disagrees the moment a bake raises between the
    texel write and the upload. The cost is one ``mjr_uploadTexture`` per plate-cache MISS, never per
    render.
    """
    import mujoco
    want = np.asarray(colours, np.float64).reshape(-1, 3)
    assert want.shape == slots["default"].shape, (
        f"substitute got {want.shape[0]} colours for a texture whose masks were taken for "
        f"{slots['default'].shape[0]}")
    flat = model.tex_data[slots["adr"]:slots["adr"] + slots["n"]].reshape(-1, slots["nch"])
    for mk, c in zip(slots["masks"], want):
        flat[mk, :3] = quantise(c)
    renderer._gl_context.make_current()
    mujoco.mjr_uploadTexture(model, renderer._mjr_context, slots["texid"])


def texel_key(slots: dict | None, colours):
    """The plate-cache key for one texel substitution: the BYTES MuJoCo stores, or ``None``.

    Keyed on the quantised bytes rather than on the floats so that two requests which compile to the
    same texture share one bake instead of baking the same picture twice -- 0.1 and 0.1003 are one
    plate. ``None`` is returned for the registered default by whatever route it was asked for, so the
    default plate keeps exactly the key it had before the leaf existed and no default render can miss
    a cache that was warm before.
    """
    if slots is None or colours is None:
        return None
    q = quantise(np.asarray(colours, np.float64).reshape(-1, 3))
    if np.array_equal(q, quantise(slots["default"])):
        return None
    return tuple(int(x) for x in q.reshape(-1))
