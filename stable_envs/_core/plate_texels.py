"""Substitute default colors in compiled MuJoCo textures and upload them to a renderer."""

from __future__ import annotations

import numpy as np

__all__ = ["quantise", "texel_slots", "substitute", "texel_key"]


def quantise(v) -> np.ndarray:
    """Convert float colors to uint8 using MuJoCo texture truncation, then clamp to [0, 255]."""
    return np.clip((np.asarray(v, np.float64) * 255.0).astype(np.int64), 0, 255).astype(np.uint8)


def texel_slots(model, texture: str | int, default_colours, masks=None) -> dict | None:
    """Locate a texture and record masks for its pristine default colors.

    Return None for a missing texture. Reuse these masks after substitutions;
    recomputing from mutated texels loses color identity. Explicit masks can
    distinguish marks from equal-colored fill texels. Every mask must be nonempty
    and match its declared default; unselected texels are left unchanged.
    """
    import mujoco

    t = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, texture) if isinstance(texture, str) else int(texture)
    if t < 0:
        return None
    adr = int(model.tex_adr[t])
    nch = int(model.tex_nchannel[t]) if hasattr(model, "tex_nchannel") else 3
    n = int(model.tex_height[t]) * int(model.tex_width[t]) * nch
    flat = np.asarray(model.tex_data[adr : adr + n]).reshape(-1, nch)
    cols = np.asarray(default_colours, np.float64).reshape(-1, 3)
    if masks is None:
        masks = [np.all(flat[:, :3] == quantise(c)[None, :], axis=1) for c in cols]
    else:
        masks = [np.asarray(mk, bool).reshape(-1) for mk in masks]
        assert len(masks) == len(cols), f"{len(masks)} masks for {len(cols)} colours"
        for i, mk in enumerate(masks):
            assert mk.shape[0] == flat.shape[0], (
                f"mask {i} is {mk.shape[0]} texels, texture {texture!r} is {flat.shape[0]}"
            )
            bad = int((flat[mk, :3] != quantise(cols[i])[None, :]).any(1).sum())
            assert bad == 0, (
                f"mask {i} selects {bad} texels that do not hold default colour {i} "
                f"({list(cols[i])}); the masks and the compiled texture disagree"
            )
    for i, mk in enumerate(masks):
        assert mk.any(), (
            f"texture {texture!r} carries none of default colour {i} ({list(cols[i])}); the texel mask is empty"
        )
    return {
        "texid": t,
        "adr": adr,
        "n": n,
        "nch": nch,
        "masks": tuple(masks),
        "default": cols.copy(),
        "covered": int(sum(int(mk.sum()) for mk in masks)),
        "texels": int(flat.shape[0]),
    }


def substitute(model, renderer, slots: dict, colours) -> None:
    """Replace selected texels and upload them to the renderer that owns this model.

    Always upload after writing: a previous failed bake may have left model
    and GL texture contents inconsistent.
    """
    import mujoco

    want = np.asarray(colours, np.float64).reshape(-1, 3)
    assert want.shape == slots["default"].shape, (
        f"substitute got {want.shape[0]} colours for a texture whose masks were taken for {slots['default'].shape[0]}"
    )
    flat = model.tex_data[slots["adr"] : slots["adr"] + slots["n"]].reshape(-1, slots["nch"])
    for mk, c in zip(slots["masks"], want):
        flat[mk, :3] = quantise(c)
    renderer._gl_context.make_current()
    mujoco.mjr_uploadTexture(model, renderer._mjr_context, slots["texid"])


def texel_key(slots: dict | None, colours):
    """Return quantized color bytes as the cache key, or None for no/default substitution."""
    if slots is None or colours is None:
        return None
    q = quantise(np.asarray(colours, np.float64).reshape(-1, 3))
    if np.array_equal(q, quantise(slots["default"])):
        return None
    return tuple(int(x) for x in q.reshape(-1))
