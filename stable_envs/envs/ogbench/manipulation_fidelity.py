import copy

import numpy as np
import torch

from . import manipulation as core

_REFERENCE = core._RES.BoundedCache(32, name="ogbench.fidelity_reference")


def draw(token, bundle, state, device, size, supersample):
    """Composite Cube transparency and moving shadows using two MJWarp passes.

    Remove the pinned shader's 0.3 shadow transmission before applying shadows
    to the static reference. This approximation is supported only for Cube."""
    import mujoco

    from .manipulation_renderer import _ConfiguredRenderer

    res = size * supersample
    alpha = core.MW.effective_alpha(bundle.model)
    transparent = np.flatnonzero((alpha > 0) & (alpha < 1))
    for suffix, hidden in (("opaque", transparent), ("static", np.union1d(transparent, bundle.movers))):
        key = token + "/fidelity/" + suffix
        if core.get_bundle(key) is None:
            alternate = copy.copy(bundle)
            alternate.unwrapped = copy.copy(bundle.unwrapped)
            alternate.unwrapped._model = copy.deepcopy(bundle.model)
            alternate.unwrapped._data = mujoco.MjData(alternate.model)
            alternate.model.geom_group[hidden] = core.MW.HIDDEN_GROUP
            alternate.pristine = {k: v.copy() for k, v in bundle.pristine.items()}
            alternate.pristine["geom_group"] = alternate.model.geom_group.copy()
            core.register_bundle(key, alternate)

    def raster(key, rows):
        renderer = _ConfiguredRenderer(key, core.get_bundle(key), device, res, 1)
        try:
            composite = renderer.draw(rows)
            nw = 1 if len(rows) == 1 else 1 << (len(rows) - 1).bit_length()
            context = core._context(key, res, nw)
            return composite, context.bgra[: len(rows)][..., [2, 1, 0]], context.seg[: len(rows)]
        finally:
            renderer.close()

    reference_key = token, res, str(device)
    reference = _REFERENCE.get(reference_key)
    if reference is None:
        reference = raster(token + "/fidelity/static", state[:1])[1].clone().float().clamp_min(1)
        _REFERENCE.put(reference_key, reference)
    movers = torch.zeros(bundle.model.ngeom + 1, dtype=torch.bool, device=device)
    movers[torch.as_tensor(np.asarray(bundle.movers) + 1, device=device)] = True
    opacity = torch.ones(bundle.model.ngeom + 1, device=device)
    for geom in transparent:
        if geom in bundle.movers:
            opacity[geom + 1] = float(alpha[geom])
    images = []
    for start in range(0, len(state), core.MJW_CHUNK):
        rows = state[start : start + core.MJW_CHUNK]
        nw = 1 << (len(rows) - 1).bit_length()
        keys = [(torch.cuda.current_device(), key, res, nw) for key in (token, token + "/fidelity/opaque")]
        # Both passes' segmentation views remain live until composition completes.
        with core._CTX_CACHE.lease(keys):
            _, first, first_seg = raster(token, rows)
            opaque, shadowed, opaque_seg = raster(token + "/fidelity/opaque", rows)
            ratio = ((shadowed.float() / reference).clamp(0, 1) - 0.3).div(0.7).clamp(0, 1)
            moving = movers[(opaque_seg + 1).long()]
            shadows = torch.where(moving[..., None], opaque.float(), opaque.float() * ratio)
            amount = opacity[(first_seg + 1).long()][..., None]
            image = torch.where(amount < 1, amount * first.float() + (1 - amount) * shadows, shadows)
            image = (
                image.reshape(len(rows), size, supersample, size, supersample, 3)
                .mean((2, 4))
                .round()
                .clamp(0, 255)
                .byte()
            )
            images.append(image)
    return torch.cat(images)
