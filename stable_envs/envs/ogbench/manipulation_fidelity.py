import copy

import numpy as np
import torch

from . import manipulation as core

_REFERENCE = core._RES.BoundedCache(256, name="ogbench.fidelity_reference")


def rotate_four_samples(rays: torch.Tensor) -> None:
    """Match NVIDIA's four MSAA positions in a 2× ray grid with top-down image rows."""
    height, width, _ = rays.shape
    plane = rays / -rays[..., 2:3]
    dx = (plane[0, 1, 0] - plane[0, 0, 0]).clone()
    dy = (plane[1, 0, 1] - plane[0, 0, 1]).clone()
    plane[..., 0] += (torch.arange(height, device=rays.device)[:, None] % 2 - 0.5) * 0.5 * dx
    plane[..., 1] += (0.5 - torch.arange(width, device=rays.device)[None, :] % 2) * 0.5 * dy
    rays.copy_(plane / torch.linalg.vector_norm(plane, dim=-1, keepdim=True))


def sample_plate(image: torch.Tensor) -> torch.Tensor:
    """Sample a cached 4× GL background at the matching four ray positions."""
    height, width, _ = image.shape
    pixels = image.reshape(height // 4, 4, width // 4, 4, 3)
    result = image.new_empty((height // 2, width // 2, 3))
    for y, x, hy, hx in ((0, 0, 1, 0), (0, 1, 0, 2), (1, 0, 3, 1), (1, 1, 2, 3)):
        result[y::2, x::2] = pixels[:, hy, :, hx, :]
    return result


def draw(token, bundle, state, device, size, supersample, worlds):
    """Composite transparency and moving shadows using two MJWarp passes.

    Remove the pinned shader's 0.3 shadow transmission before applying shadows
    to the static reference.
    """
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

    glass = token + "/fidelity/glass"
    if core.get_bundle(glass) is None:
        single = copy.copy(bundle)
        single.sample_pattern = "grid"
        core.register_bundle(glass, single)

    def raster(key, rows, worlds, *, composite=False, res=res):
        renderer = _ConfiguredRenderer(key, core.get_bundle(key), device, res, 1)
        try:
            if not composite:
                return renderer.raster(rows, worlds)
            image = renderer.draw(rows, worlds)
            nw = 1 if len(rows) == 1 else 1 << (len(rows) - 1).bit_length()
            context = core._context(key, res, nw)
            return image, context.bgra[: len(rows)][..., [2, 1, 0]], context.seg[: len(rows)]
        finally:
            renderer.close()

    # The static pass hides every mover, so its image depends only on the camera and light.
    references = {}
    for view in np.unique(worlds.view):
        first = int(np.flatnonzero(worlds.view == view)[0])
        reference_key = token, res, str(device), worlds.views[view].key
        reference = _REFERENCE.get(reference_key)
        if reference is None:
            rows = slice(first, first + 1)
            reference = raster(token + "/fidelity/static", state[rows], worlds[rows])[0].clone()
            _REFERENCE.put(reference_key, reference)
        references[int(view)] = reference.float().clamp_min(1)
    movers = torch.zeros(bundle.model.ngeom + 1, dtype=torch.bool, device=device)
    movers[torch.as_tensor(np.asarray(bundle.movers) + 1, device=device)] = True
    opacity = torch.ones(bundle.model.ngeom + 1, device=device)
    for geom in transparent:
        if bundle.nbuttons or geom in bundle.movers:
            opacity[geom + 1] = float(alpha[geom])
    images = []
    step = core.mjw_worlds()
    for start in range(0, len(state), step):
        rows = state[start : start + step]
        chunk = worlds[start : start + step]
        nw = 1 << (len(rows) - 1).bit_length()
        keys = [
            (torch.cuda.current_device(), token + "/fidelity/opaque", res, nw),
            (torch.cuda.current_device(), glass, size, nw),
        ]
        # Both passes' segmentation views remain live until composition completes.
        with core._CTX_CACHE.lease(keys):
            first, first_seg = raster(glass, rows, chunk, res=size)
            opaque, shadowed, opaque_seg = raster(token + "/fidelity/opaque", rows, chunk, composite=True)
            views = np.unique(chunk.view)
            if len(views) == 1:
                ratio = shadowed.float() / references[int(views[0])]
            else:
                ratio = torch.empty(shadowed.shape, device=shadowed.device)
                for view in views:
                    selected = torch.as_tensor(np.flatnonzero(chunk.view == view), device=shadowed.device)
                    ratio[selected] = shadowed[selected].float() / references[int(view)]
            ratio = (ratio.clamp(0, 1) - 0.3).div(0.7).clamp(0, 1)
            moving = movers[(opaque_seg + 1).long()]
            shadows = torch.where(moving[..., None], opaque.float(), opaque.float() * ratio)
            shadows = shadows.reshape(len(rows), size, supersample, size, supersample, 3).mean((2, 4))
            amount = opacity[(first_seg + 1).long()][..., None]
            image = torch.where(amount < 1, amount * first.float() + (1 - amount) * shadows, shadows)
            images.append(image.round().clamp(0, 255).byte())
    return torch.cat(images)
