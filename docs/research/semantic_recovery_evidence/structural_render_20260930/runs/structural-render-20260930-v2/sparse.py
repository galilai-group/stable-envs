"""Diagnostic: compact world/ray pairs, not the union of ray indices across worlds."""
import torch
from stable_envs.render import raycast3d as r

ORIGINAL_CAST = r.cast_nearest_torch
ORIGINAL_BLOCK = r.any_blocking_torch


def cast(kinds, sizes, pos, rot, o, d):
    dd = r._dot(d, d)
    if not r.PRIMITIVE_BOUNDS or not any(r._kind_is_bounded(int(k)) for k in kinds) or not r._unit_rays(dd):
        return ORIGINAL_CAST(kinds, sizes, pos, rot, o, d)
    b, n = o.shape[:2]
    best = torch.full((b, n), r.INF, device=o.device, dtype=o.dtype)
    normal = torch.zeros((b, n, 3), device=o.device, dtype=o.dtype)
    index = torch.zeros((b, n), device=o.device, dtype=torch.long)
    for k, kind in enumerate(map(int, kinds)):
        rad = r._bound_radius(kind, sizes[:, k:k+1]) if r._kind_is_bounded(kind) else None
        if rad is None:
            t, norm = r.intersect(kind, sizes[:, k:k+1], pos[:, k:k+1], rot[:, k:k+1], o, d)
            update = t < best
            best = torch.where(update, t, best)
            normal = torch.where(update[..., None], norm, normal)
            index = torch.where(update, k, index)
            continue
        bi, ri = torch.nonzero(r._rays_near(pos[:, k:k+1], rad, o, d, dd), as_tuple=True)
        if not bi.numel():
            continue
        t, norm = r.intersect(kind, sizes[bi, k], pos[bi, k], rot[bi, k], o[bi, ri], d[bi, ri])
        update = t < best[bi, ri]
        best[bi, ri] = torch.where(update, t, best[bi, ri])
        normal[bi, ri] = torch.where(update[..., None], norm, normal[bi, ri])
        index[bi, ri] = torch.where(update, k, index[bi, ri])
    return best, normal, index


def block(kinds, sizes, pos, rot, o, d, lim, skip=None):
    dd = r._dot(d, d)
    if not r.PRIMITIVE_BOUNDS or not any(r._kind_is_bounded(int(k)) for k in kinds) or not r._unit_rays(dd):
        return ORIGINAL_BLOCK(kinds, sizes, pos, rot, o, d, lim, skip)
    blocked = torch.zeros(o.shape[:2], device=o.device, dtype=torch.bool)
    for k, kind in enumerate(map(int, kinds)):
        rad = r._bound_radius(kind, sizes[:, k:k+1]) if r._kind_is_bounded(kind) else None
        mask = ~blocked
        if rad is not None:
            mask &= r._rays_near(pos[:, k:k+1], rad, o, d, dd)
        if skip is not None:
            mask &= skip != k
        bi, ri = torch.nonzero(mask, as_tuple=True)
        if not bi.numel():
            continue
        t, _ = r.intersect(kind, sizes[bi, k], pos[bi, k], rot[bi, k], o[bi, ri], d[bi, ri])
        blocked[bi, ri] |= (t > 0) & (t < lim.expand_as(blocked)[bi, ri])
    return blocked


def install(enabled):
    r.cast_nearest_torch = cast if enabled else ORIGINAL_CAST
    r.any_blocking_torch = block if enabled else ORIGINAL_BLOCK
