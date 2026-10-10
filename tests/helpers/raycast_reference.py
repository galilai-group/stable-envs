from __future__ import annotations

import torch

from stable_envs.render import raycast3d as R3


@torch.no_grad()
def composite_window(
    plate: torch.Tensor,
    plate_t: torch.Tensor,
    cam: R3.Camera,
    prims: R3.Primitives,
    xpos: torch.Tensor,
    xmat: torch.Tensor,
    lights: R3.Lights,
    H: int,
    W: int,
    supersample: int = 1,
    out: torch.Tensor | None = None,
    tile: int | None = None,
    mat_rgb: torch.Tensor | None = None,
    size_scale: torch.Tensor | None = None,
    shader=None,
    *,
    cover_compaction: bool = True,
) -> torch.Tensor:
    """Reference compositor that shades and merges the whole caller-sized window.

    Arguments match raycast3d.composite. ``cover_compaction`` selects union-only
    shading or all-sample shading for differential tests."""
    device = plate.device
    B = xpos.shape[0]
    ss = int(supersample)
    Rh, Rw = H * ss, W * ss
    o, d = R3.cached_rays(cam, Rh, Rw, device)
    P = prims.to(device)
    ldir, lamb, ldif, lspec = lights.to(device)

    if out is None:
        out = torch.empty((B, H, W, 3), dtype=torch.uint8, device=device)
    else:
        out = out.view(B, H, W, 3)

    base_t = plate_t if ss == 1 else R3._upsample_t(plate_t, H, W, ss)
    org = None
    if tile is not None and tile < H and tile < W:
        bs = 1.0 if size_scale is None else float(size_scale.max())
        org = _tile_origin(cam, prims, xpos, H, W, tile, device, bound_scale=bs)
    if org is None:
        N = Rh * Rw
        oo = o.expand(B, N, 3)
        dd = d.expand(B, N, 3)
        bt = base_t.expand(B, N)
    else:
        idx = R3._expand_tile(org, tile, ss, Rw, device)
        N = idx.shape[1]
        oo = o.expand(B, N, 3)
        dd = d[idx.reshape(-1)].view(B, N, 3)
        bt = base_t[idx]

    nprim = len(P["kind"])
    sz = P["size"].view(1, nprim, 3)
    if size_scale is not None:
        sz = sz * size_scale.view(B, 1, 1)
    t, n, i_hit = R3.cast_nearest(P["kind"], sz.expand(B, nprim, 3), xpos, xmat, oo, dd)
    cov = t < bt

    def by_shader(cols, ih, mat):
        rows = torch.arange(B, device=device).unsqueeze(1).expand_as(cols).reshape(-1)
        cols = cols.reshape(-1)
        ray = dd[rows, cols]
        hit = oo[rows, cols] + t[rows, cols].clamp(max=1e6).unsqueeze(-1) * ray
        c = shader(rows, hit, n[rows, cols], -ray, ih.reshape(-1), mat.reshape(-1, 3))
        return c.view(B, -1, 3)

    if cover_compaction:
        col = torch.zeros((B, N, 3), device=device, dtype=n.dtype)
        js = torch.nonzero(cov.any(0), as_tuple=False).reshape(-1)
        if js.numel():
            ih = i_hit[:, js]
            mat = P["mat"][ih] if mat_rgb is None else mat_rgb[torch.arange(B, device=device).unsqueeze(1), ih]
            if shader is not None:
                c = by_shader(js.unsqueeze(0).expand(B, -1), ih, mat)
            else:
                c = (
                    R3.shade(
                        n[:, js],
                        -dd[:, js],
                        ldir,
                        lamb,
                        ldif,
                        lspec,
                        lights.exponent,
                        mat,
                        P["tex"][ih],
                        P["spec"][ih],
                    )
                    * 255.0
                )
            col.index_copy_(1, js, torch.where(cov[:, js].unsqueeze(-1), c, torch.zeros_like(c)))
    else:  # the expression above is the identity of this one
        mat = P["mat"][i_hit] if mat_rgb is None else mat_rgb[torch.arange(B, device=device).unsqueeze(1), i_hit]
        if shader is not None:
            c = by_shader(torch.arange(N, device=device).unsqueeze(0).expand(B, -1), i_hit, mat)
        else:
            c = R3.shade(n, -dd, ldir, lamb, ldif, lspec, lights.exponent, mat, P["tex"][i_hit], P["spec"][i_hit])
            c = c * 255.0
        col = torch.where(cov.unsqueeze(-1), c, torch.zeros_like(c))

    covf = cov.to(col.dtype)
    if org is None:
        col = col.view(B, H, ss, W, ss, 3).sum((2, 4)) / (ss * ss)
        covf = covf.view(B, H, ss, W, ss).sum((2, 4)) / (ss * ss)
        base = plate.to(col.dtype).view(-1, H, W, 3)
        out.copy_(torch.clamp(torch.round(base * (1 - covf.unsqueeze(-1)) + col), 0, 255).to(torch.uint8))
        return out

    col = col.view(B, tile, ss, tile, ss, 3).sum((2, 4)) / (ss * ss)
    covf = covf.view(B, tile, ss, tile, ss).sum((2, 4)) / (ss * ss)
    oidx = R3._expand_tile(org, tile, 1, W, device).unsqueeze(-1).expand(B, tile * tile, 3)
    out.copy_(plate.view(-1, H, W, 3).expand(B, H, W, 3))
    patch = torch.gather(plate.reshape(-1, H * W, 3).to(col.dtype).expand(B, H * W, 3), 1, oidx)
    merged = patch * (1 - covf.reshape(B, -1, 1)) + col.reshape(B, -1, 3)
    out.view(B, H * W, 3).scatter_(1, oidx, torch.clamp(torch.round(merged), 0, 255).to(torch.uint8))
    return out


def _tile_origin(
    cam: R3.Camera,
    prims: R3.Primitives,
    xpos: torch.Tensor,
    H: int,
    W: int,
    tile: int,
    device,
    bound_scale: float = 1.0,
):
    """Return clamped output-pixel origins, or None if the movers exceed the window."""
    span = R3.screen_span(cam, prims, xpos, H, W, device, bound_scale)
    if span is None:
        return None
    x0, y0, w, h = span
    if w > tile or h > tile:
        return None
    return x0.clamp(0, W - tile), y0.clamp(0, H - tile)
