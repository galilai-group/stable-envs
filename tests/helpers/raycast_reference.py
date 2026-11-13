from __future__ import annotations

import numpy as np
import torch

from stable_envs.render import raycast3d as R3
from stable_envs.render.primitives import dot3


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


def _camera():
    return R3.Camera(
        pos=(0.0, -2.6, 1.1),
        forward=(0.0, 1.0, -0.35),
        up=(0.0, 0.35, 1.0),
        near=0.05,
        top=0.026,
        bottom=-0.026,
        center=0.0,
    )


def _lights():
    return R3.Lights(
        dirs=np.array([[0.3, -0.5, 0.81], [-0.6, 0.2, 0.77]]),
        ambient=np.array([0.25, 0.1]),
        diffuse=np.array([0.7, 0.4]),
        specular=np.array([0.35, 0.2]),
        exponent=12.0,
    )


def _prims(kinds, g):
    K = len(kinds)
    size = torch.rand(K, 3, generator=g).numpy() * 0.12 + 0.03
    return R3.Primitives(
        kind=np.asarray(kinds, int),
        size=size,
        mat_rgb=torch.rand(K, 3, generator=g).numpy(),
        tex_rgb=torch.rand(K, 3, generator=g).numpy() * 0.5 + 0.5,
        spec=torch.rand(K, generator=g).numpy(),
    )


#: ``t`` of a mover at this scene's default placement, from ``_camera()``. The "band" ``plate_t``
#: is centered on it so the threshold cuts through the movers.
_MOVER_T = 2.72


def scene(
    seed,
    *,
    B=5,
    K=2,
    H=41,
    W=37,
    ss=2,
    spread=0.25,
    depth=0.0,
    kinds=(R3.SPHERE, R3.CAPSULE),
    plate_t="floor",
    offset=(0.0, 0.0),
    scales=None,
):
    """A plate, its ray parameter, and a batch of mover poses in front of the camera.

    ``plate_t`` selects the occlusion threshold:

    * ``"floor"`` -- the analytic depth of a ground plane through the origin, behind the movers.
    * ``"inf"`` -- nothing static in the way, so ``INF < INF`` has to stay False.
    * ``"band"`` -- a finite per-pixel field that cuts through the movers, so a sample's coverage
      depends on which pixel's ``plate_t`` it reads. Only this mode exercises the ``bt`` gather, which
      must map ``(ty, a, tx, b) -> (y0+ty, x0+tx)``; a transposition is invisible under the other
      modes. ``y`` and ``x`` have different periods so a transposition cannot alias onto itself, and
      ``R3.INF`` never appears in the field.

    ``offset`` shifts every mover in world x and z, pushing windows against a frame edge so the origin
    clamp in ``_fit_tile`` engages. ``scales`` is a per-row ``size_scale``.
    """
    g = torch.Generator().manual_seed(seed)
    cam = _camera()
    prims = _prims(kinds[:K], g)
    plate = (torch.rand(H, W, 3, generator=g) * 255).to(torch.uint8)
    if plate_t == "inf":
        plate_t_v = torch.full((H * W,), R3.INF)
    elif plate_t == "band":
        y = torch.arange(H).view(H, 1).to(torch.float32)
        x = torch.arange(W).view(1, W).to(torch.float32)
        phase = (3.0 * y + 7.0 * x) % 11.0 / 11.0 + 0.017 * ((y * 5 + x * 2) % 3)
        plate_t_v = (_MOVER_T - 0.22 + 0.44 * phase).reshape(-1).contiguous()
        assert torch.isfinite(plate_t_v).all() and float(plate_t_v.max()) < R3.INF
    else:
        floor = R3.Primitives(
            kind=np.array([R3.PLANE]),
            size=np.zeros((1, 3)),
            mat_rgb=np.zeros((1, 3)),
            tex_rgb=np.ones((1, 3)),
            spec=np.zeros(1),
        )
        plate_t_v = R3.static_depth(cam, floor, np.zeros((1, 3)), np.eye(3).reshape(1, 3, 3), H, W, "cpu")
    xpos = torch.zeros(B, K, 3)
    xpos[..., 0] = (torch.rand(B, K, generator=g) - 0.5) * spread + offset[0]
    xpos[..., 1] = (torch.rand(B, K, generator=g) - 0.5) * spread + depth
    xpos[..., 2] = torch.rand(B, K, generator=g) * spread + 0.12 + offset[1]
    xmat = torch.eye(3).view(1, 1, 3, 3).expand(B, K, 3, 3).contiguous()
    sc = dict(
        plate=plate,
        plate_t=plate_t_v,
        cam=cam,
        prims=prims,
        xpos=xpos,
        xmat=xmat,
        lights=_lights(),
        H=H,
        W=W,
        supersample=ss,
    )
    if scales is not None:
        sc["_scales"] = torch.as_tensor(scales, dtype=torch.float32)
    return sc


def split_scene(sc):
    """The composite kwargs, and the per-row ``size_scale`` the scene carried beside them."""
    sc = dict(sc)
    return sc, sc.pop("_scales", None)


KINDS = (R3.PLANE, R3.SPHERE, R3.CAPSULE, R3.CYLINDER, R3.BOX, R3.ELLIPSOID)


def random_primitive_scene(
    rng, B, K, N, *, degenerate=False, nonunit_d=False, tangent=False, far=0.0, kinds=None, dd_at_tol=0
):
    g = torch.Generator().manual_seed(rng)
    r = lambda *s: torch.rand(*s, generator=g, dtype=torch.float32)  # noqa: E731
    # Pinned ``kinds`` cover scenes that mix bounded and unbounded kinds writing into the same
    # ``best`` / ``blocked`` on every run, rather than whenever sampling reaches them.
    kinds = [KINDS[int(v)] for v in torch.randint(0, len(KINDS), (K,), generator=g)] if kinds is None else list(kinds)
    K = len(kinds)
    sizes = r(B, K, 3) * 0.4 + 0.02
    if degenerate:
        sizes[:, ::2] = 0.0  # zero radius / half-length / extent
        sizes[0, 0] = 0.0
    pos = (r(B, K, 3) - 0.5) * 1.5
    q = torch.randn(B, K, 4, generator=g)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        -1,
    ).view(B, K, 3, 3)
    # ``far`` puts the camera far from small primitives (~100 m to a 5 cm capsule). There
    # |oc|^2|d|^2 and (oc.d)^2 are each ~1e4 and differ by ~r^2 ~ 2.5e-3, float32's resolution at that
    # magnitude, so a bound that subtracts them can cancel away a true hit.
    o = ((r(B, N, 3) - 0.5) * 0.2 + torch.tensor([0.0, -2.0, 0.4])).contiguous()
    if far:
        sizes = sizes * 0.05  # centimeter-scale primitives...
        pos = pos * 0.05
        o = o + torch.tensor([0.0, -float(far), 0.35 * float(far)])  # ...seen from far away
    if tangent:
        # aim a third of the rays exactly at each primitive's bound radius: the grazing case, which is
        # where a bound that is one ulp too small drops a pixel.
        tgt = pos[:, torch.randint(0, K, (N,), generator=g)]
        rad = (sizes[:, :, 0] + sizes[:, :, 1]).mean()
        off = torch.nn.functional.normalize(torch.randn(B, N, 3, generator=g), dim=-1)
        tgt = tgt + off * rad
        d = torch.nn.functional.normalize(tgt - o, dim=-1)
    else:
        d = torch.nn.functional.normalize((r(B, N, 3) - 0.5) + torch.tensor([0.0, 1.0, 0.0]), dim=-1)
    if nonunit_d:
        d = d * (r(B, N, 1) * 3.0)  # includes exact zeros
    if dd_at_tol:
        # Put |d|^2 just inside the admission window, the worst bundle ``_unit_rays`` lets through.
        # f backs off from 0.9 until the float32 |d|^2 the cast computes is inside.
        for f in (0.9, 0.7, 0.5, 0.3):
            m = torch.tensor((1.0 + dd_at_tol * f * R3._UNIT_D_TOL) ** 0.5, dtype=torch.float32)
            cand = (d * m).contiguous()
            dd = dot3(cand, cand)
            if bool(torch.all((dd - 1.0).abs() <= R3._UNIT_D_TOL)):
                break
        else:
            raise AssertionError("could not place |d|^2 inside the admission window")
        d = cand
        assert float((dd - 1.0).abs().max()) > R3._UNIT_D_TOL / 4, (
            "this case must sit at the edge of the window but landed near |d| = 1 "
            f"(max |dd-1| = {float((dd - 1.0).abs().max()):.3e})"
        )
    return kinds, sizes.contiguous(), pos.contiguous(), R.contiguous(), o, d.contiguous()
