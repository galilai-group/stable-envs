"""Differential fuzz: ``render_torch`` against a per-pixel reference implementation, byte for byte.

Registered scenes cover few edge cases, so this generates awkward ones: disabled slots, zero-area
and collinear polygons, degenerate padded edges, negative radii, inner discs, outlines wider than the
shape, primitives off the frame or larger than the world, flip_y, non-square worlds and outputs, and
supersample 2.

The reference implementation is inlined rather than imported so that the comparison is independent.

    python scripts/regression/raster2d/fuzz_render_torch.py --cases 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError("repo root not found")


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

from stable_envs.render import raster2d as R2  # noqa: E402
from stable_envs.render.scene import KIND_DISC, KIND_POLY, KIND_SEGMENT, Scene2D  # noqa: E402


@torch.no_grad()
def render_torch_baseline(scene, out_size, supersample: int = 1) -> torch.Tensor:
    """Per-pixel reference ``render_torch``: one masked pass per primitive."""
    H = W = int(out_size) if isinstance(out_size, int) else out_size[0]
    if not isinstance(out_size, int):
        H, W = out_size
    ss = int(supersample)
    Rh, Rw = H * ss, W * ss
    dev, dt = scene.kind.device, scene.geom.dtype
    px, py = R2._pixel_grid(scene, Rh, Rw, dev, dt)
    B, N, V = scene.batch, scene.n_prims, scene.vertices

    canvas = scene.background.view(B, 3, 1, 1).expand(B, 3, Rh, Rw).clone()
    for i in range(N):
        kind = scene.kind[:, i]
        g = scene.geom[:, i]
        fill = scene.fill[:, i].view(B, 3, 1, 1)
        line = scene.line[:, i].view(B, 3, 1, 1)
        lw = scene.line_w[:, i].view(B, 1, 1)
        p = px.unsqueeze(0)
        q = py.unsqueeze(0)

        d2 = (p - g[:, 0].view(B, 1, 1)) ** 2 + (q - g[:, 1].view(B, 1, 1)) ** 2
        is_disc = (kind == KIND_DISC).view(B, 1, 1)
        hit = is_disc & (d2 <= g[:, 2].view(B, 1, 1) ** 2)
        canvas = torch.where(hit.unsqueeze(1), fill, canvas)
        hit = is_disc & (d2 <= g[:, 3].view(B, 1, 1) ** 2) & (g[:, 3].view(B, 1, 1) > 0)
        canvas = torch.where(hit.unsqueeze(1), line, canvas)

        sdf = torch.full((B, Rh, Rw), -1e30, device=dev, dtype=dt)
        edge_d2 = torch.full((B, Rh, Rw), 1e30, device=dev, dtype=dt)
        for e in range(V):
            ax, ay = g[:, 2 * e].view(B, 1, 1), g[:, 2 * e + 1].view(B, 1, 1)
            j = (e + 1) % V
            bx, by = g[:, 2 * j].view(B, 1, 1), g[:, 2 * j + 1].view(B, 1, 1)
            ex, ey = bx - ax, by - ay
            L2 = ex * ex + ey * ey
            degen = L2 < 1e-12
            inv = torch.rsqrt(L2.clamp_min(1e-20))
            d = (p - ax) * (ey * inv) + (q - ay) * (-ex * inv)
            sdf = torch.maximum(sdf, torch.where(degen, torch.full_like(d, -1e30), d))
            t = (((p - ax) * ex + (q - ay) * ey) / L2.clamp_min(1e-20)).clamp(0, 1)
            qx, qy = p - ax - t * ex, q - ay - t * ey
            edge_d2 = torch.minimum(edge_d2, torch.where(degen, torch.full_like(qx, 1e30), qx * qx + qy * qy))
        is_poly = (kind == KIND_POLY).view(B, 1, 1)
        canvas = torch.where((is_poly & (sdf <= 0)).unsqueeze(1), fill, canvas)
        canvas = torch.where((is_poly & (lw > 0) & (edge_d2 <= lw * lw)).unsqueeze(1), line, canvas)

        ax, ay = g[:, 0].view(B, 1, 1), g[:, 1].view(B, 1, 1)
        ex, ey = g[:, 2].view(B, 1, 1) - ax, g[:, 3].view(B, 1, 1) - ay
        t = (((p - ax) * ex + (q - ay) * ey) / (ex * ex + ey * ey).clamp_min(1e-20)).clamp(0, 1)
        dx, dy = p - ax - t * ex, q - ay - t * ey
        hit = (kind == KIND_SEGMENT).view(B, 1, 1) & (dx * dx + dy * dy <= g[:, 4].view(B, 1, 1) ** 2)
        canvas = torch.where(hit.unsqueeze(1), fill, canvas)

    if ss > 1:
        canvas = torch.nn.functional.avg_pool2d(canvas, ss)
    return canvas.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def random_scene(gen: torch.Generator, device: str):
    # Drawn on the CPU and then moved, so --device cuda fuzzes the same scenes as a CPU run.
    device_out, device = device, "cpu"
    B = int(torch.randint(1, 5, (1,), generator=gen).item())
    N = int(torch.randint(1, 7, (1,), generator=gen).item())
    V = int(torch.randint(3, 6, (1,), generator=gen).item())
    rnd = lambda *s: torch.rand(*s, generator=gen, device=device)  # noqa: E731
    x0, y0 = (rnd(2) * 4 - 2).tolist()
    w, h = (rnd(2) * 3 + 0.5).tolist()
    world = (x0, y0, x0 + w, y0 + h)
    kind = torch.randint(0, 4, (B, N), generator=gen, device=device, dtype=torch.int32)
    geom = (rnd(B, N, 2 * V) - 0.5) * 6 + torch.tensor([x0 + w / 2, y0 + h / 2] * V, device=device)
    style = torch.randint(0, 9, (N,), generator=gen)
    for i, s in enumerate(style.tolist()):
        if s == 0:  # collinear polygon: every vertex on one line
            t = rnd(B, V) * 4 - 2
            geom[:, i, 0::2] = geom[:, i, 0:1] + t
            geom[:, i, 1::2] = geom[:, i, 1:2] + 2 * t
        elif s == 1:  # every vertex identical: all edges degenerate
            geom[:, i] = geom[:, i, 0:2].repeat(1, V)
        elif s == 2:  # a triangle padded to V by repeating the last vertex
            geom[:, i, 6:] = geom[:, i, 4:6].repeat(1, V - 3)
        elif s == 3:  # negative radius / negative half width
            geom[:, i, 2] = -geom[:, i, 2].abs()
            geom[:, i, 4] = -geom[:, i, 4].abs()
        elif s == 4:  # an inner disc (a ring)
            geom[:, i, 2] = 1.0
            geom[:, i, 3] = rnd(B) * 1.4  # sometimes larger than the outer radius
        elif s == 5:  # entirely off the frame
            geom[:, i] = geom[:, i] + 50.0
        elif s == 6:  # far larger than the world
            geom[:, i] = (geom[:, i] - geom[:, i].mean()) * 40
        elif s == 7:  # a disc with no inner radius at all
            geom[:, i, 3] = -1.0
    lw = torch.where(rnd(B, N) < 0.5, torch.zeros(B, N, device=device), rnd(B, N) * 0.6)
    return (
        Scene2D(
            kind=kind,
            geom=geom.contiguous(),
            fill=rnd(B, N, 3) * 255,
            line=rnd(B, N, 3) * 255,
            line_w=lw,
            background=rnd(B, 3) * 255,
            world=world,
            flip_y=bool(torch.rand(1, generator=gen).item() < 0.5),
            static_prefix=0,
        )
        .validate()
        .to(device=torch.device(device_out))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--size", type=int, default=24)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(a.seed)
    bad = 0
    for c in range(a.cases):
        sc = random_scene(gen, a.device)
        ss = 1 + (c % 3 == 2)
        size = a.size if c % 4 else (a.size, a.size + 5)  # non-square output too
        want = render_torch_baseline(sc, size, ss)
        got = R2.render_torch(sc, size, ss)
        if not torch.equal(want, got):
            bad += 1
            d = (want.to(torch.int16) - got.to(torch.int16)).abs()
            print(
                f"case {c}: {int((d.amax(-1) > 0).sum())} pixels differ, max {int(d.max())}, "
                f"B={sc.batch} N={sc.n_prims} V={sc.vertices} ss={ss} flip={sc.flip_y} "
                f"world={sc.world}"
            )
            if bad > 5:
                break
    print(f"{a.cases - bad}/{a.cases} identical" if not bad else f"FAILED: {bad} differing cases")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
