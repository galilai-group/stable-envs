"""Phase attribution for the raster2d workstream, finer than the operator trace can give.

``scratch/step6/profile_note.py`` is the canonical harness and stays the canonical harness: it times
the public path and attributes by OPERATOR. Section 8 of this workstream's brief asks for a
different cut — scene construction vs raster, primitive dispatch, grid creation,
masks/intermediates, allocation, static background, dynamic primitives, compositing, output
conversion — and an operator trace cannot separate "the grid" from "the disc test" because both are
``aten::sub``. ``profile_note.py``'s own docstring says a workstream that needs finer phases should
add real timers in its own branch and say so. This is that, and it says so.

It does not instrument the renderer. Production code with timers in it is production code with
timers in it, and this has to run against the UNMODIFIED baseline. Instead the per-phase timings
come from an instrumented REPLICA of ``render_torch``'s body, and the replica is checked against the
real function by digest on every run: if ``replica_matches_render_torch`` is false, every number
below it is describing something else and must be thrown away.

The two families in this workstream are measured differently because they are different code:

  * **Scene2D** (PushT, SimplePointMaze) — ``env.scene()`` then ``raster2d.render()``. The replica
    splits the raster into grid, canvas, per-primitive-family work, and output conversion, and
    counts how much of the frame each primitive's world-space bounding box actually covers.
  * **swm_nav torch arithmetic** (TwoRoom, TwoRoomTarget, Piecewise, PiecewiseZones) — no Scene2D.
    The phases are the env's own: ``base_image`` (static background + structure), the Gaussian
    dots, and the alpha blends.

    python scratch/step6/raster2d/attribute.py --env PushT --batch 1 64 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import stable_envs as se                                          # noqa: E402
from stable_envs.render import raster2d as R2                     # noqa: E402
from stable_envs.render.scene import KIND_DISC, KIND_POLY, KIND_SEGMENT   # noqa: E402
from stable_envs.testing import conformance as C                  # noqa: E402

SCENE_ENVS = ("PushT", "SimplePointMaze")
NAV_ENVS = ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones")


class Clock:
    """Accumulating wall-clock phases. CPU only; this workstream's target is the CPU path."""

    def __init__(self):
        self.t: dict[str, float] = {}

    def __call__(self, name):
        return _Span(self, name)

    def add(self, name, dt):
        self.t[name] = self.t.get(name, 0.0) + dt


class _Span:
    def __init__(self, clock, name):
        self.c, self.n = clock, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.c.add(self.n, (time.perf_counter() - self.t0) * 1e3)
        return False


# ------------------------------------------------------------------ Scene2D replica


def replica(scene, out_size, ss, clk) -> torch.Tensor:
    """``render_torch`` with timers. Operation for operation the same, digest-checked by the caller."""
    H = W = int(out_size) if isinstance(out_size, int) else out_size[0]
    if not isinstance(out_size, int):
        H, W = out_size
    Rh, Rw = H * ss, W * ss
    dev, dt = scene.kind.device, scene.geom.dtype
    with clk("grid"):
        px, py = R2._pixel_grid(scene, Rh, Rw, dev, dt)
    B, N, V = scene.batch, scene.n_prims, scene.vertices
    with clk("canvas alloc"):
        canvas = scene.background.view(B, 3, 1, 1).expand(B, 3, Rh, Rw).clone()
    K = int(scene.static_prefix)
    for i in range(N):
        tag = "static prefix" if i < K else "dynamic"
        kind = scene.kind[:, i]
        g = scene.geom[:, i]
        fill = scene.fill[:, i].view(B, 3, 1, 1)
        line = scene.line[:, i].view(B, 3, 1, 1)
        lw = scene.line_w[:, i].view(B, 1, 1)
        p = px.unsqueeze(0)
        q = py.unsqueeze(0)
        with clk("prim/disc"), clk(f"{tag}/disc"):
            d2 = (p - g[:, 0].view(B, 1, 1)) ** 2 + (q - g[:, 1].view(B, 1, 1)) ** 2
            is_disc = (kind == KIND_DISC).view(B, 1, 1)
            hit = is_disc & (d2 <= g[:, 2].view(B, 1, 1) ** 2)
            canvas = torch.where(hit.unsqueeze(1), fill, canvas)
            hit = is_disc & (d2 <= g[:, 3].view(B, 1, 1) ** 2) & (g[:, 3].view(B, 1, 1) > 0)
            canvas = torch.where(hit.unsqueeze(1), line, canvas)
        with clk("prim/poly"), clk(f"{tag}/poly"):
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
                edge_d2 = torch.minimum(edge_d2, torch.where(degen, torch.full_like(qx, 1e30),
                                                             qx * qx + qy * qy))
            is_poly = (kind == KIND_POLY).view(B, 1, 1)
            canvas = torch.where((is_poly & (sdf <= 0)).unsqueeze(1), fill, canvas)
            canvas = torch.where((is_poly & (lw > 0) & (edge_d2 <= lw * lw)).unsqueeze(1),
                                 line, canvas)
        with clk("prim/segment"), clk(f"{tag}/segment"):
            ax, ay = g[:, 0].view(B, 1, 1), g[:, 1].view(B, 1, 1)
            ex, ey = g[:, 2].view(B, 1, 1) - ax, g[:, 3].view(B, 1, 1) - ay
            t = (((p - ax) * ex + (q - ay) * ey) / (ex * ex + ey * ey).clamp_min(1e-20)).clamp(0, 1)
            dx, dy = p - ax - t * ex, q - ay - t * ey
            hit = ((kind == KIND_SEGMENT).view(B, 1, 1)
                   & (dx * dx + dy * dy <= g[:, 4].view(B, 1, 1) ** 2))
            canvas = torch.where(hit.unsqueeze(1), fill, canvas)
    with clk("output conversion"):
        if ss > 1:
            canvas = torch.nn.functional.avg_pool2d(canvas, ss)
        out = canvas.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    return out


def replica_v2(scene, out_size, ss, clk) -> torch.Tensor:
    """The Step 6 ``render_torch`` with timers, for the AFTER side of the same attribution.

    One script has to serve both trees, because the before and the after have to be cut the same
    way to be comparable. Which replica runs is decided by whether the renderer has the pre-pass,
    not by a flag the caller could get wrong, and both are digest-checked against the real function
    on every run.
    """
    H = W = int(out_size) if isinstance(out_size, int) else out_size[0]
    if not isinstance(out_size, int):
        H, W = out_size
    Rh, Rw = H * ss, W * ss
    dev, dt = scene.kind.device, scene.geom.dtype
    with clk("grid"):
        px, py = R2._pixel_grid(scene, Rh, Rw, dev, dt)
    B, N, V = scene.batch, scene.n_prims, scene.vertices
    flat_out = ss == 1
    with clk("canvas alloc"):
        if flat_out:
            u8 = lambda c: c.round().clamp(0, 255).to(torch.uint8).view(B, 1, 1, 3)   # noqa: E731
            canvas = u8(scene.background).expand(B, Rh, Rw, 3).clone()
        else:
            u8 = lambda c: c.view(B, 3, 1, 1)                                         # noqa: E731
            canvas = scene.background.view(B, 3, 1, 1).expand(B, 3, Rh, Rw).clone()
    with clk("plan (pre-pass)"):
        plan = R2._prim_plan(scene, Rh, Rw)
    K = int(scene.static_prefix)
    painted = 0
    for i, (has_disc, has_ring, has_poly, has_outline, has_seg, edges,
            r0, r1, c0, c1) in enumerate(plan):
        if r1 <= r0 or c1 <= c0 or not (has_disc or has_poly or has_seg):
            continue
        painted += 1
        tag = "static prefix" if i < K else "dynamic"
        kind = scene.kind[:, i]
        g = scene.geom[:, i]
        fill, line = u8(scene.fill[:, i]), u8(scene.line[:, i])
        lw = scene.line_w[:, i].view(B, 1, 1)
        p = px[r0:r1, c0:c1].unsqueeze(0)
        q = py[r0:r1, c0:c1].unsqueeze(0)
        view = ((lambda: canvas[:, r0:r1, c0:c1, :]) if flat_out
                else (lambda: canvas[:, :, r0:r1, c0:c1]))
        mdim = -1 if flat_out else 1

        def paint(hit, colour, _v=view, _m=mdim):
            w = _v()
            w.copy_(torch.where(hit.unsqueeze(_m), colour, w))

        if has_disc:
            with clk("prim/disc"), clk(f"{tag}/disc"):
                d2 = (p - g[:, 0].view(B, 1, 1)) ** 2 + (q - g[:, 1].view(B, 1, 1)) ** 2
                is_disc = (kind == KIND_DISC).view(B, 1, 1)
                paint(is_disc & (d2 <= g[:, 2].view(B, 1, 1) ** 2), fill)
                if has_ring:
                    paint(is_disc & (d2 <= g[:, 3].view(B, 1, 1) ** 2)
                          & (g[:, 3].view(B, 1, 1) > 0), line)
        if has_poly:
            with clk("prim/poly"), clk(f"{tag}/poly"):
                h, w_ = r1 - r0, c1 - c0
                sdf = torch.full((B, h, w_), -1e30, device=dev, dtype=dt)
                edge_d2 = (torch.full((B, h, w_), 1e30, device=dev, dtype=dt)
                           if has_outline else None)
                for e, mixed in edges:
                    ax, ay = g[:, 2 * e].view(B, 1, 1), g[:, 2 * e + 1].view(B, 1, 1)
                    j = (e + 1) % V
                    bx, by = g[:, 2 * j].view(B, 1, 1), g[:, 2 * j + 1].view(B, 1, 1)
                    ex, ey = bx - ax, by - ay
                    L2 = ex * ex + ey * ey
                    degen = L2 < 1e-12
                    inv = torch.rsqrt(L2.clamp_min(1e-20))
                    d = (p - ax) * (ey * inv) + (q - ay) * (-ex * inv)
                    sdf = torch.maximum(sdf, torch.where(degen, torch.full_like(d, -1e30), d)
                                        if mixed else d)
                    if has_outline:
                        t = (((p - ax) * ex + (q - ay) * ey) / L2.clamp_min(1e-20)).clamp(0, 1)
                        qx, qy = p - ax - t * ex, q - ay - t * ey
                        e2 = qx * qx + qy * qy
                        edge_d2 = torch.minimum(
                            edge_d2, torch.where(degen, torch.full_like(qx, 1e30), e2)
                            if mixed else e2)
                is_poly = (kind == KIND_POLY).view(B, 1, 1)
                paint(is_poly & (sdf <= 0), fill)
                if has_outline:
                    paint(is_poly & (lw > 0) & (edge_d2 <= lw * lw), line)
        if has_seg:
            with clk("prim/segment"), clk(f"{tag}/segment"):
                ax, ay = g[:, 0].view(B, 1, 1), g[:, 1].view(B, 1, 1)
                ex, ey = g[:, 2].view(B, 1, 1) - ax, g[:, 3].view(B, 1, 1) - ay
                t = (((p - ax) * ex + (q - ay) * ey)
                     / (ex * ex + ey * ey).clamp_min(1e-20)).clamp(0, 1)
                dx, dy = p - ax - t * ex, q - ay - t * ey
                paint((kind == KIND_SEGMENT).view(B, 1, 1)
                      & (dx * dx + dy * dy <= g[:, 4].view(B, 1, 1) ** 2), fill)
    clk.add("primitives evaluated", 0.0)
    clk.t["primitives evaluated"] = painted
    with clk("output conversion"):
        if flat_out:
            out = canvas
        else:
            if ss > 1:
                canvas = torch.nn.functional.avg_pool2d(canvas, ss)
            out = canvas.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
    return out


def coverage(scene, H: int, W: int) -> dict:
    """How much of the frame each primitive's world-space bounding box actually covers.

    The bound is the same encoding ``_bbox_kernel`` uses on the GPU: a disc is the centre plus the
    larger radius, a capsule is the endpoints grown by the half width, a polygon is the vertex hull
    grown by the outline half width, a collinear polygon (twice-signed-area exactly zero) is
    UNBOUNDED and a disabled primitive is EMPTY. Reported per primitive and summed, because
    ``sum(box area) / (N * frame area)`` is the fraction of the per-primitive full-frame work that
    could survive a bounded evaluation.
    """
    x0, y0, x1, y1 = scene.world
    k = scene.kind.to(torch.int32)
    g = scene.geom.to(torch.float64)
    lw = scene.line_w.abs().to(torch.float64)
    V = scene.vertices
    vx, vy = g[..., 0::2][..., :V], g[..., 1::2][..., :V]
    hull0x, hull1x = vx.min(-1).values, vx.max(-1).values
    hull0y, hull1y = vy.min(-1).values, vy.max(-1).values
    area2 = (vx * torch.roll(vy, -1, -1) - torch.roll(vx, -1, -1) * vy).sum(-1)
    flat = area2 == 0
    r = torch.maximum(g[..., 2].abs(), g[..., 3].clamp_min(0))
    hw = g[..., 4].abs() if g.shape[-1] > 4 else torch.zeros_like(r)
    inf = torch.full_like(r, float("inf"))
    bx0 = torch.where(k == KIND_DISC, g[..., 0] - r,
                      torch.where(k == KIND_POLY, torch.where(flat, -inf, hull0x - lw),
                                  torch.minimum(g[..., 0], g[..., 2]) - hw))
    by0 = torch.where(k == KIND_DISC, g[..., 1] - r,
                      torch.where(k == KIND_POLY, torch.where(flat, -inf, hull0y - lw),
                                  torch.minimum(g[..., 1], g[..., 3]) - hw))
    bx1 = torch.where(k == KIND_DISC, g[..., 0] + r,
                      torch.where(k == KIND_POLY, torch.where(flat, inf, hull1x + lw),
                                  torch.maximum(g[..., 0], g[..., 2]) + hw))
    by1 = torch.where(k == KIND_DISC, g[..., 1] + r,
                      torch.where(k == KIND_POLY, torch.where(flat, inf, hull1y + lw),
                                  torch.maximum(g[..., 1], g[..., 3]) + hw))
    off = k == 0
    bx0, by0 = torch.where(off, inf, bx0), torch.where(off, inf, by0)
    bx1, by1 = torch.where(off, -inf, bx1), torch.where(off, -inf, by1)
    # union over the batch -> the window one shared evaluation would have to cover
    ux0, uy0 = bx0.min(0).values, by0.min(0).values
    ux1, uy1 = bx1.max(0).values, by1.max(0).values
    sx, sy = (x1 - x0) / W, (y1 - y0) / H
    cols = ((ux1 - ux0) / sx).clamp(0, W).nan_to_num(0.0)
    rows = ((uy1 - uy0) / sy).clamp(0, H).nan_to_num(0.0)
    frac = (cols * rows / (H * W))
    return {"per_primitive_frac": [round(float(f), 6) for f in frac],
            "kinds": [int(v) for v in scene.kind.amax(0).tolist()],
            "n_prims": int(scene.n_prims),
            "mean_frac": float(frac.mean()), "sum_frac": float(frac.sum())}


# ------------------------------------------------------------------ drivers


def med(fn, repeat, warmup):
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        s.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(s)


def do_scene_env(env, states, repeat, warmup) -> dict:
    st = torch.as_tensor(states, device=env.device, dtype=torch.float32)
    size, ss = int(env.size), int(env.ss)
    row = {"family": "scene2d"}
    row["ms_render_total"] = med(lambda: env.render(states), repeat, warmup)
    row["ms_scene"] = med(lambda: env.scene(st), repeat, warmup)
    sc = env.scene(st)
    row["ms_raster"] = med(lambda: R2.render(sc, size, ss, backend=env.backend), repeat, warmup)
    row["static_prefix"] = int(sc.static_prefix)
    row["n_prims"] = int(sc.n_prims)
    row["vertices"] = int(sc.vertices)
    rep = replica_v2 if hasattr(R2, "_prim_plan") else replica
    row["replica"] = rep.__name__
    clk = Clock()
    img = rep(sc, size, ss, clk)
    clk2 = Clock()
    for _ in range(max(1, repeat // 4)):
        rep(sc, size, ss, clk2)
    n = max(1, repeat // 4)
    row["replica_matches_render_torch"] = (
        C.frames_digest(img) == C.frames_digest(R2.render_torch(sc, size, ss)))
    row["phases_ms"] = {k: round(v / n, 4) for k, v in clk2.t.items()}
    row["coverage"] = coverage(sc, size * ss, size * ss)
    return row


def do_nav_env(env, states, repeat, warmup) -> dict:
    """TwoRoom / Piecewise: no Scene2D. Phases are base image, Gaussian dots, alpha blends."""
    st = torch.as_tensor(states, device=env.device, dtype=torch.float32).reshape(-1, env.state_dim)
    n = int(st.shape[0])
    sty = env.resolve_style(None, None, batch=n)
    sl = lambda p: sty[p][0:n]                                           # noqa: E731
    row = {"family": "swm_nav"}
    row["ms_render_total"] = med(lambda: env.render(states), repeat, warmup)
    row["ms_resolve_style"] = med(lambda: env.resolve_style(None, None, batch=n), repeat, warmup)
    row["ms_base_image"] = med(lambda: env.base_image(sl, n), repeat, warmup)
    base = env.base_image(sl, n)
    row["ms_blend_dots"] = med(lambda: env.blend_dots(base.clone(), st, sl, n), repeat, warmup)
    from stable_envs.envs import swm_nav as SN
    row["ms_one_gaussian_dot"] = med(
        lambda: SN._gaussian_dot(env._gy, env._gx, st[:, 0:2], sl("agent.radius")), repeat, warmup)
    dot = SN._gaussian_dot(env._gy, env._gx, st[:, 0:2], sl("agent.radius"))
    row["ms_one_alpha_blend"] = med(lambda: SN._alpha_blend(base, dot, sl("agent.color")),
                                    repeat, warmup)
    row["base_image_is_state_independent"] = bool(
        C.frames_digest(env.base_image(sl, n)) == C.frames_digest(base))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--states-dir", default=str(ROOT / "examples" / "results" / "step4" / "states"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    pool = np.load(Path(a.states_dir) / f"{a.env}.npz")["states"]
    note = {"env": a.env, "device": a.device, "threads": a.threads,
            "host": socket.gethostname(), "cpu": platform.machine(),
            "torch": torch.__version__, "batches": {}}
    for B in a.batch:
        states = np.asarray(pool[np.arange(B) % len(pool)], np.float32)
        env = se.make(a.env, batch=B, device=a.device)
        note["size"] = int(env.size)
        fn = do_scene_env if a.env in SCENE_ENVS else do_nav_env
        note["batches"][str(B)] = fn(env, states, a.repeat, a.warmup)
        print(f"B={B}", json.dumps({k: v for k, v in note["batches"][str(B)].items()
                                    if k.startswith("ms_") or k.startswith("replica")}), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"attribution_{a.env}_{a.device}_{note['host'].split('.')[0]}.json"
    f.write_text(json.dumps(note, indent=2))
    print("wrote", f)


if __name__ == "__main__":
    main()
