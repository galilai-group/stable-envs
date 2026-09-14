"""RAYCAST-6 — per-op attribution inside ``raycast3d.composite`` at ss=2.

RAYCAST-2b left ``PointmazeLargeNavigate`` at 42.6 ms per B=64 render of which ``composite``'s own
machinery is 25.3 ms; ``shade`` is 4.1 ms and the cast 13.4 ms. "``composite``'s own machinery" is
not an actionable statement, so this script opens it: it replaces ``R3.composite`` with a
STATEMENT-FOR-STATEMENT copy of the shipped body in which every block is timed, renders through the
public path (``env.render(states)``), and reports both the wall time of each block and the SHAPES it
ran over -- because the design question is which intermediates are B x N sized and why ss=2 doubles
them, and a millisecond with no shape beside it cannot answer that.

It also records, per call, the two compaction opportunities that the design rests on:

  * ``js`` -- the sample positions covered in ANY row (what ``COVER_COMPACTION`` already uses), and
  * ``ps`` -- the OUTPUT-pixel positions with non-zero coverage in ANY row, which is the set the
    patch merge could run over instead of the whole ``tile x tile`` window.

Nothing here imports from the renderer except to read it; no renderer code is modified by this file.
Writes under ``examples/results/step6/raycast3d/experiments/``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))


def _root_tag(p: str) -> str:
    return re.sub(r"^/oscar/scratch/([^/]+)", r"$SCR", p)


# --------------------------------------------------------------------------- the timed copy

class Rec:
    """One composite call's block timings (ms) and shapes.

    A STACK rather than one ``(name, t0)`` pair, because the first version of this had neither and
    a nested block silently wrote its parent's elapsed time under the CHILD's name -- so one column
    read 0.00 ms for real work and another double-counted. An instrument that mis-attributes is
    worse than one that is missing, because the number it prints looks like a result.
    """

    def __init__(self):
        self.t = {}
        self.shape = {}
        self._stack: list = []
        self._pending = None

    def __call__(self, name):
        self._pending = name
        return self

    def __enter__(self):
        self._stack.append((self._pending, time.perf_counter()))
        self._pending = None
        return self

    def __exit__(self, *a):
        name, t0 = self._stack.pop()
        dt = (time.perf_counter() - t0) * 1e3
        self.t[name] = self.t.get(name, 0.0) + dt
        assert not self._stack, ("blocks must not nest: the total is their sum, and a nested block "
                                 "would be counted inside its parent and again on its own")
        return False


def make_instrumented(R3, sink: list, variant: str = "window"):
    """Return a ``composite`` split into timed blocks: the window body, or the compact one.

    ``variant="window"`` mirrors ``_composite_window``, which is what the integration tree ships and
    what the design was measured against. ``variant="compact"`` mirrors the RAYCAST-6 body, so the
    same block names can be read down both columns and the saving attributed statement by statement
    rather than inferred from an end-to-end ratio. Neither is the shipped code -- both are copies --
    so the numbers here are attribution, and the ratio that goes in a report comes from the
    interleaved sweep, which times the real function.
    """
    if variant == "compact":
        return _make_compact(R3, sink)
    import torch

    @torch.no_grad()
    def composite(plate, plate_t, cam, prims, xpos, xmat, lights, H, W,
                  supersample=1, out=None, tile=None, mat_rgb=None, size_scale=None):
        r = Rec()
        with r("setup"):
            device = plate.device
            B = xpos.shape[0]
            ss = int(supersample)
            Rh, Rw = H * ss, W * ss
            o, d = R3.cached_rays(cam, Rh, Rw, device)
            P = prims.to(device)
            ldir, lamb, ldif, lspec = lights.to(device)
            if out is None:
                out = torch.empty((B, H, W, 3), dtype=torch.uint8, device=device)
                r.shape["out_allocated"] = 1
            else:
                out = out.view(B, H, W, 3)
                r.shape["out_allocated"] = 0

        with r("upsample_t"):
            base_t = plate_t if ss == 1 else R3._upsample_t(plate_t, H, W, ss)

        with r("tile_origin"):
            org = None
            if tile is not None and tile < H and tile < W:
                bs = 1.0 if size_scale is None else float(size_scale.max())
                org = R3._tile_origin(cam, prims, xpos, H, W, tile, device, bound_scale=bs)

        with r("window_gather"):
            if org is None:
                N = Rh * Rw
                oo = o.expand(B, N, 3)
                dd = d.expand(B, N, 3)
                bt = base_t.expand(B, N)
                idx = None
            else:
                idx = R3._expand_tile(org, tile, ss, Rw, device)
                N = idx.shape[1]
                oo = o.expand(B, N, 3)
                dd = d[idx.reshape(-1)].view(B, N, 3)
                bt = base_t[idx]

        with r("sizes"):
            nprim = len(P["kind"])
            sz = P["size"].view(1, nprim, 3)
            if size_scale is not None:
                sz = sz * size_scale.view(B, 1, 1)

        with r("cast"):
            t, n, i_hit = R3.cast_nearest(P["kind"], sz.expand(B, nprim, 3), xpos, xmat, oo, dd)

        with r("cov"):
            cov = t < bt

        if R3.COVER_COMPACTION:
            with r("compact_nonzero"):
                col = torch.zeros((B, N, 3), device=device, dtype=n.dtype)
                js = torch.nonzero(cov.any(0), as_tuple=False).reshape(-1)
            if js.numel():
                with r("material_gather"):
                    ih = i_hit[:, js]
                    mat = (P["mat"][ih] if mat_rgb is None
                           else mat_rgb[torch.arange(B, device=device).unsqueeze(1), ih])
                    tex, spec = P["tex"][ih], P["spec"][ih]
                with r("shade"):
                    c = R3.shade(n[:, js], -dd[:, js], ldir, lamb, ldif, lspec, lights.exponent,
                                 mat, tex, spec) * 255.0
                with r("col_scatter"):
                    col.index_copy_(1, js, torch.where(cov[:, js].unsqueeze(-1), c,
                                                       torch.zeros_like(c)))
        else:
            with r("material_gather"):
                mat = (P["mat"][i_hit] if mat_rgb is None
                       else mat_rgb[torch.arange(B, device=device).unsqueeze(1), i_hit])
            with r("shade"):
                c = R3.shade(n, -dd, ldir, lamb, ldif, lspec, lights.exponent,
                             mat, P["tex"][i_hit], P["spec"][i_hit]) * 255.0
            with r("col_scatter"):
                col = torch.where(cov.unsqueeze(-1), c, torch.zeros_like(c))
            js = torch.nonzero(cov.any(0), as_tuple=False).reshape(-1)

        with r("covf"):
            covf = cov.to(col.dtype)

        if org is None:
            with r("box_avg_full"):
                col = col.view(B, H, ss, W, ss, 3).sum((2, 4)) / (ss * ss)
                covf = covf.view(B, H, ss, W, ss).sum((2, 4)) / (ss * ss)
            with r("full_merge"):
                base = plate.to(col.dtype).unsqueeze(0)
                out.copy_(torch.clamp(torch.round(base * (1 - covf.unsqueeze(-1)) + col),
                                      0, 255).to(torch.uint8))
            r.shape.update(B=B, N=N, ss=ss, H=H, W=W, tile=0, nprim=nprim,
                           js=int(js.numel()), ps=-1,
                           cov_rows_mean=float(cov.sum(1).to(torch.float64).mean()))
            sink.append(r)
            return out

        with r("box_avg_col"):
            col = col.view(B, tile, ss, tile, ss, 3).sum((2, 4)) / (ss * ss)
        with r("box_avg_cov"):
            covf = covf.view(B, tile, ss, tile, ss).sum((2, 4)) / (ss * ss)
        with r("oidx"):
            oidx = R3._expand_tile(org, tile, 1, W, device).unsqueeze(-1).expand(B, tile * tile, 3)
        with r("plate_broadcast_write"):
            out.copy_(plate.view(1, H, W, 3).expand(B, H, W, 3))
        with r("plate_promote"):
            platef = plate.reshape(1, H * W, 3).to(col.dtype)
        with r("patch_gather"):
            patch = torch.gather(platef.expand(B, H * W, 3), 1, oidx)
        with r("merge_arith"):
            merged = patch * (1 - covf.reshape(B, -1, 1)) + col.reshape(B, -1, 3)
        with r("round_cast"):
            px = torch.clamp(torch.round(merged), 0, 255).to(torch.uint8)
        with r("scatter"):
            out.view(B, H * W, 3).scatter_(1, oidx, px)

        # the two compaction sets the design rests on, measured rather than assumed
        with r("_probe_ps"):
            covp = covf.reshape(B, -1) > 0
            ps = torch.nonzero(covp.any(0), as_tuple=False).reshape(-1)
        r.shape.update(B=B, N=N, ss=ss, H=H, W=W, tile=tile, nprim=nprim,
                       js=int(js.numel()), ps=int(ps.numel()),
                       win_px=tile * tile,
                       cov_rows_mean=float(cov.sum(1).to(torch.float64).mean()),
                       covp_rows_mean=float(covp.sum(1).to(torch.float64).mean()))
        sink.append(r)
        return out

    return composite


def _make_compact(R3, sink: list):
    """The RAYCAST-6 body, same block names, so the two columns subtract."""
    import torch

    @torch.no_grad()
    def composite(plate, plate_t, cam, prims, xpos, xmat, lights, H, W,
                  supersample=1, out=None, tile=None, mat_rgb=None, size_scale=None):
        r = Rec()
        with r("setup"):
            device = plate.device
            B = xpos.shape[0]
            ss = int(supersample)
            Rh, Rw = H * ss, W * ss
            o, d = R3.cached_rays(cam, Rh, Rw, device)
            P = prims.to(device)
            ldir, lamb, ldif, lspec = lights.to(device)
            out = torch.empty((B, H, W, 3), dtype=torch.uint8, device=device) if out is None \
                else out.view(B, H, W, 3)

        with r("tile_origin"):
            org = None
            ask = tile
            if tile is not None and tile < H and tile < W:
                bs = 1.0 if size_scale is None else float(size_scale.max())
                fit = R3._fit_tile(cam, prims, xpos, H, W, tile, device, bound_scale=bs)
                if fit is not None:
                    org, tile = fit

        with r("window_gather"):
            if org is None:
                N = Rh * Rw
                oo, dd = o.expand(B, N, 3), d.expand(B, N, 3)
                obase = None
            else:
                idx = R3._expand_tile(org, tile, ss, Rw, device)
                N = idx.shape[1]
                oo = o.expand(B, N, 3)
                dd = d[idx.reshape(-1)].view(B, N, 3)
                obase = R3._expand_tile(org, tile, 1, W, device)
        with r("upsample_t"):
            if org is None:
                bt = (plate_t if ss == 1 else R3._upsample_t(plate_t, H, W, ss)).expand(B, N)
            else:
                bt = plate_t[obase].view(B, tile, 1, tile, 1) \
                    .expand(B, tile, ss, tile, ss).reshape(B, N)

        with r("sizes"):
            nprim = len(P["kind"])
            sz = P["size"].view(1, nprim, 3)
            if size_scale is not None:
                sz = sz * size_scale.view(B, 1, 1)
        with r("cast"):
            t, n, i_hit = R3.cast_nearest(P["kind"], sz.expand(B, nprim, 3), xpos, xmat, oo, dd)
        with r("cov"):
            cov = t < bt

        with r("compact_nonzero"):
            col = torch.zeros((B, N, 3), device=device, dtype=n.dtype)
            flat = torch.nonzero(cov.reshape(-1), as_tuple=False).reshape(-1)
        if flat.numel():
            with r("material_gather"):
                rows = flat.div(N, rounding_mode="floor")
                cols = flat - rows * N
                ih = i_hit[rows, cols]
                mat = P["mat"][ih] if mat_rgb is None else mat_rgb[rows, ih]
                nsel, dsel = n[rows, cols], -dd[rows, cols]
            with r("shade"):
                c = R3.shade(nsel, dsel, ldir, lamb, ldif, lspec, lights.exponent,
                             mat, P["tex"][ih], P["spec"][ih]) * 255.0
            with r("col_scatter"):
                col.view(-1, 3).index_copy_(0, flat, c)

        with r("covf"):
            covf = cov.to(col.dtype)
        if org is None:
            with r("box_avg_full"):
                col = col.view(B, H, ss, W, ss, 3).sum((2, 4)) / (ss * ss)
                covf = covf.view(B, H, ss, W, ss).sum((2, 4)) / (ss * ss)
            with r("full_merge"):
                base = plate.to(col.dtype).unsqueeze(0)
                out.copy_(torch.clamp(torch.round(base * (1 - covf.unsqueeze(-1)) + col),
                                      0, 255).to(torch.uint8))
            r.shape.update(B=B, N=N, ss=ss, H=H, W=W, tile=0, asked_tile=ask, nprim=nprim,
                           js=int(flat.numel()) // max(B, 1), ps=-1)
            sink.append(r)
            return out

        with r("box_avg_col"):
            col = col.view(B, tile, ss, tile, ss, 3).sum((2, 4)) / (ss * ss)
        with r("box_avg_cov"):
            covf = covf.view(B, tile, ss, tile, ss).sum((2, 4)) / (ss * ss)
        with r("plate_broadcast_write"):
            out.copy_(plate.view(1, H, W, 3).expand(B, H, W, 3))
        with r("oidx"):
            covf = covf.reshape(B, -1)
            ps = torch.nonzero((covf > 0).any(0), as_tuple=False).reshape(-1)
        if ps.numel():
            with r("patch_gather"):
                oidx = obase[:, ps].unsqueeze(-1).expand(B, ps.numel(), 3)
                patch = torch.gather(plate.reshape(1, H * W, 3).expand(B, H * W, 3), 1,
                                     oidx).to(col.dtype)
            with r("merge_arith"):
                merged = patch * (1 - covf[:, ps].unsqueeze(-1)) + col.reshape(B, -1, 3)[:, ps]
            with r("round_cast"):
                px = torch.clamp(torch.round(merged), 0, 255).to(torch.uint8)
            with r("scatter"):
                out.view(B, H * W, 3).scatter_(1, oidx, px)
        r.shape.update(B=B, N=N, ss=ss, H=H, W=W, tile=tile, asked_tile=ask, nprim=nprim,
                       win_px=tile * tile, ps=int(ps.numel()),
                       covered_pairs=int(flat.numel()),
                       covered_per_row=float(flat.numel()) / max(B, 1))
        sink.append(r)
        return out

    return composite


# --------------------------------------------------------------------------- driver

def run(cfg: str, batches, repeats: int, device: str, threads: int,
        variant: str = "window"):
    import numpy as np
    import torch
    import stable_envs as se
    from stable_envs.render import raycast3d as R3

    torch.set_num_threads(threads)
    npz = ROOT / "examples" / "results" / "step4" / "states" / f"{cfg}.npz"
    pool = np.asarray(np.load(npz)["states"], np.float32) if npz.is_file() else None

    orig = R3.composite
    out = {"config": cfg, "device": device, "threads": threads, "variant": variant,
           "stable_envs_from": _root_tag(se.__file__),
           "COVER_COMPACTION": bool(R3.COVER_COMPACTION),
           "PRIMITIVE_BOUNDS": bool(getattr(R3, "PRIMITIVE_BOUNDS", None)),
           "batches": {}}
    try:
        for B in batches:
            env = se.make(cfg, batch=B, device=device)
            if pool is not None:
                st = pool[np.arange(B) % len(pool)]
            else:
                st = env.sample_states(B, np.random.default_rng(0))
            env.render(st)                                      # warm every cache
            sink: list = []
            R3.composite = make_instrumented(R3, sink, variant)
            walls = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                env.render(st)
                walls.append((time.perf_counter() - t0) * 1e3)
            R3.composite = orig
            # one composite call per render here (a single plate group); fold multiple calls anyway
            per_render = max(1, len(sink) // repeats)
            keys = sorted({k for r in sink for k in r.t})
            agg = {}
            for k in keys:
                vals = []
                for i in range(repeats):
                    vals.append(sum(r.t.get(k, 0.0) for r in sink[i * per_render:(i + 1) * per_render]))
                agg[k] = float(np.median(vals))
            shp = dict(sink[-1].shape)
            tot = float(np.median(walls))
            out["batches"][str(B)] = {
                "render_ms": tot, "fps": 1e3 * B / tot,
                "composite_calls_per_render": per_render,
                "blocks_ms": agg,
                "composite_total_ms": float(sum(agg.values()) - agg.get("_probe_ps", 0.0)),
                "shapes": shp,
            }
            print(f"[{cfg}] B={B:5d} render {tot:8.2f} ms  fps {1e3*B/tot:9.1f}  "
                  f"composite {sum(agg.values()) - agg.get('_probe_ps', 0.0):8.2f} ms", flush=True)
            for k in sorted(agg, key=lambda z: -agg[z]):
                print(f"        {k:24s} {agg[k]:8.3f} ms", flush=True)
            print(f"        shapes {shp}", flush=True)
            del env
    finally:
        R3.composite = orig
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="PointmazeLargeNavigate,PointmazeMediumNavigate")
    ap.add_argument("--batches", default="16,64,256")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--variant", default="window", choices=("window", "compact"),
                    help="which body to attribute: the integration tree's, or RAYCAST-6's")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    batches = [int(x) for x in a.batches.split(",") if x]
    res = {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "variant": a.variant, "configs": {}}
    for cfg in [c for c in a.configs.split(",") if c]:
        res["configs"][cfg] = run(cfg, batches, a.repeats, a.device, a.threads, a.variant)

    dst = Path(a.out) if a.out else (ROOT / "examples" / "results" / "step6" / "raycast3d"
                                     / "experiments" / f"r6_composite_profile_{a.device}.json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
