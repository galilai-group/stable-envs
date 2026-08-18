"""Does any corrector FIT? Inference cost alone, against the microseconds the frontier leaves over.

This is the gate. A learned corrector is only interesting if the render it corrects is fast enough
that the difference pays for the network, and that difference is small: rendering a RoboCasa room
camera at 112 px with texture LOD 2 gives 18,871 f/s against a stored-frame target of 14,343, so the
whole network -- unpack, convolve, upsample, repack -- has **16.7 us per frame**. Nothing about
fidelity matters until a candidate fits in that.

Timed the way it would ship: fp16, ``channels_last``, batch 1024, replayed from a CUDA graph, with
the uint8 unpack and the uint8 repack inside the timed region because a consumer asked for uint8.
The non-learned baselines (bilinear, Lanczos, FXAA) are timed through the same path so the network's
cost is reported against something and not against zero.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import nets

#: (env, base render config, base f/s measured by probe_budget_frontier, stored-frame target f/s)
#: The base f/s are from ``logs/.../budget_<env>.json`` on the same H200 this runs on; the targets
#: are 0.9 x ``examples/results/storage/lance.json`` ``local_n``.
BASE = {
    "RoboCasaCounterToSink": {"r112_tex2": 18871, "r128_tex2": 15430, "r144_tex2": 13068,
                              "shipped": 5681, "target": 14343},
    "RoboCasaCounterToSinkWrist": {"r112_tex2": 40852, "r128_tex2": 34507, "shipped": 13273,
                                   "target": 16869},
    "HumanoidBenchH1HandBalanceSimple": {"r112": 30108, "r128": 23364, "r160": 19507,
                                         "shipped": 11290, "target": 19063},
    "HumanoidBenchH1Walk": {"r112": 43973, "r128": 35937, "r160": 31319, "shipped": 19294,
                            "target": 18950},
}


def lanczos_kernel(scale: float, a: int = 3, device="cuda", dtype=torch.float32):
    """Separable Lanczos-``a`` resampling weights for an integer or fractional upscale.

    Written out rather than taken from a library because the only Lanczos in the stack is
    ``PIL``/``torchvision``'s CPU path, and a CPU filter cannot be compared against a GPU network on
    throughput. Same sample positions as ``F.interpolate(align_corners=False)``: output pixel ``i``
    maps to input coordinate ``(i + 0.5) / scale - 0.5``.
    """
    def L(x):
        x = torch.clamp(torch.abs(x), max=float(a))
        px = torch.pi * x
        out = torch.where(x < 1e-8, torch.ones_like(x),
                          a * torch.sin(px) * torch.sin(px / a) / (px * px + 1e-20))
        return torch.where(x >= a, torch.zeros_like(x), out)
    return L


def make_lanczos_weights(n_in: int, n_out: int, a: int = 3, device="cuda"):
    """``(n_out, n_in)`` dense resampling matrix. Dense is fine: 256 x 128 floats is nothing, and one
    matmul per axis is the fastest way to run a separable filter of this size on a GPU."""
    L = lanczos_kernel(n_out / n_in, a)
    scale = n_out / n_in
    oc = (torch.arange(n_out, device=device, dtype=torch.float64) + 0.5) / scale - 0.5
    ic = torch.arange(n_in, device=device, dtype=torch.float64)
    w = L(oc[:, None] - ic[None, :])
    w = w / w.sum(1, keepdim=True)
    return w.to(torch.float16)


class Lanczos(torch.nn.Module):
    """uint8 NHWC -> uint8 NHWC via two half-precision matmuls. The non-learned upsample baseline."""

    def __init__(self, n_in: int, n_out: int, a: int = 3):
        super().__init__()
        self.register_buffer("w", make_lanczos_weights(n_in, n_out, a))
        self.n_out = n_out

    def forward(self, u8):
        x = u8.to(torch.float16)                                 # (B,H,W,3)
        x = torch.einsum("oi,bihc->bohc", self.w, x)             # rows
        x = torch.einsum("oi,bhic->bhoc", self.w, x)             # cols
        return x.round_().clamp_(0, 255).to(torch.uint8)


class BilinearOnly(torch.nn.Module):
    def __init__(self, n_out: int):
        super().__init__()
        self.n_out = n_out

    def forward(self, u8):
        x = u8.permute(0, 3, 1, 2).to(torch.float16)
        x = F.interpolate(x, size=(self.n_out, self.n_out), mode="bilinear", align_corners=False)
        return x.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


class FXAA(torch.nn.Module):
    """Bilinear upsample followed by an FXAA-shaped edge-directed blend, at full resolution.

    Not a super-resolution method and not claimed as one: it is the cheapest thing that answers the
    user's specific complaint about aliasing, and it is here so that "a network was needed" is a
    measured statement rather than an assumption. Luma is computed, a local contrast test gates the
    filter, and the blend runs along the dominant gradient direction -- the same three steps as
    Lottes' original, expressed as tensor ops.
    """

    def __init__(self, n_out: int, edge_min: float = 0.03, edge_mul: float = 0.15):
        super().__init__()
        self.n_out, self.edge_min, self.edge_mul = n_out, edge_min, edge_mul

    def forward(self, u8):
        x = u8.permute(0, 3, 1, 2).to(torch.float16).mul_(1.0 / 255.0)
        x = F.interpolate(x, size=(self.n_out, self.n_out), mode="bilinear", align_corners=False)
        luma = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        p = F.pad(luma, (1, 1, 1, 1), mode="replicate")
        n, s = p[:, :, :-2, 1:-1], p[:, :, 2:, 1:-1]
        w, e = p[:, :, 1:-1, :-2], p[:, :, 1:-1, 2:]
        lmax = torch.maximum(torch.maximum(luma, torch.maximum(n, s)), torch.maximum(w, e))
        lmin = torch.minimum(torch.minimum(luma, torch.minimum(n, s)), torch.minimum(w, e))
        rng = lmax - lmin
        gate = (rng >= torch.clamp(lmax * self.edge_mul, min=self.edge_min)).to(x.dtype)
        horiz = (torch.abs(w + e - 2 * luma) >= torch.abs(n + s - 2 * luma)).to(x.dtype)
        xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
        blur_h = 0.5 * (xp[:, :, 1:-1, :-2] + xp[:, :, 1:-1, 2:])
        blur_v = 0.5 * (xp[:, :, :-2, 1:-1] + xp[:, :, 2:, 1:-1])
        blur = horiz * blur_h + (1 - horiz) * blur_v
        y = x + gate * 0.5 * (blur - x)
        return y.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


def time_module(mod, in_size: int, batch: int, in_ch: int = 3, reps: int = 30, graph: bool = True):
    """Median wall-clock seconds per batch, from a captured CUDA graph when one can be captured."""
    x = torch.randint(0, 255, (batch, in_size, in_size, in_ch), dtype=torch.uint8, device="cuda")
    with torch.no_grad():
        for _ in range(5):
            y = mod(x)
        torch.cuda.synchronize()
        g = None
        if graph:
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        mod(x)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    y = mod(x)
            except Exception:                                     # noqa: BLE001
                g = None
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if g is not None:
                g.replay()
            else:
                mod(x)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
    out_shape = tuple(y.shape)
    del x, y, g
    torch.cuda.empty_cache()
    return float(np.median(ts)), out_shape


def end_to_end(base_fps: float, net_seconds_per_batch: float, batch: int) -> float:
    """f/s of render-then-correct, from the render's f/s and the network's seconds per batch."""
    return 1.0 / (1.0 / base_fps + net_seconds_per_batch / batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    ap.add_argument("--in-ch", type=int, default=3)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/"
                                     "rendernet/speed.json")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]

    rows = []
    baselines = []
    for r in (96, 112, 128, 144, 160):
        baselines += [(f"bilinear_{r}", BilinearOnly(256), r, 3),
                      (f"lanczos3_{r}", Lanczos(r, 256), r, 3),
                      (f"fxaa_{r}", FXAA(256), r, 3)]
    for name, mod, isz, ich in baselines:
        mod = mod.cuda().eval()
        dt_s, shape = time_module(mod, isz, a.batch, ich)
        rows.append({"name": name, "kind": "baseline", "in_size": isz, "render_size": isz,
                     "params": 0, "gflops": 0.0,
                     "seconds_per_batch": dt_s, "us_per_frame": 1e6 * dt_s / a.batch,
                     "out_shape": list(shape)})
        print(json.dumps(rows[-1]), flush=True)
        del mod
        torch.cuda.empty_cache()

    for name in nets.ZOO:
        try:
            m = nets.build(name, in_ch=a.in_ch).cuda().to(dt).eval()
            m = m.to(memory_format=torch.channels_last)
            rsz = nets.render_size(name)
            dt_s, shape = time_module(m, rsz, a.batch, a.in_ch)
            rows.append({"name": name, "kind": "net", "in_size": nets.ZOO[name][1],
                         "render_size": rsz, "params": nets.n_params(m),
                         "gflops": nets.gflops_per_frame(name, in_ch=a.in_ch),
                         "seconds_per_batch": dt_s, "us_per_frame": 1e6 * dt_s / a.batch,
                         "out_shape": list(shape)})
            del m
        except Exception as e:                                    # noqa: BLE001
            rows.append({"name": name, "error": f"{type(e).__name__}: {e}"})
        torch.cuda.empty_cache()
        print(json.dumps(rows[-1]), flush=True)

    # ---- what each candidate would deliver end to end, per cell ------------------------------
    # A corrector is priced against the render config that produces ITS OWN input resolution, which
    # is ``render_size``: a "112t128" model is fed a 112 px render and resizes internally, so pricing
    # it against a 128 px render would charge the wrong renderer.
    table = {}
    for env, b in BASE.items():
        target = b["target"]
        table[env] = []
        for r in rows:
            if "error" in r:
                continue
            want = f"r{r['render_size']}"
            key = next((k for k in b if k == want or k.startswith(want + "_")), None)
            if key is None:
                continue
            fps = end_to_end(b[key], r["seconds_per_batch"], a.batch)
            table[env].append({"corrector": r["name"], "render": key, "render_fps": b[key],
                               "fps": fps, "target": target, "clears": fps >= target,
                               "vs_shipped": fps / b["shipped"],
                               "us_per_frame": r["us_per_frame"]})
        table[env].sort(key=lambda z: -z["fps"])
    out = {"gpu": torch.cuda.get_device_name(0), "batch": a.batch, "dtype": a.dtype,
           "in_ch": a.in_ch, "rows": rows, "end_to_end": table}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for env, rs in table.items():
        print(f"\n=== {env}  target {BASE[env]['target']} f/s (shipped {BASE[env]['shipped']}) ===",
              flush=True)
        for z in rs[:14]:
            print(f"  {z['corrector']:24s} on {z['render']:10s} -> {z['fps']:8.0f} f/s "
                  f"({z['vs_shipped']:.2f}x shipped) {'CLEARS' if z['clears'] else ''}", flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
