"""Every candidate timed back to back in ONE process on ONE node, against the stored-frame target.

The per-checkpoint evaluations each measure their own throughput, and they disagree by up to 5 %
between nodes -- enough to move a candidate across a target it is within 2 % of, which several are.
A recommendation cannot rest on that. This runs the whole shortlist in a single process so the GPU,
the clocks, the driver and the warp kernel cache are identical for every row, takes 25 repetitions
instead of 5, and reports the interquartile spread so the reader can see how much of the gap between
two rows is real.

It also prices the two things a per-checkpoint run cannot: the shipped 256 px render as the same
process measures it, and the render alone at each resolution, so the corrector's share is a
subtraction rather than an estimate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

import aux as AX
import nets
from eval_corrector import aux_context, load_model
from probe_budget_frontier import ctx_cache, env_module, family_of, patched
from train import _forward_u8

CKPT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/ckpt"
#: 0.9 x ``examples/results/storage/lance.json`` ``local_n``: the stored-frame bar this must clear.
TARGET = {"RoboCasaCounterToSink": 14343, "RoboCasaCounterToSinkRight": 14277,
          "RoboCasaCounterToSinkL45S33": 14151, "RoboCasaCounterToSinkWrist": 16869,
          "HumanoidBenchH1Walk": 18950, "HumanoidBenchH1Run": 19977,
          "HumanoidBenchH1Stand": 19801, "HumanoidBenchH1Stair": 19300,
          "HumanoidBenchH1Slide": 19056, "HumanoidBenchH1SitSimple": 18778,
          "HumanoidBenchH1SitHard": 18716, "HumanoidBenchH1BalanceSimple": 19233,
          "HumanoidBenchH1HandWalk": 19746, "HumanoidBenchH1HandRun": 19710,
          "HumanoidBenchH1HandStand": 19559, "HumanoidBenchH1HandStair": 19030,
          "HumanoidBenchH1HandSlide": 18833, "HumanoidBenchH1HandSitSimple": 18552,
          "HumanoidBenchH1HandSitHard": 17844, "HumanoidBenchH1HandBalanceSimple": 19063,
          "HumanoidBenchH1HandSpoon": 16736}


def timed(fn, reps: int):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return np.asarray(ts)


def graph_of(model, low, shrink, dep=None, seg=None):
    with torch.no_grad():
        for _ in range(5):
            _forward_u8(model, low, shrink, dep, seg)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _forward_u8(model, low, shrink, dep, seg)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _forward_u8(model, low, shrink, dep, seg)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--ckpts", default="", help="comma-separated checkpoint basenames; default all")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--out", default="", help="where the JSON goes; default is the rendernet dir")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True

    import stable_envs as se
    mod = env_module(a.env)
    names = ([x for x in a.ckpts.split(",") if x] or
             sorted(os.path.basename(p)[:-3] for p in glob.glob(f"{CKPT}/{a.env}_*.pt")))
    rows = []

    # ---- the shipped 256 px render, in this process ------------------------------------------
    ctx_cache(mod).clear()
    env = se.make(a.env, batch=1, size=256, device="cuda", seed=0)
    st = torch.as_tensor(env.sample_states(64), device="cuda", dtype=torch.float32)
    st = st[torch.as_tensor(np.random.default_rng(0).integers(0, 64, a.batch), device="cuda")]
    buf = torch.empty((a.batch, 256, 256, 3), dtype=torch.uint8, device="cuda")
    ts = timed(lambda: env.render(st, size=256, out=buf), a.reps)
    rows.append({"config": "shipped_256", "render_size": 256, "tex_lod": 1,
                 "fps": a.batch / float(np.median(ts)),
                 "iqr_pct": 100 * float(np.percentile(ts, 75) - np.percentile(ts, 25))
                 / float(np.median(ts))})
    print(json.dumps(rows[-1]), flush=True)
    del env, buf
    ctx_cache(mod).clear()
    torch.cuda.empty_cache()

    # ---- each candidate, and the render it needs, under the same patch -----------------------
    render_only: dict = {}
    for name in names:
        try:
            model, ck = load_model(f"{CKPT}/{name}.pt")
            tag = ck["tag"]
            r = nets.render_size(ck["net"])
            lod = next((int(p[3:]) for p in tag.split("_") if p.startswith("tex")), 1)
            use_aux = model.aux_dim() > 0
            # An aux corrector needs a context that WRITES the buffers, which is a different context
            # and a different render rate, so it gets its own ``render_only`` key. Charging it the
            # RGB-only render rate would hide the megakernel's own 0.4-2.3 %.
            orig = (AX.patch_context(mod, lod) if use_aux
                    else patched(mod, dict(enable_backface_culling=False), lod))
            try:
                env = se.make(a.env, batch=1, size=256, device="cuda", seed=0)
                low = torch.empty((a.batch, r, r, 3), dtype=torch.uint8, device="cuda")
                dep = seg = None
                chunk = int(mod.MJW_CHUNK)
                if use_aux:
                    cls, fam = se.REGISTRY[a.env], family_of(a.env)
                    dep = torch.empty((a.batch, r, r), dtype=torch.uint8, device="cuda")
                    seg = torch.empty((a.batch, r, r), dtype=torch.int32, device="cuda")

                    def do_render():
                        for i in range(0, a.batch, chunk):
                            j = min(i + chunk, a.batch)
                            env.render(st[i:j], size=r, out=low[i:j])
                            n = j - i
                            nw = 1 if n <= 1 else min(chunk, 1 << (n - 1).bit_length())
                            d, gg = AX.read_aux(aux_context(mod, fam, cls, r, nw), n, r, 0)
                            dep[i:j] = d
                            seg[i:j] = gg
                else:
                    def do_render():
                        env.render(st, size=r, out=low)
                key = (r, lod, use_aux)
                if key not in render_only:
                    t = timed(do_render, a.reps)
                    render_only[key] = a.batch / float(np.median(t))
                    rows.append({"config": f"render_only_r{r}_tex{lod}"
                                           + ("_aux" if use_aux else ""), "render_size": r,
                                 "tex_lod": lod, "aux": use_aux, "fps": render_only[key],
                                 "iqr_pct": 100 * float(np.percentile(t, 75)
                                                        - np.percentile(t, 25)) / float(np.median(t))})
                    print(json.dumps(rows[-1]), flush=True)
                g = graph_of(model, low, a.shrink, dep, seg)

                def step():
                    do_render()
                    g.replay()

                t = timed(step, a.reps)
                fps = a.batch / float(np.median(t))
                base = render_only[key]
                rows.append({"config": name, "net": ck["net"], "aux": ck.get("aux", "none"),
                             "render_size": r, "tex_lod": lod,
                             "params": nets.n_params(model),
                             "gflops": nets.gflops_per_frame(ck["net"]),
                             "render_only_fps": base, "fps": fps,
                             "corrector_us_per_frame": 1e6 * (1 / fps - 1 / base),
                             "iqr_pct": 100 * float(np.percentile(t, 75) - np.percentile(t, 25))
                             / float(np.median(t)),
                             "target": TARGET.get(a.env), "clears": fps >= TARGET.get(a.env, 1e9),
                             "vs_shipped": fps / rows[0]["fps"]})
                print(json.dumps(rows[-1]), flush=True)
                del env, low, g, model, dep, seg
            finally:
                mod.MW.create_render_context = orig
                ctx_cache(mod).clear()
                torch.cuda.empty_cache()
        except Exception as e:                                        # noqa: BLE001
            rows.append({"config": name, "error": f"{type(e).__name__}: {e}"})
            print(json.dumps(rows[-1]), flush=True)

    p = a.out or (f"/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/"
                  f"finalfps_{a.env}.json")
    with open(p, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "batch": a.batch, "reps": a.reps,
                   "shrink": a.shrink, "target": TARGET.get(a.env), "rows": rows}, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
