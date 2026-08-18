"""Every fast-path candidate timed back to back in ONE process, against that task's stored-frame bar.

The corrector's cost is a property of the architecture and the batch, not of the task, so it is
measured once here; the RENDER's cost is a property of the task, so it is measured for every
resolution; and the composition of the two is then measured END TO END as well, because the two
numbers have to agree or one of them is wrong. Nothing is added up on paper.

25 repetitions, median, with the interquartile spread reported, because several candidates sit
within 2 % of a bar and per-node variation is a few percent.

``lightspec`` is ``STABLE_ENVS_LIGHTSPEC=1``: Meta-World compiles four any-hit BVH traversals for
three lights and a headlight and executes one, and deleting the three dead ones is worth 1.04-1.09x.
It is OFF by default on this family because it is not bit-identical on ``window-open`` and
``window-close`` (1/255 on 2 of 4,194,304 pixels, reproducibly). An opt-in fast path has already
given up bit-identity, so it takes the 1.05x.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import common as C

C.extend_zoo()
import nets                                                              # noqa: E402
from train import _forward_u8                                            # noqa: E402

BARS = json.load(open(os.path.join(C.LOGS, "bars.json")))


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


def graph_of(model, low, shrink):
    with torch.no_grad():
        for _ in range(5):
            _forward_u8(model, low, shrink)
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _forward_u8(model, low, shrink)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _forward_u8(model, low, shrink)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--nets", default="unet_112t128_w48_d2_u2,unet_112t128_w64_d3_u2,"
                                      "unet_96t128_w48_d2_u2,unet_96t128_w64_d3_u2,"
                                      "unet_112t128_w32_d2_u2,espcn_112t128_w64_d6_u2,"
                                      "espcn_112t128_w48_d4_u2")
    ap.add_argument("--res", default="128,112,96")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--noblend", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True

    import stable_envs as se
    M = C.mw()
    if a.noblend:
        M.MetaWorldBase._blend = lambda self, c, rgb, plate, n, alpha: rgb
    bar = BARS.get(a.env, {}).get("stored")
    rows = []
    out = {"env": a.env, "batch": a.batch, "reps": a.reps, "shrink": a.shrink,
           "noblend": a.noblend, "stored_bar": bar, "gpu": torch.cuda.get_device_name(0),
           "node": os.environ.get("SLURMD_NODENAME", "?"), "rows": rows}

    # ---- the shipped parity render, in this process ------------------------------------------
    C.clear_caches()
    env = se.make(a.env, batch=1, size=256, device="cuda", seed=0)
    st = torch.as_tensor(env.sample_states(64), device="cuda", dtype=torch.float32)
    st = st[torch.as_tensor(np.random.default_rng(0).integers(0, 64, a.batch), device="cuda")]
    buf = torch.empty((a.batch, 256, 256, 3), dtype=torch.uint8, device="cuda")
    t = timed(lambda: env.render(st, size=256, out=buf), a.reps)
    ship = a.batch / float(np.median(t))
    rows.append({"config": "shipped_256", "fps": ship, "vs_bar": ship / bar if bar else None})
    print(json.dumps(rows[-1]), flush=True)
    del buf, env
    C.clear_caches()

    # ---- render only, per resolution and per lightspec setting -------------------------------
    render_fps: dict = {}
    for r in [int(x) for x in a.res.split(",") if x]:
        for ls in (False, True):
            prev = os.environ.get("STABLE_ENVS_LIGHTSPEC")
            os.environ["STABLE_ENVS_LIGHTSPEC"] = "1" if ls else "0"
            C.clear_caches()
            env = se.make(a.env, batch=1, size=256, device="cuda", seed=0)
            low = torch.empty((a.batch, r, r, 3), dtype=torch.uint8, device="cuda")
            t = timed(lambda: env.render(st, size=r, out=low), a.reps)
            f = a.batch / float(np.median(t))
            render_fps[(r, ls)] = f
            up = C.upsample
            t2 = timed(lambda: up(env.render(st, size=r, out=low), 256), a.reps)
            rows.append({"config": f"render_r{r}" + ("_lightspec" if ls else ""),
                         "render_size": r, "lightspec": ls, "fps": f,
                         "fps_with_bilinear_to_256": a.batch / float(np.median(t2)),
                         "vs_bar": f / bar if bar else None,
                         "iqr_pct": 100 * float(np.percentile(t, 75) - np.percentile(t, 25))
                         / float(np.median(t))})
            print(json.dumps(rows[-1]), flush=True)
            del low, env
            if prev is None:
                os.environ.pop("STABLE_ENVS_LIGHTSPEC", None)
            else:
                os.environ["STABLE_ENVS_LIGHTSPEC"] = prev
            C.clear_caches()

    # ---- each corrector: standalone cost, then the composition, end to end -------------------
    for name in [x for x in a.nets.split(",") if x]:
        r = nets.render_size(name)
        if (r, False) not in render_fps:
            continue
        # fp16, as rendernet's own loader does: the corrector is a fp16 CUDA-graph replay,
        # and fp32 costs 5 us/frame here -- a quarter of the whole budget.
        model = nets.build(name).cuda().half().eval().to(memory_format=torch.channels_last)
        low = torch.empty((a.batch, r, r, 3), dtype=torch.uint8, device="cuda")
        g = graph_of(model, low, a.shrink)
        t = timed(g.replay, a.reps)
        us = 1e6 * float(np.median(t)) / a.batch
        for ls in (False, True):
            prev = os.environ.get("STABLE_ENVS_LIGHTSPEC")
            os.environ["STABLE_ENVS_LIGHTSPEC"] = "1" if ls else "0"
            C.clear_caches()
            env = se.make(a.env, batch=1, size=256, device="cuda", seed=0)
            g2 = graph_of(model, low, a.shrink)

            def step():
                env.render(st, size=r, out=low)
                g2.replay()

            t2 = timed(step, a.reps)
            fps = a.batch / float(np.median(t2))
            base = render_fps[(r, ls)]
            rows.append({"config": name + ("_lightspec" if ls else ""), "net": name,
                         "render_size": r, "lightspec": ls,
                         "params": nets.n_params(model),
                         "gflops": nets.gflops_per_frame(name),
                         "corrector_us_standalone": us,
                         "corrector_us_end_to_end": 1e6 * (1 / fps - 1 / base),
                         "render_only_fps": base, "fps": fps,
                         "vs_shipped": fps / ship, "stored_bar": bar,
                         "clears_bar": None if bar is None else bool(fps >= bar),
                         "iqr_pct": 100 * float(np.percentile(t2, 75) - np.percentile(t2, 25))
                         / float(np.median(t2))})
            print(json.dumps(rows[-1]), flush=True)
            del env, g2
            if prev is None:
                os.environ.pop("STABLE_ENVS_LIGHTSPEC", None)
            else:
                os.environ["STABLE_ENVS_LIGHTSPEC"] = prev
            C.clear_caches()
        del model, low, g
        torch.cuda.empty_cache()

    p = a.out or os.path.join(C.LOGS, f"fps_{a.env}" + ("_noblend" if a.noblend else "") + ".json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
