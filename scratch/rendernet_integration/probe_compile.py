"""How fast is the corrector once Inductor and a CUDA graph are allowed at it, and is it still exact?

Three questions in one process, because the answer to the third decides whether the first two matter:

1. inference microseconds per frame, for each compile mode and each fixed network batch;
2. end-to-end f/s of ``render(112 px, tex LOD 2) + corrector``, against this cell's own disk read;
3. whether the compiled, graph-replayed path is byte-identical to itself across calls and across
   batch sizes, and how far it moves from the eager path.

Nothing here writes a checkpoint. It picks the configuration that ``train_corrector.py`` then measures
fidelity through.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

PROTO = ("/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/ckpt/"
         "RoboCasaCounterToSink_unet_112t128_w48_d2_u2_sty_e300.pt")
#: 8-worker Lance read for this cell, from ``examples/results/storage/lance.json`` — the full bar.
DISK = {"RoboCasaCounterToSink": 15937, "RoboCasaCounterToSinkRight": 15863,
        "RoboCasaCounterToSinkL45S33": 15723, "RoboCasaCounterToSinkWrist": 18743,
        "HumanoidBenchH1HandBalanceSimple": 21181, "HumanoidBenchH1Walk": 21055}


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def timed(fn, reps: int = 25):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    a = np.asarray(ts)
    return float(np.median(a)), 100 * float(np.percentile(a, 75) - np.percentile(a, 25)) / float(np.median(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--proto", default=PROTO)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--modes", default="eager,default,max-autotune-no-cudagraphs")
    ap.add_argument("--net-batches", default="64,128,256")
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/"
                                     "rendernet/integration_compile.json")
    a = ap.parse_args()

    import stable_envs as se
    from stable_envs.render import rendernet as RN

    ck = torch.load(a.proto, map_location="cpu", weights_only=False)
    net, tex_lod, r = ck["net"], 2 if "tex2" in ck["tag"] else 1, RN.render_size(ck["net"])
    print(f"{a.env}: {net}, render {r} px tex_lod {tex_lod}, "
          f"{RN.n_params(RN.build(net))/1e3:.0f}k params", flush=True)
    cell = se.REGISTRY[a.env].corrector_cell_key()
    render = {"render_size": r, "out_size": 256, "tex_lod": tex_lod, "backface_cull": False}

    rows = []
    # ---- the shipped 256 px render, in this process, as the reference point ------------------
    env = se.make(a.env, batch=1, size=256, device="cuda", seed=0)
    st = torch.as_tensor(env.sample_states(64), device="cuda", dtype=torch.float32)
    st = st[torch.as_tensor(np.random.default_rng(0).integers(0, 64, a.batch), device="cuda")]
    buf = torch.empty((a.batch, 256, 256, 3), dtype=torch.uint8, device="cuda")
    dt, iqr = timed(lambda: env.render(st, size=256, out=buf), a.reps)
    rows.append({"config": "shipped_256", "fps": a.batch / dt, "iqr_pct": iqr})
    print(json.dumps(rows[-1]), flush=True)
    del buf

    # ---- the cheap render alone, through the corrected class's own profile -------------------
    Cheap = type("CheapProbe", (se.REGISTRY[a.env],),
                 dict(TEX_LOD=tex_lod, BACKFACE_CULL=False))          # not registered; profile only
    cheap = Cheap(batch=1, size=256, device="cuda", seed=0)
    low = torch.empty((a.batch, r, r, 3), dtype=torch.uint8, device="cuda")
    dt_r, iqr = timed(lambda: cheap.render(st, size=r, out=low), a.reps)
    rows.append({"config": f"render_only_r{r}_tex{tex_lod}_nocull", "fps": a.batch / dt_r,
                 "iqr_pct": iqr})
    print(json.dumps(rows[-1]), flush=True)
    d_render = digest(low)

    eager_out = None
    for nb in [int(x) for x in a.net_batches.split(",") if x]:
        for mode in [m for m in a.modes.split(",") if m]:
            try:
                m = RN.build(net)
                m.load_state_dict(ck["state_dict"])
                corr = RN.unsealed(m, cell=cell, render=render, net=net, mode=mode, net_batch=nb)
                out = torch.empty((a.batch, 256, 256, 3), dtype=torch.uint8, device="cuda")
                corr(low, out=out)                                     # compile + capture
                dt_c, iqr_c = timed(lambda: corr(low, out=out), a.reps)
                dt_e, iqr_e = timed(lambda: (cheap.render(st, size=r, out=low), corr(low, out=out)),
                                    a.reps)
                d1 = digest(out)
                corr(low, out=out)
                d2 = digest(out)
                # batch-size independence of the CORRECTOR alone: frame 0 of a full batch against
                # frame 0 rendered in batches of 1, 3, 7, 8, 64 -- the sizes the repo's
                # reproducibility suite uses, including the odd ones.
                bad = []
                for b in (1, 3, 7, 8, 64):
                    got = corr(low[:b])
                    if not torch.equal(got, out[:b]):
                        bad.append({"batch": b,
                                    "max": int((got.to(torch.int16) - out[:b].to(torch.int16))
                                               .abs().max()),
                                    "px": int(((got != out[:b]).any(-1)).sum())})
                row = {"config": f"corrector_{mode}_nb{nb}", "mode": mode, "net_batch": nb,
                       "corrector_us_per_frame": 1e6 * dt_c / a.batch,
                       "end_to_end_fps": a.batch / dt_e, "iqr_pct": iqr_e,
                       "iqr_pct_corrector": iqr_c,
                       "vs_shipped": (a.batch / dt_e) / rows[0]["fps"],
                       "disk_fps": DISK.get(a.env), "clears_disk": (a.batch / dt_e) >= DISK.get(a.env, 1e9),
                       "deterministic": d1 == d2, "digest": d1,
                       "batch_size_disagreements": bad}
                if mode == "eager" and eager_out is None:
                    eager_out = out.clone()
                elif eager_out is not None:
                    d = (out.to(torch.int16) - eager_out.to(torch.int16)).abs()
                    row["vs_eager_max"] = int(d.max())
                    row["vs_eager_frac_moved"] = float((d.amax(-1) > 0).float().mean())
                rows.append(row)
                print(json.dumps(row), flush=True)
                del corr, out
                torch.cuda.empty_cache()
            except Exception as e:                                     # noqa: BLE001
                rows.append({"config": f"corrector_{mode}_nb{nb}",
                             "error": f"{type(e).__name__}: {e}"})
                print(json.dumps(rows[-1]), flush=True)

    with open(a.out, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "env": a.env, "net": net,
                   "render_digest": d_render, "batch": a.batch, "rows": rows}, f, indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
