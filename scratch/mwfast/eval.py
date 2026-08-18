"""Score a Meta-World corrector: fidelity, throughput WITH inference, and the hallucination controls.

Every row is measured against ``mujoco.Renderer`` at 256 px -- the official frame -- in blocks of 64,
which is the draw size every declaration in this repo is measured at and the only size at which the
repo's SSIM is the same quantity (its uniform filter smooths along the frame axis).

Five rows exist for each evaluation set and they answer different questions:

* ``shipped_256`` -- the PARITY frame the family ships today. The thing a fast path is offered
  alongside, not the thing it replaces.
* ``bilinear`` -- the cheap render resized. This is what the corrector IS at initialisation (its head
  is zero-initialised), so a corrector that does not beat this has learned nothing, and a corrector
  that loses to it on any metric is actively harmful on that metric.
* ``corrector`` and ``corrector_shrink`` -- with and without the residual soft-shrink that protects
  bit-exact interiors. On a parity family this is not a nicety: the cheap frame's interior is already
  0.996-0.9999 exact and a dense residual would destroy it.
* ``jpeg_q95`` -- the storage baseline, i.e. the error a dataset consumer has already accepted, and
  the rate this whole exercise has to beat.

CONTROLS, all three of them, because the RoboCasa study's failure mode was a corrector that lowered
the mean while making more pixels wrong:

* **dynamic / static** -- the mean split by how much the OFFICIAL pixel moves across the evaluation
  states. A lookup table scores full marks on the static three quarters and nothing on the rest.
* **blank frame** -- what the network draws from its weights alone, on a constant input.
* **foreign task** -- the checkpoint applied zero-shot to a task it never saw, against bilinear and
  against the shipped render. A corrector worse than bilinear there is worse than doing nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch

import common as C
import data as D
import nets
from train import _forward_u8, predict

D.ROOT = C.DATA
C.extend_zoo()
CKPT = os.path.join(C.LOGS, "ckpt")
BARS = json.load(open(os.path.join(C.LOGS, "bars.json")))


def load_model(path: str, dtype=torch.float16):
    """fp16, ``channels_last``, eval -- the tensor a consumer receives, not a training artefact.

    The dtype is not cosmetic: at batch 1024 this architecture costs 21.6 us/frame in fp32 and
    16.4 us/frame in fp16 (``probe_fps.py``), which is the difference between clearing a bar and
    missing it, so the FIDELITY has to be measured in the same precision the throughput is.
    """
    ck = torch.load(path, map_location="cuda", weights_only=False)
    m = nets.build(ck["net"]).cuda()
    m.load_state_dict(ck["state_dict"])
    return m.to(dtype).eval().to(memory_format=torch.channels_last), ck


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


def shipped_frames(env_name: str, states, size: int = 256, chunk: int = 256,
                   style=None) -> np.ndarray:
    """The PARITY frame the family ships: the composite at full resolution.

    ``style`` is threaded through and it is not optional bookkeeping. Scoring the shipped render at
    the DEFAULT appearance against official frames drawn under the env's declared variation compares
    two different scenes and reports a number about the harness -- the second bug
    ``rendernet/README.md`` section 8 records.
    """
    import stable_envs as se
    C.clear_caches()
    env = se.make(env_name, batch=1, size=size, device="cuda", seed=0)
    out = np.empty((len(states), size, size, 3), np.uint8)
    for i in range(0, len(states), chunk):
        j = min(i + chunk, len(states))
        out[i:j] = env.render(states[i:j], size=size,
                              style=None if style is None else style[i:j]).cpu().numpy()
    del env
    C.clear_caches()
    return out


def worst_draw(ours: np.ndarray, official: np.ndarray, block: int = 64) -> dict:
    """The declaration's own reduction: metrics per 64-state draw, WORST of the draws.

    ``decl`` is four independent ``sample_states`` draws concatenated, and a ``visual_fidelity``
    entry is the worst of them -- not their mean -- so a re-declaration has to be read off the same
    reduction the shipped one was.
    """
    per = [C.metrics(ours[i:i + block], official[i:i + block], block)
           for i in range(0, len(ours), block)]
    return {"worst_mean_abs": max(p["mean_abs"] for p in per),
            "worst_within8_frac": min(p["within8_frac"] for p in per),
            "worst_ssim": min(p["ssim"] for p in per),
            "worst_interior_exact_frac": min(p["interior_exact_frac"] for p in per),
            "worst_exact_frac": min(p["exact_frac"] for p in per),
            "n_draws": len(per)}


def dyn_static(pred: np.ndarray, hr: np.ndarray, frac: float = 0.25) -> dict:
    sd = hr.astype(np.float32).std(0).max(-1)
    mask = sd >= np.quantile(sd, 1.0 - frac)
    d = np.abs(pred.astype(np.int16) - hr.astype(np.int16)).max(-1)
    return {"mean_dynamic": float(d[:, mask].mean()), "mean_static": float(d[:, ~mask].mean()),
            "dynamic_frac": float(mask.mean())}


def sbs(path: str, panels: list[tuple[str, np.ndarray]], official: np.ndarray, idx: int = 0,
        amp: int = 6):
    """One row of frames, one row of amplified |difference| maps against the official frame."""
    from PIL import Image, ImageDraw
    h = official.shape[1]
    cols = len(panels)
    im = Image.new("RGB", (cols * h, 2 * h + 22), (16, 16, 16))
    dr = ImageDraw.Draw(im)
    for k, (name, arr) in enumerate(panels):
        f = arr[idx]
        im.paste(Image.fromarray(f), (k * h, 18))
        d = np.abs(f.astype(np.int16) - official[idx].astype(np.int16)).max(-1)
        dm = np.clip(d * amp, 0, 255).astype(np.uint8)
        im.paste(Image.fromarray(np.stack([dm] * 3, -1)), (k * h, h + 20))
        mean = float(np.abs(arr[idx].astype(np.int16)
                            - official[idx].astype(np.int16)).max(-1).mean())
        dr.text((k * h + 4, 4), f"{name}  mean {mean:.2f}", fill=(230, 230, 230))
    im.save(path)
    print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--envs", default="", help="evaluate on these instead of the ckpt's own")
    ap.add_argument("--splits", default="heldout,decl")
    ap.add_argument("--eval-tag", default="", help="score against a DIFFERENT split tag "
                    "than the one trained on -- the appearance-variation 2x2")
    ap.add_argument("--shrink", type=float, default=0.5, help="the shrink used for the TIMED path")
    ap.add_argument("--shrinks", default="0.25,0.5,1.0", help="shrink values scored on fidelity")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--no-fps", action="store_true")
    ap.add_argument("--sbs", action="store_true")
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True

    model, ck = load_model(os.path.join(CKPT, a.ckpt + ".pt"))
    tag, net = a.eval_tag or ck["tag"], ck["net"]
    r = nets.render_size(net)
    trained_on = ck["envs"]
    envs = [e for e in a.envs.split(",") if e] or trained_on
    res = {"ckpt": a.ckpt, "net": net, "tag": tag, "trained_tag": ck["tag"], "render_size": r,
           "params": nets.n_params(model), "gflops": nets.gflops_per_frame(net),
           "trained_on": trained_on, "gpu": torch.cuda.get_device_name(0),
           "node": os.environ.get("SLURMD_NODENAME", "?"), "rows": []}

    for env_name in envs:
        seen = env_name in trained_on
        for split in [s for s in a.splits.split(",") if s]:
            try:
                lr, hr, states, sty, meta, _, _ = D.load_split(env_name, split, tag,
                                                              a.limit or None)
            except FileNotFoundError as e:
                print(f"skip {env_name} {split}: {e}", flush=True)
                continue
            lr_g = torch.as_tensor(lr, device="cuda")
            hr_g = torch.as_tensor(hr, device="cuda")
            base = C.upsample(lr_g, 256).cpu().numpy()
            with torch.no_grad():
                corr = predict(model, lr_g, shrink=0.0).cpu().numpy()
                shr = {s: predict(model, lr_g, shrink=s).cpu().numpy()
                       for s in [float(x) for x in a.shrinks.split(",") if x]}
            ship = shipped_frames(env_name, states, style=sty)
            jpg, nbytes = C.jpeg_roundtrip(hr, 95)
            panels = ([("shipped_256", ship), ("bilinear", base), ("corrector", corr)]
                      + [(f"corrector_shrink{s}", v) for s, v in shr.items()]
                      + [("jpeg_q95", jpg)])
            for name, arr in panels:
                row = {"env": env_name, "split": split, "config": name, "seen_task": seen,
                       "n": len(arr)}
                row.update(C.metrics(arr, hr))
                row.update(dyn_static(arr, hr))
                if split == "decl":
                    row.update(worst_draw(arr, hr))
                if name == "jpeg_q95":
                    row["bytes_per_frame"] = nbytes
                res["rows"].append(row)
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                                  for k, v in row.items()}), flush=True)
            if a.sbs:
                p = os.path.join(C.LOGS, f"sbs_{a.ckpt}__{env_name}_{split}.png")
                sbs(p, [("official", hr)] + panels, hr, idx=0)
            del lr_g, hr_g
            torch.cuda.empty_cache()

    # ---- the blank-frame control -------------------------------------------------------------
    for level in (0, 128, 255):
        blank = torch.full((8, r, r, 3), level, dtype=torch.uint8, device="cuda")
        with torch.no_grad():
            outb = predict(model, blank, shrink=0.0)
        d = (outb.to(torch.int16) - C.upsample(blank, 256).to(torch.int16)).abs().amax(-1).float()
        res["rows"].append({"config": f"blank_{level}", "kind": "control",
                            "residual_mean": float(d.mean()), "residual_max": float(d.max()),
                            "frac_moved_gt8": float((d > 8).float().mean())})
        print(json.dumps(res["rows"][-1]), flush=True)
        if a.sbs and level == 128:
            from PIL import Image
            Image.fromarray(outb[0].cpu().numpy()).save(
                os.path.join(C.LOGS, f"blank_{a.ckpt}.png"))

    # ---- throughput, corrector inside ---------------------------------------------------------
    if not a.no_fps:
        import stable_envs as se
        for env_name in envs:
            C.clear_caches()
            env = se.make(env_name, batch=1, size=256, device="cuda", seed=0)
            st = torch.as_tensor(env.sample_states(64), device="cuda", dtype=torch.float32)
            st = st[torch.as_tensor(np.random.default_rng(0).integers(0, 64, a.batch),
                                    device="cuda")]
            buf = torch.empty((a.batch, 256, 256, 3), dtype=torch.uint8, device="cuda")
            t = timed(lambda: env.render(st, size=256, out=buf), a.reps)
            ship_fps = a.batch / float(np.median(t))
            del buf
            torch.cuda.empty_cache()
            low = torch.empty((a.batch, r, r, 3), dtype=torch.uint8, device="cuda")
            t = timed(lambda: env.render(st, size=r, out=low), a.reps)
            rend_fps = a.batch / float(np.median(t))
            up = C.upsample
            t = timed(lambda: up(env.render(st, size=r, out=low), 256), a.reps)
            bil_fps = a.batch / float(np.median(t))
            g = graph_of(model, low, a.shrink)

            def step():
                env.render(st, size=r, out=low)
                g.replay()

            t = timed(step, a.reps)
            fps = a.batch / float(np.median(t))
            bar = BARS.get(env_name, {}).get("stored")
            res["rows"].append({
                "env": env_name, "kind": "throughput", "shipped_256_fps": ship_fps,
                "render_only_fps": rend_fps, "bilinear_fps": bil_fps, "corrector_fps": fps,
                "corrector_us_per_frame": 1e6 * (1 / fps - 1 / rend_fps),
                "vs_shipped": fps / ship_fps, "stored_bar": bar,
                "clears_bar": None if bar is None else bool(fps >= bar),
                "iqr_pct": 100 * float(np.percentile(t, 75) - np.percentile(t, 25))
                / float(np.median(t)), "batch": a.batch, "reps": a.reps})
            print(json.dumps(res["rows"][-1]), flush=True)
            del env, low, g
            C.clear_caches()

    p = a.out or os.path.join(C.LOGS, f"eval_{a.ckpt}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
