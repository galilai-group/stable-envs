"""M3/M4 on a GPU node: PFRocketLanding's visual fidelity, the corpus replay, and its throughput.

Writes examples/results/step7/rocket-landing/tier/<host>.json:
  * ``visual_fidelity``: ``check_visual_fidelity`` over ``--draws`` independent 64-state draws (the worst
    is what the class declares), plus ``check_visual_fidelity_under_variation``;
  * ``corpus_replay``: our render of the 200 frozen corpus states/styles against the frozen frames
    (mean_abs, within8, ssim, interior exact) -- the reference here is the RECORDED frame, no PyFlyt;
  * ``throughput``: f/s at batches 1/16/64/256 at 480 px, warm, best of two, CUDA-synchronised;
  * ``mandatory``: batch invariance, batch-size invariance, determinism, state dims (the suite's own).

The declared numbers in ``stable_envs/envs/rocket_landing.py`` are copied from here BY HAND -- a
threshold a script writes into the file it then tests is not a threshold.

Usage (GPU node, se7-rocket venv): python scratch/step7/rocket-landing/measure_tier.py --draws 4
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from importlib.metadata import version as pkg_version

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("MUJOCO_GL", "egl")
import stable_envs as se  # noqa: E402
from stable_envs.envs import rocket_landing as RL  # noqa: E402
from stable_envs.testing import conformance as C  # noqa: E402

REF = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "reference")
OUTDIR = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "tier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256])
    ap.add_argument("--skip-reference", action="store_true", help="corpus replay + throughput only (no PyFlyt)")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    R = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0],
         "versions": {k: _v(k) for k in ("torch", "nvdiffrast", "PyFlyt", "pybullet", "stable-worldmodel", "numpy", "scipy")}}
    if "PFRocketLanding" not in se.REGISTRY:
        raise SystemExit(f"PFRocketLanding not registered: {RL.missing_backend()} / {se.envs.import_errors()}")
    env = se.make("PFRocketLanding", batch=16, device="cuda")
    R["describe"] = {k: v for k, v in env.describe().items() if k not in ("visual_fidelity",)}

    # ---- corpus replay (no reference package needed)
    S = np.load(os.path.join(REF, "states.npz"))
    F = np.load(os.path.join(REF, "frames.npz"))["frames"]
    st = S["state13"]
    sty = RL.corpus_style_to_env(S["style"]).astype(np.float32)
    ours = []
    for lo in range(0, len(st), 16):
        ours.append(env.render(st[lo:lo + 16], style=sty[lo:lo + 16]).cpu().numpy())
    ours = np.concatenate(ours)
    rep = C.pixel_agreement(ours, F)
    rep["ssim"] = float(C.ssim(ours, F))
    d = np.abs(ours.astype(np.int16) - F.astype(np.int16))
    per = d.mean(axis=(1, 2, 3))
    rep["worst_frame_mean_abs"] = float(per.max())
    rep["worst_frame_index"] = int(per.argmax())
    rep["by_height"] = {}
    h = S["height"]
    for lo, hi in ((0, 12), (12, 50), (50, 260), (260, 500)):
        m = (h >= lo) & (h < hi)
        if m.any():
            rep["by_height"][f"{lo}-{hi}m"] = {"n": int(m.sum()), "mean_abs": float(d[m].mean()),
                                              "within8": float((d[m].max(-1) <= 8).mean())}
    R["corpus_replay"] = rep
    np.savez_compressed(os.path.join(OUTDIR, "corpus_replay_ours.npz"), frames=ours[:16], idx=np.arange(16))
    print("corpus replay:", json.dumps({k: v for k, v in rep.items() if k != "by_height"}), flush=True)

    # ---- mandatory checks
    R["mandatory"] = {}
    for name in ("check_batch_invariance", "check_batch_size_invariance", "check_determinism", "check_state_dims_matter"):
        fn = getattr(C, name, None)
        if fn is None:
            continue
        try:
            R["mandatory"][name] = fn(env)
        except Exception as e:                                                      # pragma: no cover
            R["mandatory"][name] = f"ERROR {type(e).__name__}: {e}"
        print(name, R["mandatory"][name], flush=True)

    # ---- throughput
    thr = {}
    for B in a.batches:
        e = se.make("PFRocketLanding", batch=B, device="cuda")
        s = torch.as_tensor(e.sample_states(B), device="cuda")
        for _ in range(3):
            e.render(s)
        torch.cuda.synchronize()
        best = 0.0
        for _ in range(2):
            t = time.perf_counter()
            reps = max(1, 512 // B)
            for _ in range(reps):
                e.render(s)
            torch.cuda.synchronize()
            best = max(best, reps * B / (time.perf_counter() - t))
        thr[str(B)] = round(best, 1)
        print("throughput", B, thr[str(B)], flush=True)
    R["throughput_fps"] = thr
    R["throughput_vs"] = {"official_fps_local_M5": 34.25, "one_core_jpeg_decode_fps_local_M5": 1190.9,
                          "note": "the two bars were measured on the Mac (findings.md G4); re-measure on this node "
                                  "for a same-machine ratio (scratch/step7/rocket-landing/phase0_probe.py G4 section)"}

    # ---- visual fidelity against the live reference (PyFlyt + TinyRenderer)
    if not a.skip_reference:
        vf = []
        for k in range(a.draws):
            env.rng = np.random.default_rng(1000 + k)
            r = C.check_visual_fidelity(env, n=a.n)
            r["draw"] = k
            vf.append(r)
            print("visual_fidelity draw", k, json.dumps(r), flush=True)
        worst = {"mean_abs": max(r["mean_abs"] for r in vf), "within8_frac": min(r["within8_frac"] for r in vf),
                 "ssim": min(r["ssim"] for r in vf)}
        R["visual_fidelity"] = {"draws": vf, "worst": worst,
                                "suggested_thresholds": {"mean_abs_max": round(worst["mean_abs"] * 1.35 + 0.5, 2),
                                                         "within8_min": round(worst["within8_frac"] - 0.04, 3),
                                                         "ssim_min": round(worst["ssim"] - 0.03, 3)}}
        try:
            R["variation_fidelity"] = C.check_visual_fidelity_under_variation(env, n=a.n, seed=7)
        except Exception as e:                                                      # pragma: no cover
            R["variation_fidelity"] = f"ERROR {type(e).__name__}: {e}"
        print("under variation:", R["variation_fidelity"], flush=True)
    R["seconds"] = round(time.time() - t0, 1)
    out = os.path.join(OUTDIR, f"{platform.node().split('.')[0]}.json")
    with open(out, "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", out)


def _v(k):
    try:
        return pkg_version(k)
    except Exception:
        return None


if __name__ == "__main__":
    main()
