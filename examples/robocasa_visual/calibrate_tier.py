"""Measure what the registered RoboCasa classes actually do, on their OWN state distribution.

The side-by-side pipeline scores a recorded trajectory; the conformance suite scores
``env.sample_states``, which is that trajectory with every free joint's yaw randomised. Those are
different distributions and the tier's declared numbers must be the ones the suite will see, so they
are measured here and copied into ``stable_envs/envs/robocasa.py``'s ``FIDELITY`` by hand -- a
threshold a script writes into the file it then tests is not a threshold.

Also prints the per-class state width and its live-qpos share, the throughput at the env's own
surface, and the registry listing with its tier marks.

Run in a venv that has torch, gymnasium, mujoco >= 3.11 and mujoco_warp (``~/venvs/se-fetch``), on a
GPU node.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import torch

import stable_envs as se
from stable_envs.testing import conformance as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--bench-batches", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--seeds", type=int, default=4, help="independent state draws per env")
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/robocasa_visual_work/"
                                     "tier_calibration.json")
    args = ap.parse_args()

    print(se.format_registry(), flush=True)
    names = [n for n, c in sorted(se.registry.REGISTRY.items())
             if getattr(c, "fidelity_tier", "parity") == "visual"]
    if not names:
        raise SystemExit(f"no visual-tier envs registered; import errors: {se.import_errors()}")

    rec = {"envs": {}}
    for name in names:
        env = se.make(name, num_envs=args.batch, device="cuda")
        cls = type(env)
        d = env.describe()
        # Several INDEPENDENT state draws, not one. A threshold set from a single draw is a threshold
        # whose margin is a guess; the spread over draws is what says how much headroom the gate
        # actually needs, and on the wrist camera it is the biggest term (a randomised object yaw is
        # most of that frame).
        draws = []
        for s in range(args.seeds):
            e2 = se.make(name, num_envs=args.batch, device="cuda", seed=s)
            draws.append(C.check_visual_fidelity(e2, n=args.n))
        r = draws[0]
        spread = {k: [min(x[k] for x in draws), max(x[k] for x in draws)]
                  for k in ("mean_abs", "within8_frac", "ssim", "interior_mean_abs",
                            "interior_exact_frac")}
        e = {"state_dim": env.state_dim, "size": env.size, "kitchen": cls.KITCHEN,
             "camera": cls.CAMERA, "fidelity_tier": d["fidelity_tier"],
             "declared": dict(env.visual_fidelity), "measured": r, "spread_over_draws": spread,
             "n_draws": args.seeds, "state_names_head": list(env.state_names[:8])}
        # Throughput at the env's own surface: states in, uint8 frames out, everything included.
        for b in args.bench_batches:
            en = se.make(name, num_envs=b, device="cuda")
            st = torch.as_tensor(en.sample_states(b), device="cuda")
            en.render(st)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            reps = max(3, int(2000 / b))
            for _ in range(reps):
                en.render(st)
            torch.cuda.synchronize()
            e[f"fps_batch{b}"] = b * reps / (time.perf_counter() - t0)
        rec["envs"][name] = e
        print(f"\n== {name} ({cls.KITCHEN} / {cls.CAMERA}) state_dim {env.state_dim} "
              f"of nq {len(_nq(cls))}", flush=True)
        print("   measured " + json.dumps({k: round(v, 4) for k, v in r.items()
                                           if isinstance(v, float)}), flush=True)
        print("   spread over %d draws " % args.seeds
              + json.dumps({k: [round(a, 4), round(b, 4)] for k, (a, b) in spread.items()}),
              flush=True)
        print("   fps " + json.dumps({f"batch{b}": round(e[f'fps_batch{b}'])
                                      for b in args.bench_batches}), flush=True)
        for key, floor, got, worse in (
                ("mean_abs", env.visual_fidelity["mean_abs_max"], r["mean_abs"], True),
                ("within8_frac", env.visual_fidelity["within8_min"], r["within8_frac"], False),
                ("ssim", env.visual_fidelity["ssim_min"], r["ssim"], False)):
            ok = (got <= floor) if worse else (got >= floor)
            print(f"   {'PASS' if ok else 'FAIL'} {key}: {got:.4f} vs declared {floor}", flush=True)

    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print("\nwrote", args.out, flush=True)


def _nq(cls):
    from stable_envs.envs.robocasa import _scene
    return _scene(cls.KITCHEN).qpos_names


if __name__ == "__main__":
    main()
