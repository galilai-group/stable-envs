"""Measure the swm navigation ports against the official envs, and record the constants they claim.

Run on a GPU node. Prints one block per env: the conformance checks that do not need pytest, the
render parity against the official CPU renderer, the step agreement, and throughput. Also runs the
three standalone measurements the module docstring of ``stable_envs/envs/swm_nav.py`` cites, so those
numbers are reproducible rather than remembered.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing import conformance as C

ENVS = ["TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones"]


def measure_reference_facts() -> dict:
    """The three claims in the module docstring, measured on the OFFICIAL env."""
    import gymnasium
    import stable_worldmodel  # noqa: F401

    out = {}
    env = gymnasium.make("swm/TwoRoom-v1")
    u = env.unwrapped
    env.reset(seed=0, options={"variation": []})
    f0 = np.ascontiguousarray(u.render())

    # 1. the Gaussian tail. The agent is (255,0,0) on white, so the GREEN channel carries the ramp:
    # 255 far away, 0 at the peak. The red channel is 255 everywhere the agent touches and is useless
    # for this, which is what made the first version of this probe report the border as the agent.
    g = f0[..., 1].astype(np.int16)
    ay, ax = np.unravel_index(int(np.argmin(g)), g.shape)
    out["agent_peak_rc"] = [int(ay), int(ax)]
    out["agent_peak_rgb"] = [int(v) for v in f0[ay, ax]]
    out["halo_254_px"] = int((g == 254).sum())
    out["distinct_green_levels"] = int(np.unique(g).size)
    ys, xs = np.nonzero(g < 255)
    out["max_touched_radius"] = float(np.sqrt(((ys - ay) ** 2 + (xs - ax) ** 2)).max())
    # how many pixels a flat disc of the nominal radius would have covered, for the contrast
    out["nominal_disc_px"] = int((((np.arange(224)[:, None] - ay) ** 2
                                  + (np.arange(224)[None, :] - ax) ** 2) <= 7.0 ** 2).sum())
    out["touched_px"] = int((g < 255).sum())

    # 2. wall.border_color is declared and never read
    u.variation_space["wall"]["border_color"].set_value(np.array([255, 255, 255], np.uint8))
    fw = np.ascontiguousarray(u.render())
    u.variation_space["wall"]["border_color"].set_value(np.array([0, 0, 0], np.uint8))
    fb = np.ascontiguousarray(u.render())
    out["border_color_max_abs_diff"] = int(np.abs(fw.astype(np.int16) - fb.astype(np.int16)).max())

    # 3. the border stroke: which rows are wall-coloured on an empty column
    col = f0[:, 5]                                     # x = 5, outside the door span and the stripe
    rows = np.nonzero((col != np.array([255, 255, 255])).any(-1))[0]
    out["border_rows_at_x5"] = [int(v) for v in rows]

    # 4. the two-blend quantisation: does the reference quantise between target and agent?
    env2 = gymnasium.make("swm/TwoRoom-v1", render_target=True)
    v = env2.unwrapped
    env2.reset(seed=0, options={"variation": []})
    v.agent_position = torch.tensor([100.0, 112.0])
    v.target_position = torch.tensor([112.0, 112.0])   # deliberately overlapping tails
    ft = np.ascontiguousarray(v.render())
    out["overlap_distinct_colors"] = int(np.unique(ft.reshape(-1, 3), axis=0).shape[0])
    return out


def bench(env, batch: int, reps: int = 20) -> float:
    st = env.sample_states(batch, rng=np.random.default_rng(0))
    st = torch.as_tensor(st, device=env.device)
    buf = torch.empty((batch, env.size, env.size, 3), dtype=torch.uint8, device=env.device)
    for _ in range(3):
        env.render(st, out=buf)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        env.render(st, out=buf)
    torch.cuda.synchronize()
    return batch * reps / (time.perf_counter() - t0)


def reference_fps(cls, size: int, n: int = 30) -> float:
    ref = C.reference_env(cls, size)
    env = cls(batch=1, size=size, device="cuda")
    st = env.sample_states(n, rng=np.random.default_rng(1))
    t0 = time.perf_counter()
    cls.reference_render(ref, st)
    return n / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/stable_envs_logs/agentB/"
                                    "swm_nav_parity.json")
    ap.add_argument("--envs", default=",".join(ENVS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--bench-batch", type=int, default=512)
    ap.add_argument("--skip-facts", action="store_true")
    args = ap.parse_args()

    report = {}
    if not args.skip_facts:
        try:
            report["reference_facts"] = measure_reference_facts()
            print("reference facts:", json.dumps(report["reference_facts"], indent=1), flush=True)
        except Exception as e:                                              # noqa: BLE001
            report["reference_facts"] = f"ERR {type(e).__name__}: {e}"
            print("reference facts FAILED:", e, flush=True)

    for name in args.envs.split(","):
        if name not in se.REGISTRY:
            print(f"{name}: NOT REGISTERED; import errors {se.envs.import_errors()}", flush=True)
            report[name] = {"error": "not registered", "import_errors": se.envs.import_errors()}
            continue
        rec = {}
        print(f"\n===== {name} =====", flush=True)
        try:
            env = se.make(name, batch=args.batch, device="cuda")
            cls = type(env)
            rec["describe"] = env.describe()
            for fn in (C.check_state_roundtrip, C.check_state_dims_matter, C.check_batch_invariance,
                       C.check_variations_do_something, C.check_per_env_variation,
                       C.check_style_is_state, C.check_default_style_is_official,
                       C.check_colors_apply_exactly, C.check_determinism):
                rec[fn.__name__] = fn(env)
                print(f"  {fn.__name__}: {rec[fn.__name__]}", flush=True)

            states = env.sample_states(64, rng=np.random.default_rng(3))
            ref = C.reference_env(cls, env.size)
            official = cls.reference_render(ref, states)
            ours = env.render(states).cpu().numpy()
            rec["shape"] = [list(ours.shape), list(official.shape)]
            rec["parity"] = C.pixel_agreement(ours, official)
            print(f"  parity: {rec['parity']}", flush=True)
            if rec["parity"]["max_abs"] != 0:
                d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
                bad = np.argwhere(d > 0)
                rec["worst_examples"] = [
                    {"idx": [int(v) for v in b],
                     "ours": [int(v) for v in ours[tuple(b)]],
                     "official": [int(v) for v in official[tuple(b)]]}
                    for b in bad[:: max(1, len(bad) // 12)][:12]]
                rec["n_bad_pixels"] = int((d > 0).sum())
                rec["bad_frames"] = int((d > 0).any((1, 2)).sum())
                print(f"  MISMATCH: {rec['n_bad_pixels']} px over {rec['bad_frames']} frames",
                      flush=True)
                for w in rec["worst_examples"]:
                    print("   ", w, flush=True)

            rec["step_vs_reference"] = C.check_step_matches_reference(env, steps=100, episodes=16)
            print(f"  step: exact_frac={rec['step_vs_reference']['exact_frac']} "
                  f"max={rec['step_vs_reference']['max_err']}", flush=True)

            rec["fps_gpu"] = bench(env if env.B == args.bench_batch else
                                   se.make(name, batch=args.bench_batch, device="cuda"),
                                   args.bench_batch)
            rec["fps_reference"] = reference_fps(cls, env.size)
            rec["speedup"] = rec["fps_gpu"] / rec["fps_reference"]
            print(f"  fps: gpu {rec['fps_gpu']:.0f}  official {rec['fps_reference']:.1f}  "
                  f"speedup {rec['speedup']:.0f}x", flush=True)
        except Exception as e:                                              # noqa: BLE001
            import traceback
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()
            print(rec["traceback"], flush=True)
        report[name] = rec

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("\nWROTE", args.out)


if __name__ == "__main__":
    main()
