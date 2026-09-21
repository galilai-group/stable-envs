"""B appearances in ONE launch: ``env.sample_variation(B)``, and what the loop it replaces costs.

Domain randomisation is normally a loop — build N envs with N configurations, or re-configure one env N
times — because appearance lives in the env object. Here it is a ``(B, variation_dim)`` float32 tensor
alongside the compact render row, so B environments with B different colours, scales and toggles
rasterize in a single kernel launch. This script measures the difference against the loop, and checks
that the batch really carries B appearances rather than silently broadcasting environment 0's.

Three timings, all on the same B states and the same B variations:

  ``batched``       one ``render`` call with a ``(B, variation_dim)`` row
  ``per-render``    B calls on one env with batch 1, one variation each
  ``per-env``       B calls, each on a freshly constructed env

    python examples/domain_randomization.py
    python examples/domain_randomization.py --env PushT --batches 64,256,1024,4096
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import stable_envs as se


def timeit(fn, budget_s: float = 1.0, min_reps: int = 3, max_reps: int = 50) -> float:
    """Median seconds per call, synchronised on both sides, stopping once the budget is spent.

    The launch is asynchronous: timing without an explicit synchronize measures how fast Python can
    enqueue work, which for this kernel is roughly two orders of magnitude off.
    """
    fn()
    torch.cuda.synchronize()
    ts = []
    spent = 0.0
    for i in range(max_reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ts.append(dt)
        spent += dt
        if i + 1 >= min_reps and spent > budget_s:
            break
    return float(np.median(ts))


def tile(frames: np.ndarray, cols: int, pad: int = 2, pad_value: int = 235) -> np.ndarray:
    n, h, w, _ = frames.shape
    rows = math.ceil(n / cols)
    canvas = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), pad_value, np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        canvas[pad + r * (h + pad) : pad + r * (h + pad) + h, pad + c * (w + pad) : pad + c * (w + pad) + w] = f
    return canvas


def distinct_frames(frames: torch.Tensor) -> int:
    """How many of the B frames are pairwise distinct, by exact byte comparison.

    Hashing the bytes rather than comparing pairs: B(B-1)/2 comparisons of 27 kB each is 4096^2 frame
    reads at B=4096, which would take longer than every measurement in this file put together.
    """
    b = frames.reshape(frames.shape[0], -1).cpu().numpy()
    return len({row.tobytes() for row in b})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PushT")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--batches", default="64,256,1024,4096")
    ap.add_argument(
        "--per-env-cap",
        type=int,
        default=64,
        help="constructing an env per variation is O(seconds); measured up to this batch only",
    )
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("docs/research/artifacts/results"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    proto = se.make(args.env, num_envs=1, size=args.size, device="cuda")
    print(f"{args.env}: render_state_dim {proto.render_state_dim}, variation_dim {proto.variation_dim}")
    print(f"variation leaves: {proto.variation_names}")

    # One state, many appearances: any difference between the frames is the variation and nothing else.
    state = proto.sample_states(1, rng)
    variation = proto.sample_variation(args.grid, rng=rng)
    env = se.make(args.env, num_envs=args.grid, size=args.size, device="cuda")
    frames = env.render(np.repeat(state, args.grid, 0), variation=variation)
    n_distinct = distinct_frames(frames)
    print(f"\ngrid: {args.grid} environments, one shared state, {args.grid} sampled variations, one launch")
    print(f"  distinct frames: {n_distinct}/{args.grid}")
    args.out.mkdir(parents=True, exist_ok=True)
    grid_path = args.out / f"domain_randomization_{args.env}.png"
    cv2.imwrite(str(grid_path), tile(frames.cpu().numpy(), cols=int(math.sqrt(args.grid)))[:, :, ::-1])
    print(f"  wrote {grid_path} ({grid_path.stat().st_size / 1e3:.0f} kB)")

    # Restricted randomisation: everything stays at the official default except the named leaves. This is
    # what makes "vary only the background" an experiment rather than a re-render of everything.
    bg_only = proto.sample_variation(
        args.grid, paths=[f"{p}" for p in proto.variation_names if p.startswith("background")], rng=rng
    )
    bg_frames = env.render(np.repeat(state, args.grid, 0), variation=bg_only)
    bg_path = args.out / f"domain_randomization_{args.env}_background_only.png"
    cv2.imwrite(str(bg_path), tile(bg_frames.cpu().numpy(), cols=int(math.sqrt(args.grid)))[:, :, ::-1])
    print(f"  background-only sheet: {distinct_frames(bg_frames)}/{args.grid} distinct, wrote {bg_path}")
    del env
    torch.cuda.empty_cache()

    rows = []
    for B in [int(x) for x in args.batches.split(",") if x]:
        states = proto.sample_states(B, rng)
        sty = proto.sample_variation(B, rng=rng)
        st = torch.as_tensor(states, device="cuda")
        sty_t = torch.as_tensor(sty, device="cuda")

        env = se.make(args.env, num_envs=B, size=args.size, device="cuda")
        probe = env.render(st[:1])  # frame shape measured, not assumed square
        out = torch.empty((B, *probe.shape[1:]), dtype=torch.uint8, device="cuda")
        batched = timeit(lambda: env.render(st, variation=sty_t, out=out))

        one = se.make(args.env, num_envs=1, size=args.size, device="cuda")
        n_loop = min(B, 256)  # the loop is linear; time a prefix and scale, rather than wait
        per_render = timeit(
            lambda: [one.render(st[i : i + 1], variation=sty_t[i : i + 1]) for i in range(n_loop)],
            budget_s=2.0,
            max_reps=5,
        ) * (B / n_loop)

        per_env = None
        if B <= args.per_env_cap:

            def loop_construct():
                for i in range(B):
                    e = se.make(args.env, num_envs=1, size=args.size, device="cuda")
                    e.render(st[i : i + 1], variation=sty_t[i : i + 1])

            per_env = timeit(loop_construct, budget_s=2.0, max_reps=3)

        distinct = distinct_frames(env.render(st, variation=sty_t)) if B <= 1024 else None
        rows.append(
            {
                "batch": B,
                "batched_ms": batched * 1e3,
                "per_render_ms": per_render * 1e3,
                "per_env_ms": None if per_env is None else per_env * 1e3,
                "speedup_vs_per_render": per_render / batched,
                "speedup_vs_per_env": None if per_env is None else per_env / batched,
                "frames_per_s": B / batched,
                "loop_frames_per_s": B / per_render,
                "distinct_frames": distinct,
            }
        )
        print(
            f"\nbatch {B}: batched {batched * 1e3:.2f} ms ({B / batched:,.0f} frames/s) | "
            f"loop of {B} renders {per_render * 1e3:.1f} ms ({B / per_render:,.0f} frames/s) | "
            f"{per_render / batched:,.0f}x"
            + ("" if per_env is None else f" | one env per style {per_env * 1e3:.0f} ms ({per_env / batched:,.0f}x)")
            + ("" if distinct is None else f" | distinct frames {distinct}/{B}")
        )
        del env, one, out
        torch.cuda.empty_cache()

    print(
        "\n| batch | batched (ms) | loop of B renders (ms) | one env per style (ms) | speedup vs loop |"
        " frames/s batched | distinct |"
    )
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        pe = "-" if r["per_env_ms"] is None else f"{r['per_env_ms']:.0f}"
        di = "-" if r["distinct_frames"] is None else f"{r['distinct_frames']}/{r['batch']}"
        print(
            f"| {r['batch']} | {r['batched_ms']:.2f} | {r['per_render_ms']:.1f} | {pe} | "
            f"{r['speedup_vs_per_render']:,.0f}x | {r['frames_per_s']:,.0f} | {di} |"
        )

    (args.out / "domain_randomization.json").write_text(
        json.dumps(
            {
                "env": args.env,
                "size": args.size,
                "variation_dim": int(proto.variation_dim),
                "variation_names": proto.variation_names,
                "grid": {"n": args.grid, "distinct": n_distinct, "path": str(grid_path)},
                "rows": rows,
                "device": torch.cuda.get_device_name(0),
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {args.out / 'domain_randomization.json'}")


if __name__ == "__main__":
    main()
