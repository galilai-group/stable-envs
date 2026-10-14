"""Render one State under many Variations in a single call.

``env.sample_variation(B)`` draws ``(B, variation_dim)`` rows; ``env.render(state, variation=rows)``
gives each lane its own appearance. The script counts distinct frames and times the batched call
against a loop of one-lane renders. Envs whose Variation lives in the State (PushT, Rocket, OGBench
Cube/Scene) take it through ``reset(options={"variation": [...]})`` instead.

    python examples/domain_randomization.py
    python examples/domain_randomization.py --env TwoRoom --num-envs 256 --device cuda
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

import stable_envs as snv


def distinct(frames: torch.Tensor) -> int:
    return len({f.tobytes() for f in frames.cpu().numpy()})


def seconds(fn, device: torch.device, reps: int) -> float:
    fn()
    times = []
    for _ in range(reps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Per-lane Variation in one render call.")
    p.add_argument("--env", default="CartPole")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args(argv)

    n = args.num_envs
    with snv.make(args.env, num_envs=n, device=args.device) as env:
        state = env.reset(seed=args.seed)
        # One State for every lane, so differences between frames come from the Variation.
        shared = state[torch.zeros(n, dtype=torch.long)]
        appearance = [p for p in env.variation_names if p not in env.dynamics_variation_leaves]
        rng = np.random.default_rng(args.seed)
        rows = torch.as_tensor(env.sample_variation(n, paths=appearance, rng=rng), device=env.device)

        frames = env.render(shared, variation=rows)
        print(f"{args.env}: {len(appearance)} appearance leaves, variation_dim {env.variation_dim}")
        print(f"default Variation: {distinct(env.render(shared))}/{n} distinct frames")
        print(f"sampled Variation: {distinct(frames)}/{n} distinct frames, shape {tuple(frames.shape)}")

        background = [p for p in appearance if p.startswith("background")]
        if background:
            bg = torch.as_tensor(env.sample_variation(n, paths=background, rng=rng), device=env.device)
            print(f"background only: {distinct(env.render(shared, variation=bg))}/{n} distinct frames")

        batched = seconds(lambda: env.render(shared, variation=rows), env.device, args.reps)
        looped = seconds(
            lambda: [env.render(shared[i : i + 1], variation=rows[i : i + 1]) for i in range(n)],
            env.device,
            args.reps,
        )
        print(f"one call: {batched * 1e3:.2f} ms; {n} one-lane calls: {looped * 1e3:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
