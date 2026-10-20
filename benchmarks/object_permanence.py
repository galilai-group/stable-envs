"""Benchmark G34 rendering from mixed, owned full-State trajectory snapshots.

Collection and the first static bake are excluded from steady-state timing.
Every timed call draws again: the benchmark does not cache rendered frames.
"""

from __future__ import annotations

import argparse
import os
from importlib.metadata import version

import numpy as np
import torch

import stable_envs as se

from ._common import (
    add_common_arguments,
    checked_device,
    configure_threads,
    finish_provenance,
    measure,
    provenance,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, batches=True)
    parser.set_defaults(envs=["MuJoCoOcclusionBall"])
    args = parser.parse_args()
    if args.envs != ["MuJoCoOcclusionBall"]:
        parser.error("This trajectory benchmark currently supports MuJoCoOcclusionBall only")
    device = checked_device(args.device)
    configure_threads(args.threads)
    record = provenance(benchmark="g34-mixed-trajectory-render", args=args, snv_version=se.__version__)
    record["methodology"] = {
        "timed_operation": "env.render(full_State, out=reused_uint8_buffer)",
        "source": "120 owned reset/post-action States; evenly spread source phases per batch",
        "setup_excluded": ["construction", "trajectory collection", "first render/static bake", "output allocation"],
        "correctness_claim": None,
    }
    record["gl_context"] = os.environ.get("MUJOCO_GL")
    record["mujoco"] = version("mujoco")
    record["resolved_device"] = str(device)
    record["thread_environment"] = {
        name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS")
    }
    record["points"] = []
    with se.MuJoCoOcclusionBall(
        device=args.device,
        backend=args.backend,
        size=args.size,
        supersample=args.supersample or 1,
    ) as env:
        initial = env.reset(seed=args.seed)
        sequence = env.step(np.zeros((1, 119, 1), np.float64), frames=None)["states"]
        bank = torch.cat([initial, sequence.squeeze(0)], dim=0)
        for batch in args.batches:
            indices = torch.linspace(0, 119, batch).round().long()
            states = bank[indices].clone()
            first = env.render(states)
            out = torch.empty_like(first)
            timing = measure(
                lambda: env.render(states, out=out), device=device, warmup=args.warmup, repeats=args.repeats
            )
            record["points"].append(
                {
                    "batch": batch,
                    "frame_shape": list(out.shape[1:]),
                    "source_frames": (indices + 1).tolist(),
                    "frames_per_second": batch * 1000 / timing["median_ms"],
                    "timing": timing,
                }
            )
    finish_provenance(record)
    write_json(record, args.output, pretty=args.pretty)


if __name__ == "__main__":
    main()
