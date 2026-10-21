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
    parser.add_argument(
        "--delivery-device",
        default=None,
        help="final training-input device; cuda includes RGB upload for a CPU renderer",
    )
    parser.set_defaults(envs=["MuJoCoOcclusionBall"])
    args = parser.parse_args()
    if args.envs != ["MuJoCoOcclusionBall"]:
        parser.error("This trajectory benchmark currently supports MuJoCoOcclusionBall only")
    device = checked_device(args.device)
    delivery = checked_device(args.delivery_device or args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if delivery.type == "cuda" and delivery.index is None:
        delivery = torch.device("cuda", torch.cuda.current_device())
    configure_threads(args.threads)
    record = provenance(benchmark="g34-mixed-trajectory-render", args=args, snv_version=se.__version__)
    record["methodology"] = {
        "timed_operation": "env.render(CPU_full_State, out=reused_buffer), then RGB copy if delivery device differs",
        "state_storage": "pageable CPU TensorDict, float64 integration + int64 lifecycle",
        "host_to_device_included": delivery.type == "cuda" or device.type == "cuda",
        "source": "120 owned reset/post-action States; evenly spread source phases per batch",
        "setup_excluded": ["construction", "trajectory collection", "first render/static bake", "output allocation"],
        "correctness_claim": None,
    }
    record["gl_context"] = os.environ.get("MUJOCO_GL")
    record["mujoco"] = version("mujoco")
    record["resolved_device"] = str(device)
    record["delivery_device"] = str(delivery)
    record["cuda_runtime"] = torch.version.cuda
    if device.type == "cuda" or delivery.type == "cuda":
        record["gpu"] = torch.cuda.get_device_name(device if device.type == "cuda" else delivery)
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
        bank = torch.cat([initial, sequence.squeeze(0)], dim=0).cpu()
        for batch in args.batches:
            indices = torch.linspace(0, 119, batch).round().long()
            states = bank[indices].clone()
            first = env.render(states)
            out = torch.empty_like(first)
            delivered = torch.empty(out.shape, dtype=out.dtype, device=delivery) if delivery != out.device else out

            def draw():
                env.render(states, out=out)
                if delivered is not out:
                    delivered.copy_(out)

            timing = measure(
                draw, device=delivery if delivery.type == "cuda" else device, warmup=args.warmup, repeats=args.repeats
            )
            record["points"].append(
                {
                    "batch": batch,
                    "frame_shape": list(out.shape[1:]),
                    "source_frames": (indices + 1).tolist(),
                    "compact_pose_upload_bytes": batch * 28 if device.type == "cuda" else 0,
                    "rgb_upload_bytes": batch * out[0].numel()
                    if (delivery.type == "cuda" and device.type == "cpu")
                    or (device.type == "cuda" and args.backend != "raycast")
                    else 0,
                    "frames_per_second": batch * 1000 / timing["median_ms"],
                    "timing": timing,
                }
            )
    finish_provenance(record)
    write_json(record, args.output, pretty=args.pretty)


if __name__ == "__main__":
    main()
