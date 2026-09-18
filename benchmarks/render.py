"""Measure stable-envs renderer throughput and batch scaling.

This is a performance benchmark, not a correctness or fidelity check. It times the public
``snv.make -> reset -> render`` surface and never imports renderer internals.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import torch

import stable_envs as snv

try:  # Support both ``python -m benchmarks.render`` and ``python benchmarks/render.py``.
    from ._common import (add_common_arguments, checked_device, configure_threads, env_kwargs,
                          error_record, finish_provenance, measure, positive_int, provenance,
                          public_env, write_json)
except ImportError:  # pragma: no cover - exercised by CLI smoke tests
    from _common import (add_common_arguments, checked_device, configure_threads, env_kwargs,
                         error_record, finish_provenance, measure, positive_int, provenance,
                         public_env, write_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser, batches=True)
    parser.add_argument(
        "--max-output-mib", type=positive_int, default=2048,
        help="skip a point whose uint8 output alone exceeds this budget (default: 2048)")
    return parser


def _point(name: str, batch: int, args: argparse.Namespace,
           device: torch.device) -> dict[str, Any]:
    env = None
    try:
        env = public_env(name, env_kwargs(args, batch))
        shape = tuple(int(x) for x in env.observation_spec.shape)
        output_mib = batch * shape[0] * shape[1] * shape[2] / 2**20
        if output_mib > args.max_output_mib:
            return {
                "status": "skipped",
                "batch": batch,
                "reason": (
                    f"uint8 output would be {output_mib:.1f} MiB, above "
                    f"--max-output-mib={args.max_output_mib}"),
                "estimated_output_mib": output_mib,
            }

        # State is the final public TensorDict API. The first render resolves the actual frame shape
        # and pays lazy setup outside the timing; warmup remains explicit and separately configured.
        state = env.reset(seed=args.seed)
        first = env.render(state)
        out = torch.empty_like(first)
        timing = measure(
            lambda: env.render(state, out=out),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        median_ms = timing["median_ms"]
        output_gb = out.numel() * out.element_size() / 1e9
        point = {
            "status": "ok",
            "batch": batch,
            "frame_shape": list(out.shape[1:]),
            "dtype": str(out.dtype).removeprefix("torch."),
            "resolved_size": int(env.size),
            "resolved_supersample": int(env.ss),
            "resolved_backend": str(env.backend),
            "frames_per_second": batch * 1_000.0 / median_ms,
            "microseconds_per_frame": median_ms * 1_000.0 / batch,
            "output_gb_per_second": output_gb * 1_000.0 / median_ms,
            "timing": timing,
        }
        if device.type == "cuda":
            point["peak_device_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        return point
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # one unsupported optional family must not erase the other rows
        return {"batch": batch, **error_record(exc)}
    finally:
        if env is not None:
            env.close()


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    device = checked_device(args.device)
    configure_threads(args.threads)
    record = provenance(
        benchmark="renderer-throughput", args=args, snv_version=snv.__version__)
    record["resolved_device"] = str(device)
    record["methodology"] = {
        "timed_operation": "env.render(public_state, out=reused_uint8_buffer)",
        "setup_excluded": ["environment construction", "reset", "first lazy render", "output allocation"],
        "synchronized": device.type in {"cuda", "mps"},
        "statistic": "median wall time; raw synchronized samples retained",
        "peak_device_memory": "CUDA max_memory_allocated of the timed region after warmup",
        "correctness_claim": None,
    }
    record["environments"] = {}
    failures = 0
    for name in args.envs:
        points = []
        for batch in args.batches:
            point = _point(name, batch, args, device)
            points.append(point)
            if point["status"] == "error":
                failures += 1
                print(
                    f"{name} batch {batch}: {point['error_type']}: {point['error']}",
                    file=sys.stderr,
                )
        record["environments"][name] = {"points": points}
    record["import_errors"] = snv.import_errors()
    finish_provenance(record)
    return record, 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        record, status = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    write_json(record, args.output, pretty=args.pretty)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
