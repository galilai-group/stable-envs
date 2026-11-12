"""Measure host-state-to-frame throughput, with an optional real-JPEG decode comparison.

The stable-envs arm includes packed host-state ingress and the public ``env.render`` call. Supplying
``--jpeg-dir`` adds a separately labeled, single-process Pillow baseline over real files. Neither
arm imports or requires stable_worldmodel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

import stable_envs as snv

try:  # Support both module and path execution.
    from ._common import (
        add_common_arguments,
        checked_device,
        configure_threads,
        env_kwargs,
        error_record,
        finish_provenance,
        measure,
        positive_int,
        provenance,
        public_env,
        write_json,
    )
except ImportError:  # pragma: no cover - exercised by CLI smoke tests
    from _common import (
        add_common_arguments,
        checked_device,
        configure_threads,
        env_kwargs,
        error_record,
        finish_provenance,
        measure,
        positive_int,
        provenance,
        public_env,
        write_json,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser, batches=False)
    parser.add_argument(
        "--state-file", default=None, help=".npy or .npz packed-state corpus; default uses env.reset(seed)"
    )
    parser.add_argument("--state-key", default="states", help="array key for an .npz state corpus")
    parser.add_argument(
        "--jpeg-dir",
        default=None,
        help="optional directory of real .jpg/.jpeg files; with multiple envs, use one subdir per env",
    )
    parser.add_argument(
        "--max-output-mib",
        type=positive_int,
        default=2048,
        help="skip an env whose uint8 output alone exceeds this budget (default: 2048)",
    )
    return parser


def _load_states(env: snv.Env, args: argparse.Namespace) -> tuple[np.ndarray | TensorDict, str]:
    if args.state_file is None:
        state = env.reset(seed=args.seed)
        if env.state_dim == 0 and hasattr(env, "to_render_state"):
            return env.to_render_state(state).cpu(), "env.reset(seed) -> structured RenderState"
        rows = env._pack_render_state(state).detach().cpu().numpy().copy()
        return rows, "env.reset(seed) -> env._pack_render_state(public State)"

    path = Path(args.state_file)
    if not path.is_file():
        raise FileNotFoundError(f"state corpus does not exist: {path}")
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if env.state_dim == 0 and hasattr(env, "to_render_state"):
                if "recipe/version" not in loaded.files:
                    raise ValueError("This environment requires a full numeric State NPZ, not packed rows")
                if len(loaded["recipe/version"]) < args.batch:
                    raise ValueError("State NPZ has fewer lanes than --batch")
                state = TensorDict(
                    {tuple(k.split("/")): torch.tensor(loaded[k][: args.batch]) for k in loaded.files},
                    [args.batch],
                    device="cpu",
                )
                return env.to_render_state(state).cpu(), "state-file -> structured RenderState"
            if args.state_key not in loaded.files:
                raise KeyError(f"{path} has keys {loaded.files}, not --state-key={args.state_key!r}")
            rows = np.asarray(loaded[args.state_key])
        finally:
            loaded.close()
    else:
        rows = np.asarray(loaded)
    if rows.ndim != 2 or rows.shape[1] != int(env.state_dim):
        raise ValueError(f"{path}: expected packed states shaped (N, {env.state_dim}), got {rows.shape}")
    if len(rows) < args.batch:
        raise ValueError(
            f"{path}: contains {len(rows)} states, fewer than --batch={args.batch}; "
            "the benchmark will not duplicate rows silently"
        )
    return np.ascontiguousarray(rows[: args.batch], dtype=np.float32), "state-file"


def _jpeg_paths(root: Path, name: str, multi_env: bool, batch: int) -> list[Path]:
    directory = root / name if multi_env else root
    if not directory.is_dir():
        raise FileNotFoundError(f"JPEG directory does not exist: {directory}")
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"})
    if len(paths) < batch:
        raise ValueError(
            f"{directory}: found {len(paths)} JPEGs, fewer than --batch={batch}; "
            "the comparison will not repeatedly decode one cached image"
        )
    return paths[:batch]


def _jpeg_baseline(paths: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("--jpeg-dir requires Pillow; install it with `python -m pip install Pillow`") from exc

    decoded_shapes: list[list[int]] = []

    def decode() -> None:
        for path in paths:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"))
                if not decoded_shapes:
                    decoded_shapes.append(list(array.shape))

    timing = measure(decode, device=torch.device("cpu"), warmup=args.warmup, repeats=args.repeats)
    median_ms = timing["median_ms"]
    return {
        "status": "ok",
        "implementation": "Pillow Image.open(...).convert('RGB') in one serial Python process",
        "files_per_call": len(paths),
        "frame_shape": decoded_shapes[0],
        "frames_per_second": len(paths) * 1_000.0 / median_ms,
        "microseconds_per_frame": median_ms * 1_000.0 / len(paths),
        "timing": timing,
        "cache_note": (
            "Files are real and reopened on every call; warmup normally leaves them in the OS page "
            "cache. This is not a cold object-store or sharded-loader measurement."
        ),
    }


def _environment(name: str, args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], int]:
    env = None
    try:
        env = public_env(name, env_kwargs(args, args.batch))
        shape = tuple(int(x) for x in env.frame_spec.shape)
        output_mib = args.batch * shape[0] * shape[1] * shape[2] / 2**20
        if output_mib > args.max_output_mib:
            return (
                {
                    "status": "skipped",
                    "batch": args.batch,
                    "reason": (
                        f"uint8 output would be {output_mib:.1f} MiB, above --max-output-mib={args.max_output_mib}"
                    ),
                    "estimated_output_mib": output_mib,
                },
                0,
            )

        host_rows, source = _load_states(env, args)
        structured = isinstance(host_rows, TensorDict)
        draw = env.render if structured else env._render

        def ingest():
            return host_rows.to(device) if structured else torch.as_tensor(host_rows, dtype=torch.float32).to(device)

        first = draw(ingest())
        out = torch.empty_like(first)

        def ingest_and_render() -> None:
            draw(ingest(), out=out)

        timing = measure(ingest_and_render, device=device, warmup=args.warmup, repeats=args.repeats)
        median_ms = timing["median_ms"]
        result: dict[str, Any] = {
            "status": "ok",
            "batch": args.batch,
            "state_source": source,
            "state_format": "structured_render_state" if structured else "packed_rows",
            "renderer_backend": env.backend,
            "packed_state_shape": None if structured else list(host_rows.shape),
            "packed_state_bytes_per_frame": None if structured else int(host_rows[0].nbytes),
            "state_bytes_per_frame": (
                sum(v[0].numel() * v.element_size() for v in host_rows.values(True, True))
                if structured
                else int(host_rows[0].nbytes)
            ),
            "frame_shape": list(out.shape[1:]),
            "frame_bytes_per_frame": int(out[0].numel() * out.element_size()),
            "resolved_size": int(env.size),
            "resolved_supersample": int(env.ss),
            "frames_per_second": args.batch * 1_000.0 / median_ms,
            "microseconds_per_frame": median_ms * 1_000.0 / args.batch,
            "timing": timing,
        }
        if device.type == "cuda":
            result["peak_device_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20

        status = 0
        if args.jpeg_dir:
            try:
                paths = _jpeg_paths(Path(args.jpeg_dir), name, len(args.envs) > 1, args.batch)
                result["jpeg_comparison"] = _jpeg_baseline(paths, args)
                baseline_fps = result["jpeg_comparison"]["frames_per_second"]
                result["render_fps_over_jpeg_fps"] = result["frames_per_second"] / baseline_fps
            except Exception as exc:
                result["jpeg_comparison"] = error_record(exc)
                status = 1
        return result, status
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # preserve a machine-readable row for an optional-family failure
        return {"batch": args.batch, **error_record(exc)}, 1
    finally:
        if env is not None:
            env.close()


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    device = checked_device(args.device)
    configure_threads(args.threads)
    record = provenance(benchmark="host-state-render-pipeline", args=args, snv_version=snv.__version__)
    record["resolved_device"] = str(device)
    record["methodology"] = {
        "timed_operation": (
            "host numeric RenderState.to(device) -> renderer(out=reused_uint8_buffer); "
            "structured projections use env.render, packed state rows use env._render"
        ),
        "setup_excluded": [
            "environment construction",
            "state corpus load/reset",
            "first lazy render",
            "output allocation",
        ],
        "synchronized": device.type in {"cuda", "mps"},
        "statistic": "median wall time; raw synchronized samples retained",
        "peak_device_memory": "CUDA max_memory_allocated of the timed region after warmup",
        "jpeg_comparison": "optional and separately timed",
        "stable_worldmodel_required": False,
        "correctness_claim": None,
    }
    record["environments"] = {}
    status = 0
    for name in args.envs:
        result, env_status = _environment(name, args, device)
        record["environments"][name] = result
        status = max(status, env_status)
        if result["status"] == "error":
            print(f"{name}: {result['error_type']}: {result['error']}", file=sys.stderr)
        comparison = result.get("jpeg_comparison")
        if comparison and comparison["status"] == "error":
            print(
                f"{name} JPEG comparison: {comparison['error_type']}: {comparison['error']}",
                file=sys.stderr,
            )
    record["import_errors"] = snv.import_errors()
    finish_provenance(record)
    return record, status


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
