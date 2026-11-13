"""Shared measurement and provenance helpers for the two public benchmark scripts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

SCHEMA = "stable-envs-benchmark/v1"
ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def positive_ints(value: str) -> list[int]:
    try:
        values = [positive_int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated positive integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one positive integer")
    return values


def names(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one environment name")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("environment names must not be repeated")
    return values


def native_int(value: str) -> int | None:
    if value.strip().lower() in {"native", "none", "default"}:
        return None
    return positive_int(value)


def add_common_arguments(
    parser: argparse.ArgumentParser, *, batches: bool, overrides: Mapping[str, Mapping[str, Any]] | None = None
) -> None:
    """Add the shared options; ``overrides`` maps an option to ``add_argument`` keywords that replace the shared ones."""

    def add(flag: str, **kwargs: Any) -> None:
        parser.add_argument(flag, **{**kwargs, **(overrides or {}).get(flag, {})})

    add("--envs", type=names, default=["PushT"], help="comma-separated canonical environment names (default: PushT)")
    if batches:
        add("--batches", type=positive_ints, default=[1, 16, 64, 256], help="comma-separated batch sizes")
    else:
        add("--batch", type=positive_int, default=64)
    add("--device", default="cpu", help="torch device, for example cpu, cuda, or cuda:1")
    add("--warmup", type=nonnegative_int, default=5, help="untimed calls before measurement")
    add("--repeats", type=positive_int, default=10, help="synchronized timed calls")
    add("--seed", type=int, default=0)
    add(
        "--threads", type=positive_int, default=1, help="torch/OpenMP/MKL CPU thread budget (recorded on every device)"
    )
    add(
        "--size",
        type=native_int,
        default=None,
        help="output size, or native to leave the environment default unchanged",
    )
    add(
        "--supersample",
        type=native_int,
        default=None,
        help="supersampling factor, or native to leave the environment default unchanged",
    )
    add("--backend", default="auto")
    add("--output", default="-", help="JSON path, or - for stdout (default)")
    add("--pretty", action="store_true", help="indent JSON output")


def checked_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid torch device {value!r}: {exc}") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                f"device {value!r} requested, but torch.cuda.is_available() is false; "
                "choose --device cpu or run in a CUDA-enabled interpreter"
            )
        index = 0 if device.index is None else device.index
        visible = torch.cuda.device_count()
        if index >= visible:
            raise ValueError(f"device {value!r} requested, but torch.cuda.device_count() is {visible}")
    if device.type == "mps":
        available = getattr(torch.backends, "mps", None)
        if available is None or not torch.backends.mps.is_available():
            raise ValueError("device 'mps' requested, but torch.backends.mps.is_available() is false")
    return device


def configure_threads(count: int) -> None:
    """Pin the process thread budget; do not override an already-exported OpenMP/MKL setting."""
    torch.set_num_threads(count)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, str(count))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def measure(fn: Callable[[], Any], *, device: torch.device, warmup: int, repeats: int) -> dict[str, Any]:
    """Measure synchronized caller-visible wall time; keep samples so spread remains visible."""
    for _ in range(warmup):
        fn()
    synchronize(device)
    # Reset after warmup so autotune and lazy setup do not count toward peak memory.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    samples_ms: list[float] = []
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples_ms.append((time.perf_counter() - start) * 1_000.0)
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    median = _percentile(samples_ms, 0.5)
    return {
        "median_ms": median,
        "min_ms": min(samples_ms),
        "p25_ms": _percentile(samples_ms, 0.25),
        "p75_ms": _percentile(samples_ms, 0.75),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
        "cpu_seconds_per_wall_second": cpu / wall if wall else 0.0,
    }


def env_kwargs(args: argparse.Namespace, batch: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_envs": batch,
        "device": args.device,
        "seed": args.seed,
    }
    # Unset options are omitted so each environment renders at its own defaults.
    if args.size is not None:
        kwargs["size"] = args.size
    if args.supersample is not None:
        kwargs["supersample"] = args.supersample
    if args.backend != "auto":
        kwargs["backend"] = args.backend
    return kwargs


def public_env(name: str, kwargs: dict[str, Any]):
    """Construct via the canonical class when ``name`` is public; otherwise ``snv.make``."""
    import stable_envs as snv

    cls = getattr(snv, name, None)
    if isinstance(cls, type) and issubclass(cls, snv.Env) and cls is not snv.Env:
        return cls(**kwargs)
    return snv.make(name, **kwargs)


def error_record(exc: BaseException) -> dict[str, str]:
    return {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_revision() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def provenance(*, benchmark: str, args: argparse.Namespace, snv_version: str) -> dict[str, Any]:
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = os.cpu_count()
    return {
        "schema": SCHEMA,
        "benchmark": benchmark,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stable_envs_version": snv_version,
        "stable_envs_git_revision": _git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": _version("numpy"),
        "host": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_affinity": affinity,
        "load_average_start": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "argv": [sys.executable, *sys.argv],
        "config": {key: value for key, value in vars(args).items() if key not in {"output", "pretty"}},
    }


def write_json(record: dict[str, Any], output: str, *, pretty: bool) -> None:
    text = json.dumps(record, indent=2 if pretty else None, sort_keys=True, allow_nan=False) + "\n"
    if output == "-":
        sys.stdout.write(text)
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def finish_provenance(record: dict[str, Any]) -> None:
    record["load_average_end"] = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
