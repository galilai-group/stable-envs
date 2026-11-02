"""Synchronized DMC native/direct-MJWarp benchmarks with separated costs.

Example: python benchmarks/dmc_mjwarp.py --output results/dmc-mjwarp.json --pretty

The default matrix is 16 DMC cells x batches 1,16,64,256,1024 x native/MJWarp
(160 records). Install psutil and nvidia-ml-py for memory measurements.
Each environment/batch/backend record runs in a fresh process so allocator
caches from earlier records cannot contaminate its process memory measurements.
An output file is atomically updated after each record, so a long sweep retains
completed measurements. Errors remain explicit records and cause a nonzero exit.
Raw MJWarp physics advances two control rounds, matching one public DMC action;
it excludes task evaluation, public snapshots, rendering and reset bookkeeping.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from benchmarks import _common as common  # noqa: E402
from stable_envs.registry import REGISTRY  # noqa: E402
from stable_envs.testing.evidence import sha256  # noqa: E402

CELLS = tuple(sorted(name for name, cls in REGISTRY.items() if cls.family == "dmcontrol" and name != "DMCReacher"))


class MemoryProbe:
    """RSS (including workers) and NVML allocations outside the timing samples.

    Requires psutil and nvidia-ml-py in the measuring interpreter. Device memory
    includes other processes; process memory includes this process and workers.
    Before/after deltas are not represented as a high-frequency allocation peak.
    """

    def __init__(self, device):
        import psutil
        import pynvml

        self.process = psutil.Process(os.getpid())
        self.nvml = pynvml
        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)

    def sample(self):
        import psutil

        processes = [self.process, *self.process.children(recursive=True)]
        rss, pids = {}, set()
        for process in processes:
            try:
                rss[str(process.pid)] = process.memory_info().rss
                pids.add(process.pid)
            except psutil.NoSuchProcess:
                continue
        info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        gpu = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        own = [entry.usedGpuMemory for entry in gpu if entry.pid in pids]
        return {
            "rss_bytes_by_pid": rss,
            "rss_parent_and_workers_bytes": sum(rss.values()),
            "nvml_device_used_bytes": info.used,
            "nvml_parent_and_workers_used_bytes": None if any(value is None for value in own) else sum(own),
        }

    def profile(self, fn, device):
        common.synchronize(device)
        before = self.sample()
        tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            output = fn()
            common.synchronize(device)
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        after = self.sample()
        del output
        deltas = {}
        for key in ("rss_parent_and_workers_bytes", "nvml_device_used_bytes", "nvml_parent_and_workers_used_bytes"):
            deltas[key] = None if before[key] is None or after[key] is None else after[key] - before[key]
        return {
            "scope": "one extra synchronized call, separate from timed samples; result retained until after sampling",
            "before": before,
            "after": after,
            "deltas": deltas,
            "tracemalloc_current_bytes": current,
            "tracemalloc_peak_bytes": peak,
        }


def _metric(fn, args, device, *, transitions=0, warmup=None, repeats=None):
    value = common.measure(
        fn,
        device=device,
        warmup=args.warmup if warmup is None else warmup,
        repeats=args.repeats if repeats is None else repeats,
    )
    if transitions:
        value["environment_transitions_per_second"] = transitions * 1000 / value["median_ms"]
    value["torch_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    value["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    value["memory"] = args.memory_probe.profile(fn, device)
    return value


def native_raw_physics(name, batch, args, device):
    """Same models/control steps, local sequential CPU lanes, without worker IPC."""
    import numpy as np

    import stable_envs as se
    from stable_envs.envs.dmcontrol.simulation.factory import make_local

    key = getattr(se, name)._simulator_key()
    started = time.perf_counter()
    source = make_local(key, batch, "cpu", args.seed)
    try:
        construction_ms = (time.perf_counter() - started) * 1000
        spec = source.envs[0]._env.action_spec()
        controls = np.broadcast_to((spec.maximum + spec.minimum) / 2, (batch, source.action_dim)).copy()
        nsubsteps = source.envs[0]._env._n_sub_steps

        def physics():
            for _ in range(2):
                for lane, env in enumerate(source.envs):
                    env.physics.set_control(controls[lane])
                    env.physics.step(nsubsteps)

        return {
            "raw_native_construction": {"wall_ms": construction_ms, "samples": 1},
            "raw_physics_two_control_rounds": _metric(physics, args, device, transitions=batch),
        }, {
            "raw_physics_scope": "standalone native local lanes, sequential CPU, no process IPC or task/snapshot/render",
            "control_substeps": nsubsteps,
        }
    finally:
        source.close()


def measure_cell(name, batch, simulation_backend, args, device):
    kwargs = common.env_kwargs(args, batch)
    kwargs["simulation_backend"] = simulation_backend
    common.synchronize(device)
    memory_before = args.memory_probe.sample()
    started = time.perf_counter()
    candidate = common.public_env(name, kwargs)
    try:
        common.synchronize(device)
        construction_ms = (time.perf_counter() - started) * 1000
        memory_after_construction = args.memory_probe.sample()
        if candidate.simulation_backend != simulation_backend:
            raise AssertionError(f"Requested {simulation_backend}, got {candidate.simulation_backend}")
        actions = torch.zeros(batch, candidate.action_dim, dtype=torch.float64, device=device)
        saved = candidate.reset(seed=args.seed).clone()
        metrics = {
            "construction": {
                "wall_ms": construction_ms,
                "samples": 1,
                "memory_before": memory_before,
                "memory_after": memory_after_construction,
            }
        }
        metrics["first_public_step"] = _metric(
            lambda: candidate.step(actions, states=None, frames=None),
            args,
            device,
            warmup=0,
            repeats=1,
            transitions=batch,
        )
        candidate.set_state(saved)
        metrics["reset"] = _metric(lambda: candidate.reset(seed=args.seed), args, device)
        metrics["restore"] = _metric(lambda: candidate.set_state(saved), args, device)
        candidate.set_state(saved)
        metrics["public_step_no_pixels"] = _metric(
            lambda: candidate.step(actions, states=None, frames=None),
            args,
            device,
            transitions=batch,
        )
        candidate.set_state(saved)
        metrics["render_saved_state"] = _metric(lambda: candidate.render(saved), args, device)
        candidate.set_state(saved)
        metrics["public_step_with_pixels"] = _metric(
            lambda: candidate.step(actions, states=None, frames=slice(-1, None)),
            args,
            device,
            transitions=batch,
        )
        adapter = candidate._simulator
        raw_description = "native raw physics measured separately after public adapter closes"
        if simulation_backend == "mjwarp":
            candidate.set_state(saved)
            controls = adapter._controls(actions)

            def raw_physics():
                adapter._runtime.advance(controls)
                adapter._runtime.advance(controls)

            metrics["raw_physics_two_control_rounds"] = _metric(raw_physics, args, device, transitions=batch)
            candidate.set_state(saved)  # raw timing intentionally bypassed adapter bookkeeping
            raw_description = "two runtime.advance calls; native controls precomputed outside timing"
        models = getattr(adapter, "_models", [])
        source = getattr(adapter, "_metadata", adapter)
        lanes = getattr(source, "envs", [])
        model = models[0] if models else None
        if model is None and lanes:
            model = lanes[0].physics.model.ptr
        substeps = getattr(adapter, "_nsubsteps", None)
        if substeps is None and lanes:
            substeps = lanes[0]._env._n_sub_steps
        metadata = {
            "execution_mode": candidate.execution_mode,
            "simulation_adapter": type(adapter).__name__,
            "simulation_worker_pids": [process.pid for process in getattr(adapter, "_processes", [])],
            "action_dim": candidate.action_dim,
            "action_dtype": str(actions.dtype),
            "output_device": str(device),
            "physics_dtype": "float64" if simulation_backend == "native" else "float32",
            "action_repeat": 2,
            "control_substeps": substeps,
            "render_size": candidate.size,
            "render_supersample": candidate.ss,
            "render_backend_request": candidate.backend,
            "raw_physics_scope": raw_description,
            "cpu_time_scope": "parent process CPU/wall ratio; excludes child simulation workers",
            "memory_scope": (
                "Torch allocator peaks; Python tracemalloc peak; sampled RSS/workers and NVML device/process usage. "
                "RSS/NVML include earlier phases in this cell and do not measure transient allocation peaks."
            ),
            "model": None
            if model is None
            else {
                "nq": model.nq,
                "nv": model.nv,
                "nu": model.nu,
                "physics_timestep": model.opt.timestep,
                "solver": int(model.opt.solver),
                "iterations": model.opt.iterations,
                "ls_iterations": model.opt.ls_iterations,
                "tolerance": model.opt.tolerance,
            },
        }
        return {"status": "ok", "metadata": metadata, "metrics": metrics}
    finally:
        candidate.close()


def measure_record(name, batch, backend, args, device):
    row = dict(env=name, batch=batch, simulation_backend=backend, worker_pid=os.getpid())
    try:
        row.update(measure_cell(name, batch, backend, args, device))
        if backend == "native":
            metrics, metadata = native_raw_physics(name, batch, args, device)
            row["metrics"].update(metrics)
            row["metadata"].update(metadata)
    except Exception as exc:
        row.update(common.error_record(exc))
    return row


def isolated_record(name, batch, backend, args):
    """Allocator caches end with each cell, including Warp and Torch caches."""
    with tempfile.TemporaryDirectory(prefix="dmc-mjwarp-benchmark-") as directory:
        output = Path(directory) / "record.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_worker",
            "--envs",
            name,
            "--batches",
            str(batch),
            "--simulation-backends",
            backend,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
            "--threads",
            str(args.threads),
            "--backend",
            args.backend,
            "--output",
            str(output),
        ]
        for option in ("size", "supersample"):
            value = getattr(args, option)
            if value is not None:
                command.extend(("--" + option, str(value)))
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if output.is_file():
            row = json.loads(output.read_text())
        else:
            row = dict(
                env=name,
                batch=batch,
                simulation_backend=backend,
                status="error",
                error_type="BenchmarkWorkerError",
                error=f"Worker exited {result.returncode} without a result",
            )
        if result.returncode and row.get("status") == "ok":
            row.update(status="error", error_type="BenchmarkWorkerError", error=f"Worker exited {result.returncode}")
        if row.get("status") != "ok":
            row["worker_stdout_tail"] = result.stdout[-4000:]
            row["worker_stderr_tail"] = result.stderr[-4000:]
        row["process_isolation"] = "fresh process per environment/batch/backend; worker exits after this record"
        return row


def benchmark_versions():
    versions = {}
    for name in ("psutil", "nvidia-ml-py", "mujoco", "mujoco-warp", "warp-lang", "dm-control"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_arguments(parser, batches=True)
    parser.set_defaults(envs=list(CELLS), batches=[1, 16, 64, 256, 1024], device="cuda")
    parser._option_string_actions["--envs"].help = "comma-separated supported DMC environment names (default: all 16)"
    parser.add_argument(
        "--simulation-backends", type=common.names, default=["native", "mjwarp"], help="native,mjwarp (default: both)"
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not CELLS:
        parser.error("Install stable-envs[dmcontrol-gpu] to benchmark DMC")
    unknown = set(args.envs) - set(CELLS)
    if unknown:
        parser.error(f"Unknown DMC cells: {sorted(unknown)}")
    if set(args.simulation_backends) - {"native", "mjwarp"}:
        parser.error("--simulation-backends accepts native,mjwarp")
    device = common.checked_device(args.device)
    if device.type != "cuda":
        parser.error("This benchmark requires actual CUDA; native baseline returns on the same CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    args.device = str(device)
    torch.cuda.set_device(device)
    common.configure_threads(args.threads)
    if args._worker:
        if len(args.envs) != 1 or len(args.batches) != 1 or len(args.simulation_backends) != 1:
            parser.error("A benchmark worker requires exactly one environment, batch and backend")
        args.memory_probe = MemoryProbe(device)
        row = measure_record(args.envs[0], args.batches[0], args.simulation_backends[0], args, device)
        common.write_json(row, args.output, pretty=args.pretty)
        return int(row["status"] != "ok")
    import stable_envs as se

    properties = torch.cuda.get_device_properties(device)
    machine = {
        "device": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "uuid": str(properties.uuid),
        "total_memory": properties.total_memory,
        "torch_cuda": torch.version.cuda,
        "versions": benchmark_versions(),
    }
    record = common.provenance(benchmark="dmc-mjwarp", args=args, snv_version=se.__version__)
    record.update(
        machine=machine,
        candidate_files={
            path.relative_to(ROOT).as_posix(): sha256(path) for path in (ROOT / "stable_envs").rglob("*.py")
        },
        benchmark_files={
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in (Path(__file__).resolve(), ROOT / "benchmarks/_common.py")
        },
        benchmark_versions=benchmark_versions(),
        process_isolation="one fresh worker per environment/batch/backend; parent hardware context may remain",
        baseline_revision=os.environ.get("SE_DMC_MJWARP_BASELINE_REVISION"),
        baseline_scope="native from this measured worktree; baseline_revision is a caller label",
        runtime_environment={
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        records=[],
        complete=False,
    )
    failures = 0
    for name in args.envs:
        for batch in args.batches:
            for backend in args.simulation_backends:
                try:
                    row = isolated_record(name, batch, backend, args)
                except Exception as exc:
                    row = dict(env=name, batch=batch, simulation_backend=backend, **common.error_record(exc))
                failures += row["status"] != "ok"
                record["records"].append(row)
                if args.output != "-":
                    common.write_json(record, args.output, pretty=args.pretty)
                print(f"{name} batch={batch} {backend}: {row['status']}", file=sys.stderr, flush=True)
    record["complete"] = True
    common.finish_provenance(record)
    common.write_json(record, args.output, pretty=args.pretty)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
