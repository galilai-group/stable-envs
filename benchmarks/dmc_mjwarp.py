"""Synchronized DMC native/direct-MJWarp benchmarks with separated costs.

Example: python benchmarks/dmc_mjwarp.py --output results/dmc-mjwarp.json --pretty

The default matrix is 17 DMC cells x batches 1,16,64,256,1024 x 3 seeds x native/MJWarp
(510 records). --extended-batches appends 2048,4096,8192 to the selected batches.
Seeded uniform actions, 20 warmups, a checkpoint restore and 100 timed calls match
the PR #28 rollout workload. Action generation and upload happen outside timing.
Use --dry-run to inspect the matrix without CUDA initialization or worker launches.
Install psutil and nvidia-ml-py for memory measurements.
Each environment/batch/seed/backend record runs in a fresh process so allocator
caches from earlier records cannot contaminate its process memory measurements.
An output file is atomically updated after each record, so a long sweep retains
completed measurements. Errors remain explicit records and cause a nonzero exit.
Raw MJWarp physics advances two control rounds, matching one public DMC action;
it excludes task evaluation, public snapshots, rendering and reset bookkeeping.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from benchmarks import _common as common  # noqa: E402
from stable_envs.registry import REGISTRY  # noqa: E402
from stable_envs.testing.evidence import sha256  # noqa: E402

# Keep the benchmark matrix available on hosts without the optional DMC packages.
CELLS = (
    "DMCAcrobotBalance",
    "DMCBallInCupCatch",
    "DMCCartpoleBalance",
    "DMCCartpoleSwingup",
    "DMCCheetahRun",
    "DMCFingerSpin",
    "DMCFingerTurnEasy",
    "DMCFingerTurnHard",
    "DMCHopperHop",
    "DMCHumanoidWalk",
    "DMCManipulatorBringBall",
    "DMCPendulumSwingup",
    "DMCPointMassEasy",
    "DMCQuadrupedWalk",
    "DMCReacher",
    "DMCSwimmerSwimmer6",
    "DMCWalkerWalk",
)
DEFAULT_BATCHES = (1, 16, 64, 256, 1024)
EXTENDED_BATCHES = (2048, 4096, 8192)
DEFAULT_SEEDS = (91000, 91010, 91020)
ACTION_SEED_OFFSET = 91717


def _seeds(value):
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated non-negative integers") from exc
    if any(seed < 0 for seed in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("seeds must be distinct non-negative integers")
    return values


def _action_array(batch, width, seed, calls):
    """Reproduce the PR harness's lane-major PCG64 draws, including batch prefixes."""
    return (
        np.random.default_rng(seed + ACTION_SEED_OFFSET)
        .uniform(-1, 1, (batch, calls, width))
        .transpose(1, 0, 2)
        .copy()
    )


def _state_sha256(state):
    """Hash the complete checkpoint with canonical leaf order, shapes and dtypes."""
    leaves = {(key,) if isinstance(key, str) else key: value for key, value in state.items(True, True)}
    digest = hashlib.sha256()
    for key, value in sorted(leaves.items()):
        array = value.detach().cpu().numpy()
        digest.update(json.dumps([key, array.shape, array.dtype.str]).encode())
        digest.update(np.ascontiguousarray(array))
    return digest.hexdigest()


def _protocol(args):
    return {
        "name": "dmc-pr28-random-rollout/v1",
        "seeds": args.seeds,
        "action_generator": "numpy.default_rng(seed + 91717).uniform(-1, 1, (batch, calls, action_dim))",
        "action_layout": "draw lane-major, transpose to contiguous [call, lane, component] float64",
        "generated_calls": max(args.warmup, args.repeats),
        "warmup_calls": args.warmup,
        "timed_calls": args.repeats,
        "warmup_actions": "prefix of the generated sequence; replay from index zero after checkpoint restore",
        "initial_state": "seeded native reset checkpoint, restored before warmup and again before timing",
        "public_step": "one action per call, two control rounds, full State/task outputs, frames=None, states=None",
        "timing": "CUDA synchronized before and after each call; action generation/upload and restore excluded",
        "backend_order": "requested order for even seed indices, reversed for odd seed indices",
    }


def _matrix(args):
    return [
        dict(env=name, batch=batch, seed=seed, simulation_backend=backend)
        for name in args.envs
        for batch in args.batches
        for index, seed in enumerate(args.seeds)
        for backend in (args.simulation_backends if index % 2 == 0 else args.simulation_backends[::-1])
    ]


def _maxrss_bytes(value):
    """Normalize the kernel high-water mark on platforms with known units."""
    if sys.platform == "darwin":
        return int(value)
    if sys.platform.startswith("linux"):
        return int(value) * 1024
    return None


def _self_peak_rss_bytes():
    try:
        import resource
    except ImportError:
        return None
    return _maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


class MemoryProbe:
    """RSS (including workers) and NVML allocations outside the timing samples.

    Requires psutil and nvidia-ml-py in the measuring interpreter. Device memory
    includes other processes; process memory includes this process and workers.
    rss_peak_self_bytes is the kernel's process-lifetime high-water mark, including
    imports and earlier phases, not a resettable per-operation or process-tree peak.
    Before/after RSS sums and NVML deltas remain point samples.
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
            "rss_peak_self_bytes": _self_peak_rss_bytes(),
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


def _metric(fn, args, device, *, transitions=0, warmup=None, repeats=None, memory_fn=None):
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
    value["memory"] = args.memory_probe.profile(fn if memory_fn is None else memory_fn, device)
    return value


def _rollout_metric(step, restore, actions, args, device, *, transitions):
    """Warm the path, then replay the timed sequence from its original checkpoint."""
    restore()
    for action in actions[: args.warmup]:
        step(action)
    restore()
    timed = iter(actions[: args.repeats])
    result = None

    def next_step():
        nonlocal result
        # Retain outputs through the CUDA synchronization, as in the PR harness.
        result = step(next(timed))
        return result

    value = _metric(
        next_step,
        args,
        device,
        transitions=transitions,
        warmup=0,
        memory_fn=lambda: step(actions[0]),
    )
    value["memory"]["action_scope"] = "one additional call with sequence index 0 after the timed rollout"
    return value


def native_raw_physics(name, batch, args, device):
    """Same models/control steps, local sequential CPU lanes, without worker IPC."""
    import stable_envs as se
    from stable_envs.envs.dmcontrol.simulation.factory import make_local

    key = getattr(se, name)._simulator_key()
    started = time.perf_counter()
    source = make_local(key, batch, "cpu", args.seed)
    try:
        construction_ms = (time.perf_counter() - started) * 1000
        spec = source.envs[0]._env.action_spec()
        saved = source.get_state()
        actions = _action_array(batch, source.action_dim, args.seed, max(args.warmup, args.repeats))
        controls = spec.minimum + (spec.maximum - spec.minimum) / 2 * (actions + 1)
        nsubsteps = source.envs[0]._env._n_sub_steps

        def physics(control):
            for _ in range(2):
                for lane, env in enumerate(source.envs):
                    env.physics.set_control(control[lane])
                    env.physics.step(nsubsteps)

        return {
            "raw_native_construction": {"wall_ms": construction_ms, "samples": 1},
            "raw_physics_two_control_rounds": _rollout_metric(
                physics, lambda: source.set_state(saved), controls, args, device, transitions=batch
            ),
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
        action_array = _action_array(batch, candidate.action_dim, args.seed, max(args.warmup, args.repeats))
        actions = torch.as_tensor(action_array, dtype=torch.float64, device=device).unbind(0)
        saved = candidate.reset(seed=args.seed).clone()
        inputs = {
            "reset_seed": args.seed,
            "action_seed": args.seed + ACTION_SEED_OFFSET,
            "action_shape": list(action_array.shape),
            "actions_sha256": hashlib.sha256(action_array).hexdigest(),
            "initial_state_sha256": _state_sha256(saved),
        }
        metrics = {
            "construction": {
                "wall_ms": construction_ms,
                "samples": 1,
                "memory_before": memory_before,
                "memory_after": memory_after_construction,
            }
        }
        metrics["first_public_step"] = _metric(
            lambda: candidate.step(actions[0], states=None, frames=None),
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
        metrics["public_step_no_pixels"] = _rollout_metric(
            lambda action: candidate.step(action, states=None, frames=None),
            lambda: candidate.set_state(saved),
            actions,
            args,
            device,
            transitions=batch,
        )
        candidate.set_state(saved)
        metrics["render_saved_state"] = _metric(lambda: candidate.render(saved), args, device)
        candidate.set_state(saved)
        metrics["public_step_with_pixels"] = _rollout_metric(
            lambda action: candidate.step(action, states=None, frames=slice(-1, None)),
            lambda: candidate.set_state(saved),
            actions,
            args,
            device,
            transitions=batch,
        )
        adapter = candidate._simulator
        raw_description = "native raw physics measured separately after public adapter closes"
        if simulation_backend == "mjwarp":
            candidate.set_state(saved)
            controls = tuple(adapter._controls(action) for action in actions)

            def raw_physics(control):
                adapter._runtime.advance(control)
                adapter._runtime.advance(control)

            metrics["raw_physics_two_control_rounds"] = _rollout_metric(
                raw_physics, lambda: candidate.set_state(saved), controls, args, device, transitions=batch
            )
            candidate.set_state(saved)  # raw timing intentionally bypassed adapter bookkeeping
            raw_description = "two runtime.advance calls; native controls precomputed outside timing"
        models = getattr(adapter, "_models", [])
        source = getattr(adapter, "lanes", getattr(adapter, "_source", adapter))
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
            "action_dtype": str(actions[0].dtype),
            "inputs": inputs,
            "native_nthread": int(adapter.b.nthread) if simulation_backend == "native" else None,
            "torch_num_threads": torch.get_num_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
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
                "Torch allocator peaks; Python tracemalloc peak; kernel process-lifetime peak RSS through each "
                "sample (rss_peak_self_bytes, excluding children); sampled RSS/workers and NVML device/process usage. "
                "Lifetime peaks include imports and earlier phases; RSS sums/NVML are point samples, not peaks."
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
    row = dict(env=name, batch=batch, seed=args.seed, simulation_backend=backend, worker_pid=os.getpid())
    try:
        row.update(measure_cell(name, batch, backend, args, device))
        if backend == "native":
            metrics, metadata = native_raw_physics(name, batch, args, device)
            row["metrics"].update(metrics)
            row["metadata"].update(metadata)
    except Exception as exc:
        row.update(common.error_record(exc))
    return row


def _run_worker(command):
    """Collect exit-time peak RSS even when a worker cannot write its JSON result.

    File-backed logs avoid pipe deadlock while wait4 collects this child's rusage.
    Linux may include the largest reaped descendant in ru_maxrss; it is never the
    peak concurrent sum of a process tree. Unsupported platforms report null.
    """
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        peak = None
        with subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr) as process:
            try:
                if hasattr(os, "wait4"):
                    _, status, usage = os.wait4(process.pid, 0)
                    process.returncode = os.waitstatus_to_exitcode(status)
                    peak = _maxrss_bytes(usage.ru_maxrss)
                else:
                    process.wait()
            except BaseException:
                process.kill()
                process.wait()
                raise
        tails = []
        for stream in (stdout, stderr):
            end = stream.seek(0, os.SEEK_END)
            stream.seek(max(0, end - 4000))
            tails.append(stream.read().decode("utf-8", errors="replace"))
        return process.returncode, peak, *tails


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
        returncode, peak, stdout_tail, stderr_tail = _run_worker(command)
        if output.is_file():
            row = json.loads(output.read_text())
        else:
            row = dict(
                env=name,
                batch=batch,
                seed=args.seed,
                simulation_backend=backend,
                status="error",
                error_type="BenchmarkWorkerError",
                error=f"Worker exited {returncode} without a result",
            )
        if returncode and row.get("status") == "ok":
            row.update(status="error", error_type="BenchmarkWorkerError", error=f"Worker exited {returncode}")
        exit_signal = None
        if returncode < 0 and os.name == "posix":
            try:
                exit_signal = signal.Signals(-returncode).name
            except ValueError:
                exit_signal = str(-returncode)  # e.g. unnamed POSIX real-time signals
        row["worker_exit"] = {"returncode": returncode, "signal": exit_signal}
        row["worker_peak_rss_bytes"] = peak
        row["worker_peak_rss_scope"] = (
            "wait4.ru_maxrss across the worker lifetime, including startup, construction and all measured phases; "
            "Linux may include the largest reaped descendant, not the concurrent process-tree sum; "
            "null when unavailable. A signal alone does not establish an OOM cause."
        )
        if row.get("status") != "ok":
            row["worker_stdout_tail"] = stdout_tail
            row["worker_stderr_tail"] = stderr_tail
        row["process_isolation"] = "fresh process per environment/batch/seed/backend; worker exits after this record"
        return row


def _check_paired_inputs(records, row):
    """Reject a successful timing comparison if its backend inputs differ."""
    if row["status"] != "ok":
        return
    peer = next(
        (
            previous
            for previous in records
            if previous["status"] == "ok"
            and all(previous[key] == row[key] for key in ("env", "batch", "seed"))
            and previous["simulation_backend"] != row["simulation_backend"]
        ),
        None,
    )
    if peer is None:
        return
    keys = ("actions_sha256", "initial_state_sha256")
    differences = [key for key in keys if peer["metadata"]["inputs"][key] != row["metadata"]["inputs"][key]]
    peer["paired_inputs_match"] = row["paired_inputs_match"] = not differences
    if differences:
        row.update(
            status="error",
            error_type="BenchmarkInputMismatch",
            error=f"Native/MJWarp inputs differ: {', '.join(differences)}",
        )


def benchmark_versions():
    versions = {}
    for name in ("psutil", "nvidia-ml-py", "mujoco", "mujoco-warp", "warp-lang", "dm-control"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_common_arguments(parser, batches=True)
    parser.set_defaults(
        envs=list(CELLS), batches=list(DEFAULT_BATCHES), device="cuda", seed=None, warmup=20, repeats=100, size=32
    )
    parser._option_string_actions["--envs"].help = "comma-separated supported DMC environment names (default: all 17)"
    parser._option_string_actions["--batches"].help = "comma-separated batch sizes (default: 1,16,64,256,1024)"
    parser._option_string_actions["--seed"].type = common.nonnegative_int
    parser._option_string_actions["--seed"].help = "one reset seed; replaces the default three-seed sweep"
    parser._option_string_actions[
        "--warmup"
    ].help = "untimed action-prefix calls before checkpoint restore (default: 20)"
    parser._option_string_actions[
        "--repeats"
    ].help = "synchronized timed calls after checkpoint restore (default: 100)"
    parser._option_string_actions["--size"].help = "render size (default: 32, as in the PR harness), or native"
    parser.add_argument(
        "--seeds",
        type=_seeds,
        help="comma-separated reset seeds (default: 91000,91010,91020); cannot combine with --seed",
    )
    parser.add_argument(
        "--extended-batches",
        action="store_true",
        help="append 2048,4096,8192 to the selected batches, without duplicates",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="write the requested matrix without checking CUDA or launching workers"
    )
    parser.add_argument(
        "--simulation-backends", type=common.names, default=["native", "mjwarp"], help="native,mjwarp (default: both)"
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.seed is not None and args.seeds is not None:
        parser.error("Use either --seed or --seeds")
    args.seeds = [args.seed] if args.seed is not None else args.seeds or list(DEFAULT_SEEDS)
    if len(args.seeds) == 1:
        args.seed = args.seeds[0]
    if args.extended_batches:
        args.batches = list(dict.fromkeys([*args.batches, *EXTENDED_BATCHES]))
    unknown = set(args.envs) - set(CELLS)
    if unknown:
        parser.error(f"Unknown DMC cells: {sorted(unknown)}")
    if set(args.simulation_backends) - {"native", "mjwarp"}:
        parser.error("--simulation-backends accepts native,mjwarp")
    if args._worker and (
        args.dry_run
        or len(args.envs) != 1
        or len(args.batches) != 1
        or len(args.seeds) != 1
        or len(args.simulation_backends) != 1
    ):
        parser.error("A benchmark worker requires one environment, batch, seed and backend, and cannot use --dry-run")
    matrix = _matrix(args)
    if args.dry_run:
        common.write_json(
            {
                "benchmark": "dmc-mjwarp",
                "dry_run": True,
                "protocol": _protocol(args),
                "record_count": len(matrix),
                "records": matrix,
            },
            args.output,
            pretty=args.pretty,
        )
        return 0
    if set(args.envs) - REGISTRY.keys():
        parser.error("Install stable-envs[dmcontrol-gpu] to benchmark the requested DMC cells")
    device = common.checked_device(args.device)
    if device.type != "cuda":
        parser.error("This benchmark requires actual CUDA; native baseline returns on the same CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    args.device = str(device)
    torch.cuda.set_device(device)
    common.configure_threads(args.threads)
    if args._worker:
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
        protocol=_protocol(args),
        candidate_files={
            path.relative_to(ROOT).as_posix(): sha256(path) for path in (ROOT / "stable_envs").rglob("*.py")
        },
        benchmark_files={
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in (Path(__file__).resolve(), ROOT / "benchmarks/_common.py")
        },
        benchmark_versions=benchmark_versions(),
        process_isolation="one fresh worker per environment/batch/seed/backend; parent hardware context may remain",
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
    for cell in matrix:
        worker_args = argparse.Namespace(**{**vars(args), "seed": cell["seed"]})
        name, batch, backend, seed = (cell[key] for key in ("env", "batch", "simulation_backend", "seed"))
        try:
            row = isolated_record(name, batch, backend, worker_args)
            _check_paired_inputs(record["records"], row)
        except Exception as exc:
            row = dict(**cell, **common.error_record(exc))
        failures += row["status"] != "ok"
        record["records"].append(row)
        if args.output != "-":
            common.write_json(record, args.output, pretty=args.pretty)
        print(f"{name} batch={batch} seed={seed} {backend}: {row['status']}", file=sys.stderr, flush=True)
    record["complete"] = True
    common.finish_provenance(record)
    common.write_json(record, args.output, pretty=args.pretty)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
