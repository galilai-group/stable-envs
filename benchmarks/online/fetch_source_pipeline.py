"""Resource-matched Fetch source/public online timings; Oscar compute nodes only.

Source P1/P8 and library run in separate interpreters. Source State snapshots,
restoration and independent correctness oracles never enter online timing.
Rendered modes require a passing same-device registration-specific image gate.
PickAndPlace is explicitly limited to no-frame rows until its render lifecycle
protocol is resolved. The mandatory library final State remains in timing.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

import stable_envs as se
from benchmarks.online.common import equal_state, state_hash
from benchmarks.online.dmc_source_pipeline import require_gl, task_hash
from benchmarks.online.fetch_parallel_source import FetchSource, ParallelFetchSource, import_source
from benchmarks.online.fetch_source_frame_gate import compare
from benchmarks.online.fetch_source_transport_gate import tasks
from tests.helpers.reference.fetch_semantics import source_state

MODES = ("source_serial", "source_parallel", "ours")
REGISTRATIONS = [
    f"swm/Fetch{task}{suffix}-v3"
    for task in ("Reach", "Push", "Slide", "PickAndPlace")
    for suffix in ("", "Dense", "Dict")
]


def copy_state_to_device(state, device):
    """Measure owned leaf copies plus packing, including a real copy on CPU."""
    return TensorDict(
        {key: value.to(device=device, copy=True) for key, value in state.items(True, True)},
        batch_size=state.batch_size,
        device=device,
    )


def require_image_gate(path, registration, device, source_manifest, *, candidate_root=None, versions=None):
    """Reject missing, stale, sibling, wrong-device or partial image evidence."""
    candidate_root = Path.cwd() if candidate_root is None else Path(candidate_root)
    gate = json.loads(path.read_text())
    assert gate["status"] == "PASS" and gate["registration"] == registration and gate["device"] == device
    assert gate["source_pin"] == source_manifest["commit"] and gate["source_files"] == source_manifest["files"]
    assert gate["batch"] == 4 and gate["source_workers"] == [1, 2]
    assert gate["source_internal_size"] == 480 and gate["output_size"] == 64
    required = {"benchmarks/online/fetch_source_frame_gate.py", "benchmarks/online/fetch_parallel_source.py"}
    assert required <= gate["candidate_files"].keys()
    assert set(gate["versions"]) == {"numpy", "mujoco", "gymnasium", "gymnasium-robotics", "torch", "tensordict"}
    for rel, digest in gate["candidate_files"].items():
        assert hashlib.sha256((candidate_root / rel).read_bytes()).hexdigest() == digest, rel
    if versions is None:
        versions = {name: importlib.metadata.version(name) for name in gate["versions"]}
    assert gate["versions"] == versions
    require_gl(gate["parent_gl"], device)
    expected = {
        (t, s, tuple(limits))
        for t in (1, 8)
        for s in ("none", "empty", "final", "all", "reverse_sparse")
        for limits in ([50] * 4, [1, 3, 2, 50])
    }
    seen = set()
    for cell in gate["cases"]:
        key = cell["horizon"], cell["selection"], tuple(cell["limits"])
        assert key not in seen
        seen.add(key)
        for field in (
            "source_parallel_exact",
            "source_render_nonmutation",
            "source_full_state_tasks",
            "output_ownership",
            "terminal_padding_state_hold",
        ):
            assert cell[field] == "PASS"
        if cell["selection"] in ("none", "empty"):
            assert cell["frames_checked"] == 0 and cell["mse_max"] is None
        else:
            selected = (
                1
                if cell["selection"] == "final"
                else (cell["horizon"] if cell["selection"] == "all" else len(range(cell["horizon"])[::-2]))
            )
            assert cell["frames_checked"] == 2 * 4 * selected
            assert 0 <= cell["mse_mean"] <= cell["mse_max"] <= 0.001
            assert math.isfinite(cell["mse_sd"]) and cell["mse_sd"] >= 0
        assert [len(group) for group in cell["source_gl"]] == [1, 2]
        for group in cell["source_gl"]:
            for identity in group:
                require_gl(identity, device)
    assert seen == expected and len(gate["worker_closures"]) == 20
    closures = set()
    for pool in gate["worker_closures"]:
        key = pool["horizon"], pool["selection"], len(pool["workers"])
        assert key not in closures
        closures.add(key)
        assert all(row["exitcode"] == 0 and not row["alive"] for row in pool["workers"])
    assert closures == {
        (t, s, p) for t in (1, 8) for s in ("none", "empty", "final", "all", "reverse_sparse") for p in (1, 2)
    }
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", "--registration", dest="registration", choices=REGISTRATIONS, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--devices", choices=("cpu", "cuda"), required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--image-gate", type=Path)
    p.add_argument("--batches", type=int, nargs="+", default=[64])
    p.add_argument("--verify-gl", action="store_true")
    p.add_argument("--gates-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if platform.system() != "Linux" or not os.environ.get("SLURM_JOB_ID") or not args.verify_gl:
        p.error("Requires an Oscar compute allocation and --verify-gl; no local GL")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 8 or any(b < 64 or b % 8 for b in args.batches) or len(set(args.batches)) != len(args.batches):
        p.error("Require eight allocated cores and unique batches >=64 divisible by eight")
    rendered = "PickAndPlace" not in args.registration
    if rendered != (args.image_gate is not None):
        p.error("Reach/Push/Slide require --image-gate; PickAndPlace is no-frame only without it")
    if args.output.exists():
        raise FileExistsError(args.output)
    import_source(args.source_root)
    from benchmarks.online.fetch_source_render_lifecycle import PIN, differences
    from tests.helpers.reference.fetch_compare import full_compare

    assert importlib.metadata.version("mujoco") == "3.11.0"
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest["commit"] == PIN
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256((args.source_root / rel).read_bytes()).hexdigest() == digest, rel
    device = args.devices
    gate_hash = require_image_gate(args.image_gate, args.registration, device, manifest) if rendered else None
    torch.set_num_threads(8 if device == "cpu" else 1)
    if device == "cuda":
        assert "3090" in torch.cuda.get_device_name()
    files = [
        *Path("benchmarks/online").glob("fetch*source*.py"),
        Path("benchmarks/online/classic_parallel_source.py"),
        Path("benchmarks/online/classic_source_pipeline.py"),
        Path("benchmarks/online/dmc_source_pipeline.py"),
        Path("benchmarks/online/common.py"),
        Path("tests/helpers/reference/fetch_semantics.py"),
        Path("tests/helpers/reference/fetch_compare.py"),
        *Path("stable_envs").rglob("*.py"),
    ]
    report = dict(
        status="RUNNING",
        registration=args.registration,
        mode=args.mode,
        device=device,
        host=platform.node(),
        job=os.environ["SLURM_JOB_ID"],
        pid=os.getpid(),
        affinity=affinity,
        gpu=torch.cuda.get_device_name() if device == "cuda" else None,
        versions={
            n: importlib.metadata.version(n)
            for n in ("numpy", "mujoco", "gymnasium", "gymnasium-robotics", "torch", "tensordict")
        },
        source_pin=PIN,
        source_files=manifest["files"],
        candidate_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        source_runtime_files={},
        image_gate_sha256=gate_hash,
        rendered_protocol=rendered,
        source_internal_size=480 if rendered else None,
        output_size=64,
        backend="opengl",
        source_gpu_renderer_available=True,
        source_physics_device="cpu",
        source_state_export_in_online_timing=False,
        source_readback_and_output_transfer_in_timing=True,
        ours_mandatory_final_state=True,
        ours_state_history=False,
        source_observations_computed=True,
        parent_torch_threads=torch.get_num_threads(),
        source_threads_per_worker=1,
        fresh_strategy_process=True,
        oracle_closed_before_population=True,
        action_policy="seed31 float64 uniform[-1,1], Bx8x4 prefix T1/T8",
        initial_state_policy="seed17+lane; restore full initial State before each sample outside timing",
        gate_only=args.gates_only,
        cells=[],
        components=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def measure(call):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = call()
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start, result

    save()
    try:
        for batch in args.batches:
            for horizon in (1, 8):
                selections = ("none", "final", "all") if rendered else ("none",)
                for selection in selections:
                    indices = (
                        None
                        if selection == "none"
                        else ((horizon - 1,) if selection == "final" else tuple(range(horizon)))
                    )
                    frame_slice = (
                        None if indices is None else (slice(-1, None) if selection == "final" else slice(None))
                    )
                    actions = np.random.default_rng(31).uniform(-1, 1, (batch, 8, 4))[:, :horizon]
                    # Source GL is confined to a child and closed before any measured population.
                    with ParallelFetchSource(
                        args.registration,
                        args.source_root,
                        batch,
                        1,
                        horizon,
                        frame_indices=indices,
                        render_size=64 if rendered else None,
                    ) as oracle:
                        initial_rows = oracle.snapshot()
                        initial = source_state(initial_rows)
                        identities = list(oracle.renderer_info()) if rendered else []
                        for identity in identities:
                            require_gl(identity, device)
                        expected = oracle.step(actions)
                        endpoint_rows = oracle.snapshot()
                        endpoint = source_state(endpoint_rows)
                        expected_later = oracle.step(-actions)
                        later_rows = oracle.snapshot()
                        later_state = source_state(later_rows)
                        report["source_runtime_files"].update(oracle.runtime_files)
                    assert all(not child.is_alive() and child.exitcode == 0 for child in oracle.vector.processes)
                    with contextlib.ExitStack() as stack:
                        if args.mode == "ours":
                            actor = stack.enter_context(
                                se.make(
                                    args.registration.removeprefix("swm/").removesuffix("-v3"),
                                    num_envs=batch,
                                    seed=17,
                                    device=device,
                                    size=64,
                                    backend="opengl",
                                )
                            )
                            assert len(actor._simulator.worker_info) == 8
                            restore = lambda: actor.set_state(initial.to(device))
                            snapshot = actor.get_state
                            call = lambda a=actions: actor.step(
                                a[:, 0] if horizon == 1 else a, states=None, frames=frame_slice
                            )
                            worker_count, actor_gl = 8, []
                        else:
                            if args.mode == "source_serial":
                                actor = stack.enter_context(
                                    contextlib.closing(
                                        FetchSource(
                                            args.registration,
                                            args.source_root,
                                            batch,
                                            render_size=64 if rendered else None,
                                        )
                                    )
                                )
                                call = lambda a=actions: {
                                    k: torch.as_tensor(v, device=device) for k, v in actor.step(a, indices).items()
                                }
                                actor_gl = [actor.renderer_info()] if rendered else []
                                # Cross-check the independent snapshot-to-State projection against
                                # the existing source-field oracle, not just another adapter export.
                                full_compare(initial, actor.envs)
                                worker_count = 1
                            else:
                                actor = stack.enter_context(
                                    ParallelFetchSource(
                                        args.registration,
                                        args.source_root,
                                        batch,
                                        8,
                                        horizon,
                                        frame_indices=indices,
                                        render_size=64 if rendered else None,
                                    )
                                )
                                call = lambda a=actions: actor.step(a, device)
                                actor_gl = list(actor.renderer_info()) if rendered else []
                                report["source_runtime_files"].update(actor.runtime_files)
                                worker_count = 8
                            restore = lambda: actor.restore(initial_rows)
                            snapshot = lambda: source_state(actor.snapshot())
                            assert not differences(initial_rows, actor.snapshot())
                        for identity in actor_gl:
                            require_gl(identity, device)
                        equal_state(snapshot().cpu(), initial)
                        restore()
                        result = call()
                        output = tasks(result, horizon) if args.mode == "ours" else result
                        errors = compare(output, expected, library=args.mode == "ours")
                        equal_state(snapshot().cpu(), endpoint)
                        if args.mode != "ours":
                            assert not differences(endpoint_rows, actor.snapshot())
                        assert bool(output["valid"].all())
                        held = {k: v.clone() for k, v in output.items()}
                        held_state = result["state"].clone() if args.mode == "ours" else None
                        following = call(-actions)
                        compare(
                            tasks(following, horizon) if args.mode == "ours" else following,
                            expected_later,
                            library=args.mode == "ours",
                        )
                        equal_state(snapshot().cpu(), later_state)
                        if args.mode != "ours":
                            assert not differences(later_rows, actor.snapshot())
                        compare(output, held)
                        if held_state is not None:
                            equal_state(result["state"], held_state)
                        cell = dict(
                            batch=batch,
                            horizon=horizon,
                            selection=selection,
                            physics_workers=worker_count,
                            source_gl=identities,
                            actor_gl=actor_gl,
                            initial_state_sha256=state_hash(initial),
                            endpoint_state_sha256=state_hash(endpoint),
                            source_tasks_sha256=task_hash(expected),
                            exact_state_task="PASS",
                            output_ownership="PASS",
                            valid_steps=batch * horizon,
                            frames_checked=batch * len(indices or ()),
                            mse_mean=statistics.mean(errors) if errors else None,
                            mse_sd=statistics.stdev(errors) if len(errors) > 1 else None,
                            mse_max=max(errors) if errors else None,
                            seconds=[],
                        )
                        del (
                            result,
                            output,
                            held,
                            held_state,
                            following,
                            expected,
                            expected_later,
                            endpoint,
                            later_state,
                        )
                        if not args.gates_only:
                            restore()
                            warmed = call()
                            del warmed
                            for _ in range(6):
                                restore()
                                elapsed, result = measure(call)
                                cell["seconds"].append(elapsed)
                                del result
                            if args.mode == "ours" and horizon == 1 and selection == "none":
                                render_input = initial.to(device)
                                operations = {
                                    "task_no_state_with_ipc_transfer": lambda: actor._simulator.advance(actions[:, 0]),
                                    "get_state": actor.get_state,
                                    "state_cpu_to_device": lambda: copy_state_to_device(initial, device),
                                }
                                if rendered:
                                    operations["render_saved_state"] = lambda: actor.render(render_input)
                                for name, fn in operations.items():
                                    restore()
                                    warmed = fn()
                                    del warmed
                                    seconds = []
                                    for _ in range(6):
                                        restore()
                                        elapsed, value = measure(fn)
                                        seconds.append(elapsed)
                                        del value
                                    report["components"].append(dict(batch=batch, operation=name, seconds=seconds))
                        report["cells"].append(cell)
                        save()
                        print(args.registration, args.mode, device, batch, horizon, selection, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
