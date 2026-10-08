"""Isolated-strategy DMC online comparison with real source software/NVIDIA GL.

Run each mode/device in a fresh interpreter in the same eight-core allocation.
The serial source oracle is closed before the measured population is constructed.
No source State export/restore or output validation is inside timing regions.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from benchmarks.online.classic_source_pipeline import image_errors
from benchmarks.online.common import PIN, equal_state, state_hash
from benchmarks.online.dmc_parallel_source import (
    NAMES,
    ParallelDMCSource,
    import_source,
    renderer_info,
    source,
    source_step,
)

MODES = ("source_serial", "source_parallel", "ours")


def require_gl(identity, device):
    if device == "cpu":
        assert identity["software"], identity
    else:
        assert "nvidia" in identity["vendor"].lower() and not identity["software"], identity


def task_hash(tasks):
    h = hashlib.sha256()
    for key, value in sorted(tasks.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def tasks_from_public(result, horizon):
    tasks = {}
    for key, value in result.exclude("state", "valid", "frame_valid").items(True, True):
        name = ".".join(key) if isinstance(key, tuple) else key
        tasks[name] = value if horizon != 1 or name == "frames" else value.unsqueeze(1)
    return tasks


def compare_tasks(actual, expected, *, library):
    assert set(actual) == set(expected)
    errors = []
    for key in expected:
        a, b = actual[key].cpu(), expected[key].cpu()
        if key == "frames" and library:
            errors = image_errors(a, b)
            assert all(value <= 0.001 for value in errors)
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=NAMES, required=True)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--devices", choices=["cpu", "cuda"], nargs=1, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[64, 256])
    p.add_argument("--verify-gl", action="store_true")
    p.add_argument("--gates-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if platform.system() != "Linux" or not args.verify_gl:
        p.error("Requires a Linux compute node and --verify-gl; public construction opens GL")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 8 or any(b < 64 or b % 8 for b in args.batches):
        p.error("Require eight allocated cores and batches >=64 divisible by eight")
    device = args.devices[0]
    torch.set_num_threads(8)
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest["commit"] == PIN
    for f, h in manifest["files"].items():
        assert hashlib.sha256((args.source_root / f).read_bytes()).hexdigest() == h, f
    import_source(args.source_root)
    files = [
        Path(__file__).relative_to(Path.cwd()),
        Path("stable_envs/_core/stepping.py"),
        Path("stable_envs/env.py"),
        Path("stable_envs/_execution/requests.py"),
        *Path("stable_envs/envs/dmcontrol").glob("*.py"),
        *Path("stable_envs/envs/dmcontrol/simulation").rglob("*.py"),
        *Path("stable_envs/_execution").rglob("*.py"),
        Path("tests/helpers/reference/dmc_semantics.py"),
        Path("benchmarks/online/dmc_parallel_source.py"),
        Path("benchmarks/online/classic_parallel_source.py"),
        Path("benchmarks/online/classic_source_pipeline.py"),
        Path("benchmarks/online/common.py"),
    ]
    versions = {
        n: importlib.metadata.version(n)
        for n in ("torch", "numpy", "gymnasium", "mujoco", "dm-control", "tensordict", "mujoco-warp", "warp-lang")
    }
    report = dict(
        status="RUNNING",
        domain=args.domain,
        mode=args.mode,
        device=device,
        host=platform.node(),
        job=os.environ.get("SLURM_JOB_ID"),
        pid=os.getpid(),
        affinity=affinity,
        gpu=torch.cuda.get_device_name() if device == "cuda" else None,
        versions=versions,
        size=64,
        source_pin=PIN,
        source_files=manifest["files"],
        source_gpu_supported=True,
        source_physics_device="cpu",
        source_render_device=device,
        source_readback_in_timing=True,
        source_state_export_in_timing=False,
        ours_mandatory_final_state=True,
        ours_state_history=False,
        parent_torch_threads=8,
        source_threads_per_worker=1,
        source_workers=8,
        oracle_closed_before_population=True,
        fresh_strategy_process=True,
        gate_only=args.gates_only,
        initial_state_policy="seed 17 + lane; restore full initial State before every sample",
        action_policy="seed 31, float64 uniform[-0.5,0.5], eight actions per lane",
        candidate_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        source_runtime_files={},
        cells=[],
        components=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def measure(call):
        sync()
        start = time.perf_counter()
        result = call()
        sync()
        return time.perf_counter() - start, result

    save()
    try:
        import mujoco
        from OpenGL import GL

        model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
        )
        with mujoco.Renderer(model, height=32, width=32) as probe:
            probe._gl_context.make_current()
            identity = dict(
                vendor=GL.glGetString(GL.GL_VENDOR).decode(), renderer=GL.glGetString(GL.GL_RENDERER).decode()
            )
            identity["software"] = any(x in identity["renderer"].lower() for x in ("llvmpipe", "softpipe", "swrast"))
            require_gl(identity, device)
            report["opengl"] = identity
        save()
        for batch in args.batches:
            for horizon in (1, 8):
                for selection, indices, frame_slice in (
                    ("none", (), None),
                    ("final", (horizon - 1,), slice(-1, None)),
                    ("all", tuple(range(horizon)), slice(None)),
                ):
                    # Materialize the independent expected result, then close its
                    # simulation/render contexts before constructing the actor.
                    with contextlib.closing(source(args.domain, batch)) as oracle:
                        initial = oracle.get_state()
                        source_gl = renderer_info(oracle)
                        require_gl(source_gl, device)
                        equal_state(initial, oracle.get_state())
                        dim = oracle.envs[0].action_space.shape[0]
                        actions = np.random.default_rng(31).uniform(-0.5, 0.5, (batch, 8, dim))[:, :horizon]
                        expected = source_step(oracle, actions, indices, "cpu")
                        endpoint = oracle.get_state()
                        for wrapper in (oracle.envs[0],):
                            import inspect

                            for cls in (type(wrapper), type(wrapper).__bases__[0], type(wrapper.env._env)):
                                f = Path(inspect.getsourcefile(cls))
                                report["source_runtime_files"][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
                    del oracle, wrapper
                    with contextlib.ExitStack() as stack:
                        if args.mode == "ours":
                            actor = stack.enter_context(
                                se.make(NAMES[args.domain], num_envs=batch, seed=17, device=device, size=64)
                            )
                            workers = (
                                len(actor._simulator.worker_info) if hasattr(actor._simulator, "worker_info") else 0
                            )
                            assert workers == (0 if args.domain == "quadruped" else 8) and actor.backend == "auto"
                            snapshot = actor.get_state
                            restore = lambda: actor.set_state(initial.to(device))
                            call = lambda a=actions: actor.step(
                                a[:, 0] if horizon == 1 else a, states=None, frames=frame_slice
                            )
                            identities = []
                        elif args.mode == "source_serial":
                            actor = source(args.domain, batch)
                            stack.callback(actor.close)
                            workers = 1
                            snapshot = actor.get_state
                            restore = lambda: actor.set_state(initial)
                            call = lambda a=actions: source_step(actor, a, indices, device)
                            identities = [renderer_info(actor)]
                        else:
                            actor = stack.enter_context(
                                ParallelDMCSource(args.domain, args.source_root, batch, 8, horizon, indices)
                            )
                            workers = len(actor.vector.processes)
                            assert workers == 8
                            snapshot = actor.snapshot
                            restore = lambda: actor.restore(initial)
                            call = lambda a=actions: actor.step(a, device)
                            identities = list(actor.renderer_info())
                            assert len(identities) == 8
                            report["source_runtime_files"].update(actor.runtime_files)
                        for identity in identities:
                            require_gl(identity, device)
                        equal_state(initial, snapshot())
                        restore()
                        result = call()
                        if args.mode == "ours":
                            assert bool(result["valid"].all())
                            if indices:
                                assert bool(result["frame_valid"].all())
                            equal_state(result["state"], endpoint)
                            tasks = tasks_from_public(result, horizon)
                        else:
                            tasks = result
                        equal_state(snapshot(), endpoint)
                        errors = compare_tasks(tasks, expected, library=args.mode == "ours")
                        held = {k: v.clone() for k, v in tasks.items()}
                        held_state = result["state"].clone() if args.mode == "ours" else None
                        later = call(-actions)
                        later_tasks = tasks_from_public(later, horizon) if args.mode == "ours" else later
                        assert not torch.equal(later_tasks["observation"], held["observation"])
                        compare_tasks(tasks, held, library=False)
                        if held_state is not None:
                            equal_state(result["state"], held_state)
                        cell = dict(
                            batch=batch,
                            horizon=horizon,
                            selection=selection,
                            source_gl=source_gl,
                            actor_gl=identities,
                            physics_workers=workers,
                            initial_state_sha256=state_hash(initial),
                            endpoint_state_sha256=state_hash(endpoint),
                            source_tasks_sha256=task_hash(expected),
                            exact_state_task="PASS",
                            output_ownership="PASS",
                            frames_checked=batch * len(indices),
                            valid_steps=batch * horizon,
                            mse_mean=statistics.mean(errors) if errors else None,
                            mse_sd=statistics.stdev(errors) if len(errors) > 1 else (0 if errors else None),
                            mse_max=max(errors) if errors else None,
                            seconds=[],
                        )
                        del result, tasks, held, held_state, later, later_tasks, expected, endpoint
                        if not args.gates_only:
                            restore()
                            warmed = call()
                            del warmed
                            for _ in range(6):
                                restore()
                                elapsed, result = measure(call)
                                if args.mode == "ours":
                                    assert bool(result["valid"].all())
                                cell["seconds"].append(elapsed)
                                del result
                            if args.mode == "ours" and horizon == 1 and selection == "none":
                                render_input = initial.to(device)
                                ops = {
                                    "task_no_state_with_ipc_transfer": lambda: actor._simulator.step(
                                        actions[:, 0], capture_state=False
                                    ),
                                    "get_state": actor.get_state,
                                    "render_saved_state": lambda: actor.render(render_input),
                                }
                                for operation, fn in ops.items():
                                    restore()
                                    warmed = fn()
                                    del warmed
                                    seconds = []
                                    for _ in range(6):
                                        restore()
                                        elapsed, result = measure(fn)
                                        seconds.append(elapsed)
                                        del result
                                    report["components"].append(
                                        dict(batch=batch, operation=operation, seconds=seconds)
                                    )
                        report["cells"].append(cell)
                        save()
                        print(args.domain, device, args.mode, batch, horizon, selection, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
