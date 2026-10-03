"""Gate column-wise full-State validation and measure its public pipeline cost.

One ordinary production population is used. The control replaces only the parent
validator with its frozen implementation; worker stepping is unchanged, and
restoration is outside timing. This isolates validation without doubling worlds.
No upstream speedup is implied. All State checks remain enabled in both modes.
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
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict

import stable_envs as se
from stable_envs._adapters.ogbench_manipulation import OGBManipulationAdapter
from tests.helpers.reference.ogbench_before_validation_columns import OGBManipulationAdapter as Before
from tests.unit.test_ogbench_state_export import equal


def control(mode):
    return (patch.object(OGBManipulationAdapter, "validate", Before.validate)
            if mode == "baseline" else contextlib.nullcontext())


def gate(family, device, batch):
    with se.make("OGB" + family, num_envs=batch, device=device, seed=17, size=64) as env:
        workers = len(env._simulator.worker_info) if hasattr(env._simulator, "worker_info") else 0
        assert workers == (8 if batch >= 64 else 0) and env.backend == "auto"
        initial = env.get_state()
        initial["rng", "environment", "has_uint32"][-1] = 1
        initial["rng", "environment", "uinteger"][-1] = 2**32 - 1
        initial["rng", "task", "has_gauss"][-1] = 1
        initial["rng", "task", "cached_gaussian"][-1] = .25
        actions = np.random.default_rng(31).uniform(-.2, .2, (batch, 8, 5))
        cases = 0
        for horizon in (1, 8):
            action = actions[:, 0].astype(np.float32) if horizon == 1 else actions
            for states, frames in ((None, None), (None, slice(-1, None)),
                    (slice(None), slice(None)), (slice(None, None, -2), slice(None, None, 3)),
                    (slice(9, 12), slice(9, 12))):
                with control("baseline"):
                    env.set_state(initial)
                    expected = env.step(action, states=states, frames=frames)
                env.set_state(initial)
                actual = env.step(action, states=states, frames=frames)
                equal(expected, actual)
                assert bool(actual["valid"].all())
                held = actual.clone()
                env.reset(41)
                equal(actual, held)
                cases += 1
        for key, value in ((("physics", "time"), -1), (("rng", "task", "position"), 625),
                           (("model", "compiled", "cube.size"), .3)):
            bad = initial.clone()
            bad[key][-1] = value
            before = env.get_state()
            errors = []
            for mode in ("baseline", "candidate"):
                with control(mode):
                    for operation in (env.render, env.set_state):
                        try:
                            operation(bad)
                        except (ValueError, TypeError) as exc:
                            errors.append((type(exc).__name__, str(exc)))
                        else:
                            raise AssertionError("Malformed final-lane State accepted")
            assert len(set(errors)) == 1 and len(errors) == 4
            equal(before, env.get_state())
        resets = []
        for mode in ("baseline", "candidate"):
            with control(mode):
                env.set_state(initial)
                resets.append(env.reset())
        equal(*resets)
        env.set_variation({"cube.size": [.025], "agent.color": [.2, .7, .4]})
        varied = env.reset(42)
        with control("baseline"):
            expected = env.step(torch.as_tensor(actions, device=device))
        env.set_state(varied)
        actual = env.step(torch.as_tensor(actions, device=device))
        equal(expected, actual)
        noncontiguous = actual["state"].repeat_interleave(2, dim=0)[::2]
        with control("baseline"):
            expected_frame = env.render(noncontiguous)
        torch.testing.assert_close(expected_frame, env.render(noncontiguous), rtol=0, atol=0)
        if workers:
            env._simulator.submit_step(actions[:, 0], capture_state=False)
            assert env._simulator._pending is not None
            for mode in ("baseline", "candidate"):
                with control(mode):
                    env._simulator.validate(noncontiguous)
            env._simulator.receive_step()
        return dict(device=device, batch=batch, workers=workers, cases=cases + 2,
                    complete_state_task_rng_lifecycle_ownership="PASS", frame_mse_vs_baseline=0,
                    rejected_render_restore="PASS", reset_variation_device_actions="PASS",
                    validation_during_inflight="PASS" if workers else "NOT_APPLICABLE")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=("Cube", "Scene"), required=True)
    p.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    p.add_argument("--batches", nargs="+", type=int, default=[64, 128])
    p.add_argument("--gate-batch", type=int, default=64)
    p.add_argument("--verify-gl", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.verify_gl and len(args.devices) != 1:
        p.error("One GL output mode per process required")
    torch.set_num_threads(1)
    files = [*Path("stable_envs/_adapters").glob("ogbench*.py"),
             *Path("stable_envs/envs/ogbench").glob("*.py"), Path("stable_envs/_stepping.py"),
             Path("stable_envs/_adapters/process_requests.py"),
             Path("tests/helpers/reference/ogbench_before_validation_columns.py"),
             Path("tests/unit/test_ogbench_validation_columns.py"), Path(__file__).relative_to(Path.cwd())]
    report = dict(status="RUNNING", family=args.domain, host=platform.node(), job=os.environ.get("SLURM_JOB_ID"),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "mujoco", "ogbench")},
                  source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                  scope="same population, frozen parent validator vs candidate; workers unchanged; not upstream speedup",
                  gates=[], components=[], pipeline=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def sync(device):
        if device == "cuda":
            torch.cuda.synchronize()

    def paired(fn, restore, device, count):
        samples = {"baseline": [], "candidate": []}
        for block in range(2):
            for mode in (("baseline", "candidate") if block == 0 else ("candidate", "baseline")):
                with control(mode):
                    restore()
                    fn()
                    for _ in range(3):
                        restore()
                        sync(device)
                        start = time.perf_counter()
                        result = fn()
                        sync(device)
                        samples[mode].append(time.perf_counter() - start)
                        if isinstance(result, TensorDict) and "valid" in result.keys():
                            assert bool(result["valid"].all())
                        del result
        return [dict(mode=mode, seconds=seconds, count=count,
                     rate_mean=statistics.mean(count / s for s in seconds),
                     rate_sd=statistics.stdev(count / s for s in seconds)) for mode, seconds in samples.items()]

    save()
    try:
        if args.verify_gl:
            import mujoco
            from OpenGL import GL

            model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
            with mujoco.Renderer(model, height=32, width=32) as probe:
                probe._gl_context.make_current()
                vendor, renderer = GL.glGetString(GL.GL_VENDOR).decode(), GL.glGetString(GL.GL_RENDERER).decode()
                software = any(x in renderer.lower() for x in ("llvmpipe", "softpipe", "swrast"))
                assert software if args.devices == ["cpu"] else "nvidia" in vendor.lower() and not software
                report["opengl"] = dict(vendor=vendor, renderer=renderer, software=software)
        for device in args.devices:
            report["gates"].append(gate(args.domain, device, args.gate_batch))
            save()
            for batch in args.batches:
                with se.make("OGB" + args.domain, num_envs=batch, device=device, seed=17, size=64) as env:
                    initial = env.get_state()
                    actions = np.random.default_rng(31).uniform(-.2, .2, (batch, 8, 5))
                    for operation, fn in (("validate_full", lambda: env._simulator.validate(initial)),
                                          ("project_full", lambda: env.to_render_state(initial)),
                                          ("render_saved_full", lambda: env.render(initial))):
                        for cell in paired(fn, lambda: None, device, batch):
                            report["components"].append(dict(device=device, batch=batch, operation=operation, **cell))
                        save()
                    for horizon in (1, 8):
                        a = actions[:, 0] if horizon == 1 else actions
                        for selection, frames in (("none", None), ("final", slice(-1, None)), ("all", slice(None))):
                            with control("baseline"):
                                env.set_state(initial)
                                expected = env.step(a, frames=frames)
                            env.set_state(initial)
                            actual = env.step(a, frames=frames)
                            equal(expected, actual)
                            assert bool(actual["valid"].all())
                            del expected, actual

                            def step():
                                return env.step(a, frames=frames)

                            for cell in paired(step, lambda: env.set_state(initial), device, batch * horizon):
                                report["pipeline"].append(dict(device=device, batch=batch, horizon=horizon,
                                    selection=selection, emitted_frames=batch * (horizon if selection == "all" else int(selection == "final")),
                                    exact_state_task_frames="PASS", **cell))
                            save()
                print(args.domain, device, batch, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
