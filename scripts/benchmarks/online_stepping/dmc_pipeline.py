"""Measure accepted DMC process routes, saved-State rendering and online overlap.

Ordinary public construction uses the default renderer. The deferred control
uses identical fused worker requests but starts each only when received. Thus
it removes overlap without changing the number of exports or messages. Task-only
rates include task packaging/IPC/transfer: they are not bare MuJoCo kernel rates.
This compares library schedules, not upstream implementations.
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
from dmc_state_export import NAMES, equal

import stable_envs as se
from stable_envs._adapters.dmcontrol_execution import process_count
from stable_envs._adapters.dmcontrol_process import DMCProcessAdapter


class Deferred:
    def __init__(self, owner):
        self.owner, self.pending = owner, None

    def submit_step(self, actions, **kwargs):
        assert self.pending is None
        self.pending = actions, kwargs

    def receive_step(self):
        actions, kwargs = self.pending
        self.pending = None
        self.owner.submit_step(actions, **kwargs)
        return self.owner.receive_step()

    def abort_step(self):
        self.owner.abort_step()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def schedule(env, mode):
    return (patch.object(env, "_step_pipeline", lambda: Deferred(env._simulator))
            if mode == "deferred" else contextlib.nullcontext())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=[d for d in NAMES if d != "quadruped"], required=True)
    p.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    p.add_argument("--batches", nargs="+", type=int, default=[64, 256, 1024])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--verify-gl", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2 or min(args.batches) < 64:
        p.error("Require >=2 repeats and batches eligible for accepted process defaults")
    if args.verify_gl and len(args.devices) != 1:
        p.error("Use one GL device mode per interpreter")
    torch.set_num_threads(1)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if process_count(args.domain, min(args.batches), cpus) != 8:
        p.error("This comparison requires the accepted eight-owner Linux default")
    files = list(Path("stable_envs/_adapters").glob("dmcontrol*.py"))
    files += list(Path("stable_envs/envs/dmcontrol").glob("*.py"))
    files += [Path("stable_envs/_stepping.py"), Path("stable_envs/_adapters/process_requests.py"),
              Path(__file__).relative_to(Path.cwd())]
    renderer_versions = {}
    for name in ("mujoco-warp", "warp-lang"):
        try:
            renderer_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            renderer_versions[name] = None
    report = dict(status="RUNNING", domain=args.domain, name=NAMES[args.domain],
                  host=platform.node(), job=os.environ.get("SLURM_JOB_ID"), affinity=cpus,
                  gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "mujoco", "dm-control", "tensordict")},
                  renderer_versions=renderer_versions,
                  source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                  scope="CPU physics, eight process owners; default 64x64 renderer; all States retained; no policy inference or upstream comparison",
                  components=[], pipeline=[], gates=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def measure(fn, device):
        sync(device)
        start = time.perf_counter()
        result = fn()
        sync(device)
        return time.perf_counter() - start, result

    def statistics_for(seconds, count):
        rates = [count / t for t in seconds]
        return dict(seconds=seconds, count_per_sample=count,
                    rate_mean=statistics.mean(rates), rate_sd=statistics.stdev(rates))

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
                save()
        for device in args.devices:
            for batch in args.batches:
                with se.make(NAMES[args.domain], num_envs=batch, seed=17, device=device, size=64) as env:
                    assert isinstance(env._simulator, DMCProcessAdapter) and len(env._simulator.worker_info) == 8
                    assert env.backend == "auto"
                    initial = env.get_state()
                    actions = np.random.default_rng(31).uniform(-.5, .5, (batch, 8, env.action_dim))
                    corpus = env.step(actions, frames=None)["states"]
                    for horizon in (1, 8):
                        a = actions[:, 0] if horizon == 1 else actions

                        def task_only():
                            result = None
                            for t in range(horizon):
                                result = env._simulator.step(actions[:, t], capture_state=False)
                            return result

                        for operation, fn, count in (("task_no_state_with_ipc", task_only, batch * horizon),
                                                      ("state_export_with_ipc_transfer", env.get_state, batch)):
                            seconds = []
                            env.set_state(initial)
                            fn()
                            for _ in range(2 * args.repeats):
                                env.set_state(initial)
                                elapsed, result = measure(fn, device)
                                seconds.append(elapsed)
                                del result
                            report["components"].append(dict(device=device, batch=batch, horizon=horizon,
                                                              operation=operation, **statistics_for(seconds, count)))
                        for selection, frames, indices in (("none", None, ()),
                                                           ("final", slice(-1, None), (horizon - 1,)),
                                                           ("all", slice(None), tuple(range(horizon)))):
                            def render_saved():
                                return torch.stack([env.render(corpus[:, t]) for t in indices], dim=1)

                            if indices:
                                render_saved()
                                seconds = []
                                for _ in range(2 * args.repeats):
                                    elapsed, image = measure(render_saved, device)
                                    seconds.append(elapsed)
                                    del image
                                report["components"].append(dict(device=device, batch=batch, horizon=horizon,
                                    operation="render_saved_" + selection, **statistics_for(seconds, batch * len(indices))))
                            env.set_state(initial)
                            with schedule(env, "deferred"):
                                expected = env.step(a, frames=frames)
                            timings = {"deferred": [], "pipeline": []}
                            for block in range(2):
                                for mode in (("deferred", "pipeline") if block == 0 else ("pipeline", "deferred")):
                                    with schedule(env, mode):
                                        env.set_state(initial)
                                        actual = env.step(a, frames=frames)
                                        equal(expected, actual)
                                        for _ in range(args.repeats):
                                            env.set_state(initial)
                                            elapsed, result = measure(lambda: env.step(a, frames=frames), device)
                                            assert bool(result["valid"].all())
                                            timings[mode].append(elapsed)
                                            del result
                            report["gates"].append(dict(device=device, batch=batch, horizon=horizon,
                                selection=selection, complete_state_task="PASS", frame_mse_vs_deferred=0 if indices else None))
                            for mode, seconds in timings.items():
                                report["pipeline"].append(dict(device=device, batch=batch, horizon=horizon,
                                    selection=selection, mode=mode, emitted_frames=batch * len(indices),
                                    steps=batch * horizon, **statistics_for(seconds, batch * horizon),
                                    fps_mean=statistics.mean([batch * len(indices) / t for t in seconds]) if indices else None,
                                    fps_sd=statistics.stdev([batch * len(indices) / t for t in seconds]) if indices else None))
                            save()
                        print(args.domain, device, batch, horizon, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
