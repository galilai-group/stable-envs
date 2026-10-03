"""Gate automatic Cube/Scene rendering against frozen, explicit backend controls."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ogbench_manipulation_process import equal

import stable_envs as se
from stable_envs.envs.ogbench import manipulation_renderer as renderer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["Cube", "Scene"], required=True)
    p.add_argument("--devices", choices=["cpu", "cuda"], nargs=1, required=True)
    p.add_argument("--verify-gl", action="store_true", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    device = args.devices[0]
    path = Path("../reference/manipulation_renderer.py")
    spec = importlib.util.spec_from_file_location("stable_envs.envs.ogbench._reference_auto", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    manifest = json.loads(Path("../source_manifest.json").read_text())
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256(Path(rel).read_bytes()).hexdigest() == digest, rel
    report = dict(status="RUNNING", domain=args.domain, device=device, host=platform.node(),
                  job=os.environ.get("SLURM_JOB_ID"), cases=[], timings=[],
                  source_sha256=manifest["files"],
                  reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "mujoco", "ogbench")},
                  gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                  scope="default route regression, not a new upstream fidelity or speedup claim")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def control(family, state, output_device, backend="auto", chunk_size=None):
        return reference.render(family, state, output_device,
                                "opengl" if device == "cpu" and backend == "auto" else backend, chunk_size)

    def images(a, b):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.dtype == torch.uint8 and a.device.type == device

    save()
    try:
        import mujoco
        from OpenGL import GL

        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
        with mujoco.Renderer(model, height=32, width=32) as probe:
            probe._gl_context.make_current()
            vendor, gl_device = GL.glGetString(GL.GL_VENDOR).decode(), GL.glGetString(GL.GL_RENDERER).decode()
            software = any(x in gl_device.lower() for x in ("llvmpipe", "softpipe", "swrast"))
            assert software if device == "cpu" else "nvidia" in vendor.lower() and not software
            report["opengl"] = dict(vendor=vendor, renderer=gl_device, software=software)
        with se.make("OGB" + args.domain, num_envs=2, device=device, seed=17) as env:
            assert env.backend == "auto"
            held = []
            for varied in (False, True):
                options = {"variation": [], "variation_values": {"cube.size": [0.025], "agent.color": [0.2, 0.7, 0.4]}}
                state = env.reset(17, options if varied else {"variation": []})
                saved = state.clone()
                for size in (64, env.size):
                    for compact in (False, True):
                        inputs = env.to_render_state(state, size=size) if compact else state
                        actual = env.render(inputs, size=size, chunk_size=1)
                        with patch.object(renderer, "render", control):
                            expected = env.render(inputs, size=size, chunk_size=1)
                        images(actual, expected)
                        out = torch.empty_like(actual)
                        assert env.render(inputs, size=size, out=out) is out
                        images(out, expected)
                        held.append((actual, actual.clone()))
                        report["cases"].append(dict(kind="render", varied=varied, size=size, compact=compact, mse=0))
                equal(state, saved)
                equal(env.get_state(), saved)
                actions = np.random.default_rng(31).uniform(-0.2, 0.2, (2, 8, 5)).astype(np.float64)
                for horizon in (1, 8):
                    action = actions[:, 0] if horizon == 1 else actions
                    for selection in ("default", None, slice(None), slice(None, None, -2), slice(9, 12)):
                        kwargs = {} if selection == "default" else dict(states=selection, frames=selection)
                        env.set_state(saved)
                        with patch.object(renderer, "render", control):
                            expected = env.step(action, **kwargs)
                        env.set_state(saved)
                        actual = env.step(action, **kwargs)
                        equal(actual, expected)
                        report["cases"].append(dict(kind="step", varied=varied, horizon=horizon,
                                                    selection=str(selection), mse=0))
                save()
            for actual, saved_image in held:
                images(actual, saved_image)
            report["owned_frames"] = "PASS"
            # Resident-State render-only samples, warmed and with alternating order.
            state = env.get_state()
            samples = {"reference": [], "auto": []}
            for repeat in range(7):
                for mode in (("reference", "auto") if repeat % 2 == 0 else ("auto", "reference")):
                    with patch.object(renderer, "render", control if mode == "reference" else renderer.render):
                        sync()
                        start = time.perf_counter()
                        frame = env.render(state)
                        sync()
                        elapsed = time.perf_counter() - start
                    if repeat:
                        samples[mode].append(elapsed)
            for mode, seconds in samples.items():
                fps = [len(state) / t for t in seconds]
                report["timings"].append(dict(mode=mode, batch=2, size=env.size, seconds=seconds,
                                              fps_mean=statistics.mean(fps), fps_sd=statistics.stdev(fps)))
            assert frame.device.type == device
        assert len(report["cases"]) == 28
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
