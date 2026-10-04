"""GPU prototype validation and saved-state render timings; no simulation timed."""

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import nvdiffrast.torch as dr
from triangle_raster import rasterize as exact_rasterize

import stable_envs as se
from stable_envs.envs import rocket_render

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1)
original_context = rocket_render._gpu_context
original_rasterize = dr.rasterize
current = []


def context(*args, **kwargs):
    renderer, chunk = original_context(*args, **kwargs)
    if not hasattr(renderer, "_research_capture"):
        clip = renderer.scene.clip

        def capture(*args, **kwargs):
            result = clip(*args, **kwargs)
            current[:] = [result[0], renderer.scene.face]
            return result

        renderer.scene.clip = capture
        renderer._research_capture = True
    return renderer, chunk


def corrected(ctx, pos, faces, resolution, **kwargs):
    assert not kwargs.get("grad_db", False)
    result = exact_rasterize(current[0], current[1], resolution[0])
    return result, torch.zeros_like(result)


raw = np.load(a.input, allow_pickle=False)
prefix = "state/2/"
state = TensorDict(
    {tuple(k[len(prefix) :].split("/")): torch.from_numpy(raw[k]) for k in raw.files if k.startswith(prefix)}, [2]
)
report = {"gpu": torch.cuda.get_device_name(0), "job": os.environ.get("SLURM_JOB_ID"), "timings": []}
rocket_render._gpu_context = context
try:
    for size in (96, 480):
        with se.make(
            "PyFlytRocketLanding",
            num_envs=2,
            device="cuda",
            size=size,
            simulator_config=dict(render_mode="rgb_array", render_resolution=(96, 96)),
        ) as env:
            projected = env.to_render_state(state, size=size)
            for batch in (1, 16, 64):
                saved = torch.cat([projected[i % 2 : i % 2 + 1] for i in range(batch)], 0)
                for label, raster in [("baseline", original_rasterize), ("triangle", corrected)]:
                    dr.rasterize = raster
                    for _ in range(3):
                        env.render(saved)
                    torch.cuda.synchronize()
                    times = []
                    for _ in range(10):
                        start = time.perf_counter()
                        env.render(saved)
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - start)
                    report["timings"].append(
                        dict(
                            size=size,
                            batch=batch,
                            kind=label,
                            seconds_mean=float(np.mean(times)),
                            seconds_sd=float(np.std(times)),
                            frames_per_second=batch / np.mean(times),
                        )
                    )
                    print(report["timings"][-1], flush=True)
    (a.output / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
    dr.rasterize = corrected
    sys.argv = ["online_source_gate.py", "--gpu", "--output", str(a.output / "online")]
    runpy.run_path(str(Path(__file__).with_name("online_source_gate.py")), run_name="__main__")
finally:
    rocket_render._gpu_context = original_context
    dr.rasterize = original_rasterize
