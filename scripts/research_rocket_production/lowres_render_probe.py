"""Preserve Rocket's low-resolution residual and localize it by source object.

Diagnostic only. Sampling-offset and supersampling trials are recorded without
changing production defaults, source images, or acceptance thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/recovery"))
from rocket_feasibility import RocketLandingEnv as Oracle
from rocket_feasibility import capture, differences

import stable_envs
from stable_envs.envs import rocket_render


def metrics(ours, source):
    per_pixel = np.mean(((ours.astype(np.float64) - source) / 255) ** 2, axis=-1)
    return dict(mse=float(per_pixel.mean()), unequal_pixels=int(np.any(ours != source, axis=-1).sum())), per_pixel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    import pybullet as bullet

    oracle_hash = hashlib.sha256(Path(inspect.getfile(Oracle)).read_bytes()).hexdigest()
    assert oracle_hash == "5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23"
    cases = []
    report = dict(status="RUNNING", cases=cases, job_id=os.environ.get("SLURM_JOB_ID"), oracle_sha256=oracle_hash)
    for size in (96, 64, 128, 224, 480):
        for seed in (17, 0, 42):
            config = dict(render_mode="rgb_array", render_resolution=(size, size))
            source = Oracle(**config)
            cpu = stable_envs.make("PyFlytRocketLanding", device="cpu", size=size, simulator_config=config)
            gpu = stable_envs.make("PyFlytRocketLanding", device="cuda", size=size, simulator_config=config)
            try:
                source.reset(seed=seed)
                cpu.reset(seed=seed)
                action = np.array([0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1], np.float32)
                source.step(action.copy())
                cpu.step(action[None], states=None, frames=None)
                assert not differences(capture(source), capture(cpu._simulator.envs[0].env))
                state = cpu.get_state()
                gpu.set_state(state)
                reference = source.render().copy()
                cpu_frame = cpu.render(state)[0].numpy()
                assert np.array_equal(reference, cpu_frame)
                ours = gpu.render()[0].cpu().numpy()
                baseline, error = metrics(ours, reference)
                client, bodies, _ = rocket_render._cpu_context(cpu)
                projected = cpu.to_render_state(state)
                view, projection = rocket_render.camera_matrices(
                    projected["renderer", "camera_pose"][0].numpy(), bullet
                )
                raw = bullet.getCameraImage(
                    size,
                    size,
                    viewMatrix=view,
                    projectionMatrix=projection,
                    renderer=bullet.ER_TINY_RENDERER,
                    physicsClientId=client,
                )
                # PyBullet builds may return flat buffers. Production reshapes
                # these explicitly; the diagnostic must do the same.
                segmentation_rgb = np.asarray(raw[2], np.uint8).reshape(size, size, 4)[..., :3]
                segmentation_rgb_exact = np.array_equal(segmentation_rgb, reference)
                segments = np.asarray(raw[4], np.int32).reshape(size, size)
                objects = segments & ((1 << 24) - 1)
                regions = {}
                for name, mask in [
                    ("background", segments < 0),
                    *[
                        (name, (segments >= 0) & (objects == body))
                        for name, body in zip(("ground", "rocket", "pad"), bodies, strict=True)
                    ],
                ]:
                    regions[name] = dict(
                        pixels=int(mask.sum()),
                        contribution_to_frame_mse=float(error[mask].sum() / error.size),
                        mse=float(error[mask].mean()) if mask.any() else None,
                    )
                arrays = dict(
                    source=reference,
                    ours=ours,
                    cpu=cpu_frame,
                    source_segments=segments,
                    segmentation_rgb=segmentation_rgb,
                    **{"state/" + "/".join(k): v.cpu().numpy() for k, v in state.items(True, True)},
                )
                trials = []
                if size == 96 and seed == 17:
                    renderer, _ = rocket_render._gpu_context(gpu, size, 1)
                    original_offset = renderer.sample_offset
                    try:
                        for index, offset in enumerate(
                            ((0.0, 0.0), (0.25, 0.25), (0.75, 0.75), (0.5, 0.0), (0.0, 0.5), (1.0, 0.5), (0.5, 1.0))
                        ):
                            renderer.sample_offset = offset
                            frame = gpu.render()[0].cpu().numpy()
                            arrays[f"offset_{index}"] = frame
                            score, _ = metrics(frame, reference)
                            trials.append(dict(kind="sample_offset", value=offset, **score))
                    finally:
                        renderer.sample_offset = original_offset
                    for ss in (2, 4):
                        frame = gpu.render(supersample=ss)[0].cpu().numpy()
                        arrays[f"supersample_{ss}"] = frame
                        score, _ = metrics(frame, reference)
                        trials.append(dict(kind="supersample", value=ss, **score))
                equal_now = gpu.render()[0].cpu().numpy()
                assert np.array_equal(equal_now, ours)
                assert all(torch.equal(v, cpu.get_state()[k]) for k, v in state.items(True, True))
                path = args.output / f"size{size}-seed{seed}.npz"
                np.savez_compressed(path, **arrays)
                cases.append(
                    dict(
                        size=size,
                        seed=seed,
                        baseline=baseline,
                        regions=regions,
                        segmentation_rgb_exact=bool(segmentation_rgb_exact),
                        regions_authoritative=bool(segmentation_rgb_exact),
                        trials=trials,
                        source_cpu_exact=True,
                        state_source_exact=True,
                        artifact=path.name,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
                (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                print(size, seed, baseline, flush=True)
            finally:
                gpu.close()
                cpu.close()
                source.close()
                if source.modified_pad_urdf:
                    Path(source.modified_pad_urdf).unlink(missing_ok=True)
    report.update(
        status="DIAGNOSTIC_COMPLETE_NOT_ACCEPTANCE",
        gpu=torch.cuda.get_device_name(0),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        production_hashes={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in list(ROOT.glob("stable_envs/envs/rocket*.py")) + [ROOT / "stable_envs/render/nvdiff.py"]
        },
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
