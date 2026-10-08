"""Public Rocket trajectory/replay and optional frozen GPU renderer acceptance."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.helpers.reference.rocket_semantics import RocketLandingEnv as Oracle
from tests.helpers.reference.rocket_semantics import capture, differences

import stable_envs
from stable_envs._adapters.rocket.spaces import nodes
from stable_envs.envs import rocket_landing as frozen
from stable_envs.testing.conformance import pixel_agreement, ssim


def equal(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    bad = [str(k) for k in a.keys(True, True) if not torch.equal(a[k].cpu(), b[k].cpu())]
    assert not bad, bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--render-cpu", action="store_true", help="Also gate the current public CPU renderer")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    records, frames, oracle_frames, projections, timings = [], [], [], [], []
    result = dict(status="RUNNING", records=records, job_id=os.environ.get("SLURM_JOB_ID"))
    device = "cuda" if args.gpu else "cpu"
    render_images = args.gpu or args.render_cpu
    resolution = 480 if render_images else 64
    versions = {k: md.version(k) for k in ["PyFlyt", "pybullet", "numpy", "gymnasium", "torch", "tensordict"]}
    production_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(ROOT.glob("stable_envs/_adapters/rocket*.py"))
        + list(ROOT.glob("stable_envs/envs/rocket*.py"))
        + [ROOT / "stable_envs/render/nvdiff.py", ROOT / "stable_envs/envs/_rocket_raster.py"]
    }
    oracle_path = Path(inspect.getfile(Oracle))
    result["oracle_source"] = {
        "path": str(oracle_path),
        "sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
    }
    assert result["oracle_source"]["sha256"] == "5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23"
    active = {}
    start = time.perf_counter()

    def check(e, oracle, dest, label, seed, dtype):
        assert not differences(capture(oracle), capture(e._simulator.envs[0].env))
        saved = e.get_state()
        frame = oracle.render().copy()
        assert np.array_equal(frame, e._simulator.envs[0].env.render())
        projected = e.to_render_state(saved)
        equal(saved, e.get_state())
        begin = time.perf_counter()
        dest.set_state(saved)
        elapsed = time.perf_counter() - begin
        equal(saved, dest.get_state())
        assert np.array_equal(frame, dest._simulator.envs[0].env.render())
        equal(projected, dest.to_render_state())
        if render_images:
            # Dynamics execute on CPU. At every checkpoint, exercise CUDA State
            # restore, full-State render, projected render, and restored rendering.
            # Intermediate source trajectories need not copy every full cache to
            # CUDA; that would measure transfer overhead, not additional physics.
            cuda_saved = dest.get_state()
            ours = dest.render(saved)
            assert torch.equal(ours, dest.render(dest.to_render_state(cuda_saved)))
            assert torch.equal(ours, dest.render())
            equal(cuda_saved, dest.get_state())
            equal(saved, e.get_state())
            metrics = pixel_agreement(ours.cpu().numpy(), frame[None])
            metrics["ssim"] = ssim(ours.cpu().numpy(), frame[None])
            metrics["normalized_mse"] = float(
                np.mean(((ours.cpu().numpy().astype(np.float64) - frame[None].astype(np.float64)) / 255) ** 2)
            )
            frames.append(ours.cpu().numpy()[0])
            oracle_frames.append(frame)
            projections.append(projected.cpu())
        else:
            metrics = None
        records.append(
            dict(
                label=label,
                seed=seed,
                dtype=dtype,
                public_step=oracle.step_count,
                physical_actions=int(saved["history", "length"][0]),
                post_terminal=int(saved["lifecycle", "post_terminal_steps"][0]),
                contact=float(oracle.landing_pad_contact),
                success=bool(oracle.info["env_complete"]),
                terminated=bool(oracle.termination),
                truncated=bool(oracle.truncation),
                state_bytes=sum(v.numel() * v.element_size() for v in saved.values(True, True)),
                exact_source_numeric=True,
                independent_restore_exact=True,
                immediate_pixels_exact=True,
                metrics=metrics,
            )
        )
        timings.append(dict(label=label, replay_seconds=elapsed))
        # Independent future continuation, then both restore their original checkpoint.
        for i in range(3):
            action = np.array([[0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1]], dtype=dtype)
            equal(e.step(action, states=None, frames=None), dest.step(action, states=None, frames=None))
        e.set_state(saved)
        dest.set_state(cuda_saved if render_images else saved)
        if render_images:
            assert torch.equal(ours, dest.render())
        equal(saved, e.get_state())
        # Invalid State must not affect the engine, including after partial staging.
        corrupt = saved.clone()
        corrupt["current", "wrapper", "reward"][0] += 1
        try:
            dest.set_state(corrupt)
        except ValueError:
            pass
        else:
            raise AssertionError("Inconsistent State accepted")
        equal(saved, dest.get_state())
        bad = np.array([[0, 0, 0, 0, 0, 2, 0]], dtype=dtype)
        for _ in range(3):
            try:
                dest.step(bad, states=None, frames=None)
            except AssertionError:
                pass
            else:
                raise AssertionError("Invalid action accepted")
        equal(saved, dest.get_state())
        equal(
            e.step(np.zeros((1, 7), dtype=dtype), states=None, frames=None),
            dest.step(np.zeros((1, 7), dtype=dtype), states=None, frames=None),
        )
        e.set_state(saved)
        # Keep destination's differing suffix history until the next independent
        # restore. A redundant second restore here adds no coverage.

    try:
        for seed in [0, 17, 42]:
            for dtype in ["float32", "float64"]:
                scenarios = [
                    ("fall", {}, {}, [0, 1, 100, 300, 301, 302, 303, 310]),
                    ("mixed", {}, {}, [0, 1, 12, 100]),
                    ("success", {"ceiling": 3.2}, {"accelerate_drop": False}, [0, 1, 6, 7, 12]),
                    (
                        "horizon",
                        {"ceiling": 500000.0},
                        {"accelerate_drop": False},
                        [0, 300, 600, 900, 1201, 1202, 1204],
                    ),
                    ("all", {}, {"variation": ["all"]}, [0, 1, 12]),
                ]
                for scenario, config, options, checkpoints in scenarios:
                    active = dict(seed=seed, dtype=dtype, scenario=scenario, config=config, options=options)
                    config = dict(config, render_mode="rgb_array", render_resolution=(resolution, resolution))
                    oracle = Oracle(**config)
                    e = stable_envs.make(
                        "PyFlytRocketLanding", device="cpu", size=resolution, simulator_config=config, seed=seed
                    )
                    dest = stable_envs.make(
                        "PyFlytRocketLanding", device=device, size=resolution, simulator_config=config, seed=42
                    )
                    try:
                        oracle.reset(seed=seed, options=options)
                        e.reset(seed=seed, options=options)
                        for step in range(max(checkpoints) + 1):
                            if step:
                                action = (
                                    np.zeros(7, dtype=dtype)
                                    if scenario not in ["mixed", "all"]
                                    else np.array([0.15, -0.1, 0.2, 1, 0.5, 0.1, -0.1], dtype=dtype)
                                )
                                src = oracle.step(action.copy())
                                out = e.step(action[None], states=None, frames=None)
                                assert out["reward"][0].item() == src[1]
                                assert out["terminated"][0].item() == bool(src[2])
                                assert out["truncated"][0].item() == bool(src[3])
                                assert out["success"][0].item() == bool(src[4]["env_complete"])
                            if step in checkpoints:
                                active["step"] = step
                                check(e, oracle, dest, scenario + "_" + str(step), seed, dtype)
                        print("trajectory", seed, dtype, scenario, "PASS", flush=True)
                    finally:
                        for env in [oracle, e, dest]:
                            env.close()
                        if oracle.modified_pad_urdf:
                            Path(oracle.modified_pad_urdf).unlink(missing_ok=True)
        # Each leaf, each parent, and parent/leaf overlap; independent combined all above.
        e = stable_envs.make(
            "PyFlytRocketLanding",
            device="cpu",
            size=resolution,
            simulator_config={"render_mode": "rgb_array", "render_resolution": (resolution, resolution)},
        )
        dest = stable_envs.make(
            "PyFlytRocketLanding",
            device=device,
            size=resolution,
            simulator_config={"render_mode": "rgb_array", "render_resolution": (resolution, resolution)},
        )
        oracle = Oracle(render_mode="rgb_array", render_resolution=(resolution, resolution))
        try:
            selections = [[p[1:].replace("/", ".")] for p, _ in nodes(oracle.variation_space) if p]
            selections += [["rocket", "rocket.body_color"], ["environment", "environment.start_tilt"]]
            for seed in [0, 17, 42]:
                for selection in selections:
                    active = dict(seed=seed, variation=selection)
                    oracle.reset(seed=seed, options={"variation": selection})
                    e.reset(seed=seed, options={"variation": selection})
                    check(e, oracle, dest, "variation_" + "_".join(selection), seed, "float64")
        finally:
            for env in [oracle, e, dest]:
                env.close()
            if oracle.modified_pad_urdf:
                Path(oracle.modified_pad_urdf).unlink(missing_ok=True)
        if render_images:
            batch = torch.cat(projections)
            e = stable_envs.make("PyFlytRocketLanding", device=device, size=resolution)
            try:
                together = e.render(batch.to(device)).cpu().numpy()
                assert np.array_equal(together, np.stack(frames))
                assert torch.equal(e.render(batch.to(device), chunk_size=7), torch.from_numpy(together).to(device))
            finally:
                e.close()
            groups = {}
            for label in sorted({r["label"] for r in records}):
                indices = [i for i, r in enumerate(records) if r["label"] == label]
                ours, reference = np.stack(frames)[indices], np.stack(oracle_frames)[indices]
                metrics = pixel_agreement(ours, reference)
                metrics["ssim"] = ssim(ours, reference)
                limits = frozen.PyFlytRocketLanding.visual_fidelity
                groups[label] = dict(
                    count=len(indices),
                    metrics=metrics,
                    passed=metrics["mean_abs"] <= limits["mean_abs_max"]
                    and metrics["within8_frac"] >= limits["within8_min"]
                    and metrics["ssim"] >= limits["ssim_min"],
                )
            np.savez_compressed(args.output / "frames.npz", ours=np.stack(frames), oracle=np.stack(oracle_frames))
            result["groups"] = groups
            result["gpu"] = torch.cuda.get_device_name(0) if args.gpu else None
            result["device"] = device
            result["per_frame_mse_pass"] = all(r["metrics"]["normalized_mse"] <= 0.001 for r in records)
            result["status"] = (
                ("PUBLIC_GPU_PASS" if args.gpu else "PUBLIC_CPU_RENDER_PASS")
                if all(g["passed"] for g in groups.values()) and result["per_frame_mse_pass"]
                else "FROZEN_RENDERER_ACCEPTANCE_BLOCKER"
            )
        else:
            result["status"] = "PUBLIC_CPU_TRAJECTORIES_PASS"
    except BaseException as error:
        result.update(status="CORRECTNESS_FAILURE", active=active, error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(
            versions=versions,
            elapsed_seconds=time.perf_counter() - start,
            restore_diagnostics=timings,
            production_hashes=production_hashes,
        )
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result["status"], len(records), flush=True)
    if result["status"] not in {"PUBLIC_GPU_PASS", "PUBLIC_CPU_TRAJECTORIES_PASS", "PUBLIC_CPU_RENDER_PASS"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
