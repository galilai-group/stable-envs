"""Pinned SWM PushT saved-pose rendering only; no physics/State acceptance.

Run capture once, then benchmark the on-disk corpus on CPU and CUDA. Namespace
loading bypasses unrelated SWM data/policy imports, without changing env code.
"""

import argparse
import hashlib
import importlib
import inspect
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from tests.helpers.reference.pusht_semantics import diagnostics, equal

from .common import PIN, profile, save_state, state_hash, write
from .paired_saved import statistics_for


def source_import(root):
    for name, sub in [("stable_worldmodel", ""), ("stable_worldmodel.envs.pusht", "envs/pusht")]:
        module = types.ModuleType(name)
        module.__path__ = [str(root / "stable_worldmodel" / sub)]
        sys.modules[name] = module
    importlib.import_module("stable_worldmodel.utils")
    importlib.import_module("stable_worldmodel.envs")
    package = sys.modules["stable_worldmodel.envs.pusht"]
    package.PushT = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
    package.PushTDiscrete = importlib.import_module("stable_worldmodel.envs.pusht.env_discrete").PushTDiscrete


def make_source(registration):
    import gymnasium as gym

    return gym.make(registration, render_mode="rgb_array", resolution=224)


def pose(raw):
    return np.array([*raw.agent.position, *raw.block.position, raw.block.angle], dtype=np.float64)


def install(raw, value):
    # Angle first avoids Pymunk's centre-of-gravity setter-order displacement.
    raw.block.angle = float(value[4])
    raw.block.position = value[2:4].tolist()
    raw.agent.position = value[:2].tolist()
    raw.space.reindex_shapes_for_body(raw.block)
    raw.space.reindex_shapes_for_body(raw.agent)
    np.testing.assert_allclose(pose(raw), value, atol=1e-10, rtol=0)


def source_hashes(root):
    paths = [
        "envs/__init__.py",
        "envs/pusht/env.py",
        "envs/pusht/env_discrete.py",
        "envs/utils.py",
        "envs/quantizer.py",
        "spaces.py",
        "utils.py",
    ]
    return {p: hashlib.sha256((root / "stable_worldmodel" / p).read_bytes()).hexdigest() for p in paths}


def capture(args):
    args.output.mkdir(parents=True, exist_ok=False)
    env = make_source(args.registration)
    poses, images, records = [], [], []
    try:
        seed, step, reset = 0, 0, True
        while len(poses) < args.frames:
            if reset:
                env.reset(seed=seed)
                env.action_space.seed(10000 + seed)
                step = 0
            raw = env.unwrapped
            before = pose(raw)
            image = env.render().copy()
            np.testing.assert_array_equal(pose(raw), before)
            poses.append(before)
            images.append(image)
            records.append(dict(seed=seed, step=step))
            if step >= 19:
                seed += 1
                reset = True
            else:
                _, _, term, trunc, _ = env.step(env.action_space.sample())
                step += 1
                reset = bool(term or trunc)
                if reset:
                    seed += 1
        path = args.output / "corpus.npz"
        np.savez_compressed(path, poses=np.stack(poses), images=np.stack(images))
        write(
            args.output / "manifest.json",
            dict(
                registration=args.registration,
                source_pin=PIN,
                source_hashes=source_hashes(args.oracle),
                corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                frames=records,
                count=len(poses),
                distinct_poses=len(np.unique(poses, axis=0)),
                resolution=224,
                profile=profile(),
                scope="Default appearance; 20-frame random-action source episodes. Renderer-only pose corpus, not restorable full State.",
            ),
        )
    finally:
        env.close()


def mse(a, b):
    # The same per-frame reduction, with bounded float64 scratch allocation.
    return np.concatenate(
        [
            np.mean(((a[i : i + 32].astype(np.float64) - b[i : i + 32].astype(np.float64)) / 255) ** 2, axis=(1, 2, 3))
            for i in range(0, len(a), 32)
        ]
    )


def check_outputs(images, indices, reference, errors, label):
    """Check every pixel after timing without a whole-corpus float64 temporary."""
    measured, position = [], 0
    for image in images:
        for start in range(0, len(image), 32):
            value = image[start : start + 32].cpu().numpy()
            selected = indices[position : position + len(value)]
            if label == "source":
                np.testing.assert_array_equal(value, reference[selected])
            else:
                item = mse(value, reference[selected])
                np.testing.assert_allclose(item, errors[selected], atol=1e-7, rtol=0)
                measured.append(item)
            position += len(value)
    assert position == len(indices)
    return np.concatenate(measured) if measured else None


def dist(x):
    return dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)), max=float(np.max(x)), count=len(x))


def benchmark(args):
    from PIL import Image

    import stable_envs as se

    args.output.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.corpus / "manifest.json").read_text())
    assert meta["registration"] == args.registration
    assert meta["source_hashes"] == source_hashes(args.oracle)
    corpus = args.corpus / "corpus.npz"
    assert hashlib.sha256(corpus.read_bytes()).hexdigest() == meta["corpus_sha256"]
    with np.load(corpus, allow_pickle=False) as data:
        poses, reference = data["poses"].copy(), data["images"].copy()
    if args.device == "cuda":
        assert "3090" in torch.cuda.get_device_name()
    name = "PushTDiscrete" if "Discrete" in args.registration else "PushT"
    env = se.make(name, device=args.device, size=224)
    result = dict(
        registration=args.registration,
        device=args.device,
        backend="pygame",
        candidate_render_device="cpu",
        status="RUNNING",
        profile=profile(),
        measurements=[],
        independent_states=len(poses),
        independent_render_poses=len(np.unique(poses, axis=0)),
        source_render_device="cpu",
        source_gpu_availability="CPU_ONLY",
        source_gpu_availability_notes="Pinned PushT._render_frame uses pygame.Surface/debug_draw, surfarray and cv2.resize; no GPU renderer/device API. Discrete inherits this method.",
        scope="RENDERER_ONLY: saved source poses and witnessed RenderState; this benchmark does not certify simulation. Default appearance, 224x224, ss=1. Both renderers run on CPU; CUDA selects output storage.",
        source_batch_method="Serial upstream render() over independent source owners, then stack. Owners are reconstructed before timing by seeded replay and checked byte-exact against disk images; this retains upstream collision-debug marks. No replay, rollout, file I/O or input upload timed.",
        candidate_render_api="PushT.render(prepared RenderState loaded from disk); rendering, validation and output transfer included; full-State capture excluded as in the existing table.",
        corpus_sha256=meta["corpus_sha256"],
        source_pin=PIN,
        source_hashes=source_hashes(args.oracle),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        production_sha256={
            str(Path(inspect.getfile(type(env))).relative_to(Path.cwd())): hashlib.sha256(
                Path(inspect.getfile(type(env))).read_bytes()
            ).hexdigest()
        },
        semantic_state_acceptance="NOT_MEASURED",
        renderer_nonmutation_passed=False,
    )
    save = lambda: write(args.output / "result.json", result)
    save()
    owners, projected = [], []
    try:
        # One source owner per pose: preparation is excluded equally from timers.
        for i, value in enumerate(poses):
            source = make_source(args.registration)
            owners.append(source)
            source.reset(seed=meta["frames"][i]["seed"])
            # Each saved source owner starts fresh, including reset-retained fields.
            env.close()
            env = se.make(name, device=args.device, size=224, seed=meta["frames"][i]["seed"])
            source.action_space.seed(10000 + meta["frames"][i]["seed"])
            for _ in range(meta["frames"][i]["step"]):
                action = source.action_space.sample()
                source.step(action.copy())
                env.step(action[None].copy(), frames=None, states=None)
            np.testing.assert_array_equal(pose(source.unwrapped), value)
            np.testing.assert_array_equal(source.render(), reference[i], err_msg=f"source replay frame {i}")
            equal(diagnostics(source.unwrapped), diagnostics(env._simulator.envs[0].env))
            projected.append(env.to_render_state().cpu())
        prepared_path = args.output / "render-states.npz"
        save_state(prepared_path, torch.cat(projected))
        projected.clear()
        with np.load(prepared_path, allow_pickle=False) as data:
            tensor = TensorDict(
                {tuple(json.loads(key)): torch.from_numpy(data[key].copy()).to(args.device) for key in data.files},
                [len(poses)], device=args.device,
            )
        result["prepared_render_states_sha256"] = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
        tensor_hash = state_hash(tensor)
        live_hash = state_hash(env.get_state())
        outputs = []
        for start in range(0, len(poses), 32):
            outputs.append(env.render(tensor[start : start + 32]).cpu().numpy())
        actual = np.concatenate(outputs)
        errors = mse(actual, reference)
        result["paired_mse"] = dist(errors)
        result["frame_mse"] = errors.tolist()
        result["quality_gate"] = {
            "mean_le_0.001": bool(errors.mean() <= 0.001),
            "all_frames_le_0.001": bool(errors.max() <= 0.001),
            "mean_le_0.0005": bool(errors.mean() <= 0.0005),
        }
        for label, i in [("worst", int(errors.argmax())), ("median", int(np.argsort(errors)[len(errors) // 2]))]:
            Image.fromarray(reference[i]).save(args.output / f"{label}-source.png")
            Image.fromarray(actual[i]).save(args.output / f"{label}-ours.png")
            Image.fromarray(np.concatenate([reference[i], actual[i]], axis=1)).save(args.output / f"{label}-pair.png")
            result[f"{label}_frame"] = dict(index=i, **meta["frames"][i], mse=float(errors[i]))
        save()
        sync = torch.cuda.synchronize if args.device == "cuda" else lambda: None
        for batch in args.batches:
            total = max(len(poses), batch)
            assert total % batch == 0
            indices = np.arange(total) % len(poses)
            chunks = [tensor[torch.as_tensor(ids, device=args.device)] for ids in indices.reshape(-1, batch)]
            source_chunks = [[owners[i] for i in ids] for ids in indices.reshape(-1, batch)]
            for destination in ["cpu", "cuda"] if args.device == "cuda" else ["cpu"]:

                def ours():
                    return [env.render(x).to(destination) for x in chunks]

                def original():
                    return [
                        torch.from_numpy(np.stack([s.render() for s in chunk])).to(destination)
                        for chunk in source_chunks
                    ]

                ours()
                original()
                sync()
                samples = {"ours": [], "source": []}
                for repeat in range(args.repeats):
                    for label, fn in (
                        [("ours", ours), ("source", original)]
                        if repeat % 2 == 0
                        else [("source", original), ("ours", ours)]
                    ):
                        sync()
                        start = time.perf_counter_ns()
                        images = fn()
                        sync()
                        samples[label].append((time.perf_counter_ns() - start) / 1e6)
                        # Check every timed output outside timers, all 1024 poses.
                        checked = check_outputs(images, indices, reference, errors, label)
                        if checked is not None:
                            measured = checked
                        del images
                stats = {side: statistics_for(v, total) for side, v in samples.items()}
                for side in stats:
                    stats[side]["fps_sd"] = statistics.stdev(1000 * total / x for x in samples[side])
                result["measurements"].append(
                    dict(
                        batch=batch,
                        frames_per_repeat=total,
                        output_device=destination,
                        **stats,
                        paired_mse=dist(measured),
                        speedup=stats["source"]["mean_ms"] / stats["ours"]["mean_ms"],
                    )
                )
                save()
                print(
                    args.registration, args.device, batch, destination, stats["ours"]["frames_per_second"], flush=True
                )
        for owner, value in zip(owners, poses):
            np.testing.assert_allclose(pose(owner.unwrapped), value, atol=1e-10, rtol=0)
        assert state_hash(env.get_state()) == live_hash
        assert state_hash(tensor) == tensor_hash
        result.update(
            status="COMPLETE",
            renderer_nonmutation_passed=True,
            state_nonmutation_passed=True,
            nonmutation_scope="Candidate live snapshot and input poses; source render poses. No exhaustive source State acceptance claimed.",
        )
        save()
    finally:
        for owner in owners:
            owner.close()
        env.close()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=["capture", "benchmark"])
    p.add_argument("registration", choices=["swm/PushT-v1", "swm/PushT-Discrete-v1"])
    p.add_argument("output", type=Path)
    p.add_argument("--oracle", required=True, type=Path)
    p.add_argument("--corpus", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--frames", type=int, default=1024)
    p.add_argument("--batches", nargs="+", type=int, default=[64, 128, 256])
    p.add_argument("--repeats", type=int, default=7)
    args = p.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    torch.set_num_threads(1)
    source_import(args.oracle)
    (capture if args.mode == "capture" else benchmark)(args)


if __name__ == "__main__":
    main()
