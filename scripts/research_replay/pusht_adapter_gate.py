"""Independent source gate for the private numeric PushT replay batch candidate.

This is a CPU simulator/replay gate, not public API or GPU renderer acceptance.
The source engines are advanced normally and are never restored from candidate
State. NPZ witnesses contain only numeric arrays (load with allow_pickle=False).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from pusht_replay_gate import PIN, action, diagnostics, equal, import_source

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from stable_envs._adapters.pusht_replay_batch import PushTReplayAdapter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[0, 1, 100, 300, 1025, 10240])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 17, 42])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == PIN
    classes = import_source(source)
    files = (
        "spaces.py",
        "utils.py",
        "envs/utils.py",
        "envs/quantizer.py",
        "envs/pusht/env.py",
        "envs/pusht/env_discrete.py",
    )
    source_hashes = {}
    for name in files:
        relative = "stable_worldmodel/" + name
        assert (source / relative).read_bytes() == subprocess.check_output(
            ["git", "-C", str(source), "show", PIN + ":" + relative]
        )
        source_hashes[relative] = sha(source / relative)
    report = dict(
        status="RUNNING",
        source_pin=PIN,
        source_never_restored=True,
        public_acceptance=False,
        gpu_renderer_acceptance=False,
        script_sha256=sha(Path(__file__)),
        source_helper_sha256=hashlib.sha256((ROOT / "tests/helpers/reference/pusht_semantics.py").read_bytes()).hexdigest(),
        helper_sha256=sha(Path(__file__).with_name("pusht_replay_gate.py")),
        production_files={
            str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "stable_envs/_adapters").glob("pusht_*.py"))
        },
        source_files=source_hashes,
        versions={
            name: md.version(name)
            for name in ("pymunk", "numpy", "pygame", "opencv-python-headless", "torch", "tensordict")
        },
        cases=[],
    )

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for discrete, cls in enumerate(classes):
            variant = "discrete" if discrete else "continuous"
            for batch in args.batches:
                for seed in args.seeds:
                    for length in args.lengths:
                        name = f"{variant}-b{batch}-s{seed}-t{length}"
                        oracle, candidate, restored = [], None, None
                        try:
                            config = dict(resolution=64)
                            candidate = PushTReplayAdapter(batch, seed=seed, discrete=bool(discrete), config=config)
                            restored = PushTReplayAdapter(
                                batch, seed=seed + 1234, discrete=bool(discrete), config=config
                            )
                            prefixes, prefix_codes = [], []
                            for lane in range(batch):
                                env = cls(**config)
                                oracle.append(env)
                                env.reset(seed=seed + lane)
                                lane_length = max(0, length - lane % 4)
                                controls, codes = [], []
                                for step in range(lane_length):
                                    control = action(env, variant, step + lane)
                                    env.step(control.copy())
                                    candidate.envs[lane].step(control.copy())
                                    controls.append(control.astype(np.float64))
                                    codes.append(
                                        {np.dtype("float32"): 0, np.dtype("float64"): 1, np.dtype("int64"): 3}[
                                            control.dtype
                                        ]
                                    )
                                prefixes.append(np.asarray(controls, np.float64).reshape(-1, 2))
                                prefix_codes.append(np.array(codes, np.uint8))
                                equal(diagnostics(env), diagnostics(candidate.envs[lane].env))
                            state = candidate.get_state()
                            # Restore through numeric disk data, not a live native/Python object.
                            state_path = args.output / (name + "-state.npz")
                            np.savez_compressed(
                                state_path, **{"/".join(k): v.cpu().numpy() for k, v in state.items(True, True)}
                            )
                            import torch
                            from tensordict import TensorDict

                            with np.load(state_path, allow_pickle=False) as data:
                                disk_state = TensorDict(
                                    {tuple(k.split("/")): torch.from_numpy(data[k].copy()) for k in data}, [batch]
                                )
                            start = time.perf_counter()
                            restored.set_state(disk_state)
                            restore_seconds = time.perf_counter() - start
                            source_frames, dest_frames = [], []
                            observations, replay_observations, tasks, replay_tasks, suffix = [], [], [], [], []
                            for lane, env in enumerate(oracle):
                                equal(diagnostics(env), diagnostics(restored.envs[lane].env))
                            for step in range(100):
                                # A batch uses one dtype at a time, alternating float32/64.
                                controls = np.stack([action(env, variant, step + length) for env in oracle])
                                suffix.append(controls.astype(np.float64))
                                expected = [
                                    env.step(control.copy()) for env, control in zip(oracle, controls, strict=True)
                                ]
                                actual = restored.step(controls)
                                source_obs = np.stack([v[0]["state"] for v in expected])
                                target_obs = np.stack([e.env._get_obs() for e in restored.envs])
                                source_task = np.array([[v[1], v[2], v[3], v[4]["n_contacts"]] for v in expected])
                                target_task = np.array(
                                    [
                                        [
                                            actual["reward"][i],
                                            actual["terminated"][i],
                                            actual["truncated"][i],
                                            env.env._get_info()["n_contacts"],
                                        ]
                                        for i, env in enumerate(restored.envs)
                                    ]
                                )
                                equal(source_obs, target_obs)
                                equal(source_task, target_task)
                                for i, env in enumerate(oracle):
                                    equal(diagnostics(env), diagnostics(restored.envs[i].env))
                                observations.append(source_obs)
                                replay_observations.append(target_obs)
                                tasks.append(source_task)
                                replay_tasks.append(target_task)
                                if step % 4 == 0 or step == 99:
                                    left = np.stack([env.render() for env in oracle])
                                    right = np.stack([env.env.render() for env in restored.envs])
                                    equal(left, right)
                                    source_frames.append(left)
                                    dest_frames.append(right)
                            artifact = args.output / (name + "-witness.npz")
                            np.savez_compressed(
                                artifact,
                                suffix=suffix,
                                source_observations=observations,
                                replay_observations=replay_observations,
                                source_tasks=tasks,
                                replay_tasks=replay_tasks,
                                source_frames=source_frames,
                                replay_frames=dest_frames,
                            )
                            report["cases"].append(
                                dict(
                                    variant=variant,
                                    batch=batch,
                                    seed=seed,
                                    max_prefix=length,
                                    future_steps=100,
                                    exact=True,
                                    cpu_private_task_mse=0.0,
                                    restore_seconds=restore_seconds,
                                    restore_timing_is_not_a_throughput_benchmark=True,
                                    state_artifact=state_path.name,
                                    state_sha256=sha(state_path),
                                    artifact=artifact.name,
                                    artifact_sha256=sha(artifact),
                                )
                            )
                            save()
                            print(name, "PASS", flush=True)
                        finally:
                            for env in oracle:
                                env.close()
                            for env in (candidate, restored):
                                if env is not None:
                                    env.close()
        report["status"] = "PASS_PRIVATE_REPLAY_ADAPTER_SOURCE_GATE"
    except BaseException:
        report["status"] = "FAIL"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    main()
