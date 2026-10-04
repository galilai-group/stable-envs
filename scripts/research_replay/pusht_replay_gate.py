"""Seeded PushT replay feasibility on the pinned, unmodified upstream engine.

Research only: this is not a production State serializer or a promotion gate.
The untouched source advances normally. Independent destinations are rebuilt
from a reset recipe and the original typed public actions, including contact
history. No native field setters, engine copies, or opaque snapshots are used.
Unseeded anchors, arbitrary persistent Variation, batching, and atomic public
validation remain separate production obligations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np

PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"


def import_source(root):
    # Avoid importing unrelated optional families from the upstream package.
    for name, sub in (
        ("stable_worldmodel", ""),
        ("stable_worldmodel.envs", "envs"),
        ("stable_worldmodel.envs.pusht", "envs/pusht"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(root / "stable_worldmodel" / sub)]
        sys.modules[name] = package
    sys.modules["stable_worldmodel"].spaces = importlib.import_module("stable_worldmodel.spaces")
    sys.modules["stable_worldmodel"].utils = importlib.import_module("stable_worldmodel.utils")
    return (
        importlib.import_module("stable_worldmodel.envs.pusht.env").PushT,
        importlib.import_module("stable_worldmodel.envs.pusht.env_discrete").PushTDiscrete,
    )


def equal(a, b, path="root"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for key in a:
            equal(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), path
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            equal(left, right, f"{path}/{index}")
    else:
        np.testing.assert_array_equal(a, b, err_msg=path, strict=True)


def diagnostics(env):
    bodies = {}
    for name in ("agent", "block"):
        body = getattr(env, name)
        bodies[name] = {
            key: np.asarray(getattr(body, key)).copy()
            for key in (
                "position",
                "velocity",
                "angle",
                "angular_velocity",
                "force",
                "torque",
                "center_of_gravity",
                "mass",
                "moment",
            )
        }

    def spaces(space):
        result = {"rng": copy.deepcopy(space._np_random.bit_generator.state) if space._np_random is not None else None}
        if hasattr(space, "spaces"):
            result["children"] = {key: spaces(child) for key, child in space.spaces.items()}
        else:
            result["value"] = copy.deepcopy(space.value)
            result["initial"] = copy.deepcopy(space.init_value)
        return result

    return dict(
        bodies=bodies,
        observation=env._get_obs().copy(),
        goal=env.goal_state.copy(),
        latest_action=copy.deepcopy(env.latest_action),
        contacts=env.n_contact_points,
        goal_image=env._goal.copy(),
        rng=copy.deepcopy(env.rng.bit_generator.state),
        gym_rng=copy.deepcopy(env.np_random.bit_generator.state),
        variation=spaces(env.variation_space),
    )


def action(env, variant, step):
    if variant == "discrete":
        # Includes both exact index boundaries; uses the actual source quantizer.
        return np.array([step % 16, (step // 8) % 16], np.int64)
    if step % 64 < 32:
        value = np.clip((np.asarray(env.block.position) - np.asarray(env.agent.position)) / 100, -1, 1)
    else:
        value = np.array([(-1.0, 1.0)[(step // 16) % 2], (1.0, -1.0)[(step // 32) % 2]])
    return value.astype(np.float32 if step % 2 == 0 else np.float64)


def rebuild(cls, seed, options, actions, codes):
    result = cls(resolution=64)
    try:
        result.reset(seed=seed, options=copy.deepcopy(options))
        for row, code in zip(actions, codes, strict=True):
            result.step(row.astype({1: np.float32, 2: np.float64, 3: np.int64}[int(code)]))
        return result
    except BaseException:
        result.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[0, 1, 100, 300, 1025, 10240])
    args = parser.parse_args()
    root = args.source.resolve()
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() == PIN
    assert md.version("pymunk") == "7.2.0"
    args.output.mkdir(parents=True, exist_ok=False)
    classes = import_source(root)
    source_files = [
        "spaces.py",
        "utils.py",
        "envs/utils.py",
        "envs/quantizer.py",
        "envs/pusht/env.py",
        "envs/pusht/env_discrete.py",
    ]
    hashes = {}
    for rel in source_files:
        path = "stable_worldmodel/" + rel
        data = (root / path).read_bytes()
        assert data == subprocess.check_output(["git", "-C", str(root), "show", PIN + ":" + path])
        hashes[path] = hashlib.sha256(data).hexdigest()
    report = dict(
        status="RUNNING",
        source_pin=PIN,
        source_files=hashes,
        versions={
            name: md.version(name) for name in ("numpy", "pymunk", "gymnasium", "pygame", "opencv-python-headless")
        },
        platform=platform.platform(),
        python=sys.version,
        cases=[],
        production_acceptance=False,
        source_never_restored=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for variant, cls in zip(("continuous", "discrete"), classes, strict=True):
            for seed in (0, 17, 42):
                for scenario in ("default", "combined", "success"):
                    options = None
                    if scenario != "default":
                        start = np.array([250, 220, 250, 250, 0, 0, 0], np.float64)
                        options = dict(
                            variation=[],
                            state=start,
                            goal_state=start.copy()
                            if scenario == "success"
                            else np.array([200, 200, 300, 300, 0, 0, 0], np.float64),
                        )
                    lengths = sorted(set(args.lengths if scenario == "default" else [0, 1, 100]))
                    source = cls(resolution=64)
                    history, codes = [], []
                    source.reset(seed=seed, options=copy.deepcopy(options))
                    contacts = successes = 0
                    try:
                        for length in lengths:
                            # Each checkpoint has its own untouched source so
                            # suffix validation cannot consume the next prefix.
                            if history:
                                source.close()
                                source = cls(resolution=64)
                                source.reset(seed=seed, options=copy.deepcopy(options))
                                history, codes = [], []
                                contacts = successes = 0
                            while len(history) < length:
                                a = action(source, variant, len(history))
                                history.append(a.astype(np.float64))
                                codes.append(
                                    {np.dtype("float32"): 1, np.dtype("float64"): 2, np.dtype("int64"): 3}[a.dtype]
                                )
                                out = source.step(a.copy())
                                contacts += int(out[4]["n_contacts"] > 0)
                                successes += int(out[2])
                            prefix = np.asarray(history, np.float64).reshape(-1, 2)
                            dtypes = np.asarray(codes, np.uint8)
                            # Exercise an unrelated environment before independent reconstruction.
                            divergent = cls(resolution=64)
                            divergent.reset(seed=seed + 101)
                            for step in range(16):
                                divergent.step(action(divergent, variant, step))
                            divergent.close()
                            begin = time.perf_counter()
                            dest = rebuild(cls, seed, options, prefix, dtypes)
                            restore_seconds = time.perf_counter() - begin
                            label = f"{variant}-{seed}-{scenario}-{length}"
                            try:
                                equal(diagnostics(source), diagnostics(dest))
                                before = diagnostics(source)
                                first = source.render().copy()
                                equal(before, diagnostics(source), "source-render-nonmutation")
                                equal(first, dest.render())
                                suffix, suffix_codes, source_observations, dest_observations = [], [], [], []
                                source_frames, dest_frames, source_tasks, dest_tasks = (
                                    [first],
                                    [dest.render().copy()],
                                    [],
                                    [],
                                )
                                # Source keeps advancing. No restore/rebuild is applied to it.
                                for _ in range(32):
                                    a = action(source, variant, len(history))
                                    history.append(a.astype(np.float64))
                                    codes.append(
                                        {np.dtype("float32"): 1, np.dtype("float64"): 2, np.dtype("int64"): 3}[a.dtype]
                                    )
                                    suffix.append(a.astype(np.float64))
                                    suffix_codes.append(codes[-1])
                                    left, right = source.step(a.copy()), dest.step(a.copy())
                                    equal(left, right, label + "/step")
                                    equal(diagnostics(source), diagnostics(dest), label + "/diagnostics")
                                    source_observations.append(left[0]["state"])
                                    dest_observations.append(right[0]["state"])
                                    source_tasks.append([left[1], left[2], left[3], left[4]["n_contacts"]])
                                    dest_tasks.append([right[1], right[2], right[3], right[4]["n_contacts"]])
                                    source_frames.append(source.render().copy())
                                    dest_frames.append(dest.render().copy())
                                    equal(source_frames[-1], dest_frames[-1], label + "/frame")
                                artifact = args.output / (label + ".npz")
                                np.savez_compressed(
                                    artifact,
                                    prefix=prefix,
                                    dtype_codes=dtypes,
                                    suffix=np.asarray(suffix),
                                    suffix_dtype_codes=np.asarray(suffix_codes, np.uint8),
                                    source_observations=source_observations,
                                    dest_observations=dest_observations,
                                    source_tasks=source_tasks,
                                    dest_tasks=dest_tasks,
                                    source_frames=source_frames,
                                    dest_frames=dest_frames,
                                )
                                report["cases"].append(
                                    dict(
                                        variant=variant,
                                        seed=seed,
                                        scenario=scenario,
                                        requested_length=length,
                                        actual_length=len(prefix),
                                        future_steps=32,
                                        contact_steps_before=contacts,
                                        success_steps_before=successes,
                                        exact_outputs=True,
                                        exact_frames=True,
                                        exact_diagnostics=True,
                                        replay_seconds=restore_seconds,
                                        artifact=artifact.name,
                                        artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                                    )
                                )
                                save()
                                print(label, len(prefix), "PASS", flush=True)
                            finally:
                                dest.close()
                    finally:
                        source.close()
        report["status"] = "PASS_BOUNDED_SEEDED_REPLAY"
    except BaseException:
        report["status"] = "FAIL"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    main()
