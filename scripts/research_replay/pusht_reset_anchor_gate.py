"""Research proof: reconstruct PushT reset physics from resolved numeric inputs.

Unlike seeded replay, this also exercises fresh-entropy reset(None). The probe
uses official source operations and typed values, not engine serialization. It
does not define the public State schema or certify a production adapter.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
from pusht_replay_gate import PIN, action, diagnostics, equal, import_source


def nodes(space, path=""):
    yield path, space
    for name, child in getattr(space, "spaces", {}).items():
        yield from nodes(child, path + "/" + name)


def anchor(env, options):
    values = copy.deepcopy(env.variation_space.value)
    start = np.concatenate(
        [
            values["agent"]["start_position"],
            values["block"]["start_position"],
            [values["block"]["angle"]],
            values["agent"]["velocity"],
        ]
    )
    options = options or {}
    return dict(
        values=values,
        state=np.asarray(options.get("state", start)).copy(),
        goal=env.goal_state.copy(),
        gym_rng=copy.deepcopy(env.np_random.bit_generator.state),
        gym_seed=env._np_random_seed,
        rng=copy.deepcopy(env.rng.bit_generator.state),
        spaces={
            path: dict(
                rng=copy.deepcopy(space._np_random.bit_generator.state) if space._np_random is not None else None,
                initial=copy.deepcopy(space.init_value) if not hasattr(space, "spaces") else None,
            )
            for path, space in nodes(env.variation_space)
        },
    )


def reconstruct(cls, saved, config):
    env = cls(**config)
    try:
        for path, space in nodes(env.variation_space):
            if not hasattr(space, "spaces"):
                space.set_init_value(copy.deepcopy(saved["spaces"][path]["initial"]))
        env.reset(
            seed=0,
            options=dict(
                variation=[],
                variation_values=copy.deepcopy(saved["values"]),
                state=saved["state"].copy(),
                goal_state=saved["goal"].copy(),
            ),
        )
        env.np_random.bit_generator.state = copy.deepcopy(saved["gym_rng"])
        env._np_random_seed = saved["gym_seed"]
        env.rng.bit_generator.state = copy.deepcopy(saved["rng"])
        for path, space in nodes(env.variation_space):
            rng = saved["spaces"][path]["rng"]
            if rng is None:
                space._np_random = None
            else:
                space.np_random.bit_generator.state = copy.deepcopy(rng)
        return env
    except BaseException:
        env.close()
        raise


def encode(value):
    if isinstance(value, dict):
        return {key: encode(child) for key, child in value.items()}
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        assert array.dtype.kind in "biuf"
        return dict(
            numeric_type="scalar" if isinstance(value, np.generic) else "array",
            dtype=array.dtype.str,
            shape=list(array.shape),
            values=array.tolist(),
        )
    assert value is None or type(value) in (int, float, str, bool), type(value)
    return value


def decode(value):
    if isinstance(value, dict):
        if "numeric_type" in value:
            result = np.asarray(value["values"], dtype=value["dtype"]).reshape(value["shape"])
            return result[()] if value["numeric_type"] == "scalar" else result
        return {key: decode(child) for key, child in value.items()}
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip() == PIN
    args.output.mkdir(parents=True, exist_ok=False)
    classes = import_source(args.source.resolve())
    report = dict(
        status="RUNNING",
        cases=[],
        production_acceptance=False,
        source_pin=PIN,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={name: md.version(name) for name in ("numpy", "pymunk", "gymnasium", "pygame")},
        helper_sha256=hashlib.sha256(Path(__file__).with_name("pusht_replay_gate.py").read_bytes()).hexdigest(),
    )

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for variant, cls in zip(("continuous", "discrete"), classes, strict=True):
            for seed in (0, 17, 42, None):
                for selection in ("default", "all", "explicit"):
                    options = None if selection == "default" else {"variation": ["all"]}
                    if selection == "explicit":
                        options = dict(
                            variation=[],
                            state=np.array([250, 220, 250, 250, 0, 0, 0], np.float64),
                            goal_state=np.array([250, 220, 250, 250, 0, 0, 0], np.float64),
                        )
                    config = dict(resolution=64)
                    source = cls(**config)
                    dest = None
                    try:
                        source.reset(seed=seed, options=copy.deepcopy(options))
                        saved = anchor(source, options)
                        label = f"{variant}-{seed}-{selection}"
                        recipe = args.output / (label + "-anchor.json")
                        recipe.write_text(json.dumps(encode(saved), indent=2, allow_nan=False) + "\n")
                        # Reconstruct from the actual persisted numeric recipe.
                        saved = decode(json.loads(recipe.read_text()))
                        begin = time.perf_counter()
                        dest = reconstruct(cls, saved, config)
                        elapsed = time.perf_counter() - begin
                        equal(diagnostics(source), diagnostics(dest), "reset-anchor")
                        source_frames, dest_frames = [source.render().copy()], [dest.render().copy()]
                        equal(source_frames[-1], dest_frames[-1])
                        controls, source_obs, dest_obs, source_tasks, dest_tasks = [], [], [], [], []
                        for step in range(100):
                            control = action(source, variant, step)
                            controls.append(control.astype(np.float64))
                            a, b = source.step(control.copy()), dest.step(control.copy())
                            equal(a, b)
                            equal(diagnostics(source), diagnostics(dest))
                            source_obs.append(a[0]["state"])
                            dest_obs.append(b[0]["state"])
                            source_tasks.append([a[1], a[2], a[3], a[4]["n_contacts"]])
                            dest_tasks.append([b[1], b[2], b[3], b[4]["n_contacts"]])
                            if step % 10 == 0:
                                source_frames.append(source.render().copy())
                                dest_frames.append(dest.render().copy())
                                equal(source_frames[-1], dest_frames[-1])
                        artifact = args.output / (label + ".npz")
                        np.savez_compressed(
                            artifact,
                            controls=controls,
                            source_observations=source_obs,
                            dest_observations=dest_obs,
                            source_tasks=source_tasks,
                            dest_tasks=dest_tasks,
                            source_frames=source_frames,
                            dest_frames=dest_frames,
                        )
                        report["cases"].append(
                            dict(
                                variant=variant,
                                seed=seed,
                                selection=selection,
                                future_steps=100,
                                exact=True,
                                reconstruct_seconds=elapsed,
                                artifact=artifact.name,
                                anchor=recipe.name,
                                anchor_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
                                artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                            )
                        )
                        save()
                        print(label, "PASS", flush=True)
                    finally:
                        source.close()
                        if dest is not None:
                            dest.close()
        report["status"] = "PASS_BOUNDED_RESET_ANCHORS"
    except BaseException:
        report["status"] = "FAIL"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    main()
