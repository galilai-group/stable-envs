"""CPU source callback/continuation audit; no GL, images, timing or promotion.

Calls the unmodified upstream _render_callback, which the inspected source
render() invokes before drawing. Compares two independent source worlds with
identical action histories, injecting only that callback into one world. This
isolates its dynamics effect; actual CPU/GPU rendering remains a separate gate.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import platform
import subprocess
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import mujoco
import numpy as np
import stable_worldmodel
import stable_worldmodel.envs  # noqa: F401
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from tests.helpers.reference.fetch_semantics import differences, snapshot

PIN = "21446f1ede6d5284e981bd7b47f432b994e6d812"
IDS = [
    f"swm/Fetch{task}{suffix}-v3"
    for task in ("Reach", "Push", "Slide", "PickAndPlace")
    for suffix in ("", "Dense", "Dict")
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(stable_worldmodel.__file__).resolve().parents[1]
    pin = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert pin == PIN
    assert not subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    assert mujoco.__version__ == "3.11.0", "Match the accepted Oscar Fetch profile; do not patch source functions"
    report = dict(
        status="RUNNING",
        source_pin=PIN,
        source_repository="https://github.com/galilai-group/stable-worldmodel",
        host=platform.node(),
        scope="Actual source CPU callback and following transitions only; no render() or GL invoked",
        no_rendering=True,
        no_timing=True,
        no_candidate_changes=True,
        source_files={},
        runtime_files={},
        call_paths={},
        helper_sha256=hashlib.sha256(Path("tests/helpers/reference/fetch_semantics.py").read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions={n: importlib.metadata.version(n) for n in ("numpy", "mujoco", "gymnasium", "gymnasium-robotics")},
        cases=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        with (
            patch.object(MujocoRenderer, "render", side_effect=AssertionError("No GL in CPU callback audit")),
            patch("mujoco.Renderer", side_effect=AssertionError("No GL in CPU callback audit")),
        ):
            for registration in IDS:
                a, b = gym.make(registration), gym.make(registration)
                try:
                    wrappers, current = [], a
                    while True:
                        wrappers.append(current)
                        if current is current.unwrapped:
                            break
                        current = current.env
                    report["call_paths"][registration] = dict(
                        wrapper_types=[f"{type(e).__module__}.{type(e).__name__}" for e in wrappers],
                        block_gripper=bool(a.unwrapped.block_gripper),
                        callback=inspect.getsource(a.unwrapped._render_callback),
                        robot_render=inspect.getsource(a.unwrapped.render),
                    )
                    for obj in wrappers:
                        for cls in type(obj).__mro__:
                            for name in (
                                "__init__",
                                "reset",
                                "step",
                                "render",
                                "_render_callback",
                                "_step_callback",
                                "_set_action",
                            ):
                                fn = cls.__dict__.get(name)
                                if inspect.isfunction(fn):
                                    path = Path(inspect.getsourcefile(fn))
                                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                                    report["runtime_files"][str(path)] = digest
                                    if path.is_relative_to(source):
                                        report["source_files"][str(path.relative_to(source))] = digest
                    for seed, dtype, mode, prefix in itertools.product(
                        (0, 17, 42), (np.float32, np.float64), ("zero", "random"), (0, 1, 8, 49)
                    ):
                        a.reset(seed=seed, options={"variation": []})
                        b.reset(seed=seed, options={"variation": []})
                        actions = (
                            np.zeros((prefix + 8, 4), dtype)
                            if mode == "zero"
                            else np.random.default_rng(90000 + seed).uniform(-1, 1, (prefix + 8, 4)).astype(dtype)
                        )
                        assert not differences(snapshot(a), snapshot(b))
                        for action in actions[:prefix]:
                            assert not differences(a.step(action), b.step(action))
                        before = snapshot(b)
                        assert not differences(snapshot(a), before)
                        target_before = b.unwrapped.model.site_pos.copy()
                        b.unwrapped._render_callback()
                        immediate = differences(before, snapshot(b))
                        targets_changed = not np.array_equal(target_before, b.unwrapped.model.site_pos)
                        first_state, first_task = None, None
                        traces = []
                        for t, action in enumerate(actions[prefix:], 1):
                            oa, ob = a.step(action), b.step(action)
                            task_delta, state_delta = differences(oa, ob), differences(snapshot(a), snapshot(b))
                            if state_delta and first_state is None:
                                first_state = t
                            if task_delta and first_task is None:
                                first_task = t
                            traces.append(dict(step=t, state_differences=state_delta, task_differences=task_delta))
                        report["cases"].append(
                            dict(
                                registration=registration,
                                seed=seed,
                                action_dtype=np.dtype(dtype).name,
                                mode=mode,
                                prefix_steps=prefix,
                                callback_state_differences=immediate,
                                target_site_visual_changed=targets_changed,
                                first_continuation_state_difference=first_state,
                                first_continuation_task_difference=first_task,
                                continuation=traces,
                            )
                        )
                    save()
                    print(registration, "48 callback/continuation cases recorded", flush=True)
                finally:
                    a.close()
                    b.close()
        assert len(report["cases"]) == 576
        report["status"] = "COMPLETE_DIAGNOSTIC_NOT_ACCEPTANCE"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
