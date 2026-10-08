import importlib
from contextlib import ExitStack, closing

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray

from tests.helpers.reference.bridges import load_source, task_columns
from tests.helpers.reference.fetch_semantics import differences, snapshot, source_state
from tests.helpers.reference.isolated import isolated_call
from tests.helpers.reference.pixels import PixelTrace
from tests.helpers.reference.profiles import Registration


def _image(source: gym.Env, device: str) -> tuple[NDArray[np.uint8], dict]:
    from OpenGL import GL

    raw = source.unwrapped
    viewer = raw.mujoco_renderer._viewers.get("rgb_array")
    if viewer is not None:
        # Gymnasium only switches contexts between render modes within one env.
        viewer.make_context_current()
    frame = np.asarray(source.render()).copy()
    raw.mujoco_renderer._viewers["rgb_array"].make_context_current()
    vendor = GL.glGetString(GL.GL_VENDOR).decode()
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    software = any(tag in renderer.lower() for tag in ("llvmpipe", "softpipe", "swrast"))
    identity = dict(
        vendor=vendor,
        renderer=renderer,
        software=software,
        internal_width=raw.width,
        internal_height=raw.height,
        output_size=64,
    )
    assert software if device == "cpu" else "nvidia" in vendor.lower() and not software, identity
    assert raw.width == raw.height == 480, identity
    assert frame.shape == (64, 64, 3) and frame.dtype == np.uint8
    return frame, identity


def _render(source: gym.Env, device: str) -> tuple[NDArray[np.uint8], dict]:
    before = snapshot(source)
    result = _image(source, device)
    delta = differences(before, snapshot(source))
    assert not delta, ("Source rendering changed continuation State", delta)
    return result


def fetch_trace(
    case: Registration, seed: int, dtype: str, horizon: int, scenario: str, device: str, source_root: str
) -> tuple[PixelTrace, list[dict]]:
    function = "_pickandplace_trace" if case.domain == "PickAndPlace" else "_fetch_trace"
    return isolated_call(__name__, function, (case, seed, dtype, horizon, scenario, device, source_root))


def _fetch_trace(
    case: Registration, seed: int, dtype: str, horizon: int, scenario: str, device: str, source_root: str
) -> tuple[PixelTrace, list[dict]]:
    """Compare rendering and no-render source lanes before extracting source-owned State."""
    if case.domain not in ("Reach", "Push", "Slide"):
        raise ValueError("PickAndPlace requires a separate render-mutation protocol")
    load_source(source_root, "fetch")
    importlib.import_module("stable_worldmodel.envs.gymnasium_robotics")
    with ExitStack() as stack:
        physics, rendered = [], []
        for lane in range(2):
            for worlds, options in (
                (physics, {}),
                (rendered, dict(render_mode="rgb_array", resolution=64)),
            ):
                source = stack.enter_context(closing(gym.make(case.registration, **options)))
                worlds.append(source)
                source.reset(seed=seed + lane, options={"variation": []})
        limit = physics[0].spec.max_episode_steps
        assert limit == 50
        prefixes = {"reached": (0, 7), "horizon": (limit - 1, limit - 3)}[scenario]
        for plain, visual, prefix in zip(physics, rendered, prefixes, strict=True):
            for _ in range(prefix):
                action = np.zeros(4, dtype=dtype)
                output = plain.step(action.copy())
                assert not differences(output, visual.step(action.copy()))
                assert not (output[2] or output[3])
            assert not differences(snapshot(plain), snapshot(visual))
        initial = source_state([snapshot(source) for source in physics])
        actions = np.random.default_rng(seed + 1000).uniform(-1, 1, (2, horizon, 4)).astype(dtype)
        active = np.ones(2, dtype=bool)
        states, frames, validity, outputs, identities = [], [], [], [], []
        previous = [None, None]
        for t in range(horizon):
            validity.append(active.copy())
            rows, images = [], []
            for lane, (plain, visual) in enumerate(zip(physics, rendered, strict=True)):
                if active[lane]:
                    output = plain.step(actions[lane, t].copy())
                    assert not differences(output, visual.step(actions[lane, t].copy()))
                    previous[lane] = output
                else:
                    output = previous[lane]
                rows.append(output)
                assert not differences(snapshot(plain), snapshot(visual))
                frame, identity = _render(visual, device)
                assert not differences(snapshot(plain), snapshot(visual))
                images.append(frame)
                if t == 0:
                    identities.append(identity)
                else:
                    assert identity == identities[lane]
            tasks = {
                **task_columns(rows),
                "success": torch.as_tensor(np.asarray([row[4]["is_success"] for row in rows])),
            }
            done = (tasks["terminated"] | tasks["truncated"]).numpy().copy()
            for column in tasks.values():
                column[~active] = 0
            outputs.append(tasks)
            states.append(source_state([snapshot(source) for source in physics]))
            frames.append(np.stack(images))
            active &= ~done
        # A final following transition catches render mutations that become observable only on step.
        for lane, (plain, visual) in enumerate(zip(physics, rendered, strict=True)):
            action = -actions[lane, -1]
            assert not differences(plain.step(action.copy()), visual.step(action.copy()))
            assert not differences(snapshot(plain), snapshot(visual))
            assert not plain.unwrapped.mujoco_renderer._viewers
        if scenario == "horizon":
            np.testing.assert_array_equal(np.stack(validity, axis=1), np.arange(horizon)[None, :] < [[1], [3]])
    return PixelTrace(
        initial,
        states,
        {key: torch.stack([row[key] for row in outputs], dim=1) for key in outputs[0]},
        np.stack(frames, axis=1),
        torch.from_numpy(np.stack(validity, axis=1)),
        actions,
    ), identities


def _pickandplace_trace(
    case: Registration, seed: int, dtype: str, horizon: int, scenario: str, device: str, source_root: str
) -> tuple[PixelTrace, list[dict]]:
    """Render disposable forward-replayed source branches; never mutate the physics trajectory."""
    assert case.domain == "PickAndPlace"
    load_source(source_root, "fetch")
    importlib.import_module("stable_worldmodel.envs.gymnasium_robotics")
    with ExitStack() as stack:
        physics = [stack.enter_context(closing(gym.make(case.registration))) for _ in range(2)]
        histories = [[], []]
        prefixes = {"reached": (0, 7), "horizon": (49, 47)}[scenario]
        for lane, source in enumerate(physics):
            assert source.spec.max_episode_steps == 50
            source.reset(seed=seed + lane, options={"variation": []})
            for _ in range(prefixes[lane]):
                action = np.zeros(4, dtype=dtype)
                output = source.step(action.copy())
                histories[lane].append(action)
                assert not (output[2] or output[3])
        initial = source_state([snapshot(source) for source in physics])
        actions = np.random.default_rng(seed + 1000).uniform(-1, 1, (2, horizon, 4)).astype(dtype)
        active = np.ones(2, bool)
        states, frames, validity, outputs, identities = [], [], [], [], []
        previous = [None, None]
        for step in range(horizon):
            validity.append(active.copy())
            rows, images = [], []
            for lane, source in enumerate(physics):
                if active[lane]:
                    previous[lane] = source.step(actions[lane, step].copy())
                    histories[lane].append(actions[lane, step].copy())
                rows.append(previous[lane])
                before = snapshot(source)
                with closing(gym.make(case.registration, render_mode="rgb_array", resolution=64)) as branch:
                    branch.reset(seed=seed + lane, options={"variation": []})
                    branch_output = None
                    for action in histories[lane]:
                        branch_output = branch.step(action.copy())
                    assert not differences(rows[-1], branch_output)
                    assert not differences(before, snapshot(branch)), (
                        "Disposable source branch did not reach matching State"
                    )
                    image, identity = _image(branch, device)
                    mutation = differences(before, snapshot(branch))
                assert not differences(before, snapshot(source)), "Renderer touched the physics source"
                assert not source.unwrapped.mujoco_renderer._viewers
                images.append(image)
                if step == 0:
                    identities.append(dict(gl=identity, oracle_mode="disposable-forward-source-branch", mutations=[]))
                else:
                    assert identity == identities[lane]["gl"]
                identities[lane]["mutations"].append(dict(step=step, differences=mutation))
            tasks = {
                **task_columns(rows),
                "success": torch.as_tensor(np.asarray([row[4]["is_success"] for row in rows])),
            }
            done = (tasks["terminated"] | tasks["truncated"]).numpy().copy()
            for column in tasks.values():
                column[~active] = 0
            outputs.append(tasks)
            states.append(source_state([snapshot(source) for source in physics]))
            frames.append(np.stack(images))
            active &= ~done
        if scenario == "horizon":
            np.testing.assert_array_equal(np.stack(validity, axis=1), np.arange(horizon)[None, :] < [[1], [3]])
        return PixelTrace(
            initial,
            states,
            {key: torch.stack([row[key] for row in outputs], dim=1) for key in outputs[0]},
            np.stack(frames, axis=1),
            torch.from_numpy(np.stack(validity, axis=1)),
            actions,
        ), identities


def fetch_mutation_witness(case: Registration, seed: int, dtype: str, device: str, source_root: str) -> dict:
    return isolated_call(__name__, "_pickandplace_mutation", (case, seed, dtype, device, source_root))


def _pickandplace_mutation(case: Registration, seed: int, dtype: str, device: str, source_root: str) -> dict:
    """Keep the rejected continuous-render source trajectory as an explicit witness."""
    assert case.domain == "PickAndPlace"
    load_source(source_root, "fetch")
    importlib.import_module("stable_worldmodel.envs.gymnasium_robotics")
    with (
        closing(gym.make(case.registration)) as plain,
        closing(gym.make(case.registration, render_mode="rgb_array", resolution=64)) as visual,
    ):
        for source in (plain, visual):
            source.reset(seed=seed, options={"variation": []})
        actions = np.random.default_rng(seed + 1000).uniform(-1, 1, (9, 4)).astype(dtype)
        for step, action in enumerate(actions[:-1]):
            assert not differences(plain.step(action.copy()), visual.step(action.copy()))
            assert not differences(snapshot(plain), snapshot(visual))
            before = snapshot(visual)
            _, identity = _image(visual, device)
            mutation = differences(before, snapshot(visual))
            if mutation:
                action = actions[step + 1]
                continuation = differences(plain.step(action.copy()), visual.step(action.copy()))
                continuation.update(differences(snapshot(plain), snapshot(visual), "/state"))
                assert continuation, "Recorded render mutation did not change the following transition"
                return dict(step=step, mutation=mutation, continuation=continuation, source_gl=identity)
        raise AssertionError("Expected PickAndPlace continuous-render mutation was not witnessed")

