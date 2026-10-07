import importlib
import pickle
import subprocess
import sys
from contextlib import ExitStack, closing
from pathlib import Path
from tempfile import TemporaryDirectory

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray

from tests.helpers.reference.bridges import load_source, task_columns
from tests.helpers.reference.fetch_semantics import differences, snapshot, source_state
from tests.helpers.reference.pixels import PixelTrace
from tests.helpers.reference.profiles import Registration


def _render(source: gym.Env, device: str) -> tuple[NDArray[np.uint8], dict]:
    from OpenGL import GL

    before = snapshot(source)
    raw = source.unwrapped
    viewer = raw.mujoco_renderer._viewers.get("rgb_array")
    if viewer is not None:
        # Gymnasium only switches contexts between render modes within one env.
        viewer.make_context_current()
    frame = np.asarray(source.render()).copy()
    delta = differences(before, snapshot(source))
    assert not delta, ("Source rendering changed continuation State", delta)
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


def fetch_trace(
    case: Registration, seed: int, dtype: str, horizon: int, scenario: str, device: str, source_root: str
) -> tuple[PixelTrace, list[dict]]:
    # Source close() terminates GLFW globally; its contexts must not share our renderer process.
    with TemporaryDirectory(prefix="fetch-source-pixels-") as directory:
        request, response = Path(directory) / "request.pkl", Path(directory) / "response.pkl"
        request.write_bytes(pickle.dumps((case, seed, dtype, horizon, scenario, device, source_root)))
        subprocess.run(
            [sys.executable, "-m", __name__, str(request), str(response)],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            timeout=120,
        )
        return pickle.loads(response.read_bytes())


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


if __name__ == "__main__":
    args = pickle.loads(Path(sys.argv[1]).read_bytes())
    Path(sys.argv[2]).write_bytes(pickle.dumps(_fetch_trace(*args)))
