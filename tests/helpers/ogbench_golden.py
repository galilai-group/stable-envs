"""Fixed render inputs and exact comparisons for the Cube/Scene GPU gate."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

FIXTURES = Path(__file__).parents[1] / "fixtures" / "ogbench_renderer"
VERSIONS = {"mujoco": "3.17.0", "ogbench": "1.2.1", "mujoco-warp": "3.17.0", "warp-lang": "1.17.0"}
CASES = (("Cube", "mjwarp"), ("Cube", "mjwarp_fidelity"), ("Scene", "mjwarp"))


def profile():
    return {
        "versions": {name: version(name) for name in VERSIONS},
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }


def fixed_states(env):
    state = env.to_render_state().cpu().expand(257).clone()
    phase = torch.arange(len(state)) % 8
    varied = phase >= 4
    state["appearance", "cube.color"][varied] = torch.tensor([[[0.2, 0.6, 0.8]]], dtype=torch.float64)
    state["appearance", "floor.color"][varied] = torch.tensor(
        [[[0.3, 0.5, 0.7], [0.7, 0.4, 0.2]]], dtype=torch.float64
    )
    angle = phase.to(torch.float64) * 0.13
    quat = state["pose", "mocap_quat"]
    quat.zero_()
    quat[:, 0] = torch.cos(angle / 2)
    quat[:, 3] = torch.sin(angle / 2)
    # The final row must differ from the first row to expose a reset chunk cursor.
    quat[-1] = torch.tensor([0.8, 0.0, 0.0, 0.6], dtype=torch.float64)
    if env.CELL == "Scene":
        state["pose", "buttons"][:, 0] = phase % 2
        state["pose", "buttons"][:, 1] = (phase // 2) % 2
    return state


def save(path, state, frames):
    values = {"/".join(key): value.cpu().numpy() for key, value in state.items(True, True)}
    np.savez_compressed(path, frames=frames, **values)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        frames = data["frames"]
        state = TensorDict(
            {tuple(key.split("/")): torch.from_numpy(data[key]) for key in data.files if key != "frames"},
            batch_size=[len(frames)],
        )
    return state, frames


def assert_frames(actual, expected):
    assert actual.dtype == expected.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)


def metadata():
    return json.loads((FIXTURES / "manifest.json").read_text())
