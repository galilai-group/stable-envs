"""Read the sealed Blender oracle and drive MuJoCo without the candidate adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/semantics/occlusion_ball"
MODEL = ROOT / "stable_envs/envs/object_permanence/assets/g34/scene.xml"
# Source is an independently evaluated Blender mesh scene; all 21 measured
# frames have IoU=1 and ROI MSE <= 6e-12 on Mesa EGL. A 1e-8 depth budget is
# 1e-4 scene units RMSE and allows float32 camera/depth readback rounding.
# At the smallest visible target (11 pixels), IoU >= .99 permits no lost pixel.
DEPTH_MSE_MAX = 1e-8
MASK_IOU_MIN = 0.99


def load_oracle():
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    archive = FIXTURE / manifest["file"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
    with np.load(archive, allow_pickle=False) as data:
        arrays = {key: value.copy() for key, value in data.items()}
    assert set(arrays) == set(manifest["arrays"])
    for key, value in arrays.items():
        assert list(value.shape) == manifest["arrays"][key]["shape"]
        assert str(value.dtype) == manifest["arrays"][key]["dtype"]
    return manifest, arrays


class NativeReference:
    """Direct-MuJoCo execution with a separately transcribed source schedule.

    Sharing the declarative XML is intentional. Pose scheduling and state
    extraction do not import candidate model or adapter helpers. ``step()`` has
    no action input: action validation is covered by the adapter unit tests.
    This helper validates engine execution; the Blender fixture validates the
    source geometry and motion independently of this shared XML.
    """

    def __init__(self, batch):
        _, self.source = load_oracle()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.data = [mujoco.MjData(self.model) for _ in range(batch)]
        self.steps = np.zeros(batch, np.int64)
        self.mask = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.width = mujoco.mj_stateSize(self.model, self.mask)
        self.reset()

    @staticmethod
    def _authored_pose(frame_index):
        # Independently transcribed from animate_occ_small_ball_behind_screen.
        # Compare this schedule against evaluated Blender separately: unlike
        # Blender's mesh transforms, MuJoCo retains float64 integration inputs.
        position = np.array([-2.90 + 5.80 * frame_index / 119.0, 0.0, 0.194])
        angle = np.float64(-0.10 * (frame_index + 1))
        return position, np.array([np.cos(angle), 0.0, np.sin(angle), 0.0])

    def reset(self):
        self.steps.fill(0)
        for data in self.data:
            mujoco.mj_resetData(self.model, data)
            data.mocap_pos[0], data.mocap_quat[0] = self._authored_pose(0)
            mujoco.mj_forward(self.model, data)
        return self.get_state()

    def get_state(self):
        rows = np.empty((len(self.data), self.width), np.float64)
        for i, data in enumerate(self.data):
            mujoco.mj_getState(self.model, data, rows[i], self.mask)
        return TensorDict(
            {
                ("physics", "integration"): torch.from_numpy(rows),
                ("lifecycle", "elapsed_steps"): torch.from_numpy(self.steps.copy()[:, None]),
            },
            [len(rows)],
            device="cpu",
        )

    def set_state(self, state):
        for i, data in enumerate(self.data):
            mujoco.mj_setState(self.model, data, state["physics", "integration"][i].numpy(), self.mask)
            mujoco.mj_forward(self.model, data)
        self.steps[:] = state["lifecycle", "elapsed_steps"][:, 0].numpy()

    def step(self):
        for i, data in enumerate(self.data):
            for _ in range(4):
                mujoco.mj_step(self.model, data)
            self.steps[i] += 1
            frame_index = self.steps[i]
            data.mocap_pos[0], data.mocap_quat[0] = self._authored_pose(frame_index)
            mujoco.mj_forward(self.model, data)
        return self.get_state()
