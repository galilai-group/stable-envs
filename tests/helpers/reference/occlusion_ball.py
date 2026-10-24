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
# Source is an independently evaluated Blender mesh scene. A 1e-8 depth budget is
# 1e-4 scene units RMSE and allows float32 camera/depth readback rounding.
# At the smallest visible target (11 pixels), IoU >= .99 permits no lost pixel.
DEPTH_MSE_MAX = 1e-8
MASK_IOU_MIN = 0.99
# Retain the original certification scope of scene-specific historical backends.
# The shared recorded-scene renderer is tested against all 120 frames separately.
LEGACY_GEOMETRY_FRAMES = (1, 30, 42, 43, 44, 45, 46, 47, 48, 55, 60, 65, 73, 74, 75, 76, 77, 78, 79, 90, 120)


def load_oracle(*, legacy=False):
    manifest = json.loads((FIXTURE / "manifest.json").read_text())
    archive = FIXTURE / manifest["file"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
    with np.load(archive, allow_pickle=False) as data:
        arrays = {key: value.copy() for key, value in data.items()}
    assert set(arrays) == set(manifest["arrays"])
    for key, value in arrays.items():
        assert list(value.shape) == manifest["arrays"][key]["shape"]
        assert str(value.dtype) == manifest["arrays"][key]["dtype"]
    if "depth_xor" in arrays:
        arrays["depth"] = (arrays["depth_xor"] ^ arrays["depth_base"].view(np.uint32)).view(np.float32)
    if legacy:
        selected = np.searchsorted(arrays["geometry_frames"], LEGACY_GEOMETRY_FRAMES)
        for name in ("geometry_frames", "depth", "object_ids"):
            arrays[name] = arrays[name][selected]
    return manifest, arrays


class LegacyBoundaryDifference(RuntimeError):
    """An explicitly bounded historical raster discrepancy, not source parity."""


def assert_legacy_boundary_gap(depths, ids, oracle, expected_frames):
    """Fail normally on new regressions; identify only the measured old gaps."""
    from stable_envs.testing.geometry import depth_mse, mask_iou

    bad = []
    for index, frame in enumerate(oracle["geometry_frames"]):
        reference = oracle["object_ids"][index]
        roi = ((reference >= 4) | (ids[index] >= 4)) & (oracle["depth"][index] > 0)
        mse = float(depth_mse(depths[index], oracle["depth"][index], roi))
        iou = float(mask_iou(ids[index] == 5, reference == 5))
        if mse > DEPTH_MSE_MAX or (np.isfinite(iou) and iou < MASK_IOU_MIN):
            assert int((ids[index] != reference).sum()) == 1
            assert mse < 6e-4  # Bounds the known one-pixel gap; NOT a parity budget.
            bad.append(int(frame))
    if bad:
        assert bad == list(expected_frames)
        raise LegacyBoundaryDifference(f"historical one-pixel source gaps at frames {bad}")


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
