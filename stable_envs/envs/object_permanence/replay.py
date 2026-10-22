"""Source-trajectory replay, independent of physics engines and training code."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from tensordict import TensorDict

from ...render._resources import Closeable
from .scenes.g34 import FRAME_COUNT, authored_pose


class G34Replay(Closeable):
    """Replay a fixed G34 trajectory; not an action-driven registered Env.

    Default poses evaluate the source nominal animation, not the test oracle.
    Optional poses are an owned CPU float32 (T,7) xyz+wxyz trajectory, using
    exactly the packaged nominal geometry/camera. Randomized scenes are not
    supported. State is a CPU TensorDict containing a frame index and trajectory
    SHA256; restoration requires the same trajectory and packaged scene version.
    No dummy actions, physics fields, rewards or RNG state are fabricated.

    Rendering is lazy and CUDA-only. Replay/state operations work without CUDA,
    nvdiffrast, Blender or MuJoCo. Close releases the owned renderer if created.
    """

    resource_kind = "object_permanence.G34Replay"

    def __init__(
        self,
        poses=None,
        *,
        size=90,
        device="cuda",
        window_size=None,
        coverage_mode="scaled",
        background_coverage="refined",
    ):
        if poses is None:
            poses = np.concatenate(authored_pose(np.arange(FRAME_COUNT)), -1)
        poses = torch.as_tensor(poses, dtype=torch.float32, device="cpu").clone().contiguous()
        if poses.ndim != 2 or poses.shape[1] != 7 or not len(poses):
            raise ValueError("poses must have nonempty shape (T, 7)")
        if not torch.isfinite(poses).all() or (poses[:, 3:].square().sum(-1) == 0).any():
            raise ValueError("poses must be finite with nonzero quaternions")
        self._poses = poses
        self._digest = torch.tensor(list(hashlib.sha256(poses.numpy().tobytes()).digest()), dtype=torch.uint8)
        self.size, self.device = size, device
        self.window_size = window_size
        self.coverage_mode, self.background_coverage = coverage_mode, background_coverage
        self._renderer = None
        self._frame = 0
        self._opened()

    def state_at(self, frames):
        """Return independent CPU States for zero-based frame indices, shape (B,)."""
        self._check_open("state_at")
        frames = torch.as_tensor(frames)
        if frames.device.type != "cpu" or frames.dtype != torch.int64 or frames.ndim != 1 or not len(frames):
            raise ValueError("frames must be a nonempty one-dimensional CPU int64 array")
        if ((frames < 0) | (frames >= len(self._poses))).any():
            raise ValueError("frame index is outside the trajectory")
        return TensorDict(
            {
                ("replay", "frame"): frames[:, None].clone(),
                ("replay", "trajectory_hash"): self._digest.expand(len(frames), -1).clone(),
            },
            [len(frames)],
            device="cpu",
        )

    def _validate(self, state):
        self._check_open("state")
        if not isinstance(state, TensorDict) or len(state.batch_size) != 1 or not state.batch_size[0]:
            raise ValueError("replay State must be a nonempty TensorDict batch [B]")
        if set(state.keys(True, True)) != {("replay", "frame"), ("replay", "trajectory_hash")}:
            raise ValueError("replay State keys differ")
        f, h = state["replay", "frame"], state["replay", "trajectory_hash"]
        if f.shape != (len(state), 1) or h.shape != (len(state), 32) or h.dtype != torch.uint8:
            raise ValueError("replay State shape/dtype differs")
        if f.dtype != torch.int64 or f.device.type != "cpu":
            raise ValueError("replay frame must be CPU int64")
        if ((f < 0) | (f >= len(self._poses))).any():
            raise ValueError("frame index is outside the trajectory")
        if h.device.type != "cpu" or not torch.equal(h, self._digest.expand(len(state), -1)):
            raise ValueError("replay State belongs to a different trajectory")

    def get_state(self):
        return self.state_at(torch.tensor([self._frame]))

    def set_state(self, state):
        """Restore one replay cursor after validating the whole checkpoint."""
        self._validate(state)
        if len(state) != 1:
            raise ValueError("set_state requires one cursor")
        self._frame = int(state["replay", "frame"][0, 0])

    def reset(self):
        self._check_open("reset")
        self._frame = 0
        return self.get_state()

    def advance(self, count=1):
        """Move the cursor forward; the final frame holds until reset or restore."""
        self._check_open("advance")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        self._frame = min(self._frame + count, len(self._poses) - 1)
        return self.get_state()

    def poses(self, state):
        """Project full replay State to owned CPU rendering rows; no engine calls."""
        self._validate(state)
        # Advanced indexing already owns its storage; no second copy is needed.
        return self._poses[state["replay", "frame"][:, 0]]

    def render(self, state=None, *, mode="rgb", out=None):
        """Render saved States without moving the live cursor; includes pose upload."""
        buffers = None if out is None else {mode: out}
        return self.render_many(state, modes=(mode,), out=buffers)[mode]

    def render_many(self, state=None, *, modes=("rgb", "depth", "ids"), out=None):
        """Render several fields from one pose upload/coverage pass; State stays CPU."""
        rows = self.poses(self.get_state() if state is None else state)
        if self._renderer is None:
            from .rendering.g34_nvdiff import G34Renderer

            self._renderer = G34Renderer(
                size=self.size,
                device=self.device,
                window_size=self.window_size,
                coverage_mode=self.coverage_mode,
                background_coverage=self.background_coverage,
            )
        return self._renderer.render_many(rows, modes=modes, out=out)

    def _release(self):
        if self._renderer is not None:
            self._renderer.close()
        self._poses = self._digest = None
