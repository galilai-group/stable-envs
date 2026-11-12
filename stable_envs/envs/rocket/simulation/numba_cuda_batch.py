"""Rocket batches stepped by the CUDA build of the Numba kernel, one GPU thread per environment."""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
from tensordict import TensorDict

from ...._execution.process import leaf_tensordict
from . import numba_cuda_physics as cuda_physics
from . import numba_physics as physics
from .actions import validate_actions
from .numba_batch import _PACKED, _VARIATION_LEAVES, RocketNumbaAdapter

_LOW = (-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0)


@cuda.jit
def _after_step(stepped, position, velocity, rotation, flame, camera, rgba, dt):
    """Write the cached link 0 pose of the stepped environments and the flame colour of the ignited ones.

    The pose is the one before the last substep, as ``numba_batch._camera_poses`` computes it.
    """
    b = cuda.grid(1)
    if b >= camera.shape[0]:
        return
    if flame[b]:
        rgba[b, 5, 0] = 1.0
        rgba[b, 5, 1] = 0.3
        rgba[b, 5, 2] = 0.3
        rgba[b, 5, 3] = 0.8
    if not stepped[b]:
        return
    for i in range(3):
        camera[b, i] = position[b, i] - dt * velocity[b, i]
    m = rotation[b]
    trace = m[0] + m[4] + m[8]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0)
        w = s * 0.5
        s = 0.5 / s
        x = (m[7] - m[5]) * s
        y = (m[2] - m[6]) * s
        z = (m[3] - m[1]) * s
    else:
        if m[0] < m[4]:
            i = 2 if m[4] < m[8] else 1
        else:
            i = 2 if m[0] < m[8] else 0
        j = (i + 1) % 3
        k = (i + 2) % 3
        s = math.sqrt(m[4 * i] - m[4 * j] - m[4 * k] + 1.0)
        qi = s * 0.5
        s = 0.5 / s
        w = (m[3 * k + j] - m[3 * j + k]) * s
        qj = (m[3 * j + i] + m[3 * i + j]) * s
        qk = (m[3 * k + i] + m[3 * i + k]) * s
        if i == 0:
            x, y, z = qi, qj, qk
        elif i == 1:
            x, y, z = qk, qi, qj
        else:
            x, y, z = qj, qk, qi
    camera[b, 3] = x
    camera[b, 4] = y
    camera[b, 5] = z
    camera[b, 6] = w


class _StateBuffer:
    """Byte layout of every State leaf in one buffer.

    The leaves of one dtype are adjacent and start at an 8-byte bound; the ``front`` leaves come first in their
    dtype. ``runs`` holds ``(dtype, first element, element count, keys)`` of the other leaves of each dtype.
    """

    def __init__(self, fields, batch, front=frozenset()):
        self.batch, self.keys, self.slots, self.runs = batch, list(fields), {}, []
        offset = 0
        for dtype in dict.fromkeys(dtype for _, dtype in fields.values()):
            keys = [key for key, (_, other) in fields.items() if other == dtype]
            keys = [key for key in keys if key in front] + [key for key in keys if key not in front]
            run, start = [], None
            for key in keys:
                shape = fields[key][0]
                if key not in front:
                    start = offset if start is None else start
                    run.append(key)
                self.slots[key] = (dtype, shape, offset)
                offset += batch * math.prod(shape) * dtype.itemsize
            if run:
                self.runs.append((dtype, start // dtype.itemsize, (offset - start) // dtype.itemsize, run))
            offset = -(-offset // 8) * 8
        self.nbytes = offset
        #: ``(key, dtype, [batch, *shape], strides, first element)`` of each leaf in State order.
        self.strides = []
        for key in self.keys:
            dtype, shape, start = self.slots[key]
            inner = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
            self.strides.append((key, dtype, (batch, *shape), (math.prod(shape), *inner), start // dtype.itemsize))

    def views(self, buffer, steps=None):
        """``{key: view}`` of the leaves in a 1-D byte ``buffer`` that holds one State or ``steps`` equal rows."""
        flats, out = {}, {}
        row = buffer.numel() // (steps or 1)
        for key, dtype, shape, strides, start in self.strides:
            flat = flats.get(dtype)
            if flat is None:
                flat = flats[dtype] = buffer.view(dtype)
            if steps is None:
                out[key] = flat.as_strided(shape, strides, start)
            else:
                out[key] = flat.as_strided(
                    (shape[0], steps, *shape[1:]), (strides[0], row // dtype.itemsize, *strides[1:]), start
                )
        return out

    def stack(self, states, device):
        """Stack States whose leaves are views of one buffer of this layout, with one copy; None for others."""
        head, tail = self.keys[0], self.keys[-1]
        head_start = self.slots[head][2] // self.slots[head][0].itemsize
        tail_start = self.slots[tail][2] // self.slots[tail][0].itemsize
        rows, size = [], None
        for state in states:
            first, last = state[head], state[tail]
            storage = first.untyped_storage()
            size = storage.nbytes() if size is None else size
            if (
                storage.nbytes() != size
                or size < self.nbytes
                or storage.device.type != device.type
                or first.storage_offset() != head_start
                or last.storage_offset() != tail_start
                or last.untyped_storage().data_ptr() != storage.data_ptr()
            ):
                return None
            rows.append(torch.empty(0, dtype=torch.uint8, device=device).set_(storage))
        return leaf_tensordict(self.views(torch.stack(rows).view(-1), len(rows)), [self.batch, len(rows)], device)


class RocketCudaAdapter(RocketNumbaAdapter):
    """Step every Rocket environment on the GPU with the CUDA build of the Numba kernel.

    The kernel fields, the camera poses and the colours stay on the device. A step and the settle of a reset
    each replay one captured CUDA graph. Resets sample the starts and seed the noise generators on the host,
    as the CPU backend does, and move them to the device in one call per field. The State has the leaves of the
    CPU backend; a step checks the actions with one host read.
    """

    def __init__(self, batch, device, seed, config=None, frame=None):
        if torch.device(device).type != "cuda":
            raise ValueError("simulation_backend='numba_cuda' requires device='cuda'")
        super().__init__(batch, device, seed, config, frame)

    def _allocate(self):
        batch, device = self.num_envs, self.device
        self.state = cuda_physics.new_state(batch, device)
        self._camera_d = torch.zeros((batch, 7), dtype=torch.float64, device=device)
        self._rgba_d = torch.zeros((batch, 6, 4), dtype=torch.float64, device=device)
        self._uploaded = None
        self._step_graph = self._reset_graph = None
        self._low = torch.tensor(_LOW, dtype=torch.float64, device=device)
        self._everyone = torch.ones(batch, dtype=torch.bool, device=device)
        self._last_d = None
        self._copies_of = self._layout = None

    def _rng_rows(self):
        return self.state["rng"].cpu().numpy()

    def _upload(self):
        """Move the host values that only resets, configuration and restores change to the device."""
        device = self.device
        self._uploaded = {
            ("recipe", "seed"): torch.from_numpy(self._seed.copy()).to(device),
            **{
                ("variation", name): torch.from_numpy(value.copy()).to(device)
                for name, value in zip(_VARIATION_LEAVES, self._values, strict=True)
            },
        }

    def _start(self, start, rng):
        device = self.device
        self.state["rng"].copy_(torch.from_numpy(rng).to(device))
        if self._reset_graph is None:
            with warnings.catch_warnings():
                # Small batches launch few blocks; the captured graph replays without the warning.
                warnings.simplefilter("ignore", NumbaPerformanceWarning)
                self._reset_graph = cuda_physics.ResetGraph(self.state)
        self._reset_graph(self._everyone, {key: torch.from_numpy(value).to(device) for key, value in start.items()})
        self._camera_d[:, :3] = self.state["position"]
        self._camera_d[:, 3:] = self.state["quaternion"]
        self._rgba_d.copy_(torch.from_numpy(self._rgba).to(device))
        self._last_d = None
        self._upload()

    def _configured(self):
        # A configuration repaints the rocket and pad colours; the flame keeps its colour.
        self._rgba_d[:, :5] = torch.from_numpy(self._rgba[:, :5]).to(self.device)
        self._upload()

    def check_step(self, actions, active):
        actions = torch.as_tensor(actions, device=self.device)
        if actions.shape != (self.num_envs, 7) or actions.dtype not in (torch.float32, torch.float64):
            validate_actions(actions.detach().cpu().numpy())
            raise ValueError("Rocket actions require shape [num_envs, 7]")
        actions = actions.to(torch.float64)
        if not bool((torch.isfinite(actions) & (actions >= self._low) & (actions <= 1)).all()):
            validate_actions(actions.cpu().numpy())
        if active is not None:
            active = torch.as_tensor(active, device=self.device)
            if active.shape != (self.num_envs,):
                raise ValueError("Rocket active mask requires [num_envs]")
            active = active.to(torch.bool)
        return actions, active

    def step(self, actions, *, active=None, capture_state=True):
        actions, active = self.check_step(actions, active)
        everyone = active is None
        active = self._everyone if everyone else active
        if self._step_graph is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NumbaPerformanceWarning)
                self._step_graph = self._capture_step()
        graph, inputs, outputs = self._step_graph
        inputs[0].copy_(actions)
        inputs[1].copy_(active)
        graph.replay()
        reward, terminated, truncated, complete = outputs
        reward, success = reward.clone(), complete.clone()
        if not everyone and self._last_d is not None:
            # Inactive environments repeat their last outputs.
            reward = torch.where(active, reward, self._last_d[0])
            success = torch.where(active, success, self._last_d[1])
        self._last_d = (reward, success)
        outputs = dict(reward=reward, success=success, terminated=terminated.clone(), truncated=truncated.clone())
        if capture_state:
            outputs["state"] = self.get_state()
        return TensorDict._new_unsafe(outputs, batch_size=torch.Size([self.num_envs]), device=self.device)

    def _capture_step(self):
        """Capture a kernel step, then the camera and flame update, in one CUDA graph; return graph, buffers, outputs."""
        state, env, device = self.state, self._template.env, self.device
        actions = torch.zeros((self.num_envs, 7), dtype=torch.float64, device=device)
        active = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        stepped = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        task = dict(
            ceiling=env.ceiling,
            max_displacement=env.max_displacement,
            max_steps=env.max_steps,
            sparse_reward=env.sparse_reward,
        )
        arrays = [
            cuda.as_cuda_array(value)
            for value in (
                stepped,
                state["position"],
                state["linear_velocity"],
                state["link_rotation"],
                state["flame_ignited"],
                self._camera_d,
                self._rgba_d,
            )
        ]
        blocks = (self.num_envs + cuda_physics.THREADS_PER_BLOCK - 1) // cuda_physics.THREADS_PER_BLOCK

        def run():
            torch.logical_and(active, ~(state["termination"] | state["truncation"]), out=stepped)
            reward, terminated, truncated, info = cuda_physics.step(state, actions, None, active, **task)
            _after_step[blocks, cuda_physics.THREADS_PER_BLOCK, cuda_physics._stream(device)](*arrays, physics.DT)
            return reward, terminated, truncated, info["env_complete"]

        graph, outputs = cuda_physics._capture(device, run)
        # The graph writes ``stepped``, so the tuple keeps it alive with the inputs.
        return graph, (actions, active, stepped), outputs

    def _leaves(self):
        """``{key tuple: device tensor}`` of every State leaf; the tensors are the live values."""
        if not hasattr(self, "_fixed"):
            self._fixed = {key: torch.from_numpy(value).to(self.device) for key, value in self._fixed_leaves().items()}
        if self._uploaded is None:
            self._upload()
        batch, state = self.num_envs, self.state
        leaves = {key: self._fixed[key] for key in self._template.recipe()}
        leaves[("recipe", "seed")] = self._uploaded[("recipe", "seed")]
        for group in ("float", "int", "bool"):
            names = [n for n, (g, _, _) in self._columns.items() if g == group]
            leaves[("physics", group)] = torch.cat([state[n].reshape(batch, -1) for n in names], 1)
        leaves[("current", "wrapper", "state")] = state["state"]
        for i in range(3):
            for leaf in ("position", "quaternion"):
                key = ("current", "physics", f"body_{i}", leaf)
                leaves[key] = state[leaf] if i == 1 else self._fixed[key]
        leaves[("current", "diagnostic", "cached_links")] = self._camera_d[:, None, :]
        leaves[("render", "rgba")] = self._rgba_d
        leaves.update({("variation", name): self._uploaded[("variation", name)] for name in _VARIATION_LEAVES})
        return leaves

    def _state_layout(self):
        if self._layout is None:
            front = frozenset(("physics", group) for group in _PACKED)
            self._layout = _StateBuffer(self.fields, self.num_envs, front)
        return self._layout

    def stack_states(self, states):
        """Stack States of this adapter along a new time axis with one copy; None for other States."""
        return self._state_layout().stack(states, self.device)

    def _copies(self):
        """The sources of the State leaves: the kernel columns of each packed leaf, the flat leaves of each run."""
        leaves, state, batch = self._leaves(), self.state, self.num_envs
        packed = [
            (("physics", group), [state[n].reshape(batch, -1) for n, (g, _, _) in self._columns.items() if g == group])
            for group in _PACKED
        ]
        runs = [
            (dtype, first, count, [leaves[key].reshape(-1) for key in keys])
            for dtype, first, count, keys in self._state_layout().runs
        ]
        return packed, runs

    def get_state(self):
        """A State whose leaves are views of one new buffer, filled by one ``torch.cat`` per packed leaf and run."""
        if self._uploaded is None:
            self._upload()
        copies = self._copies_of
        if copies is None or copies[0] is not self._uploaded:
            copies = self._copies_of = (self._uploaded, *self._copies())
        _, packed, runs = copies
        layout = self._state_layout()
        buffer = torch.empty(layout.nbytes, dtype=torch.uint8, device=self.device)
        views = layout.views(buffer)
        for key, columns in packed:
            torch.cat(columns, 1, out=views[key])
        for dtype, first, count, sources in runs:
            torch.cat(sources, out=buffer.view(dtype)[first : first + count])
        return leaf_tensordict(views, self.num_envs, self.device)

    def rgba(self):
        return self._rgba_d.clone()

    def _install(self, host):
        device = self.device
        for group in ("float", "int", "bool"):
            packed = torch.from_numpy(host[("physics", group)]).to(device)
            for name, (g, start, width) in self._columns.items():
                if g == group:
                    self.state[name].copy_(packed[:, start : start + width].reshape(self.state[name].shape))
        self._seed[:] = host[("recipe", "seed")]
        self._camera[:] = host[("current", "diagnostic", "cached_links")][:, 0]
        self._rgba[:] = host[("render", "rgba")]
        for name, value in zip(_VARIATION_LEAVES, self._values, strict=True):
            value[...] = host[("variation", name)]
        self._camera_d.copy_(torch.from_numpy(self._camera).to(device))
        self._rgba_d.copy_(torch.from_numpy(self._rgba).to(device))
        self._last_d = None
        self._upload()

    def live_fields(self):
        batch = self.num_envs
        body = np.zeros((batch, 3, 7))
        body[:, :, 6] = 1.0
        body[:, 1, :3] = self.state["position"].cpu().numpy()
        body[:, 1, 3:] = self.state["quaternion"].cpu().numpy()
        body[:, 2, 2] = 0.1
        return body, self._camera_d.cpu().numpy(), self._rgba_d.cpu().numpy()

    def close(self):
        cuda_physics.forget(self.state)
        self._step_graph = self._reset_graph = None
        super().close()
