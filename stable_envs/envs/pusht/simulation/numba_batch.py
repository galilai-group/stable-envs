"""PushT batches stepped by the Numba port of the contact solver on the CPU."""

import math

import numpy as np
import torch
from numba import njit
from tensordict import TensorDict

from .batch import PushTBatchAdapter
from .numba_physics import advance_batch, advance_serial
from .replay_batch import validate_actions

_SYMMETRIES = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, np.pi, np.pi / 2, np.pi, 2 * np.pi, np.pi / 2])
_LATEST_TAGS = np.array([[4, 0], [4, 1]], np.uint8)
_TURN = 2 * np.pi
_SUCCESS_ANGLE = np.pi / 9


@njit(cache=True)
def _finish(
    body, active, counts, contacts, segment_count, relative, actions, code, tag, latest, latest_type, present,
    history, codes, lengths, observation, goal, shape, reward, success,
):  # fmt: skip
    """Record the step of the ``active`` lanes and write their observation, reward and success.

    The same arithmetic, in the same order, as the NumPy expressions of the source task.
    """
    for lane in range(body.shape[0]):
        on = active[lane]
        n = lengths[lane]
        if on:
            contacts[lane] = counts[lane, 0]
            segment_count[lane] = counts[lane, 1]
            history[lane, n, 0] = actions[lane, 0]
            history[lane, n, 1] = actions[lane, 1]
            codes[lane, n] = code
            lengths[lane] = n + 1
            latest[lane, 0] = relative[lane, 0]
            latest[lane, 1] = relative[lane, 1]
            latest_type[lane, 0] = tag[0]
            latest_type[lane, 1] = tag[1]
            present[lane] = True
        else:
            history[lane, n, 0] = 0.0
            history[lane, n, 1] = 0.0
            codes[lane, n] = 0
        obs = observation[lane]
        obs[0], obs[1] = body[lane, 0, 0], body[lane, 0, 1]
        obs[2], obs[3] = body[lane, 1, 0], body[lane, 1, 1]
        obs[4] = body[lane, 1, 4] % _TURN
        obs[5], obs[6] = body[lane, 0, 2], body[lane, 0, 3]
        total = 0.0
        near = 0.0
        for i in range(obs.shape[0]):
            d = goal[lane, i] - obs[i]
            total += d * d
            if i == 3:
                near = total
        reward[lane] = -math.sqrt(total)
        symmetry = _SYMMETRIES[int(shape[lane])]
        angle = abs(goal[lane, 4] - obs[4]) % symmetry
        success[lane] = (math.sqrt(near) < 20) & (min(angle, symmetry - angle) < _SUCCESS_ANGLE)


class PushTNumbaAdapter(PushTBatchAdapter):
    """Step PushT environments with the Numba contact solver, in parallel over environments.

    On CPU the step updates the State columns in place. On CUDA the solver state moves to the host
    for the step and the results move back.
    """

    #: Where a step exports its State, when a ``HostAdapter`` runs this CPU adapter for an accelerator.
    export_device: torch.device | None = None

    def _refresh_current(self):
        super()._refresh_current()
        self._host = None

    def _install_columns(self, columns):
        super()._install_columns(columns)
        self._host = None

    def _views(self):
        """NumPy views of the CPU State columns that a step changes in place."""
        if self._host is not None:
            return self._host
        fixed = self._fixed
        host = {
            "body": self._body.numpy(),
            "bias": self._bias.numpy(),
            "impulse": self._impulse.numpy(),
            "identifiers": self._identifiers.numpy(),
            "tree": self._tree.numpy(),
            "tree_bounds": self._tree_bounds.numpy(),
            "tree_order": self._tree_order.numpy(),
            "geometry": tuple(value.numpy() for value in self._geometry),
        }
        for key in (
            "contacts",
            "observation",
            "render_segments",
            "render_segment_count",
            "latest/value",
            "latest/type",
            "latest_present",
            "goal/value",
        ):
            host[key] = fixed["current/" + key].numpy()
        host["shape"] = fixed["owned/variation/block/shape/current/value"].numpy()
        # NumPy State sources let the export pack with NumPy and wrap leaves with from_numpy.
        fields = {key: value.numpy() for key, value in fixed.items()}
        fields.update(("current/jax/" + key, value.numpy()) for key, value in self._solver_state().items())
        host["fields"] = fields
        self._host = host
        return host

    def _state_fields(self, width):
        if self.device.type != "cpu":
            return super()._state_fields(width)
        fields = dict(self._views()["fields"])
        lengths = self._lengths.numpy()
        fields["history/actions"] = self._actions.numpy()[:, :width]
        fields["history/dtypes"] = self._codes.numpy()[:, :width]
        fields["history/lengths"] = lengths
        fields["history/valid"] = np.arange(width) < lengths[:, None]
        return fields

    def _advance(self, target, single, active):
        host = {
            key: getattr(self, "_" + key).cpu().numpy().copy()
            for key in ("body", "bias", "impulse", "identifiers", "tree", "tree_bounds")
        }
        geometry = tuple(value.cpu().numpy() for value in self._geometry)
        counts = np.zeros((self.num_envs, 2), np.int64)
        lines = np.zeros((self.num_envs, 90, 8), np.float64)
        advance_batch(
            host["body"],
            host["bias"],
            host["impulse"],
            host["identifiers"],
            host["tree"],
            host["tree_bounds"],
            self._tree_order.cpu().numpy(),
            *geometry,
            target.cpu().numpy(),
            single,
            self._circle,
            active.cpu().numpy(),
            counts,
            lines,
        )
        arrays = (
            host["body"],
            host["bias"],
            host["impulse"],
            host["identifiers"],
            host["tree"],
            host["tree_bounds"],
            counts[:, 0],
            lines,
            counts[:, 1],
        )
        return tuple(torch.from_numpy(value).to(self.device, non_blocking=True) for value in arrays)

    def step(self, actions, *, active=None, capture_state=True):
        if self.device.type != "cpu":
            return super().step(actions, active=active, capture_state=capture_state)
        actions = actions.numpy() if torch.is_tensor(actions) else np.asarray(actions)
        if actions.shape != (self.num_envs, 2):
            raise ValueError("PushT actions require shape [num_envs, 2] and a supported dtype")
        discrete = self.discrete
        validate_actions(actions, discrete)
        everyone = active is None
        if everyone:
            active = np.ones(self.num_envs, np.bool_)
        else:
            active = active.numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or active.dtype != np.bool_:
                raise ValueError("PushT active mask requires bool [num_envs]")
        if discrete:
            polar = (actions.astype(np.float64) + 0.5) / 16
            theta = polar[:, 1] * (2 * np.pi)
            relative = np.stack((polar[:, 0] * np.cos(theta), polar[:, 0] * np.sin(theta)), 1).astype(np.float32)
        else:
            relative = actions
        host = self._views()
        body = host["body"]
        target = (body[:, 0, :2].astype(relative.dtype) + relative * 100).astype(np.float64)
        counts = np.empty((self.num_envs, 2), np.int64)
        (advance_batch if self.num_envs > 1 else advance_serial)(
            body,
            host["bias"],
            host["impulse"],
            host["identifiers"],
            host["tree"],
            host["tree_bounds"],
            host["tree_order"],
            *host["geometry"],
            target,
            relative.dtype == np.float32,
            self._circle,
            active,
            counts,
            host["render_segments"],
        )
        self._reserve_history()
        if self.discrete:
            code = (np.dtype(np.int32), np.dtype(np.int64)).index(actions.dtype) + 2
        else:
            code = (np.dtype(np.float32), np.dtype(np.float64)).index(actions.dtype)
        n = self.num_envs
        reward = np.empty(n)
        flags = np.zeros((3, n), np.bool_)
        lengths = self._lengths.numpy()
        _finish(
            body, active, counts, host["contacts"], host["render_segment_count"], relative, actions, code,
            _LATEST_TAGS[0 if relative.dtype == np.float32 else 1], host["latest/value"], host["latest/type"],
            host["latest_present"].reshape(n), self._actions.numpy(), self._codes.numpy(), lengths,
            host["observation"], host["goal/value"], host["shape"].reshape(n), reward, flags[0],
        )  # fmt: skip
        self._width = int(lengths.max())
        flags[1] = flags[0]
        success, terminated, truncated = torch.from_numpy(flags).unbind(0)
        outputs = dict(reward=torch.from_numpy(reward), success=success, terminated=terminated, truncated=truncated)
        if capture_state:
            # For an accelerator, the State moves there in the export's own copy, with no plan per history width.
            outputs["state"] = self._export(self.export_device or self._state_device)
        return TensorDict._new_unsafe(outputs, batch_size=torch.Size([self.num_envs]), device=self._state_device)

    def _reserve_history(self):
        """Widen the action history buffers when a lane's next action would not fit."""
        width = int(self._lengths.numpy().max()) + 1
        if width > self._actions.shape[1]:
            capacity = max(16, 2 * self._actions.shape[1])
            self._actions = torch.cat(
                (self._actions, self._actions.new_zeros(self.num_envs, capacity - self._actions.shape[1], 2)), 1
            )
            self._codes = torch.cat(
                (self._codes, self._codes.new_zeros(self.num_envs, capacity - self._codes.shape[1])), 1
            )
