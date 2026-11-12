"""PushT batches stepped by the Numba port of the contact solver on the CPU."""

import numpy as np
import torch
from tensordict import TensorDict

from .batch import PushTBatchAdapter
from .numba_physics import advance_batch
from .replay_batch import validate_actions

_SYMMETRIES = np.array([2 * np.pi, 2 * np.pi, 2 * np.pi, np.pi, np.pi / 2, np.pi, 2 * np.pi, np.pi / 2])
_LATEST_TAGS = np.array([[4, 0], [4, 1]], np.uint8)


class PushTNumbaAdapter(PushTBatchAdapter):
    """Step PushT environments with the Numba contact solver, in parallel over environments.

    On CPU the step updates the State columns in place. On CUDA the solver state moves to the host
    for the step and the results move back.
    """

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
        self._host = host
        return host

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
        discrete = self.template.discrete
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
        advance_batch(
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
        if everyone:
            host["contacts"][:] = counts[:, 0]
            host["render_segment_count"][:] = counts[:, 1]
        else:
            host["contacts"][active] = counts[active, 0]
            host["render_segment_count"][active] = counts[active, 1]
        self._record(host, actions, relative, active, everyone)
        observation = host["observation"]
        observation[:, 0:2] = body[:, 0, :2]
        observation[:, 2:4] = body[:, 1, :2]
        observation[:, 4] = body[:, 1, 4] % (2 * np.pi)
        observation[:, 5:7] = body[:, 0, 2:4]
        goal = host["goal/value"]
        distance = np.sqrt(((goal - observation) ** 2).sum(1))
        shape = host["shape"].astype(np.int64)
        symmetry = _SYMMETRIES[shape]
        angle = np.abs(goal[:, 4] - observation[:, 4]) % symmetry
        success = (np.sqrt(((goal[:, :4] - observation[:, :4]) ** 2).sum(1)) < 20) & (
            np.minimum(angle, symmetry - angle) < np.pi / 9
        )
        success = torch.from_numpy(success)
        outputs = dict(
            reward=torch.from_numpy(-distance),
            success=success,
            terminated=success.clone(),
            truncated=torch.zeros_like(success),
        )
        if capture_state:
            outputs["state"] = self.get_state()
        return TensorDict._new_unsafe(outputs, batch_size=torch.Size([self.num_envs]), device=self._state_device)

    def _record(self, host, actions, relative, active, everyone):
        """Append the actions of the active environments to their histories and set the latest action fields."""
        lengths = self._lengths.numpy()
        width = int(lengths.max()) + 1
        if width > self._actions.shape[1]:
            capacity = max(16, 2 * self._actions.shape[1])
            self._actions = torch.cat(
                (self._actions, self._actions.new_zeros(self.num_envs, capacity - self._actions.shape[1], 2)), 1
            )
            self._codes = torch.cat(
                (self._codes, self._codes.new_zeros(self.num_envs, capacity - self._codes.shape[1])), 1
            )
        lane = np.arange(self.num_envs)
        if self.template.discrete:
            code = (np.dtype(np.int32), np.dtype(np.int64)).index(actions.dtype) + 2
        else:
            code = (np.dtype(np.float32), np.dtype(np.float64)).index(actions.dtype)
        history, codes = self._actions.numpy(), self._codes.numpy()
        history[lane, lengths] = np.where(active[:, None], actions.astype(np.float64), 0.0)
        codes[lane, lengths] = np.where(active, code, 0)
        lengths += active
        self._width = int(lengths.max())
        tag = _LATEST_TAGS[0 if relative.dtype == np.float32 else 1]
        if everyone:
            host["latest/value"][:] = relative
            host["latest/type"][:] = tag
            host["latest_present"][:] = True
        else:
            host["latest/value"][active] = relative[active]
            host["latest/type"][active] = tag
            host["latest_present"][active] = True
