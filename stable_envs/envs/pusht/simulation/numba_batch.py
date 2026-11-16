"""PushT batches stepped by the Numba port of the contact solver on the CPU."""

import math

import numpy as np
import torch
from numba import njit
from tensordict import TensorDict

from ...._core.tensordicts import build_tensordict
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
    """Record the step of the ``active`` environments and write their observation, reward and success.

    The same arithmetic, in the same order, as the NumPy expressions of the source task.
    """
    for instance in range(body.shape[0]):
        on = active[instance]
        n = lengths[instance]
        if on:
            contacts[instance] = counts[instance, 0]
            segment_count[instance] = counts[instance, 1]
            history[instance, n, 0] = actions[instance, 0]
            history[instance, n, 1] = actions[instance, 1]
            codes[instance, n] = code
            lengths[instance] = n + 1
            latest[instance, 0] = relative[instance, 0]
            latest[instance, 1] = relative[instance, 1]
            latest_type[instance, 0] = tag[0]
            latest_type[instance, 1] = tag[1]
            present[instance] = True
        else:
            history[instance, n, 0] = 0.0
            history[instance, n, 1] = 0.0
            codes[instance, n] = 0
        obs = observation[instance]
        obs[0], obs[1] = body[instance, 0, 0], body[instance, 0, 1]
        obs[2], obs[3] = body[instance, 1, 0], body[instance, 1, 1]
        obs[4] = body[instance, 1, 4] % _TURN
        obs[5], obs[6] = body[instance, 0, 2], body[instance, 0, 3]
        total = 0.0
        near = 0.0
        for i in range(obs.shape[0]):
            d = goal[instance, i] - obs[i]
            total += d * d
            if i == 3:
                near = total
        reward[instance] = -math.sqrt(total)
        symmetry = _SYMMETRIES[int(shape[instance])]
        angle = abs(goal[instance, 4] - obs[4]) % symmetry
        success[instance] = (math.sqrt(near) < 20) & (min(angle, symmetry - angle) < _SUCCESS_ANGLE)


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
        n = self.num_envs
        host["shape"] = fixed["owned/variation/block/shape/current/value"].numpy().reshape(n)
        host["latest_present"] = host["latest_present"].reshape(n)
        host["everyone"] = np.ones(n, np.bool_)
        host["lengths"] = self._lengths.numpy()
        # NumPy State sources let the export pack with NumPy and wrap leaves with from_numpy.
        fields = {key: value.numpy() for key, value in fixed.items()}
        fields.update(("current/jax/" + key, value.numpy()) for key, value in self._solver_state().items())
        host["fields"] = fields
        self._host = host
        return host

    def _state_fields(self, width):
        if self.device.type != "cpu":
            return super()._state_fields(width)
        return {**self._views()["fields"], **self._history_fields(width)}

    def _history_fields(self, width):
        lengths = self._lengths.numpy()
        return {
            "history/actions": self._actions.numpy()[:, :width],
            "history/dtypes": self._codes.numpy()[:, :width],
            "history/lengths": lengths,
            "history/valid": np.arange(width) < lengths[:, None],
        }

    def _export(self, device):
        if self.device.type != "cpu" or torch.device(device).type != "cpu":
            return super()._export(device)
        groups = self._host_groups()
        kept = {name: np.array(groups[name]) for name in self._groups.kept}
        return self._host_state({key: groups[key] for key in self._moving}, kept)

    def _host_groups(self, history=None):
        """``{leaf key: array}`` of the State, in State order.

        The step-changed groups are new arrays and the others are views. Each step-changed group is gathered from
        ``(num_envs, -1)`` views of its fields, cut once per reset.
        """
        host = self._views()
        width = self._history_width()
        plan = host.get("export")
        if plan is None or self._layout is None:
            fields = self._state_fields(width)
            self._state_layout(fields)
            plan = host["export"] = self._export_plan(host["fields"])
        if plan:
            moving = {key: np.concatenate(views, 1) for key, views in plan.items()}
        else:
            moving = self._groups.pack(self._state_fields(width), self._moving)
        kept = self._history_fields(width if history is None else history)
        leaves = {**self._static, **moving, **{name: kept[name] for name in self._groups.kept}}
        return {key: leaves[key] for key in self._groups.order}

    def _export_plan(self, fields):
        """``{group key: [(num_envs, -1) view of each field]}`` of the step-changed groups; empty if a view copies."""
        n, plan = self.num_envs, {}
        for key in self._moving:
            views = []
            for field in self._groups.groups[key].fields:
                source = fields[field.name]
                view = source.reshape(n, -1)
                if not np.shares_memory(view, source):
                    return {}
                views.append(view)
            plan[key] = views
        return plan

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
        raw = self._advance_host(actions, active)
        n = self.num_envs
        reward = raw[: 8 * n].view(np.float64)
        flags = raw[8 * n :].view(np.bool_).reshape(3, n)
        device = self._state_device
        if self.export_device is None:
            reward, flags = torch.from_numpy(reward), torch.from_numpy(flags)
        else:
            # On an accelerator every output is already there, so the step result needs no further move.
            device = self.__dict__.get("_output_device")
            if device is None:
                device = self._output_device = build_tensordict({}, [n], self.export_device).device
            moved = torch.from_numpy(raw).to(device)
            reward, flags = moved[: 8 * n].view(torch.float64), moved[8 * n :].view(torch.bool).view(3, n)
        success, terminated, truncated = flags.unbind(0)
        outputs = dict(reward=reward, success=success, terminated=terminated, truncated=truncated)
        if capture_state:
            # For an accelerator, the State moves there in the export's own copy, with no plan per history width.
            outputs["state"] = self._export(self.export_device or self._state_device)
        return TensorDict._new_unsafe(outputs, batch_size=torch.Size([n]), device=device)

    def step_leaves(self, actions, capacity=False):
        """Step every environment; the outputs and the State as host arrays by key path, or None off the CPU.

        The arrays may be live buffers: the caller copies them before the next call. With ``capacity``, the action
        history has the width of its buffers, which changes only when they grow, so the leaf shapes stay the same
        from step to step; the caller cuts it to ``history_width()``.
        """
        if self.device.type != "cpu":
            return None
        raw = self._advance_host(actions, None)
        n = self.num_envs
        flags = raw[8 * n :].view(np.bool_).reshape(3, n)
        leaves = {
            ("reward",): raw[: 8 * n].view(np.float64),
            ("success",): flags[0],
            ("terminated",): flags[1],
            ("truncated",): flags[2],
        }
        width = self._actions.shape[1] if capacity else None
        leaves.update((("state", *key.split("/")), value) for key, value in self._host_groups(width).items())
        return leaves

    def history_width(self):
        """The width of the exported action history: the longest environment's history."""
        return self._history_width()

    def _advance_host(self, actions, active):
        """Step the environments on the host; the reward, then success, terminated and truncated, in one byte buffer."""
        actions = actions.numpy() if torch.is_tensor(actions) else np.asarray(actions)
        if actions.shape != (self.num_envs, 2):
            raise ValueError("PushT actions require shape [num_envs, 2] and a supported dtype")
        discrete = self.discrete
        validate_actions(actions, discrete)
        host = self._views()
        if active is None:
            active = host["everyone"]
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
        # Reward, then success, terminated and truncated, in one buffer that moves to an accelerator in one copy.
        raw = np.zeros(11 * n, np.uint8)
        reward = raw[: 8 * n].view(np.float64)
        flags = raw[8 * n :].view(np.bool_).reshape(3, n)
        lengths = host["lengths"]
        _finish(
            body, active, counts, host["contacts"], host["render_segment_count"], relative, actions, code,
            _LATEST_TAGS[0 if relative.dtype == np.float32 else 1], host["latest/value"], host["latest/type"],
            host["latest_present"], self._actions.numpy(), self._codes.numpy(), lengths,
            host["observation"], host["goal/value"], host["shape"], reward, flags[0],
        )  # fmt: skip
        self._width = int(lengths.max())
        flags[1] = flags[0]
        return raw

    def _reserve_history(self):
        """Widen the action history buffers when an environment's next action would not fit."""
        width = self._history_width() + 1
        if width > self._actions.shape[1]:
            capacity = max(16, 2 * self._actions.shape[1])
            self._actions = torch.cat(
                (self._actions, self._actions.new_zeros(self.num_envs, capacity - self._actions.shape[1], 2)), 1
            )
            self._codes = torch.cat(
                (self._codes, self._codes.new_zeros(self.num_envs, capacity - self._codes.shape[1])), 1
            )
