"""Direct MuJoCo Warp stepping, with pooled native DMC resets and model changes."""

from __future__ import annotations

from functools import wraps

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.protocol import SimulatorAdapter
from .mjwarp_reset import WarpResetPool
from .mjwarp_runtime import WarpRuntime
from .mjwarp_tasks import bind_task
from .state import MODEL_FIELDS


def _require_cuda(device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("simulation_backend='mjwarp' requires an available CUDA device and the GPU dependencies")


def _ordered(method):
    """Order public buffers as well as physics when the caller switches streams."""

    @wraps(method)
    def call(self, *args, **kwargs):
        if self.device.type != "cuda":
            return method(self, *args, **kwargs)
        stream = torch.cuda.current_stream(self.device)
        if self._event is not None:
            stream.wait_event(self._event)
        try:
            return method(self, *args, **kwargs)
        finally:
            if self._event is None:
                self._event = torch.cuda.Event()
            self._event.record(stream)

    return call


class DMCWarpAdapter(SimulatorAdapter):
    """Batched float32 physics with the registered DMC State and reset semantics.

    One native owner samples resets from each lane's saved RNG and model values.
    Between resets, snapshots come from the live Warp checkpoint.
    Public checkpoint columns retain their declared float64 dtype. A restored
    checkpoint is preserved exactly until physics next advances.
    """

    simulation_backend = "mjwarp"

    def __init__(self, key, num_envs, device="cuda", seed=0):
        _require_cuda(device)
        self.key, self.num_envs, self.device = key, int(num_envs), torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._closed = False
        self._event = None
        self._scratch = None
        self._runtime = None
        self._source = WarpResetPool(key, self.num_envs, seed, *WarpRuntime.model_schema())
        try:
            self.fields = self._source.fields
            self.action_dim = self._source.action_dim
            self._models = self._source.models
            owner = self._source.envs[0]
            self._task = bind_task(key, owner)
            self._nsubsteps = owner._env._n_sub_steps
            self._limits = np.full(self.num_envs, owner._env._step_limit)
            self._device_limits = torch.as_tensor(self._limits, device=self.device)[:, None]
            self._timesteps = torch.as_tensor(
                [model.opt.timestep for model in self._models], device=self.device, dtype=torch.float64
            )[:, None]
            spec = owner._env.action_spec()
            self._minimum = torch.as_tensor(spec.minimum.copy(), device=self.device, dtype=torch.float64)
            self._scale = torch.as_tensor(
                (spec.maximum - spec.minimum).copy() / 2, device=self.device, dtype=torch.float64
            )
            self._runtime = WarpRuntime(
                self._models,
                self.device,
                self._nsubsteps,
                ordered_constraints=self.key in {"cheetah", "humanoid"},
                ordered_pyramidal=self.key == "humanoid",
                ordered_elliptic=self.key == "manipulator",
            )
            # The pool already performed the registered initial resets. Warm-up
            # must not draw another source reset or change its RNG sequence.
            self._install(self._source.get_state())
            if self.device.type == "cuda":
                self._event = torch.cuda.Event()
                self._event.record(torch.cuda.current_stream(self.device))
        except BaseException:
            self.close()
            raise

    @property
    def physics_layout(self):
        return self._source.physics_layout

    @property
    def physics_dim(self):
        return self._source.physics_dim

    def _check_open(self):
        if self._closed:
            raise RuntimeError("DMCWarpAdapter is closed")

    def _install(self, state, models=None):
        snapshot = state.to(self.device).clone()
        self._runtime.load(snapshot, models=self._models if models is None else models)
        self._state = snapshot
        self._counts = state["lifecycle", "step_count"].cpu().numpy().reshape(-1).copy()
        self._pending = state["lifecycle", "reset_next_step"].cpu().numpy().reshape(-1).copy()

    @_ordered
    def configure_physics(self, values):
        self._check_open()
        prepared = self._source.prepare_physics(values)
        self._runtime.validate_models(prepared.models)
        state = self._state.clone()
        for key in self._source.physics.state_fields():
            state[key] = prepared.state[key].to(self.device).clone()
        try:
            self._runtime.load(state, models=prepared.models)
        except Exception:
            self._runtime.load(self._state, models=self._models)
            raise
        self._source.commit(prepared)
        self._models, self._state = self._source.models, state

    @_ordered
    def reset(self, seed=None):
        self._check_open()
        prepared = self._source.prepare_reset(seed)
        self._install(prepared.state, prepared.models)
        self._source.commit(prepared)
        self._models = self._source.models
        return self.get_state()

    @_ordered
    def get_state(self):
        self._check_open()
        return self._state.clone()

    def validate(self, state):
        self._check_open()
        self._source.validate(state)
        # Validate all columns before either native owners or GPU buffers change.
        for key, (_, dtype) in self.fields.items():
            if dtype == torch.float64 and not bool(torch.isfinite(state[key]).all()):
                raise ValueError(f"DMC State field {key!r} must be finite")
            if key in {"qpos", "qvel", "time", "warmstart", "activation"} or key in MODEL_FIELDS:
                if not bool(torch.isfinite(state[key].to(torch.float32)).all()):
                    raise ValueError(f"DMC State field {key!r} must be representable in Warp float32")
        keys, position = state["rng", "keys"], state["rng", "position"]
        if bool(((keys < 0) | (keys > np.iinfo(np.uint32).max)).any()):
            raise ValueError("DMC RNG keys must be uint32 values")
        if bool(((position < 0) | (position > 624)).any()):
            raise ValueError("DMC RNG position must be in [0, 624]")
        if bool((state["lifecycle", "step_count"] < 0).any()):
            raise ValueError("DMC step_count must be nonnegative")
        prepared = self._source.prepare_state(state)
        self._runtime.validate_models(prepared.models)
        return prepared

    @_ordered
    def prepare_restore(self, state):
        self.validate(state)
        self._staged_restore = state.clone()

    @_ordered
    def set_state(self, state):
        prepared = self.validate(state)
        previous = self._state
        try:
            self._install(state, prepared.models)
        except Exception:
            # A representable but physically invalid checkpoint can still make
            # forward reconstruction fail. Keep the last live checkpoint usable.
            self._runtime.load(previous, models=self._models)
            raise
        self._source.commit(prepared)
        self._models = self._source.models

    def check_step(self, actions, active=None):
        self._check_open()
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float64).detach()
        if actions.shape != (self.num_envs, self.action_dim) or not bool(torch.isfinite(actions).all()):
            raise ValueError(f"{self.key} action must have finite shape (num_envs, {self.action_dim})")
        if active is not None:
            active = torch.as_tensor(active)
            if active.shape != (self.num_envs,) or not bool(active.all()):
                raise ValueError(
                    "Accepted DMC wrappers do not expose terminal lanes; partial active masks are unsupported"
                )
        return actions, None

    def _controls(self, actions):
        # Match action_scale.Wrapper's operation order, then convert to Warp's
        # float32 physics precision. This also accepts noncontiguous actions.
        return (self._minimum + self._scale * (actions + 1)).to(torch.float32).contiguous()

    def _outputs(self, runtime, captured=None):
        if captured is not None:
            # A replay overwrites graph-owned task buffers. This is the one
            # ownership copy for outputs that leave the adapter. The round's
            # reward is consumed immediately into the public reward buffer.
            values = {"reward": captured["reward"]}
            for key, value in captured.items():
                if key != "reward":
                    values[key] = (
                        {name: column.clone() for name, column in value.items()} if key == "info" else value.clone()
                    )
            return values
        snapshot = runtime.snapshot()
        values = {"observation": self._task.observation(snapshot), "reward": self._task.reward(snapshot)}
        info = self._task.info(snapshot)
        if info is not None:
            values["success"], details = info
            values["info"] = details
        # Task methods already return owned tensors. Build the public
        # TensorDict once, after both rounds and any mixed-reset merge.
        return values

    def _save_physics(self, runtime, lanes=None):
        runtime.copy_physical_state(self._state, lanes=lanes, include_time=False)
        # The public clock follows native double additions. Warp's float32
        # integration clock is loaded from it on restoration.
        time = self._state["time"] if lanes is None else self._state["time"][lanes]
        dt = self._timesteps if lanes is None else self._timesteps[lanes]
        for _ in range(self._nsubsteps):
            time += dt
        if lanes is not None:
            self._state["time"][lanes] = time

    def _reset_lanes(self, lanes):
        # Source reset owns all RNG draws, including cached Gaussian state.
        prepared = self._source.prepare_reset(lanes=lanes)
        self._source.commit(prepared)
        self._models = self._source.models
        ids = torch.as_tensor(lanes, device=self.device)
        cumulative = self._state["task", "cumulative_reward"][ids]
        self._state[ids] = prepared.state[lanes].to(self.device)
        self._state["task", "cumulative_reward"][ids] = cumulative
        self._counts[lanes] = 0
        self._pending[lanes] = False

    def _control_step(self, controls, *, reward_only=False, controls_validated=False):
        resetting = np.flatnonzero(self._pending)
        continuing = np.flatnonzero(~self._pending)
        if not len(resetting):
            captured = self._runtime.advance(
                controls, controls_validated=controls_validated, task=self._task, reward_only=reward_only
            )
            # Only the second round's observations/info reach the caller. The
            # first still contributes reward and commits its full checkpoint.
            out = captured if reward_only else self._outputs(self._runtime, captured)
            self._save_physics(self._runtime)
        else:
            advanced = None
            if len(continuing):
                if self._scratch is None or self._scratch.num_envs != len(continuing):
                    if self._scratch is not None:
                        self._scratch.close()
                    self._scratch = WarpRuntime(
                        self._models.select(continuing),
                        self.device,
                        self._nsubsteps,
                        ordered_constraints=self.key in {"cheetah", "humanoid"},
                        ordered_pyramidal=self.key == "humanoid",
                        ordered_elliptic=self.key == "manipulator",
                    )
                ids = torch.as_tensor(continuing, device=self.device)
                self._scratch.load(self._state[ids], models=self._models.select(continuing))
                # Collect acceleration-stage sensors before reconstruction.
                advanced = self._scratch.advance(controls[ids], controls_validated=controls_validated, task=self._task)
                self._save_physics(self._scratch, ids)
            self._reset_lanes(resetting)
            self._runtime.load(self._state, models=self._models)
            out = self._outputs(self._runtime)
            out["reward"].zero_()
            if advanced is not None:
                for key, value in advanced.items():
                    if key == "info":
                        for field, column in value.items():
                            out[key][field][ids] = column
                    else:
                        out[key][ids] = value
            if reward_only:
                out = out["reward"]
        self._counts[continuing] += 1
        self._pending[continuing] = self._counts[continuing] >= self._limits[continuing]
        if not len(resetting):
            # Counts evolve deterministically, so retain the host mirror for
            # reset routing while updating the checkpoint on device. Neither
            # direction needs a transfer on ordinary rounds.
            counts = self._state["lifecycle", "step_count"]
            counts.add_(1)
            torch.ge(counts, self._device_limits, out=self._state["lifecycle", "reset_next_step"])
        else:
            self._state["lifecycle", "step_count"] = torch.as_tensor(self._counts[:, None].copy(), device=self.device)
            self._state["lifecycle", "reset_next_step"] = torch.as_tensor(
                self._pending[:, None].copy(), device=self.device
            )
        return out

    @_ordered
    def step(self, actions, *, active=None, capture_state=True):
        actions, _ = self.check_step(actions, active)
        controls = self._runtime.validate_controls(self._controls(actions))
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        reward += self._control_step(controls, reward_only=True, controls_validated=True)
        out = self._control_step(controls, controls_validated=True)
        reward += out["reward"]
        self._state["task", "cumulative_reward"] += reward[:, None]
        out["reward"] = reward
        out["score"] = self._state["task", "cumulative_reward"][:, 0] / 1000
        out["terminated"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        out["truncated"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if capture_state:
            out["state"] = self.get_state()
        return TensorDict(out, [self.num_envs], device=self.device)

    def close(self):
        if not self._closed:
            if self._event is not None:
                self._event.synchronize()
            for runtime in (self._runtime, self._scratch):
                if runtime is not None:
                    runtime.close()
            self._source.close()
            self._closed = True
