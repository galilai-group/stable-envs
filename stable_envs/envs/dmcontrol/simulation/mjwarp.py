"""DMC on CUDA: the shared ``WarpBatch`` steps MuJoCo Warp physics; native owners draw resets and model changes."""

from __future__ import annotations

import importlib.util
from functools import cache, wraps

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.protocol import SimulatorAdapter
from .mjwarp_reset import WarpResetPool, warp_batch_fields
from .mjwarp_tasks import bind_task
from .state import MODEL_FIELDS

#: Physical State columns and the ``MjData`` fields that hold them.
_PHYSICAL = {"qpos": "qpos", "qvel": "qvel", "warmstart": "qacc_warmstart"}
#: Contacts and constraints per world: the model's own sizes when larger.
CONTACTS_PER_WORLD, CONSTRAINTS_PER_WORLD = 128, 512
#: Bits of the step's device flag.
_BAD_CONTROLS, _BAD_PHYSICS = 1, 2
_MATRICES = ("xmat", "ximat", "geom_xmat", "site_xmat", "cam_xmat")


def _require_gpu_extra():
    absent = [m for m in ("mujoco_warp", "warp") if importlib.util.find_spec(m) is None]
    if absent:
        raise ModuleNotFoundError(
            f"DMC physics on CUDA steps with MuJoCo Warp, and {' and '.join(absent)} {'are' if len(absent) > 1 else 'is'} not installed: "
            "pip install 'stable-envs[dmcontrol-gpu]', or pass simulation_backend='native' to step on the CPU",
            name=absent[0],
        )


def _require_cuda(device):
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("simulation_backend='mjwarp' requires an available CUDA device and the GPU dependencies")


def _load_backend():
    try:
        import mujoco_warp
    except ImportError as error:
        raise ImportError(
            "DMC MuJoCo Warp physics requires mujoco-warp and warp-lang; install stable-envs[dmcontrol-gpu]"
        ) from error
    if any(version.split(".")[:2] != ["3", "17"] for version in (mujoco.__version__, mujoco_warp.__version__)):
        raise RuntimeError(
            "DMC MuJoCo Warp physics requires MuJoCo and mujoco-warp 3.17; install stable-envs[dmcontrol-gpu]"
        )
    return mujoco_warp


@cache
def _kernels():
    import warp as wp

    @wp.kernel
    def acc_sensors(
        sensordata: wp.array2d(dtype=float),
        index: wp.array(dtype=int),
        saved: wp.array2d(dtype=float),
        mode: int,
    ):
        world, k = wp.tid()
        i = index[k]
        if mode == 0:
            sensordata[world, i] = 0.0
        elif mode == 1:
            saved[world, k] = sensordata[world, i]
        else:
            sensordata[world, i] = saved[world, k]

    @wp.kernel
    def check_finite(
        ctrl: wp.array2d(dtype=float),
        qpos: wp.array2d(dtype=float),
        qvel: wp.array2d(dtype=float),
        act: wp.array2d(dtype=float),
        warmstart: wp.array2d(dtype=float),
        sensordata: wp.array2d(dtype=float),
        flag: wp.array(dtype=int),
    ):
        world = wp.tid()
        for i in range(ctrl.shape[1]):
            if not wp.isfinite(ctrl[world, i]):
                wp.atomic_or(flag, 0, 1)
        bad = int(0)  # noqa: UP018 - mutable Warp loop variable
        for i in range(qpos.shape[1]):
            if not wp.isfinite(qpos[world, i]):
                bad = 2
        for i in range(qvel.shape[1]):
            if not wp.isfinite(qvel[world, i]):
                bad = 2
        for i in range(act.shape[1]):
            if not wp.isfinite(act[world, i]):
                bad = 2
        for i in range(warmstart.shape[1]):
            if not wp.isfinite(warmstart[world, i]):
                bad = 2
        for i in range(sensordata.shape[1]):
            if not wp.isfinite(sensordata[world, i]):
                bad = 2
        if bad != 0:
            wp.atomic_or(flag, 0, bad)

    return acc_sensors, check_finite


def _ordered(method):
    """Order the adapter's buffers, as well as physics, when the caller switches CUDA streams."""

    @wraps(method)
    def call(self, *args, **kwargs):
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


def _own(values):
    return {key: _own(value) if isinstance(value, dict) else value.clone() for key, value in values.items()}


class LegacyStep:
    """dm_control's ``Physics.step`` with ``legacy_step``, as a ``WarpBatch`` ``step_fn``.

    ``mj_step2``, ``nsubsteps - 1`` calls of ``mj_step``, then ``mj_step1`` (RK4: ``nsubsteps`` calls of
    ``mj_step``, then ``mj_step1``): the data hold the position and velocity stages of the new state and the
    acceleration-stage sensor values of the last integration. Sensors are computed on the last substep only.
    ``flag`` collects nonfinite controls (bit 1) and nonfinite physics (bit 2) of every call.
    """

    def __init__(self, model, num_envs, nsubsteps, device):
        import warp as wp

        from ...._core.warp_cuda import require_warp_cuda

        require_warp_cuda("DMC GPU physics")
        self._mjw = _load_backend()
        self.nsubsteps = int(nsubsteps)
        self._rk4 = model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        acc = [
            index
            for sensor, stage in enumerate(model.sensor_needstage)
            if stage == mujoco.mjtStage.mjSTAGE_ACC
            for index in range(model.sensor_adr[sensor], model.sensor_adr[sensor] + model.sensor_dim[sensor])
        ]
        device = torch.device(device)
        wp_device = f"cuda:{device.index if device.index is not None else torch.cuda.current_device()}"
        self._acc_index = wp.array(np.asarray(acc, np.int32), dtype=int, device=wp_device)
        self._acc_saved = wp.zeros((int(num_envs), len(acc)), dtype=float, device=wp_device)
        self.flag = torch.zeros(1, dtype=torch.int32, device=device)
        self._flag_wp = wp.from_torch(self.flag)

    def __call__(self, m, d):
        import warp as wp

        mjw = self._mjw
        acc_sensors, check_finite = _kernels()
        flags = int(m.opt.disableflags)
        nacc = self._acc_saved.shape[1]

        def sensors(enabled):
            m.opt.disableflags = flags if enabled else flags | int(mjw.DisableBit.SENSOR)

        def acc(mode):
            if nacc:
                wp.launch(
                    acc_sensors, dim=(d.nworld, nacc), inputs=[d.sensordata, self._acc_index, self._acc_saved, mode]
                )

        last = self.nsubsteps - 1
        try:
            # MuJoCo Warp adds touch forces to the sensor values; native sensorAcc starts from zero.
            acc(0)
            if self._rk4:
                for substep in range(self.nsubsteps):
                    sensors(substep == last)
                    mjw.forward(m, d)
                    if substep == last:
                        # MuJoCo Warp's RK4 stages change the sensor values; native RK4 keeps the first stage's.
                        acc(1)
                    sensors(False)
                    mjw.rungekutta4(m, d)
            else:
                sensors(last == 0)
                mjw.step2(m, d)
                for substep in range(1, self.nsubsteps):
                    sensors(substep == last)
                    mjw.step(m, d)
                acc(1)
            sensors(True)
            mjw.step1(m, d)
            acc(2)
        finally:
            m.opt.disableflags = flags
        wp.launch(
            check_finite,
            dim=d.nworld,
            inputs=[d.ctrl, d.qpos, d.qvel, d.act, d.qacc_warmstart, d.sensordata, self._flag_wp],
        )


class _Snapshot:
    """Live ``[B, ...]`` views for the task formulas; matrices in MuJoCo's flat layout."""

    def __init__(self, batch, site_size):
        self._batch, self._site_size = batch, site_size

    def data(self, name):
        value = self._batch.data(name)
        return value.flatten(-2) if name in _MATRICES else value

    def model(self, name):
        if name == "site_size":
            return self._site_size
        return self._batch.model_field(name)


class DMCWarpAdapter(SimulatorAdapter):
    """Batched float32 MuJoCo Warp physics with the registered DMC State and reset semantics.

    Each control round runs dm_control's ``legacy_step`` (``mj_step2``, ``n - 1`` calls of ``mj_step``, then
    ``mj_step1``, with acceleration-stage sensors kept from the last integration) as one captured graph of the
    shared ``WarpBatch``. Native owners draw resets and per-environment model values from each environment's
    saved RNG; the adapter writes them into the batched model fields. The step does not synchronize with the
    host: nonfinite controls or physics raise ``PhysicsError`` one or two steps late, as a buffer overflow does.
    A restored checkpoint is returned exactly, in float64, until its environment steps again.
    """

    simulation_backend = "mjwarp"

    def __init__(self, key, num_envs, device="cuda", seed=0):
        _require_gpu_extra()
        _require_cuda(device)
        from ...._execution.mjbatch import WarpBatch

        self.key, self.num_envs, self.device = key, int(num_envs), torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._closed = False
        self._event = None
        self._task_graphs = {}
        self.batch = None
        self._model_fields, self._option_fields = warp_batch_fields()
        self._source = WarpResetPool(key, self.num_envs, seed, self._model_fields, self._option_fields)
        try:
            self.fields = self._source.fields
            self.action_dim = self._source.action_dim
            owner = self._source.envs[0]
            self._task = bind_task(key, owner)
            self._nsubsteps = owner._env._n_sub_steps
            self._limits = np.full(self.num_envs, owner._env._step_limit)
            spec = owner._env.action_spec()
            self._minimum = torch.as_tensor(spec.minimum.copy(), device=self.device, dtype=torch.float64)
            self._scale = torch.as_tensor((spec.maximum - spec.minimum).copy() / 2, device=self.device)
            models = self._source.models
            reference = models.reference
            self._physical = dict(_PHYSICAL, activation="act") if reference.na else dict(_PHYSICAL)
            self._step = LegacyStep(reference, self.num_envs, self._nsubsteps, self.device)
            self._flag = self._step.flag
            self._flag_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
            self._flag_event = torch.cuda.Event()
            self._flag_pending = False
            self.batch = WarpBatch(
                reference,
                self.num_envs,
                self.device,
                frame_skip=self._nsubsteps,
                batched_fields=(*self._model_fields, *(f"opt.{name}" for name in self._option_fields)),
                naconmax=max(CONTACTS_PER_WORLD, int(reference.nconmax)) * self.num_envs,
                njmax=max(CONSTRAINTS_PER_WORLD, int(reference.njmax)),
                step_fn=self._step,
            )
            self._timesteps = np.empty(self.num_envs)
            self._site_size = torch.empty(
                (self.num_envs, *reference.site_size.shape), dtype=torch.float32, device=self.device
            )
            self._snapshot = _Snapshot(self.batch, self._site_size)
            self._fixed = torch.as_tensor(models.fixed_indices, device=self.device)
            # Host copies of the uploaded model rows: a reset uploads only the fields that change.
            self._uploaded = {}
            self._install(self._source.get_state(), models)
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

    def _poll(self):
        if self._flag_pending and self._flag_event.query():
            self._flag_pending = False
            self._raise(int(self._flag_host[0]))

    def _raise(self, flag):
        if not flag:
            return
        from dm_control.rl.control import PhysicsError

        self._flag.zero_()
        if flag & _BAD_CONTROLS:
            raise PhysicsError("DMC MuJoCo Warp step received nonfinite controls")
        raise PhysicsError("DMC MuJoCo Warp physics produced nonfinite values")

    def _record_flag(self):
        if not self._flag_pending:
            self._flag_host.copy_(self._flag, non_blocking=True)
            self._flag_event.record()
            self._flag_pending = True

    def check_buffers(self):
        """Raise for any earlier step that overflowed a buffer or produced nonfinite values; synchronizes."""
        self._check_open()
        self.batch.check_buffers()
        self._flag_pending = False
        self._raise(int(self._flag[0]))

    @staticmethod
    def _check_float32(values, name):
        values = np.asarray(values)
        if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float32).max):
            raise ValueError(f"Model field {name!r} must be finite in float32")

    def _validate_models(self, models):
        for name in self._model_fields:
            self._check_float32(models.columns[name], name)
        for name, values in models.options.items():
            self._check_float32(values, f"opt.{name}")
        self._check_float32(models.columns["site_size"], "site_size")
        self._check_float32(models.fixed_positions, "fixed geom positions")
        self._check_float32(models.fixed_rotations, "fixed geom rotations")

    def _upload_models(self, models, lanes=None):
        """Write the model rows of ``lanes`` (all when None) into the batched fields and fixed geom poses."""
        rows = slice(None) if lanes is None else np.asarray(lanes)
        index = None if lanes is None else self._upload(rows)
        columns = {name: models.columns[name] for name in self._model_fields}
        options = {}
        for name in self._option_fields:
            if name == "impratio_invsqrt":
                options[name] = 1 / np.sqrt(np.maximum(models.options["impratio"], mujoco.mjMINVAL))
            else:
                options[name] = models.options[name]

        def write(view, values):
            values = self._upload(values).to(view.dtype)
            if index is None:
                view.copy_(values.reshape(view.shape))
            else:
                view[index] = values.reshape(len(index), *view.shape[1:])

        for key, values, field in (
            *((name, columns[name], name) for name in self._model_fields),
            *((f"opt.{name}", options[name], f"opt.{name}") for name in self._option_fields),
            ("site_size", models.columns["site_size"], None),
        ):
            new = np.asarray(values)[rows]
            old = self._uploaded.get(key)
            if old is not None and np.array_equal(old[rows], new):
                continue
            if old is None:
                self._uploaded[key] = np.array(values, copy=True)
            else:
                old[rows] = new
            write(self._site_size if field is None else self.batch.model_field(field), new)
        self._timesteps[rows] = models.options["timestep"][rows]
        # MuJoCo Warp does not recompute geoms welded to the world; write their poses from native kinematics.
        if len(self._fixed):
            for name, values in (("geom_xpos", models.fixed_positions), ("geom_xmat", models.fixed_rotations)):
                view = self.batch.data(name)
                values = self._upload(values[rows]).to(view.dtype)
                if index is None:
                    view[:, self._fixed] = values
                else:
                    view[index[:, None], self._fixed] = values

    def _write_physics(self, state, lanes=None):
        """Load the physical columns of ``state`` rows ``lanes`` (all when None) and forward every world."""
        values = {name: state[key].to(torch.float32) for key, name in self._physical.items()}
        values["time"] = state["time"].to(torch.float32)[:, 0]
        # dm_control forwards a reset with actuation disabled; the next step writes the controls again.
        self.batch.data("ctrl").zero_()
        self.batch.set_physics(values, lanes, forward=True)

    def _to_device(self, state):
        if state.device is not None and state.device.type == "cpu":
            return state.pin_memory().to(self.device, non_blocking=True)
        return state.to(self.device).clone()

    def _install(self, state, models):
        self._validate_models(models)
        host = state if state.device is not None and state.device.type == "cpu" else state.cpu()
        state = self._to_device(state)
        self._upload_models(models)
        self._write_physics(state)
        self._state = state
        self._clock = host["time"][:, 0].numpy().copy()
        self._counts = host["lifecycle", "step_count"].numpy().reshape(-1).copy()
        self._pending = host["lifecycle", "reset_next_step"].numpy().reshape(-1).copy()
        self._stale = np.zeros(self.num_envs, dtype=bool)

    def _reinstall(self):
        """Reload the last committed checkpoint after a failed change."""
        self._install(self.get_state(), self._source.models)

    @_ordered
    def configure_physics(self, values):
        self._check_open()
        prepared = self._source.prepare_physics(values)
        self._validate_models(prepared.models)
        state = self.get_state()
        for key in self._source.physics.state_fields():
            state[key] = prepared.state[key].to(self.device).clone()
        try:
            self._install(state, prepared.models)
        except Exception:
            self._reinstall()
            raise
        self._source.commit(prepared)

    @_ordered
    def reset(self, seed=None):
        self._check_open()
        prepared = self._source.prepare_reset(seed)
        self._install(prepared.state, prepared.models)
        self._source.commit(prepared)
        return self.get_state()

    @_ordered
    def get_state(self):
        self._check_open()
        if self._stale.any():
            stale = None if self._stale.all() else self._upload(np.flatnonzero(self._stale))
            for key, name in self._physical.items():
                value = self.batch.data(name)
                if stale is None:
                    self._state[key].copy_(value)
                else:
                    self._state[key][stale] = value[stale].to(torch.float64)
            self._stale[:] = False
        # The clock and lifecycle live on the host; one copy that does not wait for the queued physics.
        host = self._upload(np.stack([self._clock, self._counts, self._pending], axis=1).astype(np.float64))
        self._state["time"] = host[:, 0:1].contiguous()
        self._state["lifecycle", "step_count"] = host[:, 1:2].to(torch.int64)
        self._state["lifecycle", "reset_next_step"] = host[:, 2:3].to(torch.bool)
        return self._state.clone()

    def _upload(self, array):
        """``array`` on the device, through pinned memory, without waiting for queued work."""
        return torch.from_numpy(np.ascontiguousarray(array)).pin_memory().to(self.device, non_blocking=True)

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
        self._validate_models(prepared.models)
        return prepared

    @_ordered
    def prepare_restore(self, state):
        self.validate(state)
        self._staged_restore = state.clone()

    @_ordered
    def set_state(self, state):
        prepared = self.validate(state)
        try:
            self._install(state, prepared.models)
        except Exception:
            # A representable but physically invalid checkpoint can still fail to load. Keep the last one.
            self._reinstall()
            raise
        self._source.commit(prepared)

    def check_step(self, actions, active=None):
        actions, _ = self._prepare_step(actions, active)
        return actions, None

    def _prepare_step(self, actions, active=None):
        """Check shapes and scale actions to controls without reading device values."""
        self._check_open()
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float64).detach()
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"{self.key} action must have finite shape (num_envs, {self.action_dim})")
        if active is not None:
            active = torch.as_tensor(active)
            if active.shape != (self.num_envs,) or not bool(active.all()):
                raise ValueError(
                    "Accepted DMC wrappers do not expose terminal lanes; partial active masks are unsupported"
                )
        return actions, self._controls(actions)

    def _controls(self, actions):
        # action_scale.Wrapper's operation order, then Warp's float32.
        return (self._minimum + self._scale * (actions + 1)).to(torch.float32)

    def _outputs(self):
        s = self._snapshot
        values = {"observation": self._task.observation(s), "reward": self._task.reward(s)}
        info = self._task.info(s)
        if info is not None:
            values["success"], values["info"] = info
        return values

    def _task_values(self, full):
        """The task outputs (``full``) or the reward of the current physics, computed by a captured CUDA graph.

        The returned tensors belong to the graph: its next replay overwrites them.
        """
        entry = self._task_graphs.get(full)
        if entry is None:
            compute = self._outputs if full else (lambda: {"reward": self._task.reward(self._snapshot)})
            current = torch.cuda.current_stream(self.device)
            side = torch.cuda.Stream(self.device)
            side.wait_stream(current)
            with torch.cuda.stream(side):
                # The first calls fill the task's caches of device constants and indices.
                compute()
                compute()
            current.wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                values = compute()
            entry = self._task_graphs[full] = (graph, values)
        graph, values = entry
        graph.replay()
        return values

    def _reset_lanes(self, lanes):
        """Draw native resets for ``lanes`` and load them; the other worlds are forwarded at their state."""
        prepared = self._source.prepare_reset(lanes=lanes)
        models = prepared.models
        self._validate_models(models)
        ids = self._upload(lanes)
        rows = self._to_device(prepared.state[lanes])
        self._upload_models(models, lanes)
        self._write_physics(rows, lanes)
        self._source.commit(prepared)
        # The other worlds' physics is read back lazily; these rows keep the exact float64 reset.
        self._stale[lanes] = False
        cumulative = self._state["task", "cumulative_reward"][ids]
        self._state[ids] = rows
        self._state["task", "cumulative_reward"][ids] = cumulative
        self._clock[lanes] = prepared.state["time"][lanes, 0].numpy()
        self._counts[lanes] = 0
        self._pending[lanes] = False

    @_ordered
    def step(self, actions, *, active=None, capture_state=True):
        actions, controls = self._prepare_step(actions, active)
        self._poll()
        reward = out = None
        for last in (False, True):
            resetting = np.flatnonzero(self._pending)
            continuing = np.flatnonzero(~self._pending)
            self.batch.step(controls)
            self._record_flag()
            self._stale[:] = True
            # The reward graph and the outputs graph own separate tensors: the first round's reward stays valid.
            values = self._task_values(full=last)
            round_reward = values["reward"]
            if last:
                out = _own(values)
            if len(resetting):
                ids = self._upload(resetting)
                round_reward = round_reward.index_fill(0, ids, 0.0)
                self._reset_lanes(resetting)
                if last:
                    fresh = self._task_values(full=True)
                    for key, value in fresh.items():
                        if key == "info":
                            for field, column in value.items():
                                out[key][field][ids] = column[ids]
                        elif key != "reward":
                            out[key][ids] = value[ids]
            reward = round_reward if reward is None else reward + round_reward
            dt = self._timesteps[continuing]
            clock = self._clock[continuing]
            for _ in range(self._nsubsteps):
                clock += dt
            self._clock[continuing] = clock
            self._counts[continuing] += 1
            self._pending[continuing] = self._counts[continuing] >= self._limits[continuing]
        cumulative = self._state["task", "cumulative_reward"]
        cumulative += reward[:, None]
        out["reward"] = reward
        out["score"] = cumulative[:, 0] / 1000
        out["terminated"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        out["truncated"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if capture_state:
            out["state"] = self.get_state()
        return TensorDict(out, [self.num_envs], device=self.device)

    def close(self):
        if not self._closed:
            self._closed = True
            if self.batch is not None:
                torch.cuda.current_stream(self.device).synchronize()
                self._task_graphs.clear()
                self.batch.close()
            self._source.close()
