"""OGBench Cube and Scene lanes stepped together in one ``mujoco.batch``, equal to OGBench bit for bit."""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import torch
from ogbench.manipspace import lie, mjcf_utils

from ...._core.state_layout import StateLayout
from ...._core.tensordicts import build_tensordict, to_device
from ...._execution.mjbatch import NativeBatch, native_threads
from ...._execution.policy import cpu_affinity, process_count
from .compatibility import COMPILED_FIELDS, MODEL_VALUES, CompatibleCube, CompatibleScene, compiled_fields, values_key
from .manipulation import FLAGS, PCG_KEYS, OGBManipulationAdapter, field, groups_of, site_ids, state_schema
from .manipulation_control import BatchController
from .pcg import pcg_decode, pcg_encode

#: MjData fields of the integration State, in State order, as ``mujoco.batch`` names them.
_BATCH_NAMES = {name: "qacc_warmstart" if name == "warmstart" else name for name in FLAGS}
_BATCH_NAMES["plugin"] = "plugin_state"
_CPU = torch.device("cpu")
_INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION


def state_layout(schema, num_envs):
    """The ``StateLayout`` of the group leaves of a Cube/Scene ``StateSchema``."""
    return StateLayout(schema.leaves(), num_envs)


def live_view(schema, live, name):
    """A view of the field ``name`` in the ``{group key: array}`` buffers ``live``; writes to it change ``live``."""
    view = schema.get(live, name)
    group = live[schema.slices[name].group]
    if isinstance(group, np.ndarray):
        shared = np.shares_memory(view, group)
    else:
        shared = view.untyped_storage().data_ptr() == group.untyped_storage().data_ptr()
    if not shared:
        raise RuntimeError(f"State field {name} is not a view of its group")
    return view


def physics_model(env):
    """The lane's model with a sensor that makes every forward pass run ``mj_rnePostConstraint``.

    OGBench calls ``mj_rnePostConstraint`` once after the substeps of a step; its ``cfrc_ext`` is that of the last
    substep's forward pass, which the sensor computes. The sensor reads nothing that the physics reads.
    """
    root = env._mjcf_model
    xml = mjcf_utils.to_string(root).replace(
        "</mujoco>", '<sensor><framelinacc objtype="body" objname="world"/></sensor></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml, mjcf_utils.get_assets(root))
    copy_model_fields(env.model, model)
    return model


def _model_fields():
    from mujoco.batch import MODEL_FIELDS

    return tuple(sorted(MODEL_FIELDS))


def copy_model_fields(source, target):
    """Copy every array and option of ``source`` that ``target`` has with the same shape."""
    for name in _model_fields():
        if name.startswith("opt."):
            setattr(target.opt, name[4:], getattr(source.opt, name[4:]))
            continue
        value, out = getattr(source, name, None), getattr(target, name, None)
        if isinstance(value, np.ndarray) and isinstance(out, np.ndarray) and value.shape == out.shape:
            out[...] = value


class _IntegrationState:
    """``mujoco.batch``'s ``state`` rows for one ``MjData``: ``mjSTATE_INTEGRATION`` read and written whole."""

    def __init__(self, lane):
        self._lane = lane

    def __getitem__(self, row):
        lane = self._lane
        value = np.empty(mujoco.mj_stateSize(lane.model, _INTEGRATION))
        mujoco.mj_getState(lane.model, lane.data, value, _INTEGRATION)
        return value if isinstance(row, int | np.integer) else value[None][row]

    def __setitem__(self, row, value):
        lane = self._lane
        value = np.asarray(value, np.float64).reshape(-1, mujoco.mj_stateSize(lane.model, _INTEGRATION))
        mujoco.mj_setState(lane.model, lane.data, value[-1], _INTEGRATION)
        lane._time[0] = lane.data.time


class OneLane:
    """One lane on one ``MjData``, with the ``NativeBatch`` calls that the Cube/Scene batch makes.

    A step is OGBench's own: ``mj_step`` for ``frame_skip`` substeps, then one ``mj_rnePostConstraint``. A batch
    reads the contact forces through a sensor that runs ``mj_rnePostConstraint`` in every substep instead.

    Args:
        model: The lane's model. The Variation writes its per-lane fields into it.
        frame_skip: Substeps per step.
    """

    def __init__(self, model, frame_skip):
        self.model, self.frame_skip = model, int(frame_skip)
        self.data = mujoco.MjData(model)
        self.b = self
        self.state = _IntegrationState(self)
        # MjData holds the time as a float; the State reads it from this array.
        self._time = np.array([self.data.time])

    def bind(self, name):
        return self._time[None] if name == "time" else getattr(self.data, name)[None]

    def numpy(self, name):
        return getattr(self.data, name)[None]

    def expand(self, name):
        return getattr(self.model, name)[None]

    def step(self, active=None):
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)
        self._time[0] = self.data.time

    def forward(self, env_ids=None):
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        self.data = None


#: Smallest batch whose OGBench objects live in worker processes, unless the ``worker_min_lanes`` knob is set. A
#: reset of one lane runs milliseconds of OGBench's Python, many times the round trip to a worker, so two lanes that
#: reset in parallel take less time than one process that resets them in turn.
WORKER_MIN_LANES = 2


def make_lanes(family, num_envs, seed, reward_task_id):
    """The OGBench objects that reset, restore and vary the lanes: in worker processes from two lanes."""
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, path="ogb-sim", min_lanes=WORKER_MIN_LANES)
    if count:
        from .manipulation_process import OGBManipulationProcessAdapter

        return OGBManipulationProcessAdapter(family, num_envs, "cpu", seed, reward_task_id, processes=count, cpus=cpus)
    return OGBManipulationAdapter(family, num_envs, "cpu", seed, reward_task_id)


def metadata_env(family):
    """One reset OGBench object for the model, the controller and the ids that the step reads."""
    env = (CompatibleCube if family == "Cube" else CompatibleScene)(seed=0)
    env.reset(seed=0)
    if env._success_timing != "post" or env._mode != "task" or not env._terminate_at_goal:
        raise ValueError("The Cube/Scene batch follows OGBench's task mode with post-step success")
    return env


class LaneModels:
    """The model fields that differ between lanes, from State: compiled Variation values, Scene locks and targets.

    The methods read ``{group key: (B, size) array}`` host groups of the State.
    """

    def __init__(self, env, names, schema):
        self.family, self.model, self.schema = env.family, env.model, schema
        self.names = frozenset(names)
        self._base_key = values_key(env._model_values)
        if self.family == "Scene":
            self._lock_dofs = [int(env.model.joint(n).dofadr[0]) for n in ("drawer_slide", "window_slide")]
            self._targets = [int(env._drawer_target_site_id), int(env._window_target_site_id)]

    def _compiled(self, columns):
        return {name: self.schema.get(columns, field("model", "compiled", name)) for name in MODEL_VALUES}

    def variants(self, columns):
        """``{key: (compiled values, lane ids)}`` of the lanes whose compiled values differ from the model's."""
        compiled = self._compiled(columns)
        groups = {}
        for i in range(len(compiled[MODEL_VALUES[0]])):
            values = {name: value[i] for name, value in compiled.items()}
            key = values_key(values)
            if key != self._base_key:
                groups.setdefault(key, (values, []))[1].append(i)
        return groups

    def fields(self, columns):
        """``{name: (B, ...) values}`` of the per-lane model fields among ``names``."""
        at = self.schema.get
        n = len(at(columns, field("model", "compiled", MODEL_VALUES[0])))
        out = {}
        for values, lanes in self.variants(columns).values():
            fields, _ = compiled_fields(self.family, values)
            for name, value in fields.items():
                if name not in self.names:
                    continue
                if name not in out:
                    out[name] = np.repeat(getattr(self.model, name)[None], n, axis=0)
                out[name][lanes] = value
        if self.family == "Scene":
            damping = np.repeat(self.model.dof_damping[None], n, axis=0)
            damping[:, self._lock_dofs] = np.where(at(columns, "scene/cur_button_states") == 0, 1e6, 2.0)
            out["dof_damping"] = damping
            sites = np.repeat(self.model.site_pos[None], n, axis=0)
            sites[:, self._targets[0], 1] = at(columns, "scene/target_drawer_pos")[:, 0]
            sites[:, self._targets[1], 0] = at(columns, "scene/target_window_pos")[:, 0]
            out["site_pos"] = sites
        return out


class OGBManipulationBatchAdapter:
    """Cube/Scene lanes stepped in one ``mujoco.batch``; OGBench objects reset, restore and vary them.

    The step repeats ``ManipSpaceEnv.step``: the controller, ``mj_step`` substeps, ``mj_rnePostConstraint`` and
    the family's ``post_step``. The OGBench objects (in worker processes for large batches) hand over full State
    after a reset or restore, and the batch holds the physics until the next one.
    """

    execution_mode = "DELEGATED_CPU"

    def __init__(self, family, num_envs=1, device="cpu", seed=0, reward_task_id=None, *, threads=None, lanes=None):
        if family not in ("Cube", "Scene") or type(num_envs) is not int or num_envs < 1:
            raise ValueError("Cube/Scene require a positive integer batch")
        self.family, self.num_envs, self.device = family, num_envs, torch.device(device)
        self.cls = CompatibleCube if family == "Cube" else CompatibleScene
        self._closed = False
        self.physics = self._control = None
        # Workers spawn before the batch threads start.
        self.lanes = make_lanes(family, num_envs, seed, reward_task_id) if lanes is None else lanes
        try:
            self._meta = env = metadata_env(family)
            threads = native_threads(num_envs) if threads is None else max(1, min(int(threads), num_envs))
            if num_envs == 1:
                # The lane's model keeps the Variation's fields; _base keeps the defaults.
                self._base = copy.copy(env.model)
                self.physics = OneLane(copy.copy(env.model), int(env._n_steps))
            else:
                self._base = physics_model(env)
                self.physics = NativeBatch(
                    self._base,
                    num_envs,
                    frame_skip=int(env._n_steps),
                    outputs=("site_xpos", "site_xmat", "cfrc_ext"),
                    nthread=threads,
                )
            self._control = BatchController(env, num_envs, threads)
        except BaseException:
            self.close()
            raise
        self.b = self.physics.b
        self.fields, self.nq = self.lanes.fields, self.lanes.nq
        self.schema = schema = state_schema(family, self.fields)
        #: Physics fields in State order, which is ``mjSTATE_INTEGRATION`` order, and their batch views.
        self._physics = tuple(name for name in self.fields if name.startswith("physics/"))
        self._physics_groups = tuple(dict.fromkeys(schema.slices[name].group for name in self._physics))
        self._views = {name: self.b.bind(_BATCH_NAMES[name.split("/")[1]]) for name in self._physics}
        # The other groups live here; the step writes their fields through views cut once.
        self._live = {
            key: np.zeros((num_envs, group.size), torch.empty(0, dtype=group.dtype).numpy().dtype)
            for key, group in schema.groups.items()
            if key not in self._physics_groups
        }
        self._cut_views()
        self._layout = state_layout(schema, num_envs)
        self._all = np.arange(num_envs)
        self._models = LaneModels(env, set(_model_fields()) & {*COMPILED_FIELDS, "dof_damping", "site_pos"}, schema)
        self._expanded = {}
        p = self.physics
        self._qpos, self._qvel, self._ctrl = p.numpy("qpos"), p.numpy("qvel"), p.numpy("ctrl")
        self._mocap_pos = p.numpy("mocap_pos")
        self._out_site_xpos, self._out_site_xmat = p.numpy("site_xpos"), p.numpy("site_xmat")
        self._out_cfrc_ext = p.numpy("cfrc_ext")
        self._pos_sites, self._rot_sites = site_ids(env)
        self._pad = int(env.model.body("ur5e/robotiq/right_pad").id)
        self._cube = int(env.model.joint("object_joint_0").qposadr[0])
        self._cube_dof = int(env.model.joint("object_joint_0").dofadr[0])
        self._target = int(env._cube_target_mocap_ids[0])
        self._workspace = env._workspace_bounds
        self._object_bounds = env._object_sampling_bounds
        self._reward_task = reward_task_id is not None
        if family == "Scene":
            self._buttons = np.array(
                [int(env.model.joint(f"buttonbox_joint_{i}").qposadr[0]) for i in range(env._num_buttons)]
            )
            self._drawer = int(env.model.joint("drawer_slide").qposadr[0])
            self._window = int(env.model.joint("window_slide").qposadr[0])
            self._lock_dofs = np.array(
                [int(env.model.joint(name).dofadr[0]) for name in ("drawer_slide", "window_slide")]
            )
            self._num_button_states = int(env._num_button_states)
        # Lanes whose environment generator a cube repair advanced; the OGBench objects take it before a reset.
        self._dirty = self._stale = False
        self._load(self._lane_state())

    @property
    def envs(self):
        """The in-process OGBench objects, holding the batch's State, or the metadata object when workers own them."""
        envs = getattr(self.lanes, "envs", None)
        if envs is None:
            return [self._meta]
        if self._stale:
            self._sync_envs(envs)
        return envs

    def _cut_views(self):
        """Views of the step's fields in the live groups, cut once."""
        view = lambda name: live_view(self.schema, self._live, name)  # noqa: E731
        self._site_xpos, self._site_xmat = view("controller/site_xpos"), view("controller/site_xmat")
        self._gripper_force = view("output/gripper_force")
        self._prev_qpos, self._prev_qvel = view("output/prev_qpos"), view("output/prev_qvel")
        self._success = view("task/success")
        self._env_rng = {key: view(field("rng", "environment", key)) for key in PCG_KEYS}
        if self.family == "Scene":
            self._buttons_now, self._buttons_before = view("scene/cur_button_states"), view("scene/prev_button_states")
            self._buttons_goal = view("scene/target_button_states")
            self._drawer_goal, self._window_goal = view("scene/target_drawer_pos"), view("scene/target_window_pos")

    def _sync_envs(self, envs):
        """Write the batch's lanes into their OGBench objects, as ``OGBManipulationAdapter`` restores them."""
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        pad = "ur5e/robotiq/right_pad"
        for i, env in enumerate(envs):
            env._prev_qpos, env._prev_qvel = self._prev_qpos[i].copy(), self._prev_qvel[i].copy()
            env._success = bool(self._success[i, 0])
            if self.family == "Scene":
                env._cur_button_states = self._buttons_now[i].copy()
                env._prev_button_states = self._buttons_before[i].copy()
                env._target_button_states = self._buttons_goal[i].copy()
                env._apply_button_states()
            env.np_random.bit_generator.state = pcg_decode({key: value[i] for key, value in self._env_rng.items()})
            for _ in range(2):
                mujoco.mj_setState(env.model, env.data, self.b.state[i], spec)
                mujoco.mj_forward(env.model, env.data)
            mujoco.mj_setState(env.model, env.data, self.b.state[i], spec)
            env.data.site_xpos[self._pos_sites] = self._site_xpos[i]
            env.data.site_xmat[self._rot_sites] = self._site_xmat[i]
            env.data.body(pad).cfrc_ext[:] = self._gripper_force[i]
        self._stale = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("OGBManipulationBatchAdapter is closed")

    # Loading lanes from State.

    def _lane_state(self):
        """The State of the OGBench objects: their columns in process, else a CPU State."""
        if isinstance(self.lanes, OGBManipulationAdapter):
            return self.lanes.columns()
        return self.lanes.get_state()

    def _load(self, state):
        """Write full CPU State, or its ``{group key: array}`` columns, into the batch and the live groups."""
        self._stale = False
        columns = groups_of(self.schema, state)
        for key, value in self._live.items():
            value[...] = columns[key]
        physics = self.schema.views(columns, self._physics, numpy=True)
        parts = [physics[name].astype(np.float64).reshape(self.num_envs, -1) for name in self._physics]
        self.b.state[:] = np.concatenate(parts, axis=1)
        self._load_models(columns)
        success = self._success[:, 0]
        self._outputs = {
            "reward": np.where(success, 1.0, 0.0) if not self._reward_task else np.zeros(self.num_envs),
            "success": success.copy(),
            "terminated": success.copy(),
            "truncated": np.zeros(self.num_envs, bool),
        }

    def _load_models(self, columns):
        fields = self._models.fields(columns)
        for name in set(fields) | set(self._expanded):
            target = self._expanded.get(name)
            if target is None:
                target = self._expanded[name] = self.b.expand(name)
            values = fields[name] if name in fields else getattr(self._base, name)[None]
            # The batch expands bvh_aabb over the bodies' bounding volumes, which precede the meshes'.
            target[...] = values[:, : target.shape[1]] if name == "bvh_aabb" else values

    # State.

    def get_state(self):
        self._check_open()
        return self._state_on(self.device)

    def _state_on(self, device):
        return self._layout.export(self._state_values(), device)

    def _state_values(self):
        """``{group key: (B, size) array}`` of the State: the live groups and the packed physics groups."""
        physics = {name: (value != 0) if name == "physics/eq_active" else value for name, value in self._views.items()}
        return {**self._live, **self.schema.pack(physics, self._physics_groups)}

    def render_columns(self):
        from .render_state import projected_keys

        values = self._state_values()
        return {name: self.schema.get(values, name).copy() for name in projected_keys(self.family)}

    def validate(self, state, **kwargs):
        return self.lanes.validate(state)

    def set_state(self, state):
        self._check_open()
        self.lanes.set_state(state)
        self._dirty = False
        self._load(self._lane_state())

    def make_variation_space(self):
        return self.lanes.make_variation_space()

    def variation_values(self, layout, *, anchor=False):
        return self.lanes.variation_values(layout, anchor=anchor).to(self.device)

    # Lifecycle.

    def _push(self):
        """Give the OGBench objects the State that the batch advanced, when a cube repair drew from a generator."""
        if self._dirty:
            self.lanes.set_state(self.get_state().cpu())
            self._dirty = False

    def reset(self, seed=None, options=None):
        self._check_open()
        self._push()
        if isinstance(self.lanes, OGBManipulationAdapter):
            self._load(self.lanes.reset_columns(seed, options))
        else:
            self._load(self.lanes.reset(seed, options).cpu())
        return self.get_state()

    def configure(self, values):
        self._check_open()
        self._push()
        self.lanes.configure(values)
        current = groups_of(self.schema, self.lanes.get_state())
        for key, value in self._live.items():
            if key.split("/")[0] in ("variation", "model"):
                value[...] = current[key]
        self._load_models(self._live)

    def close(self):
        if self._closed:
            return
        self._closed = True
        for part in (self.physics, self._control, getattr(self, "lanes", None), getattr(self, "_meta", None)):
            if part is not None:
                part.close()
        self.physics = self.b = self._control = None

    # Step.

    def check_step(self, actions, active):
        actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
        if (
            actions.shape != (self.num_envs, 5)
            or actions.dtype not in (np.float32, np.float64)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Actions must be finite float32/float64 [num_envs, 5]")
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or active.dtype != np.bool_:
                raise ValueError("active must be a bool vector with shape [num_envs]")
        return actions, active

    def step(self, actions, *, active=None, capture_state=True):
        result = self.step_host(actions, active=active, capture_state=capture_state)
        # The State and the outputs move to the device in one copy.
        return result if self.device.type == "cpu" else to_device(result, self.device)

    def step_host(self, actions, *, active=None, capture_state=True):
        """``step`` with the result on the CPU, for a caller that moves several steps to the device at once."""
        self._check_open()
        actions, active = self.check_step(actions, active)
        ids = self._all if active is None else np.flatnonzero(active)
        if len(ids):
            self._advance(actions if active is None else actions[ids], ids)
            self._stale = True
        result = {key: torch.from_numpy(value.copy()) for key, value in self._outputs.items()}
        if capture_state:
            result["state"] = self._state_on(_CPU)
        return build_tensordict(result, [self.num_envs], _CPU)

    def step_leaves(self, actions, *, state=True):
        """Advance every lane one step; the State and the outputs as host arrays by key path, for one device copy.

        With ``state`` False, the State is left out. The arrays may be live buffers: the caller copies them before
        the next call.
        """
        self._check_open()
        actions, _ = self.check_step(actions, None)
        self._advance(actions, self._all)
        self._stale = True
        outputs = {(key,): value for key, value in self._outputs.items()}
        if not state:
            return outputs
        return {**outputs, **self._layout.host_leaves(self._state_values(), ("state",))}

    def _advance(self, actions, ids):
        # Every lane steps in most calls; a slice selects them without index copies.
        rows = slice(None) if ids is self._all else ids
        site_xpos, site_xmat = self._site_xpos, self._site_xmat
        # The pinch site is the first cached site row.
        self._control.control(actions, self._qpos, site_xpos[rows, 0], site_xmat[rows, 0], self._ctrl, ids)
        scene = self.family == "Scene"
        if scene:
            self._buttons_before[rows] = self._buttons_now[rows]
            button_pos = self._qpos[rows][:, self._buttons]
        self._prev_qpos[rows] = self._qpos[rows]
        self._prev_qvel[rows] = self._qvel[rows]
        self.physics.step(active=None if ids is self._all else ids)
        self._gripper_force[rows] = self._out_cfrc_ext[rows, self._pad]
        if scene:
            self._scene_post_step(ids, button_pos)
        self._cache_sites(rows)
        cube = self._reached(self._qpos[rows, self._cube : self._cube + 3] - self._mocap_pos[rows, self._target], 0.04)
        if scene:
            buttons = self._buttons_now[rows] == self._buttons_goal[rows]
            drawer = np.abs(self._qpos[rows, self._drawer] - self._drawer_goal[rows, 0]) <= 0.04
            window = np.abs(self._qpos[rows, self._window] - self._window_goal[rows, 0]) <= 0.04
            parts = np.concatenate([cube[:, None], buttons, drawer[:, None], window[:, None]], axis=1)
        else:
            parts = cube[:, None]
        success = parts.all(axis=1)
        self._success[rows, 0] = success
        out = self._outputs
        if self._reward_task:
            out["reward"][rows] = parts.sum(axis=1).astype(np.float64) - parts.shape[1]
        else:
            out["reward"][rows] = np.where(success, 1.0, 0.0)
        out["success"][rows] = success
        out["terminated"][rows] = success
        out["truncated"][rows] = False

    def _cache_sites(self, rows):
        self._site_xpos[rows] = self._out_site_xpos[rows][:, self._pos_sites]
        self._site_xmat[rows] = self._out_site_xmat[rows][:, self._rot_sites]

    def _scene_post_step(self, ids, button_pos):
        """``SceneEnv.post_step``: cube repair, button presses and locks, then ``mj_forward``."""
        cube = self._qpos[ids, self._cube : self._cube + 3]
        unhealthy = ((cube <= self._workspace[0] - 0.2) | (cube >= self._workspace[1] + 0.2)).any(axis=1)
        if unhealthy.any():
            self._repair(ids[unhealthy])
        pressed = (button_pos > -0.02) & (self._qpos[ids][:, self._buttons] <= -0.02)
        states = self._buttons_now
        states[ids] = np.where(pressed, (states[ids] + 1) % self._num_button_states, states[ids])
        damping = self._expanded.get("dof_damping")
        if damping is None:
            damping = self._expanded["dof_damping"] = self.b.expand("dof_damping")
        damping[ids[:, None], self._lock_dofs] = np.where(states[ids] == 0, 1e6, 2.0)
        self.physics.forward(None if ids is self._all else ids)

    def _repair(self, ids):
        """OGBench's reset of a cube that left the workspace, drawing from the lane's environment generator."""
        rng = np.random.Generator(np.random.PCG64())
        for i in ids:
            print("Numerical instability detected. Resetting cube positions.", flush=True)
            rng.bit_generator.state = pcg_decode({key: value[i] for key, value in self._env_rng.items()})
            xy = rng.uniform(*self._object_bounds)
            yaw = rng.uniform(0, 2 * np.pi)
            self._qpos[i, self._cube : self._cube + 3] = (*xy, 0.02)
            self._qpos[i, self._cube + 3 : self._cube + 7] = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            self._qvel[i, self._cube_dof : self._cube_dof + 6] = 0.0
            for key, value in pcg_encode(rng).items():
                self._env_rng[key][i] = value
        self._dirty = True
        self.physics.forward(ids)

    def _reached(self, delta, threshold):
        """``np.linalg.norm(delta) <= threshold`` per row, as the source decides it."""
        norm = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1] + delta[:, 2] * delta[:, 2])
        result = norm <= threshold
        for i in np.flatnonzero(np.abs(norm - threshold) <= 1e-9 * threshold):
            result[i] = np.linalg.norm(delta[i]) <= threshold
        return result
