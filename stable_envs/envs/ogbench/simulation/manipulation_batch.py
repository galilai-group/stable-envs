"""OGBench Cube and Scene lanes stepped together in one ``mujoco.batch``, equal to OGBench bit for bit."""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from ogbench.manipspace import lie, mjcf_utils
from tensordict import TensorDict

from ...._core.tensordicts import _device, build_tensordict
from ...._execution.mjbatch import NativeBatch, native_threads
from ...._execution.policy import cpu_affinity, process_count
from .compatibility import COMPILED_FIELDS, MODEL_VALUES, CompatibleCube, CompatibleScene, compiled_fields, values_key
from .manipulation import FLAGS, OGBManipulationAdapter, site_ids
from .manipulation_control import BatchController
from .pcg import pcg_decode, pcg_encode

#: MjData fields of the integration State, in State order, as ``mujoco.batch`` names them.
_BATCH_NAMES = {name: "qacc_warmstart" if name == "warmstart" else name for name in FLAGS}
_BATCH_NAMES["plugin"] = "plugin_state"


class Snapshot:
    """Owned State copies of ``fields``: one array per dtype, copied in one call, and a view of it per leaf."""

    def __init__(self, fields, num_envs):
        self._batch = torch.Size([num_envs])
        groups, self._leaves = {}, []
        for key, (shape, dtype) in fields.items():
            kind = torch.empty((), dtype=dtype).numpy().dtype
            start = groups.get(kind, 0)
            size = num_envs * int(np.prod(shape, dtype=np.int64))
            groups[kind] = start + size
            self._leaves.append((key, kind, start, start + size, (num_envs, *shape)))
        self._groups = {kind: [leaf for leaf in self._leaves if leaf[1] == kind] for kind in groups}
        self._sizes = {
            kind: [stop - start for _, _, start, stop, _ in leaves] for kind, leaves in self._groups.items()
        }
        # Parents first, so that each dictionary exists before its children.
        self._parents = sorted({key[:i] for key, *_ in self._leaves for i in range(1, len(key))}, key=len)

    def build(self, columns, device):
        """A TensorDict of copies of ``columns`` on ``device``: ``{key: (B, ...) values of the field's dtype}``.

        The values are NumPy arrays, or tensors on ``device``.
        """
        nodes = {(): {}}
        for parent in self._parents:
            nodes[parent] = nodes[parent[:-1]][parent[-1]] = {}
        for kind, leaves in self._groups.items():
            if torch.is_tensor(columns[leaves[0][0]]):
                parts = torch.cat([columns[leaf[0]].reshape(-1) for leaf in leaves]).split_with_sizes(
                    self._sizes[kind]
                )
            else:
                flat = np.concatenate([columns[leaf[0]] for leaf in leaves], axis=None, dtype=kind)
                if device.type == "cpu":
                    for key, _, start, stop, shape in leaves:
                        nodes[key[:-1]][key[-1]] = torch.from_numpy(flat[start:stop].reshape(shape))
                    continue
                # One host-to-device copy per dtype.
                parts = torch.from_numpy(flat).to(device).split_with_sizes(self._sizes[kind])
            for (key, _, _, _, shape), part in zip(leaves, parts):
                nodes[key[:-1]][key[-1]] = part.view(shape)
        # Every leaf has the batch and the device, so the TensorDict needs no checks.
        return TensorDict._new_unsafe(nodes[()], batch_size=self._batch, device=_device(device))


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


def make_lanes(family, num_envs, seed, reward_task_id):
    """The OGBench objects that reset, restore and vary the lanes: in worker processes for large batches."""
    cpus = cpu_affinity()
    count = process_count(num_envs, cpus, path="ogb-sim")
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
    """The model fields that differ between lanes, from State: compiled Variation values, Scene locks and targets."""

    def __init__(self, env, names):
        self.family, self.model = env.family, env.model
        self.names = frozenset(names)
        self._base_key = values_key(env._model_values)
        if self.family == "Scene":
            self._lock_dofs = [int(env.model.joint(n).dofadr[0]) for n in ("drawer_slide", "window_slide")]
            self._targets = [int(env._drawer_target_site_id), int(env._window_target_site_id)]

    def keys(self, columns):
        """Each lane's compiled Variation values, as ``compatibility.values_key`` keys them."""
        values = [columns["model", "compiled", name] for name in MODEL_VALUES]
        return [values_key({k: v[i] for k, v in zip(MODEL_VALUES, values)}) for i in range(len(values[0]))]

    def variants(self, columns):
        """``{key: (compiled values, lane ids)}`` of the lanes whose compiled values differ from the model's."""
        groups = {}
        for i, key in enumerate(self.keys(columns)):
            if key != self._base_key:
                groups.setdefault(key, ({k: columns["model", "compiled", k][i] for k in MODEL_VALUES}, []))[1].append(
                    i
                )
        return groups

    def fields(self, columns):
        """``{name: (B, ...) values}`` of the per-lane model fields among ``names``."""
        n = len(columns["model", "compiled", MODEL_VALUES[0]])
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
            damping[:, self._lock_dofs] = np.where(columns["scene", "cur_button_states"] == 0, 1e6, 2.0)
            out["dof_damping"] = damping
            sites = np.repeat(self.model.site_pos[None], n, axis=0)
            sites[:, self._targets[0], 1] = columns["scene", "target_drawer_pos"][:, 0]
            sites[:, self._targets[1], 0] = columns["scene", "target_window_pos"][:, 0]
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
        self._order = tuple(self.fields)
        self._views = {key[1]: self.b.bind(_BATCH_NAMES[key[1]]) for key in self._order if key[0] == "physics"}
        self._snapshot = Snapshot(self.fields, num_envs)
        self._models = LaneModels(env, set(_model_fields()) & {*COMPILED_FIELDS, "dof_damping", "site_pos"})
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
        self._load(self.lanes.get_state())

    @property
    def envs(self):
        """The in-process OGBench objects, holding the batch's State, or the metadata object when workers own them."""
        envs = getattr(self.lanes, "envs", None)
        if envs is None:
            return [self._meta]
        if self._stale:
            self._sync_envs(envs)
        return envs

    def _sync_envs(self, envs):
        """Write the batch's lanes into their OGBench objects, as ``OGBManipulationAdapter`` restores them."""
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        c = self._columns
        pad = "ur5e/robotiq/right_pad"
        for i, env in enumerate(envs):
            env._prev_qpos, env._prev_qvel = c["output", "prev_qpos"][i].copy(), c["output", "prev_qvel"][i].copy()
            env._success = bool(c["task", "success"][i, 0])
            if self.family == "Scene":
                for name in ("cur_button_states", "prev_button_states", "target_button_states"):
                    setattr(env, "_" + name, c["scene", name][i].copy())
                env._apply_button_states()
            env.np_random.bit_generator.state = pcg_decode(
                {k: c["rng", "environment", k][i] for k in ("state", "increment", "has_uint32", "uinteger")}
            )
            for _ in range(2):
                mujoco.mj_setState(env.model, env.data, self.b.state[i], spec)
                mujoco.mj_forward(env.model, env.data)
            mujoco.mj_setState(env.model, env.data, self.b.state[i], spec)
            env.data.site_xpos[self._pos_sites] = c["controller", "site_xpos"][i]
            env.data.site_xmat[self._rot_sites] = c["controller", "site_xmat"][i]
            env.data.body(pad).cfrc_ext[:] = c["output", "gripper_force"][i]
        self._stale = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError("OGBManipulationBatchAdapter is closed")

    # Loading lanes from State.

    def _load(self, state):
        """Write full CPU State into the batch, the per-lane model fields and the State columns."""
        self._stale = False
        columns = {key: state[key].numpy() for key in self._order}
        self._columns = {key: value.copy() for key, value in columns.items() if key[0] != "physics"}
        parts = []
        for key in self._order:
            if key[0] == "physics":
                parts.append(columns[key].astype(np.float64).reshape(self.num_envs, -1))
        self.b.state[:] = np.concatenate(parts, axis=1)
        self._load_models(columns)
        success = self._columns["task", "success"][:, 0]
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
        columns = dict(self._columns)
        for name, value in self._views.items():
            columns["physics", name] = (value != 0) if name == "eq_active" else value
        return self._snapshot.build(columns, self.device)

    def render_columns(self):
        from .render_state import projected_keys

        state = self.get_state()
        return {key: state[key].detach().cpu().numpy().copy() for key in projected_keys(self.family)}

    def validate(self, state, **kwargs):
        return self.lanes.validate(state)

    def set_state(self, state):
        self._check_open()
        self.lanes.set_state(state)
        self._dirty = False
        self._load(self.lanes.get_state())

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
        self._load(self.lanes.reset(seed, options).cpu())
        return self.get_state()

    def configure(self, values):
        self._check_open()
        self._push()
        self.lanes.configure(values)
        current = self.lanes.get_state()
        for key in self._order:
            if key[0] in ("variation", "model"):
                self._columns[key] = current[key].numpy().copy()
        self._load_models({**self._columns})

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
        self._check_open()
        actions, active = self.check_step(actions, active)
        ids = np.arange(self.num_envs) if active is None else np.flatnonzero(active)
        if len(ids):
            self._advance(actions[ids], ids)
            self._stale = True
        result = {key: torch.from_numpy(value.copy()).to(self.device) for key, value in self._outputs.items()}
        if capture_state:
            result["state"] = self.get_state()
        return build_tensordict(result, [self.num_envs], self.device)

    def _advance(self, actions, ids):
        c = self._columns
        site_xpos, site_xmat = c["controller", "site_xpos"], c["controller", "site_xmat"]
        # The pinch site is the first cached site row.
        self._control.control(actions, self._qpos, site_xpos[ids, 0], site_xmat[ids, 0], self._ctrl, ids)
        scene = self.family == "Scene"
        if scene:
            c["scene", "prev_button_states"][ids] = c["scene", "cur_button_states"][ids]
            button_pos = self._qpos[ids][:, self._buttons]
        c["output", "prev_qpos"][ids] = self._qpos[ids]
        c["output", "prev_qvel"][ids] = self._qvel[ids]
        self.physics.step(active=ids)
        c["output", "gripper_force"][ids] = self._out_cfrc_ext[ids, self._pad]
        if scene:
            self._scene_post_step(ids, button_pos)
        self._cache_sites(ids)
        cube = self._reached(self._qpos[ids, self._cube : self._cube + 3] - self._mocap_pos[ids, self._target], 0.04)
        if scene:
            buttons = c["scene", "cur_button_states"][ids] == c["scene", "target_button_states"][ids]
            drawer = np.abs(self._qpos[ids, self._drawer] - c["scene", "target_drawer_pos"][ids, 0]) <= 0.04
            window = np.abs(self._qpos[ids, self._window] - c["scene", "target_window_pos"][ids, 0]) <= 0.04
            parts = np.concatenate([cube[:, None], buttons, drawer[:, None], window[:, None]], axis=1)
        else:
            parts = cube[:, None]
        success = parts.all(axis=1)
        c["task", "success"][ids, 0] = success
        out = self._outputs
        if self._reward_task:
            out["reward"][ids] = parts.sum(axis=1).astype(np.float64) - parts.shape[1]
        else:
            out["reward"][ids] = np.where(success, 1.0, 0.0)
        out["success"][ids] = success
        out["terminated"][ids] = success
        out["truncated"][ids] = False

    def _cache_sites(self, ids):
        c = self._columns
        c["controller", "site_xpos"][ids] = self._out_site_xpos[ids][:, self._pos_sites]
        c["controller", "site_xmat"][ids] = self._out_site_xmat[ids][:, self._rot_sites]

    def _scene_post_step(self, ids, button_pos):
        """``SceneEnv.post_step``: cube repair, button presses and locks, then ``mj_forward``."""
        c = self._columns
        cube = self._qpos[ids, self._cube : self._cube + 3]
        unhealthy = ((cube <= self._workspace[0] - 0.2) | (cube >= self._workspace[1] + 0.2)).any(axis=1)
        if unhealthy.any():
            self._repair(ids[unhealthy])
        pressed = (button_pos > -0.02) & (self._qpos[ids][:, self._buttons] <= -0.02)
        states = c["scene", "cur_button_states"]
        states[ids] = np.where(pressed, (states[ids] + 1) % self._num_button_states, states[ids])
        damping = self._expanded.get("dof_damping")
        if damping is None:
            damping = self._expanded["dof_damping"] = self.b.expand("dof_damping")
        damping[ids[:, None], self._lock_dofs] = np.where(states[ids] == 0, 1e6, 2.0)
        self.physics.forward(ids)

    def _repair(self, ids):
        """OGBench's reset of a cube that left the workspace, drawing from the lane's environment generator."""
        rng = np.random.Generator(np.random.PCG64())
        columns = self._columns
        for i in ids:
            print("Numerical instability detected. Resetting cube positions.", flush=True)
            rng.bit_generator.state = pcg_decode(
                {k: columns["rng", "environment", k][i] for k in ("state", "increment", "has_uint32", "uinteger")}
            )
            xy = rng.uniform(*self._object_bounds)
            yaw = rng.uniform(0, 2 * np.pi)
            self._qpos[i, self._cube : self._cube + 3] = (*xy, 0.02)
            self._qpos[i, self._cube + 3 : self._cube + 7] = lie.SO3.from_z_radians(yaw).wxyz.tolist()
            self._qvel[i, self._cube_dof : self._cube_dof + 6] = 0.0
            for key, value in pcg_encode(rng).items():
                columns["rng", "environment", key][i] = value
        self._dirty = True
        self.physics.forward(ids)

    def _reached(self, delta, threshold):
        """``np.linalg.norm(delta) <= threshold`` per row, as the source decides it."""
        norm = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1] + delta[:, 2] * delta[:, 2])
        result = norm <= threshold
        for i in np.flatnonzero(np.abs(norm - threshold) <= 1e-9 * threshold):
            result[i] = np.linalg.norm(delta[i]) <= threshold
        return result
