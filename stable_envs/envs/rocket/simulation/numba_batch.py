"""Rocket batches stepped by the Numba rigid-body kernel on the CPU."""

from __future__ import annotations

import copy
import functools

import numpy as np
import torch
from numba import njit
from tensordict import TensorDict

from ...._core.space_nodes import nodes
from ...._execution.process import leaf_tensordict
from ...._execution.protocol import SimulatorAdapter
from . import numba_physics as physics
from .actions import validate_actions
from .replay import IMMEDIATE, PAINTED, RocketReplay
from .task import RocketLandingEnv

#: dtype codes of Variation values: Python float, float32, float64, uint8 (as ``replay.value_code``).
_CODES = (None, np.dtype("float32"), np.dtype("float64"), np.dtype("uint8"))
_SPAWN_LIMIT = 2.0
_ENTRY_SPEED = -30.0
_IGNITION_RGBA = np.array([1.0, 0.3, 0.3, 0.8])
#: ``render/rgba`` rows of the painted Variation leaves.
_RGBA_ROW = {path: i for i, path in enumerate(PAINTED)}
_ROCKET_PARTS = PAINTED[:4]
#: State leaves ``physics/<group>`` that hold the kernel fields of each dtype.
_PACKED = {"float": np.dtype(np.float64), "int": np.dtype(np.int64), "bool": np.dtype(np.bool_)}
_VARIATION_LEAVES = ("value", "value_dtype", "init_value", "init_value_dtype")


def _code(value):
    if type(value) is float:
        return 0
    dtype = np.asarray(value).dtype
    if dtype not in _CODES[1:]:
        raise ValueError("Unsupported Rocket Variation representation")
    return _CODES.index(dtype)


def _decode(value, code):
    if int(code) == 0:
        return float(value)
    return np.asarray(value).astype(_CODES[int(code)])


@njit(cache=True)
def _camera_poses(camera, rows, position, velocity, rotation, dt):
    """Write the link 0 pose that PyBullet caches before the last substep into ``camera[rows]``.

    The position is the one before the last semi-implicit Euler update. The quaternion comes from the cached
    row-major rotation, as Bullet's ``btMatrix3x3::getRotation`` makes it.
    """
    q = np.empty(4)
    for row in rows:
        for i in range(3):
            camera[row, i] = position[row, i] - dt * velocity[row, i]
        m = rotation[row]
        trace = m[0] + m[4] + m[8]
        if trace > 0.0:
            s = np.sqrt(trace + 1.0)
            q[3] = s * 0.5
            s = 0.5 / s
            q[0] = (m[7] - m[5]) * s
            q[1] = (m[2] - m[6]) * s
            q[2] = (m[3] - m[1]) * s
        else:
            if m[0] < m[4]:
                i = 2 if m[4] < m[8] else 1
            else:
                i = 2 if m[0] < m[8] else 0
            j = (i + 1) % 3
            k = (i + 2) % 3
            s = np.sqrt(m[4 * i] - m[4 * j] - m[4 * k] + 1.0)
            q[i] = s * 0.5
            s = 0.5 / s
            q[3] = (m[3 * k + j] - m[3 * j + k]) * s
            q[j] = (m[3 * j + i] + m[3 * i + j]) * s
            q[k] = (m[3 * k + i] + m[3 * i + k]) * s
        for i in range(4):
            camera[row, 3 + i] = q[i]


@functools.cache
def _scene():
    """A TinyRenderer scene, used to draw frames and to read the colours PyBullet draws."""
    from ..render import tiny_scene

    return tiny_scene()


@functools.cache
def _default_rgba():
    """The colours of the source's unpainted rocket, pad and flame, in ``render/rgba`` row order."""
    import pybullet as bullet

    client, (_, rocket, pad), _ = _scene()
    visuals = {v[1]: np.array(v[7]) for v in bullet.getVisualShapeData(rocket, physicsClientId=client)}
    pad_rgba = np.array(bullet.getVisualShapeData(pad, physicsClientId=client)[0][7])
    return np.stack([visuals[-1], visuals[2], visuals[6], visuals[1], pad_rgba, visuals[9]])


@functools.cache
def _painted_rgba(path, value):
    """The ``render/rgba`` row that ``replay.paint`` draws for colour leaf ``path`` at uint8 ``value``.

    The colour goes through PyBullet as the source paints it, so it has the same rounding.
    """
    import pybullet as bullet

    client, (_, rocket, pad), _ = _scene()
    rgb = np.asarray(value, np.float64) / 255.0
    if path == "pad.color":
        rgba = [float(np.float32(float(f"{c:.3f}"))) for c in rgb] + [1.0]
        body, link = pad, -1
    else:
        rgba = list(rgb) + [1.0]
        part = path.split(".")[1].removesuffix("_color")
        body = rocket
        # ``RocketReplay._read_rgba`` reads the fin, leg and booster colours from these links.
        link = {"body": -1, "fin": 2, "leg": 6, "booster": 1}[part]
    bullet.changeVisualShape(body, link, rgbaColor=rgba, physicsClientId=client)
    row = next(v[7] for v in bullet.getVisualShapeData(body, physicsClientId=client) if v[1] == link)
    return np.array(row)


class _Template:
    """Configuration and Variation space of a batch, with the reset checks of ``RocketReplay``."""

    check_reset = RocketReplay.check_reset

    def __init__(self, config):
        self.config = dict(config or {})
        if self.config.get("angle_representation", "quaternion") != "quaternion":
            raise ValueError("Registered Rocket uses quaternion observations")
        if self.config.get("render_mode") not in (None, "rgb_array"):
            raise ValueError("Registered Rocket supports None or rgb_array")
        if self.config.get("agent_hz", 40) != 40:
            raise ValueError("Rocket simulation_backend='numba' steps at agent_hz=40; use 'pybullet'")
        self.env = RocketLandingEnv(**self.config)
        if self.env.max_steps + 2 < 2:
            raise ValueError("Rocket horizon must be positive")
        self.names = [p[1:].replace("/", ".") for p, _ in nodes(self.env.variation_space) if p]
        self.leaves = [
            (p[1:].replace("/", "."), node)
            for p, node in nodes(self.env.variation_space)
            if not hasattr(node, "spaces")
        ]
        self.registered = {path: self.env.variation_space._walk(path).init_value.copy() for path in PAINTED}

    def recipe(self):
        e = self.env
        return {
            ("recipe", "config"): np.array(
                [
                    e.ceiling,
                    e.max_displacement,
                    self.config.get("max_duration_seconds", 30),
                    self.config.get("agent_hz", 40),
                ],
                np.float64,
            ),
            ("recipe", "flags"): np.array(
                [e.sparse_reward, e.angle_representation, int(e.render_mode is not None)], np.int64
            ),
            ("recipe", "render_resolution"): np.array(e.render_resolution, np.int64),
        }


class RocketNumbaAdapter(SimulatorAdapter):
    """Step every Rocket environment in one Numba kernel call.

    State holds the physical values of the kernel, the position of each environment's PCG64 thrust-noise
    stream, the Variation values and the drawn colours. ``set_state`` copies them back without a replay.
    The noise of each environment comes from its own generator, as in the source: a buffer holds the draws
    of the next agent steps and is refilled from the generator when it runs out.
    """

    action_dim = 7

    def __init__(self, batch, device, seed, config=None, frame=None):
        if type(batch) is not int or batch < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs, self.device = batch, torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Rocket supports CPU dynamics and CPU/CUDA State storage")
        self.frame = frame
        self._template = _Template(config)
        # The kernel fields are views of one array per dtype, so that a State copies three arrays.
        self._packed = {}
        self.state = {}
        #: ``{field: (group, first column, width)}`` in the packed arrays.
        self._columns = {}
        for group, dtype in _PACKED.items():
            names = [k for k, (_, d) in physics.FIELDS.items() if d == dtype]
            widths = [int(np.prod(physics.FIELDS[k][0], dtype=np.int64)) for k in names]
            packed = self._packed[group] = np.zeros((batch, sum(widths)), dtype)
            for name, start, width in zip(names, np.cumsum([0, *widths[:-1]]), widths, strict=True):
                self.state[name] = packed[:, start : start + width].reshape(batch, *physics.FIELDS[name][0])
                self._columns[name] = (group, int(start), width)
        if self.state.keys() != physics.FIELDS.keys():
            raise RuntimeError("Rocket kernel field of an unsupported dtype")
        self._camera = np.zeros((batch, 7))
        self._rgba = np.zeros((batch, 6, 4))
        self._seed = np.full(batch, -1, np.int64)
        self._last = None
        # Variation leaves: ``{path: [value, value code, init value, init code]}``, views of four packed arrays
        # with one row per environment and the leaves in ``template.leaves`` order.
        leaves = self._template.leaves
        widths = [int(np.prod(node.shape, dtype=np.int64)) for _, node in leaves]
        self._values = [np.zeros((batch, sum(widths))), np.zeros((batch, len(leaves)), np.uint8)]
        self._values += [self._values[0].copy(), self._values[1].copy()]
        self._variation = {}
        for i, ((path, node), start, width) in enumerate(
            zip(leaves, np.cumsum([0, *widths[:-1]]), widths, strict=True)
        ):
            self._variation[path] = [
                array[:, start : start + width].reshape(batch, *node.shape) if k % 2 == 0 else array[:, i]
                for k, array in enumerate(self._values)
            ]
            for k in (0, 2):
                self._variation[path][k][:] = np.asarray(node.init_value, np.float64)
                self._variation[path][k + 1][:] = _code(node.init_value)
        self.fields = {key: (value.shape[1:], torch.from_numpy(value).dtype) for key, value in self._leaves().items()}
        self.reset(seed)

    @property
    def template(self):
        """The configuration and Variation space that describe the batch."""
        return self._template

    @property
    def zero_state_leaves(self):
        return frozenset()

    # -- Variation values -------------------------------------------------------------------------------------

    def _value(self, path, lane, init=False):
        value, code, init_value, init_code = self._variation[path]
        return _decode(init_value[lane], init_code[lane]) if init else _decode(value[lane], code[lane])

    def _set(self, path, lane, value, init=False):
        column = self._variation[path]
        offset = 2 if init else 0
        column[offset][lane] = np.asarray(value, np.float64)
        column[offset + 1][lane] = _code(value)

    def configure(self, values):
        from .adapter import check_values

        check_values(self._template, values, self.num_envs)
        for lane, row in enumerate(values):
            for path, value in row.items():
                node = self._template.env.variation_space._walk(path)
                value = node._coerce(copy.deepcopy(value)) if hasattr(node, "_coerce") else value
                self._set(path, lane, value, init=True)
                if path in IMMEDIATE:
                    self._set(path, lane, value)
                if path in PAINTED:
                    self._rgba[lane, _RGBA_ROW[path]] = _painted_rgba(path, tuple(np.asarray(value).tolist()))

    def variation_rows(self, attribute, keys):
        index = 2 if attribute == "init_value" else 0
        return np.concatenate([self._variation[k][index].reshape(self.num_envs, -1) for k in keys], 1)

    # -- reset ------------------------------------------------------------------------------------------------

    def _generator(self, lane):
        """The lane's thrust-noise generator at the position of its next draw."""
        generator = np.random.Generator(np.random.PCG64(0))
        generator.bit_generator.state = physics.generator_state(self.state["rng"][lane])
        return generator

    def _sample_variation(self, lane, seed, selection, values):
        """Draw the Variation values of one environment as ``RocketLandingEnv.reset`` does; return them."""
        space = self._spaces[lane] if hasattr(self, "_spaces") else None
        if space is None:
            if not hasattr(self, "_spaces"):
                self._spaces = [None] * self.num_envs
            space = self._spaces[lane] = copy.deepcopy(self._template.env.variation_space)
        for path, _ in self._template.leaves:
            node = space._walk(path)
            node._init_value = values[path] if path in values else self._value(path, lane, init=True)
        space.seed(seed)
        space.reset()
        space.update(selection)
        return {path: copy.deepcopy(space._walk(path).value) for path, _ in self._template.leaves}

    def reset(self, seed=None, options=None, values=None):
        batch, template = self.num_envs, self._template
        if values is not None and len(values) != batch:
            raise ValueError("Expected one Rocket Variation row per lane")
        for lane in range(batch):
            template.check_reset(
                None if seed is None else seed + lane, options, None if values is None else values[lane]
            )
        options = dict(options or {})
        selection = list(options.get("variation", []))
        randomize = options.get("randomize_drop", False)
        accelerate = options.get("accelerate_drop", True)
        perturb = options.get("randoimize_drop", False)
        controls_offset = any(
            k == "all" or k == "environment" or k.startswith("environment.start_horizontal_offset") for k in selection
        )
        rocket_painted = bool(selection) and ("all" in selection or any("rocket" in k for k in selection))
        pad_painted = rocket_painted or bool(selection) and any("pad" in k for k in selection)
        for path, column in self._variation.items():
            column[0][:], column[1][:] = column[2], column[3]
        if selection or values:
            for lane in range(batch):
                row = {} if values is None else values[lane]
                if selection:
                    lane_seed = None if seed is None else seed + lane
                    for path, value in self._sample_variation(lane, lane_seed, selection, row).items():
                        self._set(path, lane, value)
                else:
                    for path, value in row.items():
                        self._set(path, lane, value)
        height, height_code = self._variation["environment.start_height_ratio"][:2]
        ceiling = template.env.ceiling
        z = np.where(
            height_code == 1,
            (np.float32(ceiling) * height.astype(np.float32)).astype(np.float64),
            ceiling * height,
        )
        offset = self._variation["environment.start_horizontal_offset"][0]
        tilt = self._variation["environment.start_tilt"][0]
        start_pos = np.concatenate([offset, z[:, None]], 1)
        start_orn = np.concatenate([tilt, np.zeros((batch, 1))], 1)
        spawn = np.clip(offset.astype(np.float32), -_SPAWN_LIMIT, _SPAWN_LIMIT).astype(np.float64)
        linear = np.zeros((batch, 3))
        generators = []
        for lane in range(batch):
            if seed is None:
                generator = self._generator(lane) if self.state["rng"][lane].any() else np.random.default_rng()
                self._seed[lane] = -1
            else:
                generator = np.random.default_rng(seed + lane)
                self._seed[lane] = seed + lane
            if randomize or perturb:
                if randomize:
                    limit = template.env.max_displacement * 0.1
                    start_pos[lane, :2] = generator.uniform(-limit, limit, size=(2,))
                    start_pos[lane, 2] = generator.uniform(ceiling * 0.8, ceiling * 0.9)
                    start_orn[lane] = generator.uniform(-0.3, 0.3, size=(3,))
                if perturb:
                    # The task's ``resetBasePositionAndOrientation`` then zeroes these velocities; only the draws stay.
                    generator.uniform(-5.0, 5.0, size=(3,))
                    generator.uniform(-0.5, 0.5, size=(3,))
                if randomize and not controls_offset:
                    spawn[lane] = generator.uniform(-_SPAWN_LIMIT, _SPAWN_LIMIT, size=2)
            generators.append(generator)
        if accelerate:
            linear[:, 2] = _ENTRY_SPEED
        position = np.concatenate([spawn, start_pos[:, 2:]], 1)
        everyone = np.ones(batch, np.bool_)
        physics.start_state(
            self.state,
            everyone,
            dict(
                start_pos=start_pos,
                start_orn=start_orn,
                position=position,
                linear_velocity=linear,
                angular_velocity=np.zeros((batch, 3)),
            ),
        )
        self.state["rng"][:] = physics.rng_words(generators)
        physics.settle(self.state, None, everyone)
        # The source saves and restores the engine after a reset, which caches the link poses at the reset pose.
        self._camera[:, :3] = self.state["position"]
        self._camera[:, 3:] = self.state["quaternion"]
        self._paint(selection, rocket_painted, pad_painted, values)
        self._last = None
        return self.get_state()

    def _paint(self, selection, rocket_painted, pad_painted, values):
        """Set the drawn colours of a reset: the source's own painting, then any configured or explicit colour."""
        self._rgba[:] = _default_rgba()
        for path in PAINTED:
            value, _, init, _ = self._variation[path]
            painted = (rocket_painted if path in _ROCKET_PARTS else pad_painted) | (
                init != self._template.registered[path]
            ).any(1)
            if values is not None:
                painted |= np.array([path in row for row in values])
            for lane in np.flatnonzero(painted):
                self._rgba[lane, _RGBA_ROW[path]] = _painted_rgba(path, tuple(value[lane].astype(np.uint8).tolist()))

    # -- step -------------------------------------------------------------------------------------------------

    def check_step(self, actions, active):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        if not isinstance(actions, np.ndarray) or actions.shape != (self.num_envs, 7):
            raise ValueError("Rocket actions require shape [num_envs, 7]")
        validate_actions(actions)
        if active is not None:
            active = np.asarray(active.detach().cpu() if torch.is_tensor(active) else active, dtype=bool)
        return actions, active

    def step(self, actions, *, active=None, capture_state=True):
        actions, active = self.check_step(actions, active)
        everyone = active is None
        if everyone:
            active = np.ones(self.num_envs, np.bool_)
        state = self.state
        stepped = active & ~(state["termination"] | state["truncation"])
        reward, terminated, truncated, info = physics.step(
            state,
            actions.astype(np.float64, copy=False),
            None,
            active,
            ceiling=self._template.env.ceiling,
            max_displacement=self._template.env.max_displacement,
            max_steps=self._template.env.max_steps,
            sparse_reward=self._template.env.sparse_reward,
        )
        # The camera reads the link pose PyBullet cached before the last substep.
        _camera_poses(
            self._camera,
            np.flatnonzero(stepped),
            state["position"],
            state["linear_velocity"],
            state["link_rotation"],
            physics.DT,
        )
        self._rgba[state["flame_ignited"], 5] = _IGNITION_RGBA
        if not everyone and self._last is not None:
            # Inactive environments repeat their last outputs.
            reward = np.where(active, reward, self._last[0])
            success = np.where(active, info["env_complete"], self._last[1])
        else:
            success = info["env_complete"]
        self._last = (reward, success)
        device = self.device
        host = {("reward",): reward, ("success",): success, ("terminated",): terminated, ("truncated",): truncated}
        if capture_state:
            host.update({("state", *key): value for key, value in self._leaves().items()})
        moved = self._to_device(host)
        outputs = {key[0]: moved[key] for key in host if len(key) == 1}
        if capture_state:
            outputs["state"] = leaf_tensordict(
                {key[1:]: value for key, value in moved.items() if key[0] == "state"}, self.num_envs, device
            )
        return TensorDict._new_unsafe(outputs, batch_size=torch.Size([self.num_envs]), device=device)

    def _to_device(self, host):
        """``{key: host array}`` -> ``{key: new tensor on the device}``; on CUDA, in one copy from pinned memory."""
        if self.device.type == "cpu":
            return {key: torch.from_numpy(np.array(value, copy=True)) for key, value in host.items()}
        layout = getattr(self, "_staging", None)
        if layout is None or layout[0] != tuple(host):
            offsets, total = [], 0
            for value in host.values():
                offsets.append(total)
                total += -(-value.nbytes // 8) * 8
            buffer = torch.empty(total, dtype=torch.uint8, pin_memory=True)
            layout = self._staging = (tuple(host), offsets, buffer, buffer.numpy())
        _, offsets, buffer, staging = layout
        for value, start in zip(host.values(), offsets, strict=True):
            staging[start : start + value.nbytes].view(value.dtype).reshape(value.shape)[...] = value
        moved = buffer.to(self.device)
        return {
            key: moved[start : start + value.nbytes].view(torch.from_numpy(value[:0]).dtype).view(value.shape)
            for (key, value), start in zip(host.items(), offsets, strict=True)
        }

    # -- State ------------------------------------------------------------------------------------------------

    def _leaves(self):
        """``{key tuple: host array}`` of every State leaf; the arrays are views of the live values."""
        if not hasattr(self, "_fixed"):
            batch = self.num_envs
            fixed = {
                key: np.tile(value, (batch,) + (1,) * value.ndim) for key, value in self._template.recipe().items()
            }
            identity = np.zeros((batch, 4))
            identity[:, 3] = 1.0
            pad = np.zeros((batch, 3))
            pad[:, 2] = 0.1
            # The ground and the pad have fixed bases.
            for i, position in ((0, np.zeros((batch, 3))), (2, pad)):
                fixed[("current", "physics", f"body_{i}", "position")] = position
                fixed[("current", "physics", f"body_{i}", "quaternion")] = identity
            self._fixed = fixed
        fixed, state = self._fixed, self.state
        leaves = {key: fixed[key] for key in self._template.recipe()}
        leaves[("recipe", "seed")] = self._seed
        leaves.update({("physics", group): value for group, value in self._packed.items()})
        leaves[("current", "wrapper", "state")] = state["state"]
        for i in range(3):
            for leaf in ("position", "quaternion"):
                key = ("current", "physics", f"body_{i}", leaf)
                leaves[key] = state[leaf] if i == 1 else fixed[key]
        leaves[("current", "diagnostic", "cached_links")] = self._camera[:, None, :]
        leaves[("render", "rgba")] = self._rgba
        leaves.update(
            {("variation", name): value for name, value in zip(_VARIATION_LEAVES, self._values, strict=True)}
        )
        return leaves

    def get_state(self):
        return leaf_tensordict(self._to_device(self._leaves()), self.num_envs, self.device)

    def rgba(self):
        return torch.from_numpy(self._rgba.copy()).to(self.device)

    def validate(self, state):
        """Check a ``[num_envs]`` State; return its host leaves."""
        batch = self.num_envs
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([batch]):
            raise ValueError("Rocket full State requires batch [num_envs]")
        leaves = dict(state.items(True, True))
        if leaves.keys() != self.fields.keys():
            raise ValueError("Rocket State keys differ")
        for key, (shape, dtype) in self.fields.items():
            if leaves[key].shape != (batch, *shape) or leaves[key].dtype != dtype:
                raise ValueError(f"Invalid Rocket State leaf: {key}")
        host = {key: value.detach().cpu().numpy() for key, value in leaves.items()}
        for key, value in host.items():
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError(f"Invalid Rocket State leaf: {key}")
        for key, value in self._template.recipe().items():
            if not (host[key] == value).all():
                raise ValueError("Rocket configuration mismatch: " + "/".join(key))
        group, start, width = self._columns["rng"]
        # A PCG64 increment is odd.
        if not (host[("physics", group)][:, start + width - 1] & 1).all():
            raise ValueError("Invalid Rocket noise generator State")
        if (host[("recipe", "seed")] < -1).any():
            raise ValueError("Rocket seed/counter range")
        rgba = host[("render", "rgba")]
        if ((rgba < 0) | (rgba > 1)).any():
            raise ValueError("Rocket render/rgba must lie in [0, 1]")
        low, high = self._bounds()
        for name in ("value", "init_value"):
            if not np.isin(host[("variation", name + "_dtype")], (0, 1, 2, 3)).all():
                raise ValueError("Invalid Variation representation")
            value = host[("variation", name)]
            if ((value < low) | (value > high)).any():
                raise ValueError("Invalid Rocket Variation value")
        return host

    def _bounds(self):
        """The lowest and highest value of every column of ``variation/value``."""
        nodes = [node for _, node in self._template.leaves]
        low = np.concatenate([np.broadcast_to(np.asarray(n.low, np.float64), n.shape).reshape(-1) for n in nodes])
        high = np.concatenate([np.broadcast_to(np.asarray(n.high, np.float64), n.shape).reshape(-1) for n in nodes])
        return low, high

    def prepare_restore(self, state):
        if self._staged_restore is not None:
            raise RuntimeError("A Rocket restore is already prepared")
        self._staged_restore = self.validate(state)

    def commit_restore(self):
        host = self._staged_restore
        if host is None:
            raise RuntimeError("No Rocket restore is prepared")
        self._staged_restore = None
        self._install(host)

    def set_state(self, state):
        self._install(self.validate(state))

    def _install(self, host):
        for group, value in self._packed.items():
            value[...] = host[("physics", group)]
        self._seed[:] = host[("recipe", "seed")]
        self._camera[:] = host[("current", "diagnostic", "cached_links")][:, 0]
        self._rgba[:] = host[("render", "rgba")]
        for name, value in zip(_VARIATION_LEAVES, self._values, strict=True):
            value[...] = host[("variation", name)]
        self._last = None

    # -- frames -----------------------------------------------------------------------------------------------

    def live_fields(self):
        """Every environment's body poses, camera link pose and colours, as ``render.project`` reads a State."""
        batch = self.num_envs
        body = np.zeros((batch, 3, 7))
        body[:, :, 6] = 1.0
        body[:, 1, :3] = self.state["position"]
        body[:, 1, 3:] = self.state["quaternion"]
        body[:, 2, 2] = 0.1
        return body, self._camera.copy(), self._rgba.copy()

    @property
    def frame_shape(self):
        return None if self.frame is None else (self.frame[0], self.frame[0], 3)

    def draw(self, body, links, colors, size, ss):
        """Draw explicit body poses, camera link poses and colours with TinyRenderer."""
        from ..render import draw_tiny

        return draw_tiny(_scene(), body, links, colors, size, ss)

    def frame_fields(self):
        """Every environment's body poses, camera link pose, colours and float32 camera view matrix."""
        import pybullet

        from ..render import frame_views

        body, links, colors = self.live_fields()
        return {"body": body, "link": links, "rgba": colors, "view": frame_views(links, pybullet)}

    def close(self):
        self._staged_restore = None
