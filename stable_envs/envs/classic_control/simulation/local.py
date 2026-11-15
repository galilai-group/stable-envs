"""Gymnasium classic-control simulation with complete continuation state."""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from ...._core.state_schema import Field, StateSchema
from ...._core.tensordicts import build_tensordict, to_device
from ...._execution.process import PER_LANE
from ...._execution.protocol import SimulatorAdapter

_NUMPY_DTYPES = {torch.float64: np.float64, torch.int64: np.int64, torch.bool: np.bool_}

#: State field names that code outside the schema refers to.
ELAPSED = "lifecycle/elapsed_steps"
BEYOND = "task/steps_beyond_terminated"
HAS_BEYOND = "task/has_steps_beyond_terminated"
HAS_LAST_U = "task/has_last_u"
IS_FLOAT32 = "task/state_is_float32"
RNG_STATE, RNG_INCREMENT, RNG_HAS_UINT32, RNG_UINTEGER = (
    "rng/state",
    "rng/increment",
    "rng/has_uint32",
    "rng/uinteger",
)

_RNG_FIELDS = {
    RNG_STATE: (4, torch.int64),
    RNG_INCREMENT: (4, torch.int64),
    RNG_HAS_UINT32: (1, torch.int64),
    RNG_UINTEGER: (1, torch.int64),
}

_PHYSICS_SPACES = {
    "cartpole": {
        "gravity": (4.9, 14.7, 9.8),
        "masscart": (0.5, 1.5, 1.0),
        "masspole": (0.05, 0.5, 0.1),
        "length": (0.25, 0.75, 0.5),
        "force_mag": (5.0, 15.0, 10.0),
        "tau": (0.01, 0.04, 0.02),
    },
    "pendulum": {
        "g": (5.0, 15.0, 10.0),
        "m": (0.5, 1.5, 1.0),
        "l": (0.5, 1.5, 1.0),
        "max_torque": (1.0, 4.0, 2.0),
        "max_speed": (4.0, 12.0, 8.0),
        "dt": (0.025, 0.1, 0.05),
    },
    "acrobot": {
        "link_length_1": (0.5, 1.5, 1.0),
        "link_length_2": (0.5, 1.5, 1.0),
        "link_mass_1": (0.5, 1.5, 1.0),
        "link_mass_2": (0.5, 1.5, 1.0),
        "link_com_pos_1": (0.25, 0.75, 0.5),
        "link_com_pos_2": (0.25, 0.75, 0.5),
        "link_moi": (0.5, 1.5, 1.0),
    },
    "mountain_car": {
        "gravity": (0.00125, 0.00375, 0.0025),
        "force": (0.0005, 0.0015, 0.001),
        "max_speed": (0.035, 0.105, 0.07),
    },
    "mountain_car_continuous": {
        "power": (0.00075, 0.00225, 0.0015),
        "max_speed": (0.035, 0.105, 0.07),
    },
}

#: State fields that only a reset, a restore or a physics change sets: the PCG64 generator (classic-control steps
#: draw no random numbers) and the physical constants.
_STATIC_PREFIXES = ("rng/", "physics/")

_PHYSICS_ATTRIBUTES = {
    "acrobot": {
        "link_length_1": "LINK_LENGTH_1",
        "link_length_2": "LINK_LENGTH_2",
        "link_mass_1": "LINK_MASS_1",
        "link_mass_2": "LINK_MASS_2",
        "link_com_pos_1": "LINK_COM_POS_1",
        "link_com_pos_2": "LINK_COM_POS_2",
        "link_moi": "LINK_MOI",
    }
}


def physics_field(name: str) -> str:
    """The State field that holds the physical constant ``name``."""
    return f"physics/{name}"


def _physics_fields(domain: str) -> dict:
    return {physics_field(name): (1, torch.float64) for name in _PHYSICS_SPACES[domain]}


CLASSIC_FIELDS = {
    "cartpole": {
        "x": (1, torch.float64),
        "x_dot": (1, torch.float64),
        "theta": (1, torch.float64),
        "theta_dot": (1, torch.float64),
        ELAPSED: (1, torch.int64),
        BEYOND: (1, torch.int64),
        HAS_BEYOND: (1, torch.bool),
        **_RNG_FIELDS,
        **_physics_fields("cartpole"),
    },
    "pendulum": {
        "theta": (1, torch.float64),
        "theta_dot": (1, torch.float64),
        # ``last_u`` keeps the caller's float dtype; float64 holds float32 and float64 actions losslessly.
        "torque": (1, torch.float64),
        HAS_LAST_U: (1, torch.bool),
        ELAPSED: (1, torch.int64),
        **_RNG_FIELDS,
        **_physics_fields("pendulum"),
    },
    "acrobot": {
        "theta1": (1, torch.float64),
        "theta2": (1, torch.float64),
        "dtheta1": (1, torch.float64),
        "dtheta2": (1, torch.float64),
        ELAPSED: (1, torch.int64),
        **_RNG_FIELDS,
        **_physics_fields("acrobot"),
    },
    "mountain_car": {
        "position": (1, torch.float64),
        "velocity": (1, torch.float64),
        ELAPSED: (1, torch.int64),
        **_RNG_FIELDS,
        **_physics_fields("mountain_car"),
    },
    "mountain_car_continuous": {
        "position": (1, torch.float64),
        "velocity": (1, torch.float64),
        # Reset installs float64 state and every step float32; the dtype changes the next NumPy update.
        IS_FLOAT32: (1, torch.bool),
        ELAPSED: (1, torch.int64),
        **_RNG_FIELDS,
        **_physics_fields("mountain_car_continuous"),
    },
}

_GROUP_DTYPE = {torch.float64: "float64", torch.int64: "int64", torch.bool: "bool"}


def _schema(domain: str) -> StateSchema:
    """The State of ``domain``: one group per dtype, with the simulation variables first in the float64 group."""
    fields = [
        Field(name, (width,), dtype, f"{domain}/{_GROUP_DTYPE[dtype]}")
        for dtype in _GROUP_DTYPE
        for name, (width, kind) in CLASSIC_FIELDS[domain].items()
        if kind == dtype
    ]
    return StateSchema(f"ClassicControl.{domain}", fields)


#: The State schema of each domain.
STATES = {domain: _schema(domain) for domain in CLASSIC_FIELDS}


#: The State fields that hold each task's ``state`` array, in its order.
_STATE_NAMES = {
    "cartpole": ("x", "x_dot", "theta", "theta_dot"),
    "pendulum": ("theta", "theta_dot"),
    "acrobot": ("theta1", "theta2", "dtheta1", "dtheta2"),
    "mountain_car": ("position", "velocity"),
    "mountain_car_continuous": ("position", "velocity"),
}

_ENV_IDS = {
    "cartpole": "CartPole-v1",
    "pendulum": "Pendulum-v1",
    "acrobot": "Acrobot-v1",
    "mountain_car": "MountainCar-v0",
    "mountain_car_continuous": "MountainCarContinuous-v0",
}


def _limbs(value: int) -> np.ndarray:
    """Split a PCG64 unsigned 128-bit integer into four uint32 limbs, least significant first."""
    value = int(value)
    return np.asarray([(value >> (32 * i)) & 0xFFFF_FFFF for i in range(4)], dtype=np.int64)


def _from_limbs(value) -> int:
    parts = np.asarray(value, dtype=np.int64).reshape(4)
    return sum(int(part) << (32 * i) for i, part in enumerate(parts))


def _host_fields(domain: str, state: TensorDict) -> dict[str, np.ndarray]:
    """``{field name: (rows, width) array}`` of a validated State, as views of its host group tensors."""
    return STATES[domain].views(state.cpu(), numpy=True)


def _rng_state(fields: dict[str, np.ndarray], lane: int) -> dict:
    """The PCG64 state of one lane, as NumPy's ``bit_generator.state`` takes it."""
    return {
        "bit_generator": "PCG64",
        "state": {"state": _from_limbs(fields[RNG_STATE][lane]), "inc": _from_limbs(fields[RNG_INCREMENT][lane])},
        "has_uint32": fields[RNG_HAS_UINT32][lane, 0].item(),
        "uinteger": fields[RNG_UINTEGER][lane, 0].item(),
    }


def validate_classic_state(domain: str, state, *, batch: int | None = None) -> None:
    """Require the complete, typed resume schema; compact render rows fail."""
    if not isinstance(state, TensorDict):
        raise TypeError("Expected full State (TensorDict); compact RenderState rows cannot restore dynamics")
    if len(state.batch_size) != 1 or (batch is not None and state.batch_size != torch.Size([batch])):
        raise ValueError(f"Expected State batch [{batch or 'N'}], got {list(state.batch_size)}")
    STATES[domain].validate(state, finite=True)


class ClassicControlAdapter(SimulatorAdapter):
    """B independent pinned Gymnasium simulators, with no rendering responsibility."""

    process_methods = {"configure": PER_LANE}
    process_attributes = ("domain", "fields")

    def __init__(self, domain: str, num_envs: int, device="cpu", seed: int = 0):
        if domain not in CLASSIC_FIELDS:
            raise ValueError(f"Unsupported classic-control domain {domain!r}")
        self.domain = domain
        self.fields = CLASSIC_FIELDS[domain]
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.envs = [gym.make(_ENV_IDS[domain]) for _ in range(self.num_envs)]
        self._physics_names = tuple(_PHYSICS_SPACES[domain])
        self._physics_attributes = {
            name: _PHYSICS_ATTRIBUTES.get(domain, {}).get(name, name) for name in self._physics_names
        }
        self._physics = np.asarray(
            [
                [float(getattr(env.unwrapped, self._physics_attributes[name])) for name in self._physics_names]
                for env in self.envs
            ],
            np.float64,
        ).reshape(self.num_envs, len(self._physics_names))
        self._physics_keys = tuple(physics_field(name) for name in self._physics_names)
        self._static_keys = frozenset(key for key in self.fields if key.startswith(_STATIC_PREFIXES))
        from .packed import PackedState

        self._packed = PackedState(domain, self.num_envs)
        self._bases = [env.unwrapped for env in self.envs]
        self._static_loaded = False
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("ClassicControlAdapter is closed")

    def configure(self, values: Mapping[str, np.ndarray]) -> None:
        """Apply each lane's physics variation values to its Gymnasium env."""
        self._check_open()
        columns = {name: np.asarray(rows) for name, rows in values.items()}
        for i in range(self.num_envs):
            self._apply_physics(i, {name: float(rows[i]) for name, rows in columns.items()})

    def _apply_physics(self, lane: int, values: Mapping[str, float]) -> None:
        self._static_loaded = False
        base = self.envs[lane].unwrapped
        for name, value in values.items():
            setattr(base, self._physics_attributes[name], value)
            self._physics[lane, self._physics_names.index(name)] = value
        if self.domain == "cartpole":
            base.total_mass = base.masspole + base.masscart
            base.polemass_length = base.masspole * base.length

    def reset(self, seed: int | None = None, options: dict | None = None) -> TensorDict:
        self._check_open()
        for i, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else int(seed) + i, options=options)
        self._static_loaded = False
        return ClassicControlAdapter.get_state(self)

    def _state_values(self, env, static: bool = True):
        """The State values of one lane; ``static=False`` leaves out the generator."""
        base = env.unwrapped
        values: dict = {ELAPSED: [env._elapsed_steps]}
        if static:
            rng = base.np_random.bit_generator.state
            if rng["bit_generator"] != "PCG64":
                raise RuntimeError(f"Expected PCG64, got {rng['bit_generator']!r}")
            values[RNG_STATE] = _limbs(rng["state"]["state"])
            values[RNG_INCREMENT] = _limbs(rng["state"]["inc"])
            values[RNG_HAS_UINT32] = [rng["has_uint32"]]
            values[RNG_UINTEGER] = [rng["uinteger"]]
        if self.domain == "cartpole":
            values.update(dict(zip(("x", "x_dot", "theta", "theta_dot"), np.asarray(base.state))))
            beyond = base.steps_beyond_terminated
            values[BEYOND] = [0 if beyond is None else beyond]
            values[HAS_BEYOND] = [beyond is not None]
        elif self.domain == "pendulum":
            values["theta"], values["theta_dot"] = np.asarray(base.state)
            last_u = getattr(base, "last_u", None)
            values["torque"] = [0.0 if last_u is None else last_u]
            values[HAS_LAST_U] = [last_u is not None]
        elif self.domain == "acrobot":
            values.update(dict(zip(("theta1", "theta2", "dtheta1", "dtheta2"), np.asarray(base.state))))
        else:
            values["position"], values["velocity"] = np.asarray(base.state)
            if self.domain == "mountain_car_continuous":
                values[IS_FLOAT32] = [np.asarray(base.state).dtype == np.float32]

        return values

    def _columns(self, keys) -> dict:
        """``{key: (num_envs, width) array}`` of ``keys``, read from each lane's Gymnasium environment."""
        columns = {
            key: np.empty((self.num_envs, self.fields[key][0]), dtype=_NUMPY_DTYPES[self.fields[key][1]])
            for key in keys
            if key not in self._physics_keys
        }
        lane_columns = list(columns.items())
        for index, env in enumerate(self.envs):
            values = self._state_values(env, static=any(key.startswith("rng/") for key in columns))
            for key, column in lane_columns:
                column[index] = values[key]
        for j, key in enumerate(self._physics_keys):
            if key in keys:
                columns[key] = self._physics[:, j : j + 1].copy()
        return columns

    def _load_static(self) -> None:
        if not self._static_loaded:
            views = self._packed.views
            for key, value in self._columns(self._static_keys).items():
                views[key][...] = value
            self._static_loaded = True

    def _write_moving(self) -> None:
        """Copy the fields that steps change from all lanes into the packed State."""
        n, bases, views = self.num_envs, self._bases, self._packed.views
        state = np.array([base.state for base in bases], dtype=np.float64).reshape(n, -1)
        for i, name in enumerate(_STATE_NAMES[self.domain]):
            views[name][:, 0] = state[:, i]
        views[ELAPSED][:, 0] = np.fromiter((env._elapsed_steps for env in self.envs), np.int64, n)
        if self.domain == "cartpole":
            beyond = [base.steps_beyond_terminated for base in bases]
            views[BEYOND][:, 0] = [b or 0 for b in beyond]
            views[HAS_BEYOND][:, 0] = [b is not None for b in beyond]
        elif self.domain == "pendulum":
            last = [getattr(base, "last_u", None) for base in bases]
            views["torque"][:, 0] = [0.0 if u is None else u for u in last]
            views[HAS_LAST_U][:, 0] = [u is not None for u in last]
        elif self.domain == "mountain_car_continuous":
            narrow = [np.asarray(base.state).dtype == np.float32 for base in bases]
            views[IS_FLOAT32][:, 0] = narrow

    def get_state(self) -> TensorDict:
        self._check_open()
        self._load_static()
        self._write_moving()
        return self._packed.export(self.device)

    def stack_states(self, states) -> TensorDict | None:
        """Stack States that ``get_state`` made with one copy per dtype; None for other States."""
        return self._packed.stack(states)

    def validate(self, state: TensorDict) -> None:
        """Check the schema and every lane's PCG64 state; change no simulator."""
        self._check_open()
        validate_classic_state(self.domain, state, batch=self.num_envs)
        probe = np.random.PCG64(0)
        for i in range(self.num_envs):
            probe.state = _rng_state(_host_fields(self.domain, state), i)

    def set_state(self, state: TensorDict) -> None:
        self._check_open()
        validate_classic_state(self.domain, state, batch=self.num_envs)
        host = _host_fields(self.domain, state)
        for i, env in enumerate(self.envs):
            base = env.unwrapped
            self._apply_physics(
                i,
                {
                    name: host[key][i, 0].item()
                    for name, key in zip(self._physics_names, self._physics_keys, strict=True)
                },
            )
            if self.domain == "cartpole":
                base.state = np.asarray([host[name][i, 0].item() for name in _STATE_NAMES["cartpole"]], np.float64)
                base.steps_beyond_terminated = host[BEYOND][i, 0].item() if host[HAS_BEYOND][i, 0].item() else None
            elif self.domain == "pendulum":
                base.state = np.asarray([host["theta"][i, 0].item(), host["theta_dot"][i, 0].item()], np.float64)
                if host[HAS_LAST_U][i, 0].item():
                    base.last_u = np.float64(host["torque"][i, 0].item())
                else:
                    base.last_u = None
            elif self.domain == "acrobot":
                base.state = np.asarray([host[name][i, 0].item() for name in _STATE_NAMES["acrobot"]], np.float64)
            else:
                dtype = (
                    np.float32
                    if self.domain == "mountain_car_continuous" and host[IS_FLOAT32][i, 0].item()
                    else np.float64
                )
                values = [host["position"][i, 0].item(), host["velocity"][i, 0].item()]
                # Only the continuous variant's array dtype affects continuation.
                base.state = np.asarray(values, dtype=dtype)

            env._elapsed_steps = host[ELAPSED][i, 0].item()
            # A state exists only after reset, so restore the wrapper's reset flag too.
            env.env._has_reset = True
            base.np_random.bit_generator.state = _rng_state(host, i)
        self._static_loaded = False

    def _actions(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        array = np.asarray(actions)
        if array.shape != (self.num_envs, 1):
            raise ValueError(f"{self.domain} action must have shape ({self.num_envs}, 1), got {array.shape}")
        if self.domain in {"cartpole", "acrobot", "mountain_car"}:
            # ``item`` keeps the int/float distinction that Gymnasium's Discrete.contains checks.
            return [array[i, 0].item() for i in range(self.num_envs)]
        return [array[i] for i in range(self.num_envs)]

    def check_step(self, actions, active):
        actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
        # Keep source scalar types and Python-list action indexing, including valid negative indices.
        converted = ClassicControlAdapter._actions(self, actions)
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or active.dtype != np.bool_:
                raise ValueError("active must be a bool vector with shape [num_envs]")
        if self.domain == "acrobot":
            # Index the torque table as Acrobot does, so an invalid action raises before any lane steps.
            for i, action in enumerate(converted):
                if active is None or active[i]:
                    [-1.0, 0.0, 1.0][action]
        return actions, active

    def probe_actions(self):
        return np.ones((1, 1), np.int64 if self.domain in {"cartpole", "acrobot", "mountain_car"} else np.float32)

    @staticmethod
    def _idle_output(env):
        space = env.observation_space
        return np.zeros(space.shape, space.dtype), 0.0, False, False, {}

    def step(self, actions, *, active=None, capture_state=True) -> TensorDict:
        active = None if active is None else np.asarray(active.detach().cpu(), dtype=bool)
        self._check_open()
        # An inactive lane does not advance and reports zero task outputs, as Env.step does.
        outputs = [
            env.step(action) if active is None or active[i] else self._idle_output(env)
            for i, (env, action) in enumerate(zip(self.envs, self._actions(actions), strict=True))
        ]
        observations = np.stack([out[0] for out in outputs])
        host = {
            "reward": torch.from_numpy(np.fromiter((out[1] for out in outputs), np.float64, self.num_envs)),
            "terminated": torch.from_numpy(np.fromiter((out[2] for out in outputs), np.bool_, self.num_envs)),
            "truncated": torch.from_numpy(np.fromiter((out[3] for out in outputs), np.bool_, self.num_envs)),
            "observation": torch.from_numpy(observations),
            # ``info['state']`` is a float32 copy of the observation.
            "info_state": torch.from_numpy(observations.astype(np.float32)),
        }
        out = build_tensordict(host, [self.num_envs], "cpu")
        if self.device.type != "cpu":
            out = to_device(out, self.device)
        if capture_state:
            out["state"] = self.get_state()
        return out

    def close(self) -> None:
        if not self._closed:
            for env in self.envs:
                env.close()
            self._closed = True
