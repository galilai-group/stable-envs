"""Exact historical Gymnasium classic-control execution for recovered cells.

The renderer remains native to :mod:`stable_envs.envs.classic_control`.  This
module owns only authoritative CPU execution and the explicit resume state for
the five pinned SWM wrappers.  It imports Gymnasium directly;
production never imports ``stable_worldmodel``.
"""

from __future__ import annotations

from collections.abc import Mapping

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from stable_envs.spaces import Box, Dict
from stable_envs.specs import Spec

_NUMPY_DTYPES = {torch.float64: np.float64, torch.int64: np.int64, torch.bool: np.bool_}

_RNG_FIELDS = {
    ("rng", "state"): (4, torch.int64),
    ("rng", "increment"): (4, torch.int64),
    ("rng", "has_uint32"): (1, torch.int64),
    ("rng", "uinteger"): (1, torch.int64),
}

CLASSIC_FIELDS = {
    "cartpole": {
        "x": (1, torch.float64),
        "x_dot": (1, torch.float64),
        "theta": (1, torch.float64),
        "theta_dot": (1, torch.float64),
        ("lifecycle", "elapsed_steps"): (1, torch.int64),
        ("task", "steps_beyond_terminated"): (1, torch.int64),
        ("task", "has_steps_beyond_terminated"): (1, torch.bool),
        **_RNG_FIELDS,
    },
    "pendulum": {
        "theta": (1, torch.float64),
        "theta_dot": (1, torch.float64),
        # ``last_u`` retains the caller's accepted float dtype.  Float64 is a
        # lossless common representation for both the declared float32 Action
        # and Gymnasium's warning-only acceptance of float64 arrays.
        "torque": (1, torch.float64),
        ("task", "has_last_u"): (1, torch.bool),
        ("lifecycle", "elapsed_steps"): (1, torch.int64),
        **_RNG_FIELDS,
    },
    "acrobot": {
        "theta1": (1, torch.float64),
        "theta2": (1, torch.float64),
        "dtheta1": (1, torch.float64),
        "dtheta2": (1, torch.float64),
        ("lifecycle", "elapsed_steps"): (1, torch.int64),
        **_RNG_FIELDS,
    },
    "mountain_car": {
        "position": (1, torch.float64),
        "velocity": (1, torch.float64),
        ("lifecycle", "elapsed_steps"): (1, torch.int64),
        **_RNG_FIELDS,
    },
    "mountain_car_continuous": {
        "position": (1, torch.float64),
        "velocity": (1, torch.float64),
        # Reset installs float64 state; every step installs float32 state.  The
        # dtype changes the next NumPy update, so it is continuation state.
        ("task", "state_is_float32"): (1, torch.bool),
        ("lifecycle", "elapsed_steps"): (1, torch.int64),
        **_RNG_FIELDS,
    },
}

_ENV_IDS = {
    "cartpole": "CartPole-v1",
    "pendulum": "Pendulum-v1",
    "acrobot": "Acrobot-v1",
    "mountain_car": "MountainCar-v0",
    "mountain_car_continuous": "MountainCarContinuous-v0",
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


def _limbs(value: int) -> np.ndarray:
    """A PCG64 unsigned 128-bit integer as four named-base uint32 limbs."""
    value = int(value)
    return np.asarray([(value >> (32 * i)) & 0xFFFF_FFFF for i in range(4)], dtype=np.int64)


def _from_limbs(value) -> int:
    parts = np.asarray(value, dtype=np.int64).reshape(4)
    return sum(int(part) << (32 * i) for i, part in enumerate(parts))


def validate_classic_state(domain: str, state, *, batch: int | None = None) -> None:
    """Require the complete, typed resume schema; compact render rows fail."""
    fields = CLASSIC_FIELDS[domain]
    if not isinstance(state, TensorDict):
        raise TypeError("Expected full State (TensorDict); compact RenderState rows cannot restore dynamics")
    if len(state.batch_size) != 1 or (batch is not None and state.batch_size != torch.Size([batch])):
        raise ValueError(f"Expected State batch [{batch or 'N'}], got {list(state.batch_size)}")
    have, want = set(state.keys(True, True)), set(fields)
    if have != want:
        missing, extra = sorted(want - have, key=str), sorted(have - want, key=str)
        raise ValueError(
            "Full State keys do not match the classic-control resume schema"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else "")
        )
    n = int(state.batch_size[0])
    for key, (width, dtype) in fields.items():
        value = state[key]
        if value.shape != (n, width):
            raise ValueError(f"State field {key!r} has invalid shape {tuple(value.shape)}")
        if value.dtype != dtype:
            raise ValueError(f"State field {key!r} has dtype {value.dtype}; expected {dtype}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"State field {key!r} contains non-finite values")


class ClassicControlAdapter:
    """B independent pinned Gymnasium simulators, with no rendering responsibility."""

    def __init__(self, domain: str, num_envs: int, device="cpu", seed: int = 0):
        if domain not in CLASSIC_FIELDS:
            raise ValueError(f"Unsupported recovered classic-control domain {domain!r}")
        self.domain = domain
        self.fields = CLASSIC_FIELDS[domain]
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.envs = [gym.make(_ENV_IDS[domain]) for _ in range(self.num_envs)]
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("ClassicControlAdapter is closed")

    def configure(self, values: Mapping[str, np.ndarray]) -> None:
        """Apply one exact SWM physical-Variation row to each Gymnasium env."""
        self._check_open()
        for i, env in enumerate(self.envs):
            base = env.unwrapped
            for name, rows in values.items():
                attribute = _PHYSICS_ATTRIBUTES.get(self.domain, {}).get(name, name)
                setattr(base, attribute, float(np.asarray(rows)[i]))
            if self.domain == "cartpole":
                base.total_mass = base.masspole + base.masscart
                base.polemass_length = base.masspole * base.length

    def reset(self, seed: int | None = None, options: dict | None = None) -> TensorDict:
        self._check_open()
        for i, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else int(seed) + i, options=options)
        return self.get_state()

    def _state_values(self, env):
        base = env.unwrapped
        rng = base.np_random.bit_generator.state
        if rng["bit_generator"] != "PCG64":
            raise RuntimeError(f"Expected PCG64, got {rng['bit_generator']!r}")
        values: dict = {
            ("lifecycle", "elapsed_steps"): [env._elapsed_steps],
            ("rng", "state"): _limbs(rng["state"]["state"]),
            ("rng", "increment"): _limbs(rng["state"]["inc"]),
            ("rng", "has_uint32"): [rng["has_uint32"]],
            ("rng", "uinteger"): [rng["uinteger"]],
        }
        if self.domain == "cartpole":
            values.update(dict(zip(("x", "x_dot", "theta", "theta_dot"), np.asarray(base.state))))
            beyond = base.steps_beyond_terminated
            values[("task", "steps_beyond_terminated")] = [0 if beyond is None else beyond]
            values[("task", "has_steps_beyond_terminated")] = [beyond is not None]
        elif self.domain == "pendulum":
            values["theta"], values["theta_dot"] = np.asarray(base.state)
            last_u = getattr(base, "last_u", None)
            values["torque"] = [0.0 if last_u is None else last_u]
            values[("task", "has_last_u")] = [last_u is not None]
        elif self.domain == "acrobot":
            values.update(dict(zip(("theta1", "theta2", "dtheta1", "dtheta2"), np.asarray(base.state))))
        else:
            values["position"], values["velocity"] = np.asarray(base.state)
            if self.domain == "mountain_car_continuous":
                values[("task", "state_is_float32")] = [np.asarray(base.state).dtype == np.float32]

        return values

    def get_state(self) -> TensorDict:
        self._check_open()
        # The pinned field schema supplies shapes/dtypes once for the batch.
        # Assign source values directly, without a temporary array per lane/field.
        columns = {key: np.empty((self.num_envs, width), dtype=_NUMPY_DTYPES[dtype])
                   for key, (width, dtype) in self.fields.items()}
        for index, env in enumerate(self.envs):
            values = self._state_values(env)
            for key, column in columns.items():
                column[index] = values[key]
        # Fresh arrays belong to this snapshot alone, including on CPU.
        return TensorDict({key: torch.from_numpy(value).to(self.device) for key, value in columns.items()},
                          [self.num_envs], device=self.device)

    def set_state(self, state: TensorDict) -> None:
        self._check_open()
        validate_classic_state(self.domain, state, batch=self.num_envs)
        host = state.cpu()
        for i, env in enumerate(self.envs):
            row, base = host[i], env.unwrapped
            if self.domain == "cartpole":
                base.state = np.asarray(
                    [row[name].item() for name in ("x", "x_dot", "theta", "theta_dot")], np.float64
                )
                base.steps_beyond_terminated = (
                    row["task", "steps_beyond_terminated"].item()
                    if row["task", "has_steps_beyond_terminated"].item()
                    else None
                )
            elif self.domain == "pendulum":
                base.state = np.asarray([row["theta"].item(), row["theta_dot"].item()], np.float64)
                if row["task", "has_last_u"].item():
                    base.last_u = np.float64(row["torque"].item())
                else:
                    base.last_u = None
            elif self.domain == "acrobot":
                base.state = np.asarray(
                    [row[name].item() for name in ("theta1", "theta2", "dtheta1", "dtheta2")], np.float64
                )
            else:
                dtype = (
                    np.float32
                    if self.domain == "mountain_car_continuous" and row["task", "state_is_float32"].item()
                    else np.float64
                )
                values = [row["position"].item(), row["velocity"].item()]
                # MountainCar itself stores a tuple after stepping, but only
                # the scalar values affect continuation.  The continuous
                # variant's ndarray dtype is semantically significant.
                base.state = np.asarray(values, dtype=dtype)

            env._elapsed_steps = row["lifecycle", "elapsed_steps"].item()
            # A public State can only be captured after reset.  Restore that
            # wrapper precondition as well as the numerical continuation.
            env.env._has_reset = True
            rng_state = {
                "bit_generator": "PCG64",
                "state": {
                    "state": _from_limbs(row["rng", "state"].numpy()),
                    "inc": _from_limbs(row["rng", "increment"].numpy()),
                },
                "has_uint32": row["rng", "has_uint32"].item(),
                "uinteger": row["rng", "uinteger"].item(),
            }
            base.np_random.bit_generator.state = rng_state

    def _actions(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        array = np.asarray(actions)
        if array.shape != (self.num_envs, 1):
            raise ValueError(f"Recovered {self.domain} action must have shape ({self.num_envs}, 1), got {array.shape}")
        if self.domain in {"cartpole", "acrobot", "mountain_car"}:
            # ``item`` retains the integer-vs-float distinction checked by
            # Gymnasium's Discrete.contains assertion.
            return [array[i, 0].item() for i in range(self.num_envs)]
        return [array[i] for i in range(self.num_envs)]

    def step(self, actions, *, active=None) -> TensorDict:
        active = None if active is None else np.asarray(active.detach().cpu(), dtype=bool)
        self._check_open()
        outputs = [env.step(action) if active is None or active[i] else self._last_outputs[i]
                   for i, (env, action) in enumerate(zip(self.envs, self._actions(actions), strict=True))]
        self._last_outputs = outputs
        observations = np.stack([out[0] for out in outputs])
        return TensorDict(
            {
                "state": self.get_state(),
                "reward": torch.tensor([out[1] for out in outputs], dtype=torch.float64, device=self.device),
                "terminated": torch.tensor([out[2] for out in outputs], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([out[3] for out in outputs], dtype=torch.bool, device=self.device),
                "observation": torch.as_tensor(observations, device=self.device).clone(),
                # SWM's only numeric info entry is an explicit float32 copy
                # of the observation under ``info['state']``.
                "info_state": torch.as_tensor(observations, dtype=torch.float32, device=self.device).clone(),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def close(self) -> None:
        if not self._closed:
            for env in self.envs:
                env.close()
            self._closed = True


class ClassicControlSemantics:
    """Compose an existing exact renderer with the authoritative CPU simulator."""

    execution_mode = "DELEGATED_CPU"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .._adapters.classic_execution import make_classic_adapter

        self._simulator = make_classic_adapter(
            self.CLASSIC_DOMAIN, self.B, self.device, self._seed, ClassicControlAdapter
        )
        self._configure_simulator()
        self.reset(seed=self._seed)

    def _with_physics_variation(self, base):
        physics = {
            name: Box(low=low, high=high, shape=(1,), dtype=np.float32, init_value=[default])
            for name, (low, high, default) in _PHYSICS_SPACES[self.CLASSIC_DOMAIN].items()
        }
        return Dict({**dict(base.items()), "physics": Dict(physics)})

    @property
    def dynamics_variation_leaves(self):
        return frozenset(f"physics.{name}" for name in _PHYSICS_SPACES[self.CLASSIC_DOMAIN])

    def _configure_simulator(self):
        layout = {name: (off, width) for name, off, width in self.variation_space.flat_layout()}
        style = self.get_variation()
        values = {
            name: style[:, layout[f"physics.{name}"][0]].detach().cpu().numpy()
            for name in _PHYSICS_SPACES[self.CLASSIC_DOMAIN]
        }
        self._simulator.configure(values)

    def set_variation(self, variation=None, **overrides):
        result = super().set_variation(variation, **overrides)
        if hasattr(self, "_simulator"):
            self._configure_simulator()
        return result

    def _layout(self):
        offsets = {self.sim_state_names[index]: packed for packed, index in enumerate(self.render_from_sim)}
        return tuple(
            (
                key if isinstance(key, tuple) else (key,),
                width,
                offsets.get(key) if isinstance(key, str) else None,
            )
            for key, (width, _) in CLASSIC_FIELDS[self.CLASSIC_DOMAIN].items()
        )

    def _layout_dtypes(self):
        return {
            key if isinstance(key, tuple) else (key,): dtype
            for key, (_, dtype) in CLASSIC_FIELDS[self.CLASSIC_DOMAIN].items()
        }

    def _validate_state(self, state, *, batch=None):
        validate_classic_state(self.CLASSIC_DOMAIN, state, batch=batch)

    @property
    def action_spec(self):
        if self.CLASSIC_DOMAIN == "cartpole":
            return Spec(shape=(1,), dtype=torch.int64, low=0, high=1, discrete_n=2, device=self.device)
        if self.CLASSIC_DOMAIN in {"acrobot", "mountain_car"}:
            return Spec(shape=(1,), dtype=torch.int64, low=0, high=2, discrete_n=3, device=self.device)
        bound = 2.0 if self.CLASSIC_DOMAIN == "pendulum" else 1.0
        return Spec(shape=(1,), dtype=torch.float32, low=-bound, high=bound, device=self.device)

    def _sync_authoritative_state(self, state):
        sim = torch.cat(
            [state[name].to(self.device, torch.float64).reshape(self.B, 1) for name in self.sim_state_names], dim=1
        )
        super().set_sim_state(sim)

    def sim_state(self):
        raise NotImplementedError(
            f"{type(self).__name__}.sim_state is unavailable for recovered semantics; use get_state()"
        )

    def set_sim_state(self, value):
        raise NotImplementedError(
            f"{type(self).__name__}.set_sim_state cannot restore lifecycle/RNG; use set_state(full_state)"
        )

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        state = self._simulator.reset(seed=seed, options=options)
        self._sync_authoritative_state(state)
        return state

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)
        self._sync_authoritative_state(state)

    def _step_pipeline(self):
        owner = self._simulator
        return owner if hasattr(owner, "submit_step") else None

    def _step(self, actions, *, active=None):
        self._check_open("step")
        result = self._simulator.step(actions, active=active)
        self._sync_authoritative_state(result["state"])
        return result


    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()
        super()._close()
