"""Pinned stable-worldmodel State and task oracle for the five classic-control envs."""

from __future__ import annotations

import importlib

import numpy as np
import torch
from tensordict import TensorDict

WRAPPERS = {
    "cartpole": ("cartpole", "CartPoleWrapper"),
    "pendulum": ("pendulum", "PendulumWrapper"),
    "acrobot": ("acrobot", "AcrobotWrapper"),
    "mountain_car": ("mountain_car", "MountainCarWrapper"),
    "mountain_car_continuous": ("mountain_car", "MountainCarContinuousWrapper"),
}
PHYSICAL = {
    "cartpole": ("x", "x_dot", "theta", "theta_dot"),
    "pendulum": ("theta", "theta_dot"),
    "acrobot": ("theta1", "theta2", "dtheta1", "dtheta2"),
    "mountain_car": ("position", "velocity"),
    "mountain_car_continuous": ("position", "velocity"),
}
# stable-worldmodel's physics leaves and the Gymnasium attribute each one writes.
PHYSICS = {
    "cartpole": {name: name for name in ("gravity", "masscart", "masspole", "length", "force_mag", "tau")},
    "pendulum": {name: name for name in ("g", "m", "l", "max_torque", "max_speed", "dt")},
    "acrobot": {
        name: name.upper()
        for name in (
            "link_length_1",
            "link_length_2",
            "link_mass_1",
            "link_mass_2",
            "link_com_pos_1",
            "link_com_pos_2",
            "link_moi",
        )
    },
    "mountain_car": {name: name for name in ("gravity", "force", "max_speed")},
    "mountain_car_continuous": {name: name for name in ("power", "max_speed")},
}
COMMON = {
    ("lifecycle", "elapsed_steps"): (1, torch.int64),
    ("rng", "state"): (4, torch.int64),
    ("rng", "increment"): (4, torch.int64),
    ("rng", "has_uint32"): (1, torch.int64),
    ("rng", "uinteger"): (1, torch.int64),
}
FIELDS = {domain: {**{key: (1, torch.float64) for key in fields}, **COMMON} for domain, fields in PHYSICAL.items()}
FIELDS["cartpole"].update(
    {("task", "steps_beyond_terminated"): (1, torch.int64), ("task", "has_steps_beyond_terminated"): (1, torch.bool)}
)
FIELDS["pendulum"].update({"torque": (1, torch.float64), ("task", "has_last_u"): (1, torch.bool)})
FIELDS["mountain_car_continuous"][("task", "state_is_float32")] = (1, torch.bool)
for domain, names in PHYSICS.items():
    FIELDS[domain].update({("physics", name): (1, torch.float64) for name in names})


def _limbs(value):
    value = int(value)
    return np.asarray([(value >> (32 * i)) & 0xFFFF_FFFF for i in range(4)], np.int64)


def _from_limbs(value):
    value = np.asarray(value, np.int64).reshape(4)
    return sum(int(part) << (32 * i) for i, part in enumerate(value))


class NativeClassicReference:
    def __init__(self, domain, num_envs=1, options=None, *, render_mode=None):
        module_name, class_name = WRAPPERS[domain]
        module = importlib.import_module(f"stable_worldmodel.envs.gymnasium_control.{module_name}")
        wrapper = getattr(module, class_name)
        self.domain = domain
        self.num_envs = int(num_envs)
        self.options = options
        self.envs = [wrapper(render_mode=render_mode) for _ in range(self.num_envs)]

    def reset(self, seed=None):
        for i, wrapper in enumerate(self.envs):
            options = self.options[i] if isinstance(self.options, list) else self.options
            wrapper.reset(seed=None if seed is None else int(seed) + i, options=options)
        return self.get_state()

    def get_state(self):
        rows = []
        for wrapper in self.envs:
            env, base = wrapper.env, wrapper.env.unwrapped
            rng = base.np_random.bit_generator.state
            assert rng["bit_generator"] == "PCG64"
            values = {
                ("lifecycle", "elapsed_steps"): [env._elapsed_steps],
                ("rng", "state"): _limbs(rng["state"]["state"]),
                ("rng", "increment"): _limbs(rng["state"]["inc"]),
                ("rng", "has_uint32"): [rng["has_uint32"]],
                ("rng", "uinteger"): [rng["uinteger"]],
            }
            physical = np.asarray(base.state)
            values.update(dict(zip(PHYSICAL[self.domain], physical, strict=True)))
            if self.domain == "cartpole":
                beyond = base.steps_beyond_terminated
                values[("task", "steps_beyond_terminated")] = [0 if beyond is None else beyond]
                values[("task", "has_steps_beyond_terminated")] = [beyond is not None]
            elif self.domain == "pendulum":
                last_u = getattr(base, "last_u", None)
                values["torque"] = [0.0 if last_u is None else last_u]
                values[("task", "has_last_u")] = [last_u is not None]
            elif self.domain == "mountain_car_continuous":
                values[("task", "state_is_float32")] = [physical.dtype == np.float32]
            for name, attribute in PHYSICS[self.domain].items():
                values[("physics", name)] = [float(getattr(base, attribute))]
            row = TensorDict({}, batch_size=[])
            for key, (width, dtype) in FIELDS[self.domain].items():
                row[key] = torch.as_tensor(values[key], dtype=dtype).reshape(width).clone()
            rows.append(row)
        return torch.stack(rows)

    def set_state(self, state):
        host = state.cpu()
        for i, wrapper in enumerate(self.envs):
            row, env, base = host[i], wrapper.env, wrapper.env.unwrapped
            for name, attribute in PHYSICS[self.domain].items():
                setattr(base, attribute, row["physics", name].item())
            if self.domain == "cartpole":
                base.total_mass = base.masspole + base.masscart
                base.polemass_length = base.masspole * base.length
            dtype = (
                np.float32
                if self.domain == "mountain_car_continuous" and row["task", "state_is_float32"].item()
                else np.float64
            )
            base.state = np.asarray([row[name].item() for name in PHYSICAL[self.domain]], dtype=dtype)
            if self.domain == "cartpole":
                base.steps_beyond_terminated = (
                    row["task", "steps_beyond_terminated"].item()
                    if row["task", "has_steps_beyond_terminated"].item()
                    else None
                )
            elif self.domain == "pendulum":
                base.last_u = np.float64(row["torque"].item()) if row["task", "has_last_u"].item() else None
            env._elapsed_steps = row["lifecycle", "elapsed_steps"].item()
            env.env._has_reset = True
            base.np_random.bit_generator.state = {
                "bit_generator": "PCG64",
                "state": {
                    "state": _from_limbs(row["rng", "state"].numpy()),
                    "inc": _from_limbs(row["rng", "increment"].numpy()),
                },
                "has_uint32": row["rng", "has_uint32"].item(),
                "uinteger": row["rng", "uinteger"].item(),
            }

    def _actions(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        assert actions.shape == (self.num_envs, 1)
        if self.domain in {"cartpole", "acrobot", "mountain_car"}:
            return [actions[i, 0].item() for i in range(self.num_envs)]
        return [actions[i] for i in range(self.num_envs)]

    def step(self, actions):
        outputs = [env.step(action) for env, action in zip(self.envs, self._actions(actions), strict=True)]
        observations = np.stack([out[0] for out in outputs])
        return TensorDict(
            {
                "state": self.get_state(),
                "reward": torch.tensor([out[1] for out in outputs], dtype=torch.float64),
                "terminated": torch.tensor([out[2] for out in outputs], dtype=torch.bool),
                "truncated": torch.tensor([out[3] for out in outputs], dtype=torch.bool),
                "observation": torch.as_tensor(observations).clone(),
                "info_state": torch.as_tensor(observations, dtype=torch.float32).clone(),
            },
            batch_size=[self.num_envs],
        )

    def close(self):
        for env in self.envs:
            env.close()
