"""Independent pinned-SWM oracle for the remaining classic-control defaults."""

from __future__ import annotations

import importlib

import numpy as np
import torch
from tensordict import TensorDict

WRAPPERS = {
    "acrobot": ("acrobot", "AcrobotWrapper"),
    "mountain_car": ("mountain_car", "MountainCarWrapper"),
    "mountain_car_continuous": ("mountain_car", "MountainCarContinuousWrapper"),
}

_RNG = {
    ("rng", "state"): (4, torch.int64),
    ("rng", "increment"): (4, torch.int64),
    ("rng", "has_uint32"): (1, torch.int64),
    ("rng", "uinteger"): (1, torch.int64),
}

FIELDS = {
    "acrobot": {
        "theta1": (1, torch.float64),
        "theta2": (1, torch.float64),
        "dtheta1": (1, torch.float64),
        "dtheta2": (1, torch.float64),
        ("lifecycle", "elapsed_steps"): (1, torch.int64),
        **_RNG,
    },
    "mountain_car": {
        "position": (1, torch.float64),
        "velocity": (1, torch.float64),
        ("lifecycle", "elapsed_steps"): (1, torch.int64),
        **_RNG,
    },
    "mountain_car_continuous": {
        "position": (1, torch.float64),
        "velocity": (1, torch.float64),
        ("task", "state_is_float32"): (1, torch.bool),
        ("lifecycle", "elapsed_steps"): (1, torch.int64),
        **_RNG,
    },
}


def _limbs(value):
    value = int(value)
    return np.asarray([(value >> (32 * i)) & 0xFFFF_FFFF for i in range(4)], np.int64)


def _from_limbs(value):
    value = np.asarray(value, np.int64).reshape(4)
    return sum(int(part) << (32 * i) for i, part in enumerate(value))


class NativeClassicRemainingReference:
    """Call only the real wrappers and project their complete continuation state."""

    def __init__(self, domain, num_envs=1, options=None):
        module_name, class_name = WRAPPERS[domain]
        module = importlib.import_module(f"stable_worldmodel.envs.gymnasium_control.{module_name}")
        wrapper = getattr(module, class_name)
        self.domain = domain
        self.num_envs = int(num_envs)
        self.options = options
        self.envs = [wrapper() for _ in range(self.num_envs)]

    def reset(self, seed=None):
        for i, wrapper in enumerate(self.envs):
            wrapper.reset(seed=None if seed is None else int(seed) + i, options=self.options)
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
            state = np.asarray(base.state)
            if self.domain == "acrobot":
                values.update(dict(zip(("theta1", "theta2", "dtheta1", "dtheta2"), state)))
            else:
                values["position"], values["velocity"] = state
                if self.domain == "mountain_car_continuous":
                    values[("task", "state_is_float32")] = [state.dtype == np.float32]
            row = TensorDict({}, batch_size=[])
            for key, (width, dtype) in FIELDS[self.domain].items():
                row[key] = torch.as_tensor(values[key], dtype=dtype).reshape(width).clone()
            rows.append(row)
        return torch.stack(rows)

    def set_state(self, state):
        host = state.cpu()
        for i, wrapper in enumerate(self.envs):
            row, env, base = host[i], wrapper.env, wrapper.env.unwrapped
            if self.domain == "acrobot":
                base.state = np.asarray(
                    [row[name].item() for name in ("theta1", "theta2", "dtheta1", "dtheta2")], np.float64
                )
            else:
                dtype = (
                    np.float32
                    if self.domain == "mountain_car_continuous" and row["task", "state_is_float32"].item()
                    else np.float64
                )
                base.state = np.asarray([row["position"].item(), row["velocity"].item()], dtype=dtype)
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
        if self.domain in {"acrobot", "mountain_car"}:
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
