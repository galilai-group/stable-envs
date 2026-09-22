"""Independent pinned-SWM oracle for the recovered classic-control defaults."""

from __future__ import annotations

import importlib

import numpy as np
import torch
from tensordict import TensorDict

WRAPPERS = {
    "cartpole": ("cartpole", "CartPoleWrapper"),
    "pendulum": ("pendulum", "PendulumWrapper"),
}

FIELDS = {
    "cartpole": {
        "x": (1, torch.float64),
        "x_dot": (1, torch.float64),
        "theta": (1, torch.float64),
        "theta_dot": (1, torch.float64),
        ("lifecycle", "elapsed_steps"): (1, torch.int64),
        ("task", "steps_beyond_terminated"): (1, torch.int64),
        ("task", "has_steps_beyond_terminated"): (1, torch.bool),
        ("rng", "state"): (4, torch.int64),
        ("rng", "increment"): (4, torch.int64),
        ("rng", "has_uint32"): (1, torch.int64),
        ("rng", "uinteger"): (1, torch.int64),
    },
    "pendulum": {
        "theta": (1, torch.float64),
        "theta_dot": (1, torch.float64),
        "torque": (1, torch.float64),
        ("task", "has_last_u"): (1, torch.bool),
        ("lifecycle", "elapsed_steps"): (1, torch.int64),
        ("rng", "state"): (4, torch.int64),
        ("rng", "increment"): (4, torch.int64),
        ("rng", "has_uint32"): (1, torch.int64),
        ("rng", "uinteger"): (1, torch.int64),
    },
}


def _limbs(value):
    value = int(value)
    return np.asarray([(value >> (32 * i)) & 0xFFFF_FFFF for i in range(4)], np.int64)


def _from_limbs(value):
    value = np.asarray(value, np.int64).reshape(4)
    return sum(int(part) << (32 * i) for i, part in enumerate(value))


class NativeClassicReference:
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
            if self.domain == "cartpole":
                values.update(dict(zip(("x", "x_dot", "theta", "theta_dot"), np.asarray(base.state))))
                beyond = base.steps_beyond_terminated
                values[("task", "steps_beyond_terminated")] = [0 if beyond is None else beyond]
                values[("task", "has_steps_beyond_terminated")] = [beyond is not None]
            else:
                values["theta"], values["theta_dot"] = np.asarray(base.state)
                last_u = getattr(base, "last_u", None)
                values["torque"] = [0.0 if last_u is None else last_u]
                values[("task", "has_last_u")] = [last_u is not None]
            row = TensorDict({}, batch_size=[])
            for key, (width, dtype) in FIELDS[self.domain].items():
                row[key] = torch.as_tensor(values[key], dtype=dtype).reshape(width).clone()
            rows.append(row)
        return torch.stack(rows)

    def set_state(self, state):
        host = state.cpu()
        for i, wrapper in enumerate(self.envs):
            row, env, base = host[i], wrapper.env, wrapper.env.unwrapped
            if self.domain == "cartpole":
                base.state = np.asarray(
                    [row[name].item() for name in ("x", "x_dot", "theta", "theta_dot")], np.float64
                )
                base.steps_beyond_terminated = (
                    row["task", "steps_beyond_terminated"].item()
                    if row["task", "has_steps_beyond_terminated"].item()
                    else None
                )
            else:
                base.state = np.asarray([row["theta"].item(), row["theta_dot"].item()], np.float64)
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
        if self.domain == "cartpole":
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
