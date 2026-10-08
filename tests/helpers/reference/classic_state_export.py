"""Frozen 6ce4e829 classic State exporter; only the function name is adapted."""

import numpy as np
import torch
from tensordict import TensorDict

from stable_envs.envs.classic_control.simulation.local import _limbs


def classic_state(self) -> TensorDict:
    self._check_open()
    rows = []
    for env in self.envs:
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

        row = TensorDict({}, batch_size=[], device=self.device)
        for key, (width, dtype) in self.fields.items():
            row[key] = torch.as_tensor(values[key], dtype=dtype, device=self.device).reshape(width).clone()
        rows.append(row)
    return torch.stack(rows)
