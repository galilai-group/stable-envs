"""Per-lane classic State exporter that reads every field from the Gymnasium env itself."""

import numpy as np
import torch
from tensordict import TensorDict

from stable_envs.envs.classic_control.simulation.local import (
    BEYOND,
    ELAPSED,
    HAS_BEYOND,
    HAS_LAST_U,
    IS_FLOAT32,
    RNG_HAS_UINT32,
    RNG_INCREMENT,
    RNG_STATE,
    RNG_UINTEGER,
    STATES,
    _limbs,
)


def classic_state(self) -> TensorDict:
    self._check_open()
    rows = []
    for env in self.envs:
        base = env.unwrapped
        rng = base.np_random.bit_generator.state
        if rng["bit_generator"] != "PCG64":
            raise RuntimeError(f"Expected PCG64, got {rng['bit_generator']!r}")
        values: dict = {
            ELAPSED: [env._elapsed_steps],
            RNG_STATE: _limbs(rng["state"]["state"]),
            RNG_INCREMENT: _limbs(rng["state"]["inc"]),
            RNG_HAS_UINT32: [rng["has_uint32"]],
            RNG_UINTEGER: [rng["uinteger"]],
        }
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
        for name, key in zip(self._physics_names, self._physics_keys, strict=True):
            values[key] = [float(getattr(base, self._physics_attributes[name]))]

        row = TensorDict({}, batch_size=[], device=self.device)
        for key, (width, dtype) in self.fields.items():
            row[tuple(key.split("/"))] = (
                torch.as_tensor(values[key], dtype=dtype, device=self.device).reshape(width).clone()
            )
        rows.append(row)
    return STATES[self.domain].fold(torch.stack(rows), [len(rows)], self.device)
