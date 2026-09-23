import json

import numpy as np
from dm_control import suite

from stable_envs.envs.dmcontrol.humanoid import DMCHumanoidStand
from stable_worldmodel.envs.dmcontrol.humanoid import HumanoidDMControlWrapper

swm = HumanoidDMControlWrapper(seed=0)
obs, info = swm.reset(seed=0)
walk = suite.load("humanoid", "walk", task_kwargs={"random": 0})
stand = suite.load("humanoid", "stand", task_kwargs={"random": 0})
walk.reset()
stand.reset()


def differences(a, b):
    out = []
    compared = 0
    for name in dir(a):
        if name.startswith("_") or not hasattr(b, name):
            continue
        try:
            x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        except Exception:
            continue
        if x.dtype.kind not in "biufc" or y.dtype.kind not in "biufc":
            continue
        if x.shape != y.shape:
            out.append({"name": name, "shape": [list(x.shape), list(y.shape)]})
            continue
        compared += 1
        if not np.array_equal(x, y, equal_nan=True):
            out.append(
                {
                    "name": name,
                    "count": int(np.count_nonzero(x != y)),
                    "max": float(np.max(np.abs(x.astype(float) - y.astype(float)))),
                }
            )
    return compared, out


p = swm.env.physics
cw, dw = differences(p.model, walk.physics.model)
cs, ds = differences(p.model, stand.physics.model)
base = swm.env._env
result = {
    "dims": {"nq": p.model.nq, "nv": p.model.nv, "nu": p.model.nu, "na": p.model.na},
    "action": {
        "shape": list(swm.action_space.shape),
        "dtype": str(swm.action_space.dtype),
        "low": swm.action_space.low.tolist(),
        "high": swm.action_space.high.tolist(),
        "repeat": swm.action_repeat,
    },
    "observation": {
        "flat_shape": list(obs.shape),
        "flat_dtype": str(obs.dtype),
        "keys": {
            key: list(np.asarray(value).shape)
            for key, value in base.task.get_observation(p).items()
        },
    },
    "info": {
        key: list(np.asarray(value).shape) for key, value in info.items() if key != "env_name"
    },
    "task": {key: value for key, value in base.task.__dict__.items() if key != "_random"},
    "control_timestep": base.control_timestep(),
    "step_limit": int(base._step_limit),
    "n_sub_steps": int(base._n_sub_steps),
    "qpos_names": list(p.named.data.qpos.axes.row.names),
    "model_vs_stock_walk": {"compared": cw, "differences": dw},
    "model_vs_stock_stand": {"compared": cs, "differences": ds},
    "stock_tasks": {"walk": walk.task.__dict__, "stand": stand.task.__dict__},
    "stable_existing_load_task": DMCHumanoidStand._load_env(0).task.__dict__,
}
print(json.dumps(result, default=str, indent=2))
swm.close()
walk.close()
stand.close()
