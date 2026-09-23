import json

import numpy as np
from dm_control import suite

from stable_worldmodel.envs.dmcontrol.humanoid import HumanoidDMControlWrapper

swm = HumanoidDMControlWrapper(seed=0)
swm.reset(seed=0)
stock = suite.load("humanoid", "walk", task_kwargs={"random": 0})
stock.reset()
actual = swm.env.physics.model.geom_friction
expected = stock.physics.model.geom_friction
rows = np.where(np.any(actual != expected, axis=1))[0]
print(
    json.dumps(
        [
            {
                "id": int(index),
                "name": swm.env.physics.model.id2name(int(index), "geom"),
                "swm": actual[index].tolist(),
                "stock": expected[index].tolist(),
            }
            for index in rows
        ],
        indent=2,
    )
)
swm.close()
stock.close()
