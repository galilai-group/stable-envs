import json

import numpy as np

from stable_worldmodel.envs.dmcontrol.humanoid import HumanoidDMControlWrapper

walk = HumanoidDMControlWrapper(task="walk", seed=0)
stand = HumanoidDMControlWrapper(task="stand", seed=0)
ow, iw = walk.reset(seed=0)
os, is_ = stand.reset(seed=0)
initial = {
    "obs_equal": bool(np.array_equal(ow, os)),
    "qpos_equal": bool(np.array_equal(iw["qpos"], is_["qpos"])),
    "qvel_equal": bool(np.array_equal(iw["qvel"], is_["qvel"])),
    "walk_reward_function": float(walk.env.task.get_reward(walk.env.physics)),
    "stand_reward_function": float(stand.env.task.get_reward(stand.env.physics)),
}
action = np.zeros(21)
sw, ss = walk.step(action), stand.step(action)
public = {
    "walk_reward": float(sw[1]),
    "stand_reward": float(ss[1]),
    "observation_equal_after": bool(np.array_equal(sw[0], ss[0])),
    "qpos_equal_after": bool(np.array_equal(sw[4]["qpos"], ss[4]["qpos"])),
    "qvel_equal_after": bool(np.array_equal(sw[4]["qvel"], ss[4]["qvel"])),
    "walk_done": [sw[2], sw[3]],
    "stand_done": [ss[2], ss[3]],
}
p = walk.env.physics.model
print(
    json.dumps(
        {
            "initial": initial,
            "zero_action_public_step": public,
            "model_counts": {
                "nmocap": p.nmocap,
                "nuserdata": p.nuserdata,
                "nsensordata": p.nsensordata,
                "npluginstate": getattr(p, "npluginstate", None),
            },
            "physics_state_shape": list(walk.env.physics.get_state().shape),
        },
        indent=2,
    )
)
walk.close()
stand.close()
