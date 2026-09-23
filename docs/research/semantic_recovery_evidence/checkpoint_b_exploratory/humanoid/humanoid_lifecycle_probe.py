import json

import numpy as np

from stable_worldmodel.envs.dmcontrol.humanoid import HumanoidDMControlWrapper

wrapper = HumanoidDMControlWrapper(seed=5)
wrapper.reset(seed=5)
env = wrapper.env._env
initial = env.task.random.get_state()
records = {}
for public_step in range(1, 503):
    output = wrapper.step(np.zeros(21))
    if public_step in (499, 500, 501, 502):
        records[str(public_step)] = {
            "step_count": int(env._step_count),
            "reset_next_step": bool(env._reset_next_step),
            "time": float(env.physics.data.time),
            "reward": float(output[1]),
            "terminated": bool(output[2]),
            "truncated": bool(output[3]),
            "score": float(output[4]["score"]),
        }
final = env.task.random.get_state()
rng_changed = any(not np.array_equal(a, b) for a, b in zip(initial[1:], final[1:], strict=True))
print(json.dumps({"records": records, "rng_changed": rng_changed}, indent=2))
wrapper.close()
