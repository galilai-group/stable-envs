import json

import numpy as np

from stable_worldmodel.envs.dmcontrol.humanoid import HumanoidDMControlWrapper


def snap(wrapper):
    env, physics = wrapper.env._env, wrapper.env.physics
    return {
        "qpos": physics.data.qpos.copy(),
        "qvel": physics.data.qvel.copy(),
        "time": float(physics.data.time),
        "warm": physics.data.qacc_warmstart.copy(),
        "step": int(env._step_count),
        "next": bool(env._reset_next_step),
        "cum": float(wrapper._cumulative_reward),
        "rng": env.task.random.get_state(),
    }


def restore(wrapper, state, warm):
    env, physics = wrapper.env._env, wrapper.env.physics
    physics.data.qpos[:] = state["qpos"]
    physics.data.qvel[:] = state["qvel"]
    physics.data.time = state["time"]
    physics.forward()
    if warm:
        physics.data.qacc_warmstart[:] = state["warm"]
    env._step_count = state["step"]
    env._reset_next_step = state["next"]
    wrapper._cumulative_reward = state["cum"]
    env.task.random.set_state(state["rng"])


def error(actual, expected):
    values = [
        np.max(np.abs(actual[0].astype(float) - expected[0].astype(float))),
        abs(float(actual[1]) - float(expected[1])),
        float(actual[2] != expected[2]),
        float(actual[3] != expected[3]),
    ]
    for key in actual[4]:
        if key == "env_name":
            continue
        left, right = np.asarray(actual[4][key]), np.asarray(expected[4][key])
        if np.issubdtype(left.dtype, np.number):
            values.append(
                float(np.max(np.abs(np.nan_to_num(left.astype(float) - right.astype(float)))))
            )
    return max(values)


found = None
for prefix in range(1, 33):
    candidate = HumanoidDMControlWrapper(seed=0)
    reference = HumanoidDMControlWrapper(seed=0)
    try:
        candidate.reset(seed=0)
        reference.reset(seed=0)
        actions = np.random.default_rng(200).uniform(-1, 1, (64, 21))
        for action in actions[:prefix]:
            assert error(candidate.step(action), reference.step(action)) == 0
        saved = snap(candidate)
        for action in -actions[prefix : prefix + 9]:
            candidate.step(action)
        control_different = not np.array_equal(
            candidate.env.physics.data.ctrl, reference.env.physics.data.ctrl
        )
        restore(candidate, saved, False)
        bad = [candidate.step(action) for action in actions[prefix : prefix + 8]]
        expected = [reference.step(action) for action in actions[prefix : prefix + 8]]
        bad_error = max(error(actual, wanted) for actual, wanted in zip(bad, expected, strict=True))
        restore(candidate, saved, True)
        good = [candidate.step(action) for action in actions[prefix : prefix + 8]]
        good_error = max(
            error(actual, wanted) for actual, wanted in zip(good, expected, strict=True)
        )
        if bad_error > 0:
            found = {
                "seed": 0,
                "prefix": prefix,
                "different_actions": 9,
                "continuation": 8,
                "ctrl_different_before_restore": control_different,
                "error_without_warmstart": bad_error,
                "error_with_warmstart": good_error,
            }
            break
    finally:
        candidate.close()
        reference.close()
print(json.dumps(found, indent=2))
