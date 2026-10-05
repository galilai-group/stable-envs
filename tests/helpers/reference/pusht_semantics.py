"""Independent source PushT native-engine and entropy witnesses."""

import copy

import numpy as np


def equal(a, b, path="root"):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for key in a:
            equal(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), path
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            equal(left, right, f"{path}/{index}")
    else:
        np.testing.assert_array_equal(a, b, err_msg=path, strict=True)


def diagnostics(env):
    bodies = {}
    for name in ("agent", "block"):
        body = getattr(env, name)
        bodies[name] = {
            key: np.asarray(getattr(body, key)).copy()
            for key in (
                "position",
                "velocity",
                "angle",
                "angular_velocity",
                "force",
                "torque",
                "center_of_gravity",
                "mass",
                "moment",
            )
        }

    def spaces(space):
        result = {"rng": copy.deepcopy(space._np_random.bit_generator.state) if space._np_random is not None else None}
        if hasattr(space, "spaces"):
            result["children"] = {key: spaces(child) for key, child in space.spaces.items()}
        else:
            result["value"] = copy.deepcopy(space.value)
            result["initial"] = copy.deepcopy(space.init_value)
        return result

    return dict(
        bodies=bodies,
        observation=env._get_obs().copy(),
        goal=env.goal_state.copy(),
        latest_action=copy.deepcopy(env.latest_action),
        contacts=env.n_contact_points,
        goal_image=env._goal.copy(),
        rng=copy.deepcopy(env.rng.bit_generator.state),
        gym_rng=copy.deepcopy(env.np_random.bit_generator.state),
        variation=spaces(env.variation_space),
    )
