"""Numeric replay values preserve Python/NumPy representation and dtype tags.

Scalar-versus-array distinctions affect source arithmetic. PCG64 words and
seed metadata remain integers; replay recipes contain no engine handles.
"""

from __future__ import annotations

import copy

import numpy as np

DTYPES = tuple(np.dtype(v) for v in ("float32", "float64", "uint8", "int32", "int64", "bool"))


def put_value(result, key, value):
    array = np.asarray(value)
    if type(value) is float:
        representation = 0
    elif type(value) is int:
        representation = 1
    elif type(value) is bool:
        representation = 2
    elif isinstance(value, np.generic):
        representation = 3
    elif isinstance(value, np.ndarray):
        representation = 4
    else:
        raise ValueError(f"Unsupported PushT numeric representation: {type(value)}")
    if array.dtype not in DTYPES or not np.isfinite(array).all():
        raise ValueError("Unsupported PushT numeric dtype/value")
    payload = array.astype(np.float64)
    if not np.array_equal(payload.astype(array.dtype), array):
        raise ValueError("PushT numeric value cannot be represented losslessly")
    result[key + "/value"] = payload
    result[key + "/type"] = np.array([representation, DTYPES.index(array.dtype)], np.uint8)


def get_value(state, key):
    value, tag = state[key + "/value"], state[key + "/type"]
    if tag.dtype != np.uint8 or tag.shape != (2,) or tag[0] > 4 or tag[1] >= len(DTYPES):
        raise ValueError("Invalid PushT numeric type tag")
    representation, dtype = int(tag[0]), DTYPES[int(tag[1])]
    if value.dtype != np.float64 or not np.isfinite(value).all():
        raise ValueError("Invalid PushT numeric payload")
    if representation < 4 and value.shape:
        raise ValueError("Scalar representation requires scalar payload")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = value.astype(dtype)
    if not np.array_equal(converted, value):
        raise ValueError("Lossy PushT numeric payload")
    if representation < 3:
        expected = (np.dtype("float64"), np.dtype("int64"), np.dtype("bool"))[representation]
        if dtype != expected:
            raise ValueError("Invalid Python scalar dtype tag")
        return (float, int, bool)[representation](converted)
    return converted[()] if representation == 3 else converted.copy()


def encode_rng(state):
    if state is None:
        return np.zeros(11, np.int64)
    if state["bit_generator"] != "PCG64":
        raise ValueError("PushT requires PCG64")
    return np.array(
        [
            1,
            *[(state["state"][k] >> (32 * i)) & 0xFFFFFFFF for k in ("state", "inc") for i in range(4)],
            state["has_uint32"],
            state["uinteger"],
        ],
        np.int64,
    )


def decode_rng(row):
    if row.dtype != np.int64 or row.shape != (11,) or row[0] not in (0, 1):
        raise ValueError("Invalid PushT PCG64 presence/shape/dtype")
    if not row[0]:
        if np.any(row):
            raise ValueError("Absent PushT PCG64 must have zero padding")
        return None
    if np.any(row[1:] < 0) or np.any(row[1:] > 0xFFFFFFFF) or row[9] not in (0, 1) or not row[5] % 2:
        raise ValueError("Invalid PushT PCG64 words/increment/cache")
    return dict(
        bit_generator="PCG64",
        state={k: sum(int(row[1 + j * 4 + i]) << (32 * i) for i in range(4)) for j, k in enumerate(("state", "inc"))},
        has_uint32=int(row[9]),
        uinteger=int(row[10]),
    )


def rng_of(node):
    return None if node._np_random is None else copy.deepcopy(node._np_random.bit_generator.state)


def install_rng(node, state):
    node._np_random = None if state is None else np.random.default_rng(0)
    if state is not None:
        node._np_random.bit_generator.state = copy.deepcopy(state)


def encode_seed(seed):
    # Gym's fresh entropy is commonly 128 bits, rather than a signed int64.
    if seed is None:
        return np.array([2, 0, 0, 0, 0], np.int64)
    if not isinstance(seed, (int, np.integer)) or not -1 <= seed < 2**128:
        raise ValueError("Unsupported Gym seed metadata")
    return np.array([int(seed == -1), *[(max(0, int(seed)) >> (32 * i)) & 0xFFFFFFFF for i in range(4)]], np.int64)


def decode_seed(row):
    if row.shape != (5,) or row.dtype != np.int64 or row[0] not in (0, 1, 2):
        raise ValueError("Invalid Gym seed metadata")
    if np.any(row[1:] < 0) or np.any(row[1:] > 0xFFFFFFFF) or row[0] and np.any(row[1:]):
        raise ValueError("Invalid Gym seed words")
    return (None if row[0] == 2 else -1) if row[0] else sum(int(row[i + 1]) << (32 * i) for i in range(4))


def capture(env):
    from .contacts import capture_contacts

    result = {}
    for name in ("agent", "block"):
        body = getattr(env, name)
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
        ):
            value = np.asarray(getattr(body, key), np.float64)
            if key in ("mass", "moment"):
                # Encode infinite kinematic mass/inertia as a flag plus finite zero padding.
                infinite = np.isposinf(value)
                result[f"body/{name}/{key}_infinite"] = np.asarray(infinite, np.bool_)
                value = np.where(infinite, 0.0, value)
            result[f"body/{name}/{key}"] = value.copy()
    result["observation"] = env._get_obs().copy()
    result["contacts"] = np.array(env.n_contact_points, np.int64)
    result["render_segments"], result["render_segment_count"] = capture_contacts(env.space)
    result["goal_image"] = env._goal.copy()
    result["latest_present"] = np.array(env.latest_action is not None, np.bool_)
    put_value(result, "latest", np.zeros(2, np.float64) if env.latest_action is None else env.latest_action)
    put_value(result, "goal", env.goal_state)
    return result
