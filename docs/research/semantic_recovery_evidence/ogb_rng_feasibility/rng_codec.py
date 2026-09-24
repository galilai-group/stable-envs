"""Research numeric RNG encoding, independent of any public State implementation."""

from __future__ import annotations

import copy

import numpy as np


def field(value, dtype, shape):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(f"expected {dtype} {shape}")
    return value


def u128(value):
    return np.frombuffer(int(value).to_bytes(16, "little"), dtype=np.uint8).copy()


def encode_pcg(state):
    assert state["bit_generator"] == "PCG64"
    return {
        "state": u128(state["state"]["state"]),
        "increment": u128(state["state"]["inc"]),
        "has_uint32": np.array(state["has_uint32"], np.uint8),
        "uinteger": np.array(state["uinteger"], np.uint32),
    }


def decode_pcg(leaves):
    if set(leaves) != {"state", "increment", "has_uint32", "uinteger"}:
        raise ValueError("PCG64 keys")
    state = int.from_bytes(field(leaves["state"], np.uint8, (16,)).tobytes(), "little")
    increment = int.from_bytes(field(leaves["increment"], np.uint8, (16,)).tobytes(), "little")
    flag = int(field(leaves["has_uint32"], np.uint8, ()))
    cached = int(field(leaves["uinteger"], np.uint32, ()))
    if flag not in (0, 1) or increment % 2 != 1:
        raise ValueError("PCG64 flag/increment")
    return {
        "bit_generator": "PCG64",
        "state": {"state": state, "inc": increment},
        "has_uint32": flag,
        "uinteger": cached,
    }


def encode_mt(state):
    assert state[0] == "MT19937"
    return {
        "key": np.array(state[1], np.uint32, copy=True),
        "position": np.array(state[2], np.int64),
        "has_gauss": np.array(state[3], np.uint8),
        "cached_gaussian": np.array(state[4], np.float64),
    }


def decode_mt(leaves):
    if set(leaves) != {"key", "position", "has_gauss", "cached_gaussian"}:
        raise ValueError("MT19937 keys")
    key = field(leaves["key"], np.uint32, (624,)).copy()
    position = int(field(leaves["position"], np.int64, ()))
    flag = int(field(leaves["has_gauss"], np.uint8, ()))
    cached = float(field(leaves["cached_gaussian"], np.float64, ()))
    if not 0 <= position <= 624 or flag not in (0, 1) or not np.isfinite(cached):
        raise ValueError("MT19937 position/flag/cache")
    return ("MT19937", key, position, flag, cached)


def encode_seed(value):
    if value is None:
        kind, raw = 0, b""
    else:
        value = int(value)
        kind = 2 if value < 0 else 1
        value = abs(value)
        raw = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "little")
    return {
        "kind": np.array(kind, np.uint8),
        "length": np.array(len(raw), np.int64),
        "bytes": np.frombuffer(raw, np.uint8).copy(),
    }


def decode_seed(leaves):
    if set(leaves) != {"kind", "length", "bytes"}:
        raise ValueError("seed metadata keys")
    kind = int(field(leaves["kind"], np.uint8, ()))
    length = int(field(leaves["length"], np.int64, ()))
    raw = field(leaves["bytes"], np.uint8, (length,))
    if kind == 0 and length == 0:
        return None
    if kind not in (1, 2) or length < 1 or (length > 1 and raw[-1] == 0):
        raise ValueError("seed metadata encoding")
    value = int.from_bytes(raw.tobytes(), "little")
    return -value if kind == 2 else value


def numeric_roundtrip(state):
    result = copy.deepcopy(state)
    numeric = {}

    def pcg(name, original):
        numeric[name] = encode_pcg(original)
        return decode_pcg(numeric[name])

    for namespace, key in [("python", "_np_random_seed"), ("ownership", "entropy_seed"), ("ownership", "action_seed")]:
        numeric["seed/" + key] = encode_seed(state[namespace][key])
        result[namespace][key] = decode_seed(numeric["seed/" + key])
    result["rng"] = pcg("environment", state["rng"])
    for name in ["action_parent", "variation_parent"]:
        result["ownership"][name] = pcg(name, state["ownership"][name])
    numeric["task"] = encode_mt(state["ownership"]["task_rng"])
    result["ownership"]["task_rng"] = decode_mt(numeric["task"])
    for name in ["action_box", "last_box"]:
        if state[name] is not None and state[name]["rng"] is not None:
            result[name]["rng"] = pcg(name, state[name]["rng"])
    for name, value in state["variation"].items():
        if value["rng"] is not None:
            result["variation"][name]["rng"] = pcg(name, value["rng"])
    return result, numeric
