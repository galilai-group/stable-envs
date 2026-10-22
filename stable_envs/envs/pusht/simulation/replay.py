from __future__ import annotations

import copy
from importlib.metadata import version

import numpy as np

from ...._core.space_nodes import nodes
from .numeric import (
    capture,
    decode_rng,
    decode_seed,
    encode_rng,
    encode_seed,
    get_value,
    install_rng,
    put_value,
    rng_of,
    rng_view,
)
from .task import PushT, PushTDiscrete

ACTION_DTYPES = (np.dtype("float32"), np.dtype("float64"), np.dtype("int32"), np.dtype("int64"))
HISTORY_KEYS = {"history/actions", "history/dtypes", "history/valid"}


def validate_seed(seed):
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**128
    ):
        raise ValueError("PushT seed must be None or a nonnegative integer below 2**128")


class PushTReplay:
    """Restore one CPU PushT engine from a numeric reset anchor and action history.

    Each step stores an owned action with its original dtype. Export copies the
    finite prefix without imposing a horizon; restoration replays it in O(T) and
    checks all exported fields before returning the new engine.
    """

    def __init__(self, *, discrete=False, config=None, seed=0, options=None, initialize=True):
        if version("pymunk") != "7.2.0":
            raise RuntimeError("PushT replay requires official pymunk==7.2.0")
        self.discrete = bool(discrete)
        self.config = dict(resolution=224, render_action=False, with_target=True)
        if config is not None:
            if set(config) - set(self.config):
                raise ValueError("Unsupported private PushT replay configuration")
            self.config.update(config)
        if type(self.config["resolution"]) is not int or self.config["resolution"] < 1:
            raise ValueError("PushT resolution must be a positive integer")
        if any(type(self.config[k]) is not bool for k in ("render_action", "with_target")):
            raise ValueError("PushT render flags must be bool")
        self.env = (PushTDiscrete if self.discrete else PushT)(**self.config)
        self.history = []
        if initialize:
            try:
                self._reset(seed, options)
            except BaseException:
                self.close()
                raise

    def validate_action(self, action):
        allowed = ACTION_DTYPES[2:] if self.discrete else ACTION_DTYPES[:2]
        if not isinstance(action, np.ndarray) or action.shape != (2,) or action.dtype not in allowed:
            raise ValueError("PushT action requires two float32/float64 values or two int32/int64 discrete indices")
        if not np.isfinite(action).all():
            raise ValueError("PushT action must be finite")
        if self.discrete and (np.any(action < 0) or np.any(action >= 16)):
            raise ValueError("PushT discrete indices must be in [0, 16)")
        if not self.discrete and (np.any(action < -1) or np.any(action > 1)):
            raise ValueError("PushT continuous actions must be in [-1, 1]")

    def _reset(self, seed, options):
        validate_seed(seed)
        options = copy.deepcopy(options or {})
        if set(options) - {"variation", "variation_values", "state", "goal_state"}:
            raise ValueError("Unsupported PushT reset option")
        for key in ("state", "goal_state"):
            if key in options:
                value = np.asarray(options[key])
                if value.shape != (7,) or value.dtype not in ACTION_DTYPES[:2] or not np.isfinite(value).all():
                    raise ValueError("Explicit PushT reset state/goal requires seven finite float32/float64 values")
                options[key] = value.copy()
        output = self.env.reset(seed=seed, options=options)
        values = self.env.variation_space.value
        start = options.get(
            "state",
            np.concatenate(
                [
                    values["agent"]["start_position"],
                    values["block"]["start_position"],
                    [values["block"]["angle"]],
                    values["agent"]["velocity"],
                ]
            ),
        )
        self.anchor = self._metadata()
        self.anchor["start"] = np.asarray(start, np.float64).copy()
        put_value(self.anchor, "goal", self.env.goal_state)
        self.anchor["latest_present"] = np.array(self.env.latest_action is not None, np.bool_)
        put_value(
            self.anchor,
            "latest",
            np.zeros(2, np.float64) if self.env.latest_action is None else self.env.latest_action,
        )
        self.history = []
        return output

    def reset_candidate(self, seed=None, options=None):
        target = type(self)(discrete=self.discrete, config=self.config, initialize=False)
        try:
            target.env.variation_space = copy.deepcopy(self.env.variation_space)
            install_rng(target.env, rng_of(self.env))
            target.env._np_random_seed = self.env._np_random_seed
            install_rng(target.env.action_space, rng_of(self.env.action_space))
            target.env.latest_action = copy.deepcopy(self.env.latest_action)
            target._reset(seed, options)
            return target
        except BaseException:
            target.close()
            raise

    def _metadata(self):
        e = self.env
        result = {
            "gym_rng": encode_rng(rng_view(e)),
            "rng": encode_rng(e.rng.bit_generator.state),
            "gym_seed": encode_seed(e._np_random_seed),
            "action_rng": encode_rng(rng_view(e.action_space)),
        }
        for path, node in nodes(e.variation_space):
            result["variation" + path + "/rng"] = encode_rng(rng_view(node))
            if not hasattr(node, "spaces"):
                put_value(result, "variation" + path + "/initial", node.init_value)
                put_value(result, "variation" + path + "/current", node.value)
        return result

    def _install_metadata(self, data):
        e = self.env
        install_rng(e, decode_rng(data["gym_rng"]))
        e._np_random_seed = decode_seed(data["gym_seed"])
        e.rng.bit_generator.state = decode_rng(data["rng"])
        install_rng(e.action_space, decode_rng(data["action_rng"]))
        for path, node in nodes(e.variation_space):
            install_rng(node, decode_rng(data["variation" + path + "/rng"]))
            if not hasattr(node, "spaces"):
                node.set_init_value(get_value(data, "variation" + path + "/initial"), reset_value=False)
                node.set_value(get_value(data, "variation" + path + "/current"))
        goal = e.variation_space["goal"]
        e.goal_pose = np.concatenate([goal["position"].value, [goal["angle"].value]])

    def step(self, action):
        self.validate_action(action)
        owned = action.copy()
        output = self.env.step(owned.copy())
        self.history.append(owned)
        return output

    def get_state(self):
        result = {
            "recipe/version": np.array(2, np.int64),
            "recipe/config": np.array(
                [self.discrete, self.config["resolution"], self.config["render_action"], self.config["with_target"]],
                np.int64,
            ),
            "history/actions": np.asarray(self.history, np.float64).reshape(-1, 2).copy(),
            "history/dtypes": np.array([ACTION_DTYPES.index(a.dtype) for a in self.history], np.uint8),
            "history/lengths": np.array(len(self.history), np.int64),
            "history/valid": np.ones(len(self.history), np.bool_),
        }
        for prefix, data in (("anchor/", self.anchor), ("current/", capture(self.env)), ("owned/", self._metadata())):
            result.update({prefix + k: v.copy() for k, v in data.items()})
        return result

    def validate(self, state):
        template = self.get_state()
        if not isinstance(state, dict) or state.keys() != template.keys():
            raise ValueError("PushT State keys differ")
        for key, expected in template.items():
            value = state[key]
            if not isinstance(value, np.ndarray) or value.dtype != expected.dtype:
                raise ValueError(f"Invalid PushT State dtype: {key}")
            if key not in HISTORY_KEYS and value.shape != expected.shape:
                raise ValueError(f"Invalid PushT State shape: {key}")
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite PushT State: {key}")
        for key in ("recipe/version", "recipe/config"):
            if not np.array_equal(state[key], template[key]):
                raise ValueError("PushT State version/configuration mismatch")
        actions, codes, valid = (state[k] for k in ("history/actions", "history/dtypes", "history/valid"))
        length = int(state["history/lengths"])
        if (
            actions.ndim != 2
            or actions.shape[1:] != (2,)
            or codes.shape != (len(actions),)
            or valid.shape != codes.shape
        ):
            raise ValueError("Invalid PushT history shapes")
        if not 0 <= length <= len(actions) or not np.array_equal(valid, np.arange(len(actions)) < length):
            raise ValueError("PushT history validity must describe a contiguous prefix")
        if np.any(actions[length:]) or np.any(codes[length:]):
            raise ValueError("PushT history padding must be zero")
        for action, code in zip(actions[:length], codes[:length], strict=True):
            if code >= len(ACTION_DTYPES):
                raise ValueError("Invalid PushT action dtype code")
            with np.errstate(over="ignore", invalid="ignore"):
                typed = action.astype(ACTION_DTYPES[int(code)])
            if not np.array_equal(typed, action):
                raise ValueError("Lossy PushT action representation")
            self.validate_action(typed)
        for key in state:
            if key.endswith("/type"):
                get_value(state, key[:-5])
            elif key.endswith(("/rng", "_rng")):
                rng = decode_rng(state[key])
                if key in ("anchor/rng", "owned/rng") and rng is None:
                    raise ValueError("PushT environment RNG cannot be absent")
            elif key.endswith("/gym_seed"):
                decode_seed(state[key])
        for prefix in ("anchor", "owned"):
            for path, node in nodes(self.env.variation_space):
                if not hasattr(node, "spaces"):
                    for label in ("initial", "current"):
                        if not node.contains(get_value(state, prefix + "/variation" + path + "/" + label)):
                            raise ValueError("PushT Variation value outside source space")

    def reconstruct(self, state):
        self.validate(state)
        target = type(self)(discrete=self.discrete, config=self.config, initialize=False)
        anchor = {k.removeprefix("anchor/"): v.copy() for k, v in state.items() if k.startswith("anchor/")}
        try:
            values = {}
            for path, node in nodes(target.env.variation_space):
                if hasattr(node, "spaces"):
                    continue
                node.set_init_value(get_value(anchor, "variation" + path + "/initial"), reset_value=False)
                branch = values
                names = path.strip("/").split("/")
                for name in names[:-1]:
                    branch = branch.setdefault(name, {})
                branch[names[-1]] = get_value(anchor, "variation" + path + "/current")
            target.env.latest_action = get_value(anchor, "latest") if anchor["latest_present"] else None
            target.env.reset(
                seed=0,
                options=dict(
                    variation=[],
                    variation_values=values,
                    state=anchor["start"].copy(),
                    goal_state=get_value(anchor, "goal"),
                ),
            )
            target._install_metadata(anchor)
            target.anchor = anchor
            length = int(state["history/lengths"])
            for action, code in zip(state["history/actions"][:length], state["history/dtypes"][:length], strict=True):
                target.step(action.astype(ACTION_DTYPES[int(code)]))
            target._install_metadata({k.removeprefix("owned/"): v for k, v in state.items() if k.startswith("owned/")})
            actual = target.get_state()
            for key, value in actual.items():
                expected = state[key][:length] if key in HISTORY_KEYS else state[key]
                if not np.array_equal(value, expected):
                    raise ValueError(f"PushT replay disagrees with supplied State: {key}")
            return target
        except BaseException:
            target.close()
            raise

    def close(self):
        self.env.close()
