from __future__ import annotations

import copy
from functools import cache
from importlib.metadata import version

import numpy as np

from ...._core.space_nodes import nodes
from .contacts import ContactSegments
from .numeric import (
    capture,
    decode_rng,
    decode_seed,
    encode_rng,
    encode_seed,
    episode_fields,
    get_value,
    install_rng,
    put_value,
    rng_of,
    rng_view,
)
from .task import PushT, PushTDiscrete

ACTION_DTYPES = (np.dtype("float32"), np.dtype("float64"), np.dtype("int32"), np.dtype("int64"))
HISTORY_KEYS = {"history/actions", "history/dtypes", "history/valid"}


@cache
def _pymunk_version():
    return version("pymunk")


def validate_seed(seed):
    if seed is not None and (
        isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**128
    ):
        raise ValueError("PushT seed must be None or a nonnegative integer below 2**128")


def shallow_copy(obj):
    """``copy.copy`` for plain instances, without the pickle protocol."""
    clone = object.__new__(type(obj))
    clone.__dict__.update(obj.__dict__)
    return clone


def copy_space(space, memo=None, generators=True):
    """Return a Variation space copy that owns its values and generators, like ``copy.deepcopy``.

    Bounds, shapes and constraint functions are not changed after construction, so the copy shares them.
    With ``generators=False`` the copy shares the generators; the caller must replace them before a draw.
    """
    memo = {} if memo is None else memo
    clone = shallow_copy(space)
    if generators:
        clone._np_random = copy.deepcopy(space._np_random, memo)
    if hasattr(space, "spaces"):
        clone.spaces = type(space.spaces)(
            (key, copy_space(child, memo, generators)) for key, child in space.spaces.items()
        )
    else:
        clone._init_value = copy.deepcopy(space._init_value, memo)
        clone._value = copy.deepcopy(space._value, memo)
    return clone


def reset_options(seed, options):
    """Check a reset seed and options; return an owned copy of the options."""
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
    return options


class PushTReplay:
    """Restore one CPU PushT engine from a numeric reset anchor and action history.

    Each step stores an owned action with its original dtype. Export copies the
    finite prefix without imposing a horizon; restoration replays it in O(T) and
    checks all exported fields before returning the new engine. Fields that change
    only at reset, restore or a Variation change are cached until ``changed``.
    """

    def __init__(self, *, discrete=False, config=None, seed=0, options=None, initialize=True, engine=None):
        if _pymunk_version() != "7.2.0":
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
        self.env = (PushTDiscrete if self.discrete else PushT)(**self.config) if engine is None else engine
        self.length = 0
        self._actions = np.zeros((0, 2), np.float64)
        self._codes = np.zeros(0, np.uint8)
        self._contacts = ContactSegments()
        self.version = 0
        self._owned = self._episode = None
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
        options = reset_options(seed, options)
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
        metadata = self._metadata()
        self.anchor = dict(metadata)
        self.anchor["start"] = np.asarray(start, np.float64).copy()
        put_value(self.anchor, "goal", self.env.goal_state)
        self.anchor["latest_present"] = np.array(self.env.latest_action is not None, np.bool_)
        put_value(
            self.anchor,
            "latest",
            np.zeros(2, np.float64) if self.env.latest_action is None else self.env.latest_action,
        )
        self.length = 0
        self.changed()
        # Nothing has drawn from a generator or changed a Variation since the anchor was read.
        self._owned = metadata
        return output

    def reset_candidate(self, seed=None, options=None):
        # Reset rebinds the engine's scene, generators and episode fields, so a shallow copy
        # shares only objects that it does not change in place.
        engine = shallow_copy(self.env)
        engine._canvas = None
        # An integer seed replaces every Variation generator before the first draw.
        engine.variation_space = copy_space(self.env.variation_space, generators=not isinstance(seed, int))
        engine.action_space = shallow_copy(self.env.action_space)
        target = type(self)(discrete=self.discrete, config=self.config, initialize=False, engine=engine)
        try:
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

    def changed(self):
        """Drop the cached reset, restore and Variation fields after any of them changes."""
        self._owned = self._episode = None
        self.version += 1

    def metadata(self):
        """Return the cached ``owned`` fields; callers copy them."""
        if self._owned is None:
            self._owned = self._metadata()
        return self._owned

    def episode(self):
        """Return the cached ``current`` fields that are fixed for the episode; callers copy them."""
        if self._episode is None:
            self._episode = episode_fields(self.env)
        return self._episode

    def recipe(self):
        return {
            "recipe/version": np.array(2, np.int64),
            "recipe/config": np.array(
                [self.discrete, self.config["resolution"], self.config["render_action"], self.config["with_target"]],
                np.int64,
            ),
        }

    def static_fields(self):
        """Return every State field that only ``changed`` can change, uncopied."""
        result = self.recipe()
        result.update(("anchor/" + k, v) for k, v in self.anchor.items())
        result.update(("current/" + k, v) for k, v in self.episode().items())
        result.update(("owned/" + k, v) for k, v in self.metadata().items())
        return result

    def _install_metadata(self, data):
        self.changed()
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
        return self.step_checked(action)

    def step_checked(self, action):
        """Step with an action that ``validate_action`` accepts."""
        output = self.env.step(action.copy())
        self._record(action)
        return output

    def advance(self, action):
        """Step with an accepted action; return only the reward and success of ``step``."""
        self.env.advance(action.copy())
        self._record(action)
        return self.evaluate()

    def evaluate(self):
        """Return the reward and success that ``step`` reports for the current scene."""
        success, distance = self.env.eval_state(self.env.goal_state, self.env._get_obs())
        return -distance, success

    def _record(self, action):
        if self.length == len(self._codes):
            capacity = max(16, 2 * self.length)
            self._actions = np.concatenate([self._actions, np.zeros((capacity - self.length, 2), np.float64)])
            self._codes = np.concatenate([self._codes, np.zeros(capacity - self.length, np.uint8)])
        self._actions[self.length] = action
        self._codes[self.length] = ACTION_DTYPES.index(action.dtype)
        self.length += 1

    def history(self):
        """Return owned copies of the recorded actions and their dtype codes."""
        return self._actions[: self.length].copy(), self._codes[: self.length].copy()

    def get_state(self):
        actions, codes = self.history()
        result = self.recipe()
        result.update(
            {
                "history/actions": actions,
                "history/dtypes": codes,
                "history/lengths": np.array(self.length, np.int64),
                "history/valid": np.ones(self.length, np.bool_),
            }
        )
        result.update(("anchor/" + k, v.copy()) for k, v in self.anchor.items())
        current = capture(self.env, self._contacts, self.episode())
        result.update(("current/" + k, v) for k, v in current.items())
        result.update(("owned/" + k, v.copy()) for k, v in self.metadata().items())
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
            # ``validate`` has checked every typed action.
            for action, code in zip(state["history/actions"][:length], state["history/dtypes"][:length], strict=True):
                target.step_checked(action.astype(ACTION_DTYPES[int(code)]))
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
