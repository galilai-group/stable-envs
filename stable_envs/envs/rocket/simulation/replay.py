from __future__ import annotations

import copy
from importlib.metadata import version

import numpy as np
import pybullet as p

from ...._core.space_nodes import nodes
from .actions import validate_action
from .numeric import capture
from .task import RocketLandingEnv

DTYPES = (None, np.dtype("float32"), np.dtype("float64"), np.dtype("uint8"))
#: Colour leaves in ``render/rgba`` row order; they and the undrawn sky act on the live episode.
PAINTED = ("rocket.body_color", "rocket.fin_color", "rocket.leg_color", "rocket.booster_color", "pad.color")
IMMEDIATE = (*PAINTED, "environment.sky_color")


def paint(env, path):
    """Draw one colour leaf's current value the way the source's reset does."""
    value = env.variation_space._walk(path).value / 255.0
    client = env.env._client
    if path == "pad.color":
        # The source writes a 3-decimal URDF material, which PyBullet holds in float32.
        body, links = env.landing_pad_id, [-1]
        rgba = [float(np.float32(float(f"{c:.3f}"))) for c in value] + [1.0]
    else:
        body, part = env.env.drones[0].Id, path.split(".")[1].removesuffix("_color")
        links = (
            [-1]
            if part == "body"
            else [
                i
                for i in range(p.getNumJoints(body, physicsClientId=client))
                if part in p.getJointInfo(body, i, physicsClientId=client)[12].decode("utf-8").lower()
            ]
        )
        rgba = list(value) + [1.0]
    for link in links:
        p.changeVisualShape(body, link, rgbaColor=rgba, physicsClientId=client)


def recolor_flame(env):
    """Apply the flame colour change that the source's ``step`` makes only when it has a render mode.

    The pixel env renders, so its flame turns ``ignition_color`` at ignition. With a render mode the
    source has already made the change and this does nothing.
    """
    if env.ignition_state != env.previous_render_ignition_state:
        env.previous_render_ignition_state = env.ignition_state
        env.env.changeVisualShape(env.env.drones[0].Id, 9, rgbaColor=env.ignition_color)


def encode_rng(state):
    if state is None:
        return np.zeros(11, np.int64)
    if state["bit_generator"] != "PCG64":
        raise ValueError("Rocket requires PCG64")
    return np.array(
        [
            1,
            *[(state["state"][k] >> (32 * i)) & 0xFFFFFFFF for k in ["state", "inc"] for i in range(4)],
            state["has_uint32"],
            state["uinteger"],
        ],
        np.int64,
    )


def decode_rng(row):
    if row.shape != (11,) or row.dtype != np.int64 or row[0] not in (0, 1):
        raise ValueError("Invalid PCG64 presence/shape/dtype")
    if not row[0]:
        if np.any(row):
            raise ValueError("Absent PCG64 must have zero padding")
        return None
    if np.any(row[1:] < 0) or np.any(row[1:] > 0xFFFFFFFF) or row[9] not in (0, 1) or not row[5] % 2:
        raise ValueError("Invalid PCG64 words/increment/cache")
    return dict(
        bit_generator="PCG64",
        state={k: sum(int(row[1 + j * 4 + i]) << (32 * i) for i in range(4)) for j, k in enumerate(["state", "inc"])},
        has_uint32=int(row[9]),
        uinteger=int(row[10]),
    )


def generator_state(node):
    return None if node._np_random is None else copy.deepcopy(node._np_random.bit_generator.state)


def install_generator(node, state):
    node._np_random = None if state is None else np.random.default_rng(0)
    if state is not None:
        node._np_random.bit_generator.state = copy.deepcopy(state)


def value_code(value):
    if type(value) is float:
        return 0
    dtype = np.asarray(value).dtype
    if dtype not in DTYPES[1:]:
        raise ValueError("Unsupported Rocket Variation representation")
    return DTYPES.index(dtype)


def decode_value(value, code):
    code = int(code)
    if code not in range(4) or code == 0 and value.shape:
        raise ValueError("Invalid Variation representation")
    if code == 0:
        return float(value)
    converted = value.astype(DTYPES[code])
    if not np.array_equal(converted, value):
        raise ValueError("Lossy Variation representation")
    return converted.copy() if value.shape else converted[()]


class RocketReplay:
    """Own one CPU engine with a numeric, bounded replay recipe.

    Restoration rebuilds native contact caches through source steps, then verifies
    the resulting fields before swapping engines. State contains no native handles;
    restore cost grows with the recorded physical action count.
    """

    def __init__(self, config=None, seed=0, options=None, *, initialize=True):
        for name, required in [("PyFlyt", "0.29.0"), ("pybullet", "3.2.7")]:
            if version(name) != required:
                raise RuntimeError(f"Rocket requires {name}=={required}")
        self.config = dict(config or {})
        if self.config.get("angle_representation", "quaternion") != "quaternion":
            raise ValueError("Registered Rocket uses quaternion observations")
        if self.config.get("render_mode") not in (None, "rgb_array"):
            raise ValueError("Registered Rocket supports None or rgb_array")
        self.env = RocketLandingEnv(**self.config)
        self.capacity = self.env.max_steps + 2
        if self.capacity < 2:
            raise ValueError("Rocket horizon must be positive")
        self.names = [p[1:].replace("/", ".") for p, _ in nodes(self.env.variation_space) if p]
        self.registered = {path: self.env.variation_space._walk(path).init_value.copy() for path in PAINTED}
        if initialize:
            try:
                self.reset(seed, options)
            except BaseException:
                self.close()
                raise

    def reset(self, seed=None, options=None, values=None):
        """``values`` sets leaves for this episode only; the drawn colours among them are painted."""
        if seed is not None and (
            isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**63
        ):
            raise ValueError("Rocket seed must be None or a nonnegative int64")
        options = copy.deepcopy(options or {})
        # ``randoimize_drop`` is PyFlyt's spelling of its random entry velocity option, passed through as is.
        if set(options) - {"variation", "randomize_drop", "accelerate_drop", "randoimize_drop"}:
            raise ValueError("Unsupported Rocket reset option")
        selection = options.get("variation", [])
        if not isinstance(selection, (list, tuple)) or any(k not in ["all", *self.names] for k in selection):
            raise ValueError("Invalid Rocket Variation selection")
        for key in ["randomize_drop", "accelerate_drop", "randoimize_drop"]:
            if key in options and type(options[key]) is not bool:
                raise ValueError("Reset flags must be bool")
        values = dict(values or {})
        for path, value in values.items():
            node = self.env.variation_space._walk(path) if path in self.names else None
            if node is None or hasattr(node, "spaces") or not node.contains(value):
                raise ValueError(f"Invalid Rocket Variation value: {path}")
            if any(k == "all" or path == k or path.startswith(k + ".") for k in selection):
                raise ValueError(f"Rocket Variation {path} is both sampled and set")
        # An explicit value is this episode's default, so the replay anchor reproduces it.
        defaults = {path: self.env.variation_space._walk(path).init_value for path in values}
        for path, value in values.items():
            self.env.variation_space._walk(path)._init_value = copy.deepcopy(value)
        try:
            self.seed = -1 if seed is None else int(seed)
            self.options = options
            self.anchor_values = {
                p: copy.deepcopy(node.init_value)
                for p, node in nodes(self.env.variation_space)
                if not hasattr(node, "spaces")
            }
            output = self.env.reset(seed=seed, options=copy.deepcopy(options))
        finally:
            for path, value in defaults.items():
                self.env.variation_space._walk(path)._init_value = value
        # Unlike the source, a configured colour default is drawn by every reset, not only sampling ones.
        for path in PAINTED:
            if path in values or not np.array_equal(
                self.env.variation_space._walk(path).init_value, self.registered[path]
            ):
                paint(self.env, path)
        self.history = np.zeros((self.capacity, 7), np.float64)
        self.dtypes = np.zeros(self.capacity, np.uint8)
        self.length = self.post = 0
        self.last_dtype = 2
        return output

    def configure(self, row):
        for path, value in row.items():
            node = self.env.variation_space._walk(path)
            node.set_init_value(copy.deepcopy(value), reset_value=False)
            if path in IMMEDIATE:
                node.set_value(copy.deepcopy(value))
            if path in PAINTED:
                paint(self.env, path)

    def step(self, action):
        validate_action(action)
        physical = not (self.env.termination or self.env.truncation)
        if physical and self.length >= self.capacity:
            raise RuntimeError("Source-derived physical action bound exceeded")
        if self.env.step_count >= np.iinfo(np.int64).max - 1:
            raise OverflowError("Rocket lifecycle counter exhausted int64")
        result = self.env.step(action.copy())
        recolor_flame(self.env)
        self.last_dtype = 1 if action.dtype == np.float32 else 2
        if physical:
            self.history[self.length] = action
            self.dtypes[self.length] = self.last_dtype
            self.length += 1
        else:
            self.post += 1
        return result

    def get_state(self):
        e = self.env
        state = {
            "recipe/version": np.array(1, np.int64),
            "recipe/seed": np.array(self.seed, np.int64),
            "recipe/config": np.array(
                [
                    e.ceiling,
                    e.max_displacement,
                    self.config.get("max_duration_seconds", 30),
                    self.config.get("agent_hz", 40),
                ],
                np.float64,
            ),
            "recipe/flags": np.array(
                [e.sparse_reward, e.angle_representation, int(e.render_mode is not None)], np.int64
            ),
            "recipe/render_resolution": np.array(e.render_resolution, np.int64),
            "recipe/reset_flags": np.array(
                [
                    self.options.get("randomize_drop", False),
                    self.options.get("accelerate_drop", True),
                    self.options.get("randoimize_drop", False),
                ],
                np.bool_,
            ),
            "recipe/variation_mask": np.array(
                [name in self.options.get("variation", []) for name in ["all", *self.names]], np.bool_
            ),
            "recipe/main_rng": encode_rng(e._main_anchor),
            "history/actions": self.history.copy(),
            "history/dtypes": self.dtypes.copy(),
            "history/length": np.array(self.length, np.int64),
            "history/valid": np.arange(self.capacity) < self.length,
            "lifecycle/post_terminal_steps": np.array(self.post, np.int64),
            "lifecycle/last_action_dtype": np.array(self.last_dtype, np.uint8),
            "rng/action_space": encode_rng(generator_state(e.action_space)),
        }
        visuals = {v[1]: np.array(v[7]) for v in e.env.getVisualShapeData(e.env.drones[0].Id)}
        state["render/rgba"] = np.stack(
            [
                visuals[-1],
                visuals[2],
                visuals[6],
                visuals[1],
                np.array(e.env.getVisualShapeData(e.landing_pad_id)[0][7]),
                visuals[9],
            ]
        )
        for key, value in capture(e).items():
            # These two source fields change bool/float types on the first step.
            if key in ["wrapper/previous_render_ignition_state", "wrapper/ignition_state"] or value.dtype.kind == "f":
                value = value.astype(np.float64)
            state["current/" + key] = value.copy()
        for path, node in nodes(e.variation_space):
            state["anchor/variation" + path + "/rng"] = encode_rng(e._variation_anchor[path])
            state["variation" + path + "/rng"] = encode_rng(generator_state(node))
            if not hasattr(node, "spaces"):
                for label, value in [
                    ("value", node.value),
                    ("init_value", node.init_value),
                    ("anchor_value", self.anchor_values[path]),
                ]:
                    state["variation" + path + "/" + label] = np.asarray(value, np.float64).copy()
                    state["variation" + path + "/" + label + "_dtype"] = np.array(value_code(value), np.uint8)
        return state

    def validate(self, state):
        template = self.get_state()
        if not isinstance(state, dict) or state.keys() != template.keys():
            raise ValueError("Rocket State keys differ")
        for key, value in state.items():
            if (
                not isinstance(value, np.ndarray)
                or value.shape != template[key].shape
                or value.dtype != template[key].dtype
            ):
                raise ValueError("Rocket State shape/dtype: " + key)
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError("Rocket State nonfinite/nonnumeric: " + key)
        for key in ["recipe/version", "recipe/config", "recipe/flags", "recipe/render_resolution"]:
            if not np.array_equal(state[key], template[key]):
                raise ValueError("Rocket configuration mismatch: " + key)
        n, post = int(state["history/length"]), int(state["lifecycle/post_terminal_steps"])
        if not 0 <= n <= self.capacity or post < 0 or n + post >= np.iinfo(np.int64).max:
            raise ValueError("Rocket history/counter range")
        if int(state["recipe/seed"]) < -1:
            raise ValueError("Invalid reset seed")
        if not np.array_equal(state["history/valid"], np.arange(self.capacity) < n):
            raise ValueError("Invalid history mask")
        if np.any(state["history/actions"][n:]) or np.any(state["history/dtypes"][n:]):
            raise ValueError("Nonzero history padding")
        if not np.isin(state["history/dtypes"][:n], [1, 2]).all() or state["lifecycle/last_action_dtype"] not in [
            1,
            2,
        ]:
            raise ValueError("Invalid action dtype code")
        if int(state["current/wrapper/step_count"]) != n + post:
            raise ValueError("Inconsistent lifecycle counter")
        if post and not (state["current/wrapper/termination"] or state["current/wrapper/truncation"]):
            raise ValueError("Post-terminal history requires terminal State")
        for i in range(n):
            action = state["history/actions"][i].astype(DTYPES[int(state["history/dtypes"][i])])
            if not np.array_equal(action, state["history/actions"][i]):
                raise ValueError("Lossy action dtype conversion")
            validate_action(action)
        for key, value in state.items():
            if key.endswith("/rng") or key in ["recipe/main_rng", "rng/action_space"]:
                decode_rng(value)
        if decode_rng(state["recipe/main_rng"]) is None:
            raise ValueError("Missing main reset RNG")
        return n, post

    def reconstruct(self, state):
        """Validate the recipe and return a matching new engine, leaving this owner untouched."""
        n, post = self.validate(state)
        staged = RocketReplay(self.config, initialize=False)
        try:
            for path, node in nodes(staged.env.variation_space):
                if hasattr(node, "spaces"):
                    continue
                value = decode_value(
                    state["variation" + path + "/anchor_value"], state["variation" + path + "/anchor_value_dtype"]
                )
                if not node.contains(value):
                    raise ValueError("Invalid reset Variation anchor")
                node._init_value = value
            staged.env._replay_anchor = dict(
                main=decode_rng(state["recipe/main_rng"]),
                variation={
                    p: decode_rng(state["anchor/variation" + p + "/rng"]) for p, _ in nodes(staged.env.variation_space)
                },
            )
            flags = state["recipe/reset_flags"]
            options = dict(
                randomize_drop=bool(flags[0]),
                accelerate_drop=bool(flags[1]),
                randoimize_drop=bool(flags[2]),
                variation=[
                    name for name, on in zip(["all", *self.names], state["recipe/variation_mask"], strict=True) if on
                ],
            )
            staged.reset(None if state["recipe/seed"] == -1 else int(state["recipe/seed"]), options)
            del staged.env._replay_anchor
            for i in range(n):
                staged.step(state["history/actions"][i].astype(DTYPES[int(state["history/dtypes"][i])]))
            if post:
                staged.step(state["current/wrapper/action"].astype(DTYPES[int(state["lifecycle/last_action_dtype"])]))
                staged.post = post
                staged.env.step_count = n + post
            rng = np.array(
                [1, *state["current/rng/state"], *state["current/rng/inc"], *state["current/rng/cached"]], np.int64
            )
            staged.env.np_random.bit_generator.state = decode_rng(rng)
            install_generator(staged.env.action_space, decode_rng(state["rng/action_space"]))
            for path, node in nodes(staged.env.variation_space):
                install_generator(node, decode_rng(state["variation" + path + "/rng"]))
                if not hasattr(node, "spaces"):
                    value = decode_value(
                        state["variation" + path + "/init_value"], state["variation" + path + "/init_value_dtype"]
                    )
                    if not node.contains(value):
                        raise ValueError("Invalid future Variation value")
                    node._init_value = value
            staged._adopt_appearance(state)
            actual = staged.get_state()
            mismatch = [key for key in state if not np.array_equal(state[key], actual[key])]
            if mismatch:
                raise ValueError("Rocket replay mismatch: " + ", ".join(mismatch))
            return staged
        except BaseException:
            staged.close()
            raise

    def _adopt_appearance(self, state):
        """Install appearance values set during the episode and repaint what they changed.

        The caller's full comparison then rejects any rgba the values do not imply.
        """
        replayed = self.get_state()
        for path in IMMEDIATE:
            key = "variation/" + path.replace(".", "/") + "/value"
            value = decode_value(state[key], state[key + "_dtype"])
            node = self.env.variation_space._walk(path)
            if not node.contains(value):
                raise ValueError("Invalid current Variation value")
            node.set_value(value)
            if path in PAINTED:
                i = PAINTED.index(path)
                if not np.array_equal(state[key], replayed[key]) or not np.array_equal(
                    state["render/rgba"][i], replayed["render/rgba"][i]
                ):
                    paint(self.env, path)

    def set_state(self, state):
        staged = self.reconstruct(state)
        old = self.env
        self.__dict__.update(staged.__dict__)
        old.close()

    def close(self):
        self.env.close()
