"""Research-only numeric reset recipe + bounded public-action replay on official Bullet."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/recovery"))
from rocket_feasibility import RocketLandingEnv, capture


def rng_numeric(rng):
    if rng is None:
        return np.zeros(11, np.uint64)
    d = rng.bit_generator.state
    if d["bit_generator"] != "PCG64":
        raise ValueError("Only pinned PCG64 accepted")
    return np.array(
        [
            1,
            *[(d["state"][k] >> (32 * i)) & 0xFFFFFFFF for k in ["state", "inc"] for i in range(4)],
            d["has_uint32"],
            d["uinteger"],
        ],
        np.uint64,
    )


def install_rng(rng, row):
    if row[0] != 1:
        raise ValueError("Expected initialized PCG64 stream")
    rng.bit_generator.state = dict(
        bit_generator="PCG64",
        state={k: sum(int(row[1 + j * 4 + i]) << (32 * i) for i in range(4)) for j, k in enumerate(["state", "inc"])},
        has_uint32=int(row[9]),
        uinteger=int(row[10]),
    )


def walk(space, path=""):
    yield path, space
    for k, v in getattr(space, "spaces", {}).items():
        yield from walk(v, path + "/" + k)


class Replay:
    """No production API claim. Candidate imports stage a new independent environment."""

    def __init__(self, config=None, seed=17, options=None):
        self.config = dict(config or {})
        self.env = RocketLandingEnv(**self.config)
        self.options = copy.deepcopy(options or {})
        allowed = {"variation", "randomize_drop", "accelerate_drop", "randoimize_drop"}
        if set(self.options) - allowed:
            raise ValueError("Unaudited reset option")
        if seed is None:
            raise ValueError("Unseeded reset recipe requires separate entropy-anchor proof")
        self.seed = int(seed)
        if not 0 <= self.seed < 2**63:
            raise ValueError("Bounded seed encoding")
        self.names = [p[1:].replace("/", ".") for p, _ in walk(self.env.variation_space) if p]
        if set(self.options.get("variation", [])) - set(self.names) - {"all"}:
            raise ValueError("Unknown variation selection")
        self.env.reset(seed=self.seed, options=copy.deepcopy(self.options))
        # Check occurs before increment: first time-limit physics step begins at max_steps+1.
        self.capacity = self.env.max_steps + 2
        if self.capacity < 2:
            raise ValueError("Unsupported nonpositive physics horizon")
        self.history = np.zeros((self.capacity, 7), np.float64)
        self.dtypes = np.zeros(self.capacity, np.uint8)
        self.length = 0
        self.post = 0
        self.last_dtype = np.uint8(2)
        self.anchor_rng = rng_numeric(self.env.np_random)

    def step(self, action):
        action = np.asarray(action)
        if (
            action.shape != (7,)
            or action.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or not np.isfinite(action).all()
        ):
            raise ValueError("Action shape/dtype/nonfinite")
        if np.any(action < self.env.action_space.low) or np.any(action > self.env.action_space.high):
            raise ValueError("Action outside authoritative bounds")
        code = 1 if action.dtype == np.float32 else 2
        if not (self.env.termination or self.env.truncation):
            if self.length == self.capacity:
                raise ValueError("Physics action bound exceeded")
            self.history[self.length] = action
            self.dtypes[self.length] = code
            self.length += 1
        else:
            self.post += 1
        self.last_dtype = np.uint8(code)
        return self.env.step(action.copy())

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
            "recipe/reset_anchor_rng": self.anchor_rng.copy(),
            "history/actions": self.history.copy(),
            "history/dtypes": self.dtypes.copy(),
            "history/length": np.array(self.length, np.int64),
            "history/mask": np.arange(self.capacity) < self.length,
            "lifecycle/post_terminal_steps": np.array(self.post, np.int64),
            "lifecycle/last_action_dtype": np.array(self.last_dtype, np.uint8),
        }
        state.update(
            {"current/" + k: v.astype(np.float64) if v.dtype.kind == "f" else v.copy() for k, v in capture(e).items()}
        )
        for path, node in walk(e.variation_space):
            state["variation" + path + "/rng"] = rng_numeric(getattr(node, "_np_random", None))
            if not hasattr(node, "spaces"):
                state["variation" + path + "/value"] = np.asarray(node.value, dtype=np.float64).copy()
                state["variation" + path + "/init_value"] = np.asarray(node.init_value, dtype=np.float64).copy()
        return state

    def validate(self, state):
        template = self.get_state()
        if set(template) != set(state):
            raise ValueError("Candidate keys differ")
        for k, v in state.items():
            if not isinstance(v, np.ndarray) or v.dtype != template[k].dtype or v.shape != template[k].shape:
                raise ValueError("Candidate shape/dtype: " + k)
            if v.dtype.kind not in "biuf" or not np.isfinite(v).all():
                raise ValueError("Candidate nonfinite/nonnumeric: " + k)
        for k in ["recipe/version", "recipe/config", "recipe/flags", "recipe/render_resolution"]:
            if not np.array_equal(state[k], template[k]):
                raise ValueError("Incompatible constructor configuration: " + k)
        n = int(state["history/length"])
        post = int(state["lifecycle/post_terminal_steps"])
        if not 0 <= n <= self.capacity or post < 0 or n + post >= np.iinfo(np.int64).max:
            raise ValueError("History/counter range")
        if not np.array_equal(state["history/mask"], np.arange(self.capacity) < n):
            raise ValueError("History mask")
        if np.any(state["history/actions"][n:]) or np.any(state["history/dtypes"][n:]):
            raise ValueError("Nonzero padding")
        if not np.isin(state["history/dtypes"][:n], [1, 2]).all() or int(state["lifecycle/last_action_dtype"]) not in [
            1,
            2,
        ]:
            raise ValueError("Action dtype code")
        if (
            int(state["recipe/seed"]) < 0
            or post
            and not (bool(state["current/wrapper/termination"]) or bool(state["current/wrapper/truncation"]))
        ):
            raise ValueError("Seed/terminal counter")
        if int(state["current/wrapper/step_count"]) != n + post:
            raise ValueError("Step counter mismatch")
        for k, row in state.items():
            if k.endswith("/rng") or k == "recipe/reset_anchor_rng":
                if row[0] not in [0, 1] or np.any(row[1:] > 0xFFFFFFFF) or row[9] > 1:
                    raise ValueError("PCG64 words")
        if np.any(state["history/actions"][:n] < self.env.action_space.low) or np.any(
            state["history/actions"][:n] > self.env.action_space.high
        ):
            raise ValueError("History action bounds")
        return n, post

    def set_state(self, state):
        n, post = self.validate(state)
        flags = state["recipe/reset_flags"]
        options = dict(
            randomize_drop=bool(flags[0]),
            accelerate_drop=bool(flags[1]),
            randoimize_drop=bool(flags[2]),
            variation=[k for k, on in zip(["all", *self.names], state["recipe/variation_mask"]) if on],
        )
        staged = Replay(self.config, seed=int(state["recipe/seed"]), options=options)
        try:
            for i in range(n):
                staged.step(
                    state["history/actions"][i].astype(np.float32 if state["history/dtypes"][i] == 1 else np.float64)
                )
            if post:
                last = state["current/wrapper/action"].astype(
                    np.float32 if state["lifecycle/last_action_dtype"] == 1 else np.float64
                )
                staged.step(last)
                staged.post = post
                staged.env.step_count = n + post
            # Future RNG consumption outside ordinary physics is independently restorable.
            current = {"rng/" + k: state["current/rng/" + k] for k in ["state", "inc", "cached"]}
            row = np.array([1, *current["rng/state"], *current["rng/inc"], *current["rng/cached"]], np.uint64)
            install_rng(staged.env.np_random, row)
            for path, node in walk(staged.env.variation_space):
                row = state["variation" + path + "/rng"]
                if row[0]:
                    install_rng(node.np_random, row)
                elif getattr(node, "_np_random", None) is not None:
                    raise ValueError("Unexpected generator materialization")
            actual = staged.get_state()
            mismatch = [k for k in state if not np.array_equal(state[k], actual[k])]
            if mismatch:
                raise ValueError("Replay mismatch: " + ",".join(mismatch))
        except BaseException:
            staged.close()
            raise
        prior = self.env
        self.__dict__.update(staged.__dict__)
        prior.close()

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
