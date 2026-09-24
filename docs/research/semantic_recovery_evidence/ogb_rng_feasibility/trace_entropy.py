"""Scoped diagnostic instrumentation of unchanged authoritative entropy calls."""

from __future__ import annotations

import copy
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from ogbench.manipspace.envs.env import CustomMuJoCoEnv
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from ownership import GETTER, CubeEnv, SceneEnv, previous, spaces

OUT = Path(__file__).resolve().parent


class Tracer:
    def __init__(self):
        self.events = []
        self.paths = {}
        self.stage = "construction"
        self.boxes = []  # Keep identities alive, so id reuse cannot conflate objects.

    def log(self, kind, **values):
        row = {"index": len(self.events), "stage": self.stage, "kind": kind, **copy.deepcopy(values)}
        self.events.append(row)
        return row

    def rng_state(self, obj):
        rng = getattr(obj, "_np_random", None)
        return None if rng is None else copy.deepcopy(rng.bit_generator.state)

    def install(self, stack):
        tracer = self
        for cls in [CubeEnv, SceneEnv]:
            original_constructor = cls.__init__

            def constructor(env, *args, original_constructor=original_constructor, **kwargs):
                original_constructor(env, *args, **kwargs)
                for path, space in spaces(env.variation_space):
                    tracer.paths[id(space)] = path
                tracer.log("source_constructor_complete", env_id=id(env))

            stack.enter_context(patch.object(cls, "__init__", constructor))
        original_seed = seeding.np_random

        def new_rng(seed=None):
            gen, actual = original_seed(seed)
            tracer.log(
                "entropy_factory",
                requested_seed=seed,
                actual_seed=actual,
                rng_type=type(gen.bit_generator).__name__,
                rng_id=id(gen),
                state=gen.bit_generator.state,
            )
            return RNG(gen, tracer), actual

        stack.enter_context(patch.object(seeding, "np_random", new_rng))
        original_default = np.random.default_rng

        def default_rng(seed=None):
            gen = original_default(seed)
            tracer.log("default_rng", seed=seed, rng_id=id(gen), state=gen.bit_generator.state)
            return RNG(gen, tracer)

        stack.enter_context(patch.object(np.random, "default_rng", default_rng))
        original_init = gym.spaces.Box.__init__

        def box_init(box, *args, **kwargs):
            original_init(box, *args, **kwargs)
            tracer.boxes.append(box)
            tracer.log(
                "box_construct",
                box_id=id(box),
                shape=box.shape,
                dtype=str(box.dtype),
                low=box.low,
                high=box.high,
                rng_state=tracer.rng_state(box),
            )

        stack.enter_context(patch.object(gym.spaces.Box, "__init__", box_init))

        def getter(env):
            tracer.log("action_space_access", env_id=id(env))
            box = GETTER(env)
            tracer.paths[id(box)] = "action_space"
            tracer.log("action_space_return", box_id=id(box), rng_state=tracer.rng_state(box))
            return box

        stack.enter_context(patch.object(ManipSpaceEnv, "action_space", property(getter)))
        original_sample = gym.spaces.Box.sample

        def sample(box, *args, **kwargs):
            event = tracer.log(
                "box_sample", box_id=id(box), path=tracer.paths.get(id(box), "unmapped"), before=tracer.rng_state(box)
            )
            value = original_sample(box, *args, **kwargs)
            event.update(after=tracer.rng_state(box), value=copy.deepcopy(value))
            return value

        stack.enter_context(patch.object(gym.spaces.Box, "sample", sample))
        for cls in [gym.spaces.Dict, gym.spaces.Space]:
            original = cls.seed

            def seed(space, seed=None, original=original, cls=cls):
                tracer.log(
                    "space_seed",
                    method=cls.__name__ + ".seed",
                    space_id=id(space),
                    path=tracer.paths.get(id(space), "unmapped"),
                    seed=seed,
                )
                return original(space, seed)

            stack.enter_context(patch.object(cls, "seed", seed))
        original_randint = np.random.randint

        def randint(*args, **kwargs):
            event = tracer.log("global_randint", args=args, kwargs=kwargs, before=np.random.get_state())
            value = original_randint(*args, **kwargs)
            event.update(after=copy.deepcopy(np.random.get_state()), value=copy.deepcopy(value))
            return value

        stack.enter_context(patch.object(np.random, "randint", randint))
        original_reset = gym.Env.reset

        def gym_reset(env, *args, **kwargs):
            tracer.log("gym_env_reset", args=args, kwargs=kwargs, env_id=id(env))
            return original_reset(env, *args, **kwargs)

        stack.enter_context(patch.object(gym.Env, "reset", gym_reset))
        for cls, method in [
            (CustomMuJoCoEnv, "compile_model_and_data"),
            (ManipSpaceEnv, "set_control"),
            (ManipSpaceEnv, "step"),
            (CubeEnv, "initialize_episode"),
            (SceneEnv, "initialize_episode"),
        ]:
            original = getattr(cls, method)

            def call(env, *args, original=original, method=method, **kwargs):
                tracer.log("source_enter", method=method, args=args)
                old_stage = tracer.stage
                tracer.stage = method
                try:
                    return original(env, *args, **kwargs)
                finally:
                    tracer.stage = old_stage
                    tracer.log("source_exit", method=method)

            stack.enter_context(patch.object(cls, method, call))


class RNG:
    def __init__(self, generator, tracer):
        self.generator, self.tracer = generator, tracer

    @property
    def bit_generator(self):
        return self.generator.bit_generator

    def __getattr__(self, name):
        attr = getattr(self.generator, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            event = self.tracer.log(
                "generator_call",
                rng_id=id(self.generator),
                method=name,
                args=args,
                kwargs=kwargs,
                before=self.bit_generator.state,
            )
            value = attr(*args, **kwargs)
            event.update(after=copy.deepcopy(self.bit_generator.state), value=copy.deepcopy(value))
            return value

        return call


def main():
    output = []
    for name, cls in [("Cube", CubeEnv), ("Scene", SceneEnv)]:
        for label, options in [
            ("default", None),
            ("gym_make_default", None),
            ("fixed_variation", {"variation": []}),
            ("sampled_variation", {"variation": ["cube.start_position"], "task_id": 2}),
        ]:
            tracer = Tracer()
            with ExitStack() as stack:
                tracer.install(stack)
                env = gym.make(f"swm/OGB{name}-v0") if label == "gym_make_default" else cls()
                raw = env.unwrapped
                for path, space in spaces(raw.variation_space):
                    tracer.paths[id(space)] = path
                tracer.stage = "metadata_read"
                unused = env.action_space
                assert unused._np_random is None
                tracer.stage = "reset"
                reset_output = env.reset(seed=17, options=options)
                rng_ids = {"environment": id(raw.np_random.generator)}
                for path, space in spaces(raw.variation_space):
                    if space._np_random is not None:
                        rng_ids[path] = id(space._np_random.generator)
                for box in tracer.boxes:
                    if box._np_random is not None:
                        rng_ids["box/" + str(id(box))] = id(box._np_random.generator)
                row = {
                    "family": name,
                    "case": label,
                    "seed": 17,
                    "options": options,
                    "events": tracer.events,
                    "rng_owners": rng_ids,
                    "reset_output": reset_output,
                    "engine": previous.engine(raw),
                }
                output.append(row)
                env.close()
            action_samples = [e for e in row["events"] if e["kind"] == "box_sample" and e["path"] == "action_space"]
            assert len(action_samples) == (4 if label == "gym_make_default" else 2)
            print(name, label, "events", len(tracer.events), "settling_samples", len(action_samples), flush=True)
    (OUT / "entropy_traces.json").write_text(json.dumps(previous.encode(output), indent=2) + "\n")
    assert gym.spaces.Box.sample is previous.gym.spaces.Box.sample
    print("OGB_ENTROPY_TRACE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
