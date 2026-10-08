"""Accepted oracle witnesses for explicit-JAX and controlled-entropy families."""

import hashlib
import pickle

import numpy as np

from .common import state_hash


class CraftaxBridge:
    def __init__(self, row, seed, options):
        from tests.helpers.reference.craftax_source import oracle

        self.name = row["name"]
        self.options = {k: v for k, v in options.items() if not k.startswith("_")}
        self.source = oracle(self.name)
        self.camera = "registered pixel observation (no camera control)"
        self.helper = "tests/helpers/reference/craftax_source.py:assert_engine"

    def reset(self, seed):
        return self.source.reset(seed=seed, options=self.options)

    def step(self, action):
        return self.source.step(int(action[0, 0]))

    def proof(self, env):
        from tests.helpers.reference.craftax_source import assert_engine

        assert_engine(self.source, env, env.get_state())
        return dict(
            kind="ACCEPTED_SEMANTIC_WITNESS",
            equal=True,
            helper=self.helper,
            candidate_state_hash=state_hash(env.get_state()),
            oracle_state_hash=self.snapshot(),
            compared="all numeric upstream engine pytree leaves and wrapper PRNG key; source output cache retained in render nonmutation hash",
        )

    def snapshot(self):
        from stable_envs._adapters.craftax import numeric_leaves

        return state_hash(
            {
                **{"engine/" + "/".join(k): v for k, v in numeric_leaves(self.source._state)},
                "key": np.asarray(self.source._key),
                "last_obs": self.source._last_obs,
            }
        )

    def render(self):
        return self.source.render()

    def close(self):
        self.source.close()


class ManipulationBridge:
    def __init__(self, row, seed, options):
        self.name = row["name"]
        self.options = {k: v for k, v in options.items() if not k.startswith("_")}
        from tests.helpers.reference.ogbench_manipulation_semantics import load_oracle

        self.module = load_oracle()
        self.source = getattr(self.module, "Owned" + self.name[3:])(entropy_seed=seed)
        self.camera = "front_pixels"
        self.helper = "tests/helpers/reference/ogbench_manipulation_semantics.py:assert_oracle"

    def reset(self, seed):
        return self.source.reset(seed=seed, options=self.options)

    def step(self, action):
        return self.source.step(action[0])

    def proof(self, env):
        from tests.helpers.reference.ogbench_manipulation_semantics import assert_oracle

        assert_oracle(env._simulator.envs[0], self.source, self.module)
        return dict(
            kind="ACCEPTED_SEMANTIC_WITNESS",
            equal=True,
            helper=self.helper,
            candidate_state_hash=state_hash(env.get_state()),
            oracle_state_hash=self.snapshot(),
            compared="engine, observation, reset/step info, model arrays, PCG64, task MT19937, action-space and Variation entropy ownership",
        )

    def snapshot(self):
        return hashlib.sha256(pickle.dumps(self.module.result(self.source, None))).hexdigest()

    def render(self):
        return self.source.render(camera="front_pixels")

    def close(self):
        self.source.close()


class RocketBridge:
    def __init__(self, row, seed, options):
        from tests.helpers.reference.rocket_semantics import RocketLandingEnv

        self.name = row["name"]
        self.options = {k: v for k, v in options.items() if not k.startswith("_")}
        self.source = RocketLandingEnv(**options["_config"])
        self.camera = "registered PyFlyt fixed camera at 480x480"
        self.helper = "tests/helpers/reference/rocket_semantics.py:capture/differences"

    def reset(self, seed):
        return self.source.reset(seed=seed, options=self.options)

    def step(self, action):
        return self.source.step(action[0])

    def proof(self, env):
        from tests.helpers.reference.rocket_semantics import capture, differences

        candidate = capture(env._simulator.envs[0].env)
        source = capture(self.source)
        assert not differences(candidate, source)
        return dict(
            kind="ACCEPTED_SEMANTIC_WITNESS",
            equal=True,
            helper=self.helper,
            candidate_state_hash=state_hash(env.get_state()),
            oracle_state_hash=state_hash(source),
            compared="all 91 independent numeric source diagnostics, bodies/joints/contact/lifecycle/controller/task/PCG64; candidate full replay State retained",
        )

    def snapshot(self):
        from tests.helpers.reference.rocket_semantics import capture

        return state_hash(capture(self.source))

    def render(self):
        return self.source.render()

    def close(self):
        from pathlib import Path

        self.source.close()
        if self.source.modified_pad_urdf:
            Path(self.source.modified_pad_urdf).unlink(missing_ok=True)


def rocket_plans(row):
    from stable_envs._adapters.rocket.spaces import nodes
    from tests.helpers.reference.rocket_semantics import RocketLandingEnv

    config = {"render_mode": "rgb_array", "render_resolution": (480, 480)}
    source = RocketLandingEnv(**config)
    selections = [[p[1:].replace("/", ".")] for p, _ in nodes(source.variation_space) if p]
    selections += [["rocket", "rocket.body_color"], ["environment", "environment.start_tilt"]]
    source.close()
    plans = []
    for seed in [0, 17, 42]:
        for dtype in ["float32", "float64"]:
            for label, extra, opts, marks in [
                ("fall", {}, {}, [0, 1, 100, 300, 301, 302, 303, 310]),
                ("mixed", {}, {}, [0, 1, 12, 100]),
                ("success", {"ceiling": 3.2}, {"accelerate_drop": False}, [0, 1, 6, 7, 12]),
                ("horizon", {"ceiling": 500000.0}, {"accelerate_drop": False}, [0, 300, 600, 900, 1201, 1202, 1204]),
                ("all", {}, {"variation": ["all"]}, [0, 1, 12]),
            ]:
                plans.append((seed, dtype, dict(opts, _config=dict(config, **extra), _marks=marks, _label=label)))
        for selection in selections:
            plans.append(
                (
                    seed,
                    "float64",
                    dict(variation=selection, _config=config, _marks=[0], _label="variation_" + "_".join(selection)),
                )
            )
    return plans


def manipulation_plans(row, varied):
    interactions = ["cube"] if row["name"] == "OGBCube" else ["button", "drawer", "window"]
    plans = [
        (s, d, dict(task_id=1, variation=[], _interaction=i))
        for s in [0, 17, 42]
        for d in ["float32", "float64"]
        for i in interactions
    ]
    variants = {
        "agent": {"agent.color": np.array([0.2, 0.7, 0.3])},
        "cube": {"cube.color": np.array([[0.1, 0.8, 0.3]])},
        "floor": {"floor.color": np.array([[0.7, 0.15, 0.2], [0.12, 0.6, 0.8]])},
        "camera": {"camera.angle_delta": np.array([[4.0, -3.0]])},
        "light": {"light.intensity": np.array([0.9])},
        "size-low": {"cube.size": np.array([0.01])},
        "size-high": {"cube.size": np.array([0.03])},
    }
    if row["name"] == "OGBScene":
        variants["locks"] = {"lock_color": np.array([[0.2, 0.8, 0.3], [0.1, 0.2, 0.9]])}
    for label, values in variants.items():
        plans.append(
            (
                17,
                "float64",
                dict(
                    task_id=1, variation=[], variation_values=values, _label=label, _marks=[0, 1, 12], _reset_count=2
                ),
            )
        )
    return plans


def contact_action(bridge, interaction, step, dtype):
    raw = bridge.source
    if step == 0:
        if interaction == "cube":
            target = raw.data.joint("object_joint_0").qpos[:3].copy()
            target[2] = 0.035
        elif interaction == "button":
            target = raw.data.site_xpos[raw._button_site_ids[0]].copy()
            target[2] = 0.06
        else:
            target = raw.data.site_xpos[getattr(raw, "_" + interaction + "_site_id")].copy()
        bridge.contact_target = target
    goal = bridge.contact_target.copy()
    if step < 12:
        goal[2] += 0.09
    if step >= 32:
        goal[{"drawer": 1, "window": 0}.get(interaction, 2)] += {"drawer": -0.12, "window": 0.18, "cube": 0.15}.get(
            interaction, 0.0
        )
    action = np.zeros((1, 5), dtype=dtype)
    action[0, :3] = np.clip((goal - raw.data.site_xpos[raw._pinch_site_id]) / 0.05, -1, 1)
    action[0, 4] = 1.0 if interaction == "button" or step >= 24 else -1.0
    return action


def bridge_for(row, seed, options):
    from .bridges import Bridge

    if row["name"].startswith("Craftax"):
        return CraftaxBridge(row, seed, options)
    if row["name"] in {"OGBCube", "OGBScene"}:
        return ManipulationBridge(row, seed, options)
    if row["name"] == "PyFlytRocketLanding":
        return RocketBridge(row, seed, options)
    return Bridge(row, seed, options)
