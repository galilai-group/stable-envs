"""PyFlyt Rocket start leaves and reset colours against pinned stable_worldmodel."""

from __future__ import annotations

import numpy as np

import stable_envs as se

ROCKET_START = {
    "environment.start_height_ratio": np.asarray([0.9, 0.75, 0.93]),
    "environment.start_horizontal_offset": np.asarray([[0.0, 0.0], [1.5, -2.0], [-0.7, 1.2]], np.float32),
    "environment.start_tilt": np.asarray([[0.0, 0.0], [0.15, -0.1], [-0.2, 0.05]], np.float32),
}


def _swm_leaf(env, path):
    node = env.variation_space
    for part in path.split("."):
        node = node[part]
    return node


class _RocketLanes:
    """One SWM Rocket per lane; ``defaults`` are installed with SWM's ``set_init_value``."""

    def __init__(self, defaults=None):
        from stable_worldmodel.envs.rocket_landing.pyflyt_rocketlanding import RocketLandingEnv

        self.envs = [RocketLandingEnv() for _ in range(3)]
        for lane, env in enumerate(self.envs):
            for path, values in (defaults or {}).items():
                _swm_leaf(env, path).set_init_value(values[lane])

    def reset(self, seed, options=None):
        for lane, env in enumerate(self.envs):
            env.reset(seed=seed + lane, options=options)

    def step(self, actions):
        return [env.step(action) for env, action in zip(self.envs, actions, strict=True)]

    def view(self):
        """The source-visible part of our State: observation, materials and Variation values."""
        out = {"observation": np.stack([env.state for env in self.envs])}
        visuals = [{v[1]: v[7] for v in env.env.getVisualShapeData(env.env.drones[0].Id)} for env in self.envs]
        out["rgba"] = np.stack(
            [
                np.asarray([v[-1], v[2], v[6], v[1], env.env.getVisualShapeData(env.landing_pad_id)[0][7]])
                for v, env in zip(visuals, self.envs, strict=True)
            ]
        )
        for path in ROCKET_VIEWED:
            out[path] = np.stack([np.asarray(_swm_leaf(env, path).value, np.float64) for env in self.envs])
        return out

    def close(self):
        for env in self.envs:
            env.close()


ROCKET_VIEWED = (*ROCKET_START, "rocket.body_color", "rocket.fin_color", "pad.color")


def _rocket_view(state):
    out = {
        "observation": state["current", "wrapper", "state"].cpu().numpy(),
        "rgba": state["render", "rgba"][:, :5].cpu().numpy(),
    }
    for path in ROCKET_VIEWED:
        out[path] = state[("variation", *path.split("."), "value")].cpu().numpy()
    return out


def _rocket_same(ours, theirs):
    for key, value in theirs.items():
        assert np.array_equal(ours[key].reshape(value.shape), value), (key, ours[key], value)


def _rocket_rollout(env, lanes, actions):
    """Per-step observation, reward and termination must be bit-identical."""
    for action in actions:
        out = env.step(action, frames=None)
        theirs = lanes.step(action)
        assert np.array_equal(_rocket_view(out["state"])["observation"], np.stack([r[0] for r in theirs]))
        assert out["reward"].tolist() == [float(r[1]) for r in theirs]
        assert out["terminated"].tolist() == [bool(r[2]) for r in theirs]


def rocket_reset_leaf_gate(seed=31, steps=40) -> dict:
    """Rocket's start leaves per lane, explicit and sampled, and the colours reset can set, against SWM."""
    low = np.asarray([-1, -1, -1, 0, 0, -1, -1], np.float64)
    actions = np.random.default_rng(seed).uniform(low, 1.0, (steps, 3, 7))
    evidence = {"stable_env": "PyFlytRocketLanding", "leaves": sorted(ROCKET_START), "seed": seed, "steps": steps}
    with se.make("PyFlytRocketLanding", num_envs=3, device="cpu") as env:
        # Explicit per-lane values; SWM reads only init values, so its lanes take them as defaults.
        ours = env.reset(seed=seed, options={"variation_values": ROCKET_START})
        lanes, default = _RocketLanes(ROCKET_START), _RocketLanes()
        try:
            lanes.reset(seed)
            default.reset(seed)
            _rocket_same(_rocket_view(ours), lanes.view())
            start = lanes.view()["observation"][:, 10:13] - default.view()["observation"][:, 10:13]
            assert not start[0].any() and np.abs(start[1:]).max(-1).min() > 0.1, start
            _rocket_rollout(env, lanes, actions)
        finally:
            lanes.close()
            default.close()
        evidence["explicit"] = {"start_shift_vs_default_m": np.abs(start).max(-1).tolist(), "exact": True}

        # Sampled per lane under seed + lane, then a plain reset back to the registered defaults.
        options = {"variation": sorted(ROCKET_START)}
        ours = env.reset(seed=seed + 1, options=options)
        lanes = _RocketLanes()
        try:
            lanes.reset(seed + 1, options)
            view = _rocket_view(ours)
            _rocket_same(view, lanes.view())
            assert len({float(h) for h in view["environment.start_height_ratio"]}) == 3
            _rocket_rollout(env, lanes, actions)
            ours = env.reset(seed=seed + 2)
            lanes.reset(seed + 2)
            _rocket_same(_rocket_view(ours), lanes.view())
            _rocket_rollout(env, lanes, actions[:10])
        finally:
            lanes.close()
        evidence["sampled"] = {"heights": view["environment.start_height_ratio"].tolist(), "exact": True}

        # Colours set for one reset are drawn as SWM draws the same values once a colour reset paints.
        colours = {
            "rocket.body_color": np.asarray([[255, 204, 0], [12, 140, 250], [200, 30, 90]], np.uint8),
            "pad.color": np.asarray([[200, 200, 200], [104, 60, 3], [240, 132, 195]], np.uint8),
        }
        options = {"variation": ["rocket.fin_color"]}
        ours = env.reset(seed=seed + 3, options={**options, "variation_values": colours})
        lanes = _RocketLanes(colours)
        try:
            lanes.reset(seed + 3, options)
            _rocket_same(_rocket_view(ours), lanes.view())
        finally:
            lanes.close()
        evidence["colours"] = {"leaves": sorted(colours), "exact": True}
    return evidence
