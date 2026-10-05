"""Research-only independent source bridges; no renderer or dynamics replacements."""

import importlib
import inspect

import numpy as np

from .common import equal_state, state_hash

CLASSIC = dict(
    CartPole="cartpole",
    Pendulum="pendulum",
    Acrobot="acrobot",
    MountainCar="mountain_car",
    MountainCarContinuous="mountain_car_continuous",
)
NAV = dict(TwoRoom="two_room", Piecewise="piecewise", SimplePointMaze="simple_point_maze")
DMC = dict(
    DMCAcrobotBalance="acrobot",
    DMCCartpoleSwingup="cartpole",
    DMCPendulumSwingup="pendulum",
    DMCReacherHard="reacher",
    DMCCheetahRun="cheetah",
    DMCWalkerWalk="walker",
    DMCHopperHop="hopper",
    DMCHumanoidWalk="humanoid",
    DMCFingerTurnHard="finger",
    DMCBallInCupCatch="ball_in_cup",
    DMCManipulatorBringBall="manipulator",
    DMCQuadrupedWalk="quadruped",
)


class Bridge:
    def __init__(self, row, seed, options):
        self.name = row["name"]
        self.options = {k: v for k, v in options.items() if not k.startswith("_")}
        name = self.name
        if name.startswith("Fetch"):
            import gymnasium as gym
            import stable_worldmodel.envs  # noqa: F401

            self.source = gym.make(row["registration"], render_mode="rgb_array", resolution=224, width=224, height=224)
            self.ref = None
            self.camera = "MujocoFetchEnv free camera: distance=2.5 azimuth=132 elevation=-14 lookat=(1.3,.75,.55)"
            self.helper = "scripts/recovery/fetch_public_gate.py:full_compare"
            return
        if name in CLASSIC:
            module, cls, args = "native_classic_semantics", "NativeClassicReference", [CLASSIC[name]]
        elif name in NAV:
            module, cls, args = "native_navigation_semantics", "NativeNavigationReference", [NAV[name]]
        elif name in DMC:
            module, cls, args = "dmc_semantics", "DMCReference", [DMC[name]]
        elif name == "PointMazeLargeNavigate":
            module, cls, args = "ogbench_maze_semantics", "OGBMazeReference", []
        else:
            raise NotImplementedError(name)
        cls = getattr(importlib.import_module("tests.helpers.reference." + module), cls)
        kw = {"options": options} if "options" in inspect.signature(cls).parameters else {"seed": seed}
        self.ref = cls(*args, **kw)
        if name in CLASSIC:
            original = self.ref.envs[0]
            self.ref.envs[0] = type(original)(render_mode="rgb_array")
            original.close()
        self.source = self.ref.envs[0]
        self.helper = inspect.getsourcefile(cls)
        self.camera = self.source.camera_id if name in DMC else "source default"

    def reset(self, seed):
        if self.ref is None:
            return self.source.reset(seed=seed, options=self.options)
        if self.name == "PointMazeLargeNavigate" and self.options:
            self.ref._global_rng[0] = np.random.RandomState(seed).get_state()
            self.ref._in_legacy_rng(0, self.source.reset, seed=seed, options=self.options)
            return self.ref.get_state()
        return self.ref.reset(seed)

    def step(self, action):
        if self.ref is None:
            return self.source.step(action[0])
        return self.ref.step(action)

    def proof(self, env):
        if self.ref is None:
            from scripts.recovery.fetch_public_gate import full_compare

            full_compare(env.get_state(), [self.source])
            return dict(
                kind="ACCEPTED_SEMANTIC_WITNESS",
                equal=True,
                helper=self.helper,
                candidate_state_hash=state_hash(env.get_state()),
                oracle_state_hash=self.snapshot(),
                compared="all public Fetch fields, controller caches, time, reset anchor, lifecycle and PCG64",
            )
        return dict(equal_state(env.get_state(), self.ref.get_state()), helper=self.helper)

    def snapshot(self):
        if self.ref is None:
            return state_hash(self.numeric_state())
        return state_hash(self.ref.get_state())

    def numeric_state(self):
        from scripts.recovery.fetch_pickandplace_adapter_probe import oracle

        row = oracle([self.source])[0]
        u = self.source.unwrapped
        values = {k: np.asarray(v).copy() for k, v in row.items() if k != "rng"}
        values.update(time=np.array([u.data.time]), initial_gripper_xpos=u.initial_gripper_xpos.copy())
        r = row["rng"]
        for key in ["state", "inc"]:
            values["rng/" + key] = np.array([(r["state"][key] >> (32 * i)) & 0xFFFFFFFF for i in range(4)], np.int64)
        values["rng/cache"] = np.array([r["has_uint32"], r["uinteger"]], np.int64)
        return values

    def render(self):
        if self.name in DMC:
            return self.source.render(width=224, height=224, camera_id=self.camera)
        if self.name == "SimplePointMaze":
            return self.source.render("rgb_array")
        return self.source.render()

    def close(self):
        (self.ref or self.source).close()


def checkpoints(name):
    if name.startswith("Craftax"):
        return [0, 1, 2, 5, 10, 25, 50, 100, 300]
    if name in {"OGBCube", "OGBScene"}:
        return [0, 1, 12, 24, 32, 50]
    if name in DMC:
        return [0, 1, 10, 50, 499, 501]
    if name.startswith("Fetch"):
        return [0, 1, 10, 25, 49, 50]
    if name in CLASSIC:
        if name == "CartPole":
            return [0, 1, 3, 5, 10, 500]
        limit = dict(CartPole=500, Pendulum=200, Acrobot=500, MountainCar=200, MountainCarContinuous=999)[name]
        return [0, 1, 10, limit // 2, limit - 1, limit]
    return [0, 1, 8, 32, 64, 128]


def action_sequence(source, name, seed, dtype, length):
    space = source.action_space
    rng = np.random.default_rng(90000 + seed)
    if hasattr(space, "n"):
        return rng.integers(0, space.n, (length, 1, 1), dtype=np.int64)
    low, high = np.asarray(space.low), np.asarray(space.high)
    assert np.isfinite(low).all() and np.isfinite(high).all()
    return rng.uniform(low, high, (length, 1, *space.shape)).astype(dtype)
