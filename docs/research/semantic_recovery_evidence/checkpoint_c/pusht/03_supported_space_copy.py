"""Test Pymunk's supported Space.copy() at a cached wall-contact state."""

import numpy as np
import pymunk

from stable_worldmodel.envs.pusht.env import PushT


OPTIONS = {
    "variation": [],
    "state": np.array([100, 100, 400, 400, 0, 0, 0], np.float64),
    "goal_state": np.array([200, 200, 300, 300, 0, 0, 0], np.float64),
}


def bodies(space):
    agent = next(body for body in space.bodies if body.body_type == pymunk.Body.KINEMATIC)
    block = next(body for body in space.bodies if body.body_type == pymunk.Body.DYNAMIC)
    return agent, block


def step(space, action):
    agent, _ = bodies(space)
    target = agent.position + tuple(action * 100)
    for _ in range(10):
        acceleration = 100 * (target - agent.position) + 20 * (-agent.velocity)
        agent.velocity += acceleration * 0.01
        space.step(0.01)


def vector(space):
    agent, block = bodies(space)
    return np.array(
        [
            *agent.position,
            *block.position,
            block.angle % (2 * np.pi),
            *agent.velocity,
            *block.velocity,
            block.angular_velocity,
        ],
        np.float64,
    )


env = PushT()
env.reset(seed=0, options=OPTIONS)
source = env.space
copied = source.copy()
print("before_exact", bool(np.array_equal(vector(source), vector(copied))))
print("source_cached_arbiters", len(source._get_arbiters()), "copy_cached_arbiters", len(copied._get_arbiters()))
step(source, np.zeros(2, np.float32))
step(copied, np.zeros(2, np.float32))
expected, actual = vector(source), vector(copied)
print("after_exact", bool(np.array_equal(expected, actual)))
print("source_after", expected.tolist())
print("copy_after", actual.tolist())
print("max_error", float(np.max(np.abs(expected - actual))))
print("delta", (expected - actual).tolist())
env.close()
