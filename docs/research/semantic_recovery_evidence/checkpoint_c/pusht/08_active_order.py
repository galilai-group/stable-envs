"""Compare semantic arbiter traversal order after contact reconstruction."""

import numpy as np
import pymunk

from stable_worldmodel.envs.pusht.env import PushT


SAFE = {
    "variation": [],
    "state": np.array([100, 100, 250, 250, 0, 0, 0], np.float64),
    "goal_state": np.array([200, 200, 300, 300, 0, 0, 0], np.float64),
}


def snapshot(env):
    return [
        (tuple(b.position), tuple(b.velocity), b.angle, b.angular_velocity, tuple(b.force), b.torque)
        for b in (env.agent, env.block)
    ]


def restore(env, state):
    for b, values in zip((env.agent, env.block), state, strict=True):
        position, velocity, angle, angular_velocity, force, torque = values
        b.angle = angle
        b.position = position
        b.velocity = velocity
        b.angular_velocity = angular_velocity
        b.force = force
        b.torque = torque
        env.space.reindex_shapes_for_body(b)


def key(env, shape):
    if isinstance(shape, pymunk.Circle):
        return "agent-circle"
    if isinstance(shape, pymunk.Poly):
        vertices = tuple(sorted(tuple(v) for v in shape.get_vertices()))
        return "block-poly:" + repr(vertices)
    if isinstance(shape, pymunk.Segment):
        return "wall:" + repr((tuple(shape.a), tuple(shape.b)))
    return type(shape).__name__


def arbiter_key(env, raw):
    shapes = {shape._shape: shape for shape in env.space.shapes}
    return key(env, shapes[raw.a]), key(env, shapes[raw.b])


def space_order(env):
    return [arbiter_key(env, raw) for raw in env.space._get_arbiters()]


def body_order(env, body):
    result = []
    body.each_arbiter(lambda arbiter: result.append(arbiter_key(env, arbiter._arbiter)))
    return result


source, target = PushT(), PushT()
source.reset(seed=0)
for step_index in range(30):
    action = ((np.array(source.block.position) - np.array(source.agent.position)) / 100).astype(np.float64)
    output = source.step(action)
    if output[4]["n_contacts"]:
        break
target.reset(seed=0, options=SAFE)
state = snapshot(source)
restore(target, state)
target.space.step(0.01)
restore(target, state)
print("contact_step", step_index + 1)
print("source_space_order", space_order(source))
print("target_space_order", space_order(target))
print("source_agent_order", body_order(source, source.agent))
print("target_agent_order", body_order(target, target.agent))
print("source_block_order", body_order(source, source.block))
print("target_block_order", body_order(target, target.block))
source.close()
target.close()
