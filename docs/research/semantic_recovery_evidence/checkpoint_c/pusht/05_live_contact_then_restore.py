"""Rebuild active membership by one detection step, then restore explicit fields."""

import numpy as np
import pymunk
from pymunk._chipmunk_cffi import lib

from stable_worldmodel.envs.pusht.env import PushT


CONTACT = {
    "variation": [],
    "state": np.array([100, 100, 400, 400, 0, 0, 0], np.float64),
    "goal_state": np.array([200, 200, 300, 300, 0, 0, 0], np.float64),
}
SAFE = CONTACT | {"state": np.array([100, 100, 250, 250, 0, 0, 0], np.float64)}


def body_snapshot(env):
    return [
        (
            tuple(body.position),
            tuple(body.velocity),
            body.angle,
            body.angular_velocity,
            tuple(body.force),
            body.torque,
        )
        for body in (env.agent, env.block)
    ]


def restore_bodies(env, snapshot):
    for body, (position, velocity, angle, angular_velocity, force, torque) in zip(
        (env.agent, env.block), snapshot, strict=True
    ):
        body.angle = angle
        body.position = position
        body.velocity = velocity
        body.angular_velocity = angular_velocity
        body.force = force
        body.torque = torque
        env.space.reindex_shapes_for_body(body)


def shape_key(env, shape):
    if isinstance(shape, pymunk.Segment):
        return ("segment", tuple(shape.a), tuple(shape.b), shape.radius)
    if isinstance(shape, pymunk.Circle):
        return ("circle", "agent" if shape.body is env.agent else "other", tuple(shape.offset), shape.radius)
    if isinstance(shape, pymunk.Poly):
        return (
            "poly",
            "block" if shape.body is env.block else "other",
            tuple(sorted(tuple(v) for v in shape.get_vertices())),
            shape.radius,
        )
    raise TypeError(type(shape))


def arbiter_map(env):
    result = {}
    shapes = {shape._shape: shape for shape in env.space.shapes}
    for raw in env.space._get_arbiters():
        key = (shape_key(env, shapes[raw.a]), shape_key(env, shapes[raw.b]))
        result[key] = raw
    return result


def copy_arbiter(source, target):
    for field in ("e", "u", "surface_vr", "n", "swapped", "stamp", "state", "processCollision"):
        setattr(target, field, getattr(source, field))
    assert source.count == target.count
    for index in range(source.count):
        sc, tc = source.contacts[index], target.contacts[index]
        for field in ("r1", "r2", "nMass", "tMass", "bounce", "jnAcc", "jtAcc", "jBias", "bias", "hash"):
            setattr(tc, field, getattr(sc, field))


def vector(output):
    observation, reward, terminated, truncated, info = output
    return np.r_[
        observation["state"], reward, float(terminated), float(truncated), float(info["n_contacts"])
    ]


source, target = PushT(), PushT()
source.reset(seed=0, options=CONTACT)
target.reset(seed=0, options=SAFE)
snapshot = body_snapshot(source)
restore_bodies(target, snapshot)
print("before_detection_target_arbiters", len(target.space._get_arbiters()))
target.space.step(0.01)
print("after_detection_target_arbiters", len(target.space._get_arbiters()))
restore_bodies(target, snapshot)
source_arbiters, target_arbiters = arbiter_map(source), arbiter_map(target)
print("arbiter_keys_equal", source_arbiters.keys() == target_arbiters.keys())
for key in source_arbiters:
    copy_arbiter(source_arbiters[key], target_arbiters[key])
lib.cpSpaceSetTimestamp(target.space._space, lib.cpSpaceGetTimestamp(source.space._space))
lib.cpSpaceSetCurrentTimeStep(
    target.space._space, lib.cpSpaceGetCurrentTimeStep(source.space._space)
)
lib.cpSpaceSetShapeIDCounter(
    target.space._space, lib.cpSpaceGetShapeIDCounter(source.space._space)
)
source_membership, target_membership = [], []
source.block.each_arbiter(lambda arbiter: source_membership.append(arbiter))
target.block.each_arbiter(lambda arbiter: target_membership.append(arbiter))
print("block_membership", len(source_membership), len(target_membership))
action = np.zeros(2, np.float32)
expected, actual = vector(source.step(action)), vector(target.step(action))
print("exact", bool(np.array_equal(expected, actual)))
print("max_error", float(np.max(np.abs(expected - actual))))
print("expected", expected.tolist())
print("actual", actual.tolist())
print("delta", (expected - actual).tolist())
source.close()
target.close()
