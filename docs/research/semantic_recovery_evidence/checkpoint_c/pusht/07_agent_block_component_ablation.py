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


def copy_arbiter(source, target, mode):
    if mode in ("header", "full"):
        for field in ("e", "u", "surface_vr", "n", "swapped", "stamp", "state", "processCollision"):
            setattr(target, field, getattr(source, field))
    assert source.count == target.count
    for index in range(source.count):
        sc, tc = source.contacts[index], target.contacts[index]
        fields = ()
        if mode == "geometry":
            fields = ("r1", "r2", "nMass", "tMass", "bounce", "bias", "hash")
        elif mode == "accumulators":
            fields = ("jnAcc", "jtAcc", "jBias")
        elif mode == "contacts":
            fields = ("r1", "r2", "nMass", "tMass", "bounce", "jnAcc", "jtAcc", "jBias", "bias", "hash")
        elif mode == "full":
            fields = ("r1", "r2", "nMass", "tMass", "bounce", "jnAcc", "jtAcc", "jBias", "bias", "hash")
        for field in fields:
            setattr(tc, field, getattr(sc, field))


def vector(output):
    observation, reward, terminated, truncated, info = output
    return np.r_[
        observation["state"], reward, float(terminated), float(truncated), float(info["n_contacts"])
    ]


for mode in ("none", "header", "geometry", "accumulators", "contacts", "full"):
    source, target = PushT(), PushT()
    source.reset(seed=0)
    contact_step = None
    for step_index in range(30):
        chase = ((np.array(source.block.position) - np.array(source.agent.position)) / 100).astype(np.float64)
        contact_output = source.step(chase)
        if contact_output[4]["n_contacts"]:
            contact_step = step_index + 1
            break
    assert contact_step is not None
    target.reset(seed=0, options=SAFE)
    snapshot = body_snapshot(source)
    restore_bodies(target, snapshot)
    target.goal_state = source.goal_state.copy()
    target._goal = source._goal.copy()
    target.space.step(0.01)
    restore_bodies(target, snapshot)
    source_arbiters, target_arbiters = arbiter_map(source), arbiter_map(target)
    assert source_arbiters.keys() == target_arbiters.keys()
    for key in source_arbiters:
        copy_arbiter(source_arbiters[key], target_arbiters[key], mode)
    lib.cpSpaceSetTimestamp(target.space._space, lib.cpSpaceGetTimestamp(source.space._space))
    lib.cpSpaceSetCurrentTimeStep(target.space._space, lib.cpSpaceGetCurrentTimeStep(source.space._space))
    lib.cpSpaceSetShapeIDCounter(target.space._space, lib.cpSpaceGetShapeIDCounter(source.space._space))
    actions = [
        np.array([0.173456789, -0.081234567], np.float64),
        np.array([-0.07, 0.11], np.float32),
        np.array([0.03, -0.02], np.float32),
    ]
    errors = []
    deltas = []
    exact = True
    for action in actions:
        expected, actual = vector(source.step(action)), vector(target.step(action))
        errors.append(float(np.max(np.abs(expected - actual))))
        deltas.append((expected - actual).tolist())
        exact &= bool(np.array_equal(expected, actual))
    print(
        "mode", mode, "contact_step", contact_step,
        "source_arbiter_count", len(source_arbiters),
        "exact", exact,
        "errors", errors,
        "deltas", deltas,
    )
    source.close()
    target.close()
