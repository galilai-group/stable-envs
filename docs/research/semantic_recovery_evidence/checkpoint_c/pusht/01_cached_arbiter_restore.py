"""Controlled wall contact: test explicit cached-arbiter restoration."""

import copy
import numpy as np
import pymunk
from pymunk._chipmunk_cffi import lib
from pymunk.arbiter import _arbiter_from_dict, _arbiter_to_dict

from stable_worldmodel.envs.pusht.env import PushT


CONTACT_OPTIONS = {
    "variation": [],
    "state": np.array([100, 100, 400, 400, 0, 0, 0], np.float64),
    "goal_state": np.array([200, 200, 300, 300, 0, 0, 0], np.float64),
}
SAFE_OPTIONS = {
    "variation": [],
    "state": np.array([100, 100, 250, 250, 0, 0, 0], np.float64),
    "goal_state": CONTACT_OPTIONS["goal_state"],
}


def body_fields(source, target):
    for sb, tb in ((source.agent, target.agent), (source.block, target.block)):
        tb.angle = sb.angle
        tb.position = tuple(sb.position)
        tb.velocity = tuple(sb.velocity)
        tb.angular_velocity = sb.angular_velocity
        tb.force = tuple(sb.force)
        tb.torque = sb.torque
        target.space.reindex_shapes_for_body(tb)
    target.goal_state = source.goal_state.copy()
    target._goal = source._goal.copy()


def shape_key(env, shape):
    if isinstance(shape, pymunk.Segment):
        return ("segment", tuple(shape.a), tuple(shape.b), shape.radius)
    if isinstance(shape, pymunk.Circle):
        return ("circle", "agent" if shape.body is env.agent else "other", tuple(shape.offset), shape.radius)
    if isinstance(shape, pymunk.Poly):
        body = "block" if shape.body is env.block else "other"
        return ("poly", body, tuple(sorted(tuple(v) for v in shape.get_vertices())), shape.radius)
    raise TypeError(type(shape))


def arbiter_dicts(source, target):
    destination = {shape_key(target, s): s for s in target.space.shapes}
    result = []
    for raw in source.space._get_arbiters():
        d = _arbiter_to_dict(raw, source.space)
        d = copy.deepcopy({k: v for k, v in d.items() if k not in ("a", "b")}) | {
            "a": destination[shape_key(source, d["a"])],
            "b": destination[shape_key(source, d["b"])],
        }
        result.append(d)
    return result


def add_arbiters(target, dictionaries):
    assert not target.space._get_arbiters()
    for d in dictionaries:
        raw = _arbiter_from_dict(d, target.space)
        lib.cpSpaceAddCachedArbiter(target.space._space, raw)


def output_vector(output):
    obs, reward, terminated, truncated, info = output
    return np.r_[obs["state"], reward, float(terminated), float(truncated), float(info["n_contacts"])]


def run(copy_arbiters, copy_space_scalars):
    source, target = PushT(), PushT()
    source.reset(seed=0, options=CONTACT_OPTIONS)
    target.reset(seed=0, options=SAFE_OPTIONS)
    dictionaries = arbiter_dicts(source, target)
    body_fields(source, target)
    if copy_space_scalars:
        lib.cpSpaceSetTimestamp(target.space._space, lib.cpSpaceGetTimestamp(source.space._space))
        lib.cpSpaceSetCurrentTimeStep(
            target.space._space, lib.cpSpaceGetCurrentTimeStep(source.space._space)
        )
        lib.cpSpaceSetShapeIDCounter(
            target.space._space, lib.cpSpaceGetShapeIDCounter(source.space._space)
        )
    if copy_arbiters:
        add_arbiters(target, dictionaries)
    action = np.zeros(2, np.float32)
    expected = output_vector(source.step(action))
    actual = output_vector(target.step(action))
    print(
        "case",
        {"copy_arbiters": copy_arbiters, "copy_space_scalars": copy_space_scalars},
        "source_arbiter_count",
        len(dictionaries),
        "max_error",
        float(np.max(np.abs(expected - actual))),
        "delta",
        (expected - actual).tolist(),
        "exact",
        bool(np.array_equal(expected, actual)),
    )
    source.close()
    target.close()


for copy_arbiters, copy_space_scalars in ((False, False), (False, True), (True, False), (True, True)):
    run(copy_arbiters, copy_space_scalars)
