"""Verify exposed cached-arbiter fields and active membership after restoration."""

import copy
import numpy as np
import pymunk
from pymunk._chipmunk_cffi import lib
from pymunk.arbiter import _arbiter_from_dict, _arbiter_to_dict

from stable_worldmodel.envs.pusht.env import PushT


CONTACT = {
    "variation": [],
    "state": np.array([100, 100, 400, 400, 0, 0, 0], np.float64),
    "goal_state": np.array([200, 200, 300, 300, 0, 0, 0], np.float64),
}
SAFE = CONTACT | {"state": np.array([100, 100, 250, 250, 0, 0, 0], np.float64)}


def body_fields(source, target):
    for sb, tb in ((source.agent, target.agent), (source.block, target.block)):
        tb.angle = sb.angle
        tb.position = tuple(sb.position)
        tb.velocity = tuple(sb.velocity)
        tb.angular_velocity = sb.angular_velocity
        tb.force = tuple(sb.force)
        tb.torque = sb.torque
        target.space.reindex_shapes_for_body(tb)


def shape_key(env, shape):
    if isinstance(shape, pymunk.Segment):
        return ("segment", tuple(shape.a), tuple(shape.b), shape.radius)
    if isinstance(shape, pymunk.Circle):
        return ("circle", "agent" if shape.body is env.agent else "other", tuple(shape.offset), shape.radius)
    if isinstance(shape, pymunk.Poly):
        body = "block" if shape.body is env.block else "other"
        return ("poly", body, tuple(sorted(tuple(v) for v in shape.get_vertices())), shape.radius)
    raise TypeError(type(shape))


def normalized(env, raw):
    d = _arbiter_to_dict(raw, env.space)
    return {
        **{k: copy.deepcopy(v) for k, v in d.items() if k not in ("a", "b")},
        "a": shape_key(env, d["a"]),
        "b": shape_key(env, d["b"]),
    }


source, target = PushT(), PushT()
source.reset(seed=0, options=CONTACT)
target.reset(seed=0, options=SAFE)
body_fields(source, target)
destination = {shape_key(target, shape): shape for shape in target.space.shapes}
source_dicts = [normalized(source, raw) for raw in source.space._get_arbiters()]
for normalized_dict in source_dicts:
    d = copy.deepcopy(normalized_dict)
    d["a"] = destination[d["a"]]
    d["b"] = destination[d["b"]]
    lib.cpSpaceAddCachedArbiter(target.space._space, _arbiter_from_dict(d, target.space))
target_dicts = [normalized(target, raw) for raw in target.space._get_arbiters()]


def body_arbiters(body):
    result = []
    body.each_arbiter(lambda arbiter: result.append(normalized(source if body is source.block else target, arbiter._arbiter)))
    return result


print("space_count", len(source_dicts), len(target_dicts))
print("space_dict_exact", source_dicts == target_dicts)
print("source_dict", source_dicts)
print("target_dict", target_dicts)
print("block_membership_count", len(body_arbiters(source.block)), len(body_arbiters(target.block)))
print("block_membership_exact", body_arbiters(source.block) == body_arbiters(target.block))
print("sleeping", source.agent.is_sleeping, target.agent.is_sleeping, source.block.is_sleeping, target.block.is_sleeping)
print("timestamps", lib.cpSpaceGetTimestamp(source.space._space), lib.cpSpaceGetTimestamp(target.space._space))
print("current_dt", lib.cpSpaceGetCurrentTimeStep(source.space._space), lib.cpSpaceGetCurrentTimeStep(target.space._space))
source.close()
target.close()
