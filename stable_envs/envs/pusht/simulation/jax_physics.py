"""PushT contact solver adapted from fracapuano/envX at d26c817.

Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
See licenses/APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.

The wall contacts (``SegmentToPoly``, ``ContactPoints``) and the arbiter contact cache port Munk2D at commit
ade7ed72849e. Copyright (c) 2025 Victor Blomqvist, Copyright (c) 2007-2024 Scott Lembcke and Howling Moon
Software. MIT license; see licenses/Munk2D-MIT.txt.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def perpendicular(v):
    return jnp.stack((-v[..., 1], v[..., 0]), axis=-1)


def rotate(v, angle):
    c, s = jnp.cos(angle), jnp.sin(angle)
    return jnp.stack((c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]), -1)


def circle_polygon(center, radius, vertices):
    edges = jnp.roll(vertices, -1, axis=0) - vertices
    normals = -perpendicular(edges) / jnp.linalg.norm(edges, axis=-1, keepdims=True)
    separation = jnp.sum((center - vertices) * normals, axis=-1)
    face = jnp.argmax(separation)
    start, edge = vertices[face], edges[face]
    t = jnp.clip(jnp.dot(center - start, edge) / jnp.dot(edge, edge), 0, 1)
    point = start + t * edge
    delta = center - point
    distance = jnp.maximum(jnp.linalg.norm(delta), 1e-12)
    inside = separation[face] <= 0
    normal = jnp.where(inside, normals[face], delta / distance)
    depth = radius - jnp.where(inside, separation[face], distance)
    # Munk2D hashes every circle contact as 0.
    return point, -normal, depth, jnp.zeros_like(face)


#: Munk2D's default ``collisionSlop`` and ``collisionBias`` come from float literals: ``0.1f`` and ``(1 - 0.1f)**60``.
SLOP = float(np.float32(0.1))
BIAS = 1 - (float(np.float32(1) - np.float32(0.1)) ** 60) ** 0.01
#: The four wall segments (a, b) of the Pymunk scene, each of radius ``WALL_RADIUS``.
WALLS = (
    ((5.0, 506.0), (5.0, 5.0)),
    ((5.0, 5.0), (506.0, 5.0)),
    ((506.0, 5.0), (506.0, 506.0)),
    ((5.0, 506.0), (506.0, 506.0)),
)
WALL_RADIUS = 2.0


def segment_polygon(a, b, vertices):
    """Munk2D ``SegmentToPoly`` for a static segment ``a``-``b`` and a counterclockwise quad.

    Returns two contact rows in Munk2D's order (the point on the quad, the normal from the segment into the
    quad, the depth or -1 for no contact) and the id ``4 * end + vertex`` that Munk2D hashes for the
    cached impulse.
    """
    a, b = jnp.asarray(a, vertices.dtype), jnp.asarray(b, vertices.dtype)
    delta = b - a
    length = jnp.linalg.norm(delta)
    tangent_normal = jnp.stack((delta[1], -delta[0])) / length
    edges = jnp.roll(vertices, -1, axis=0) - vertices
    outward = -perpendicular(edges) / jnp.linalg.norm(edges, axis=-1, keepdims=True)
    # The axis of least separation (EPA when the cores overlap); n points from the segment into the quad.
    projection = vertices @ tangent_normal
    c = a @ tangent_normal
    face = jnp.minimum(outward @ a, outward @ b) - jnp.sum(outward * vertices, axis=-1)
    separations = jnp.concatenate((jnp.stack((projection.min() - c, c - projection.max())), face))
    axes = jnp.concatenate((jnp.stack((tangent_normal, -tangent_normal)), -outward))
    pick = jnp.argmax(separations)
    normal, distance = axes[pick], separations[pick]
    # Separated cores: GJK's closest points, from each vertex to the segment and each end to each edge.
    t = jnp.clip((vertices - a) @ delta / (length * length), 0, 1)
    to_vertex = vertices - (a + t[:, None] * delta)
    ends = jnp.stack((a, b))
    s = jnp.clip(
        jnp.sum((ends[None] - vertices[:, None]) * edges[:, None], -1) / jnp.sum(edges * edges, -1)[:, None], 0, 1
    )
    to_edge = vertices[:, None] + s[..., None] * edges[:, None] - ends[None]
    gaps = jnp.concatenate((to_vertex[:, None], to_edge), 1).reshape(-1, 2)
    closest = gaps[jnp.argmin(jnp.sum(gaps * gaps, -1))]
    gap = jnp.linalg.norm(closest)
    apart = distance > 0
    normal = jnp.where(apart, closest / jnp.where(apart, gap, 1), normal)
    distance = jnp.where(apart, gap, distance)
    # Support edges: the segment's face toward the quad, and the quad's edge most against n.
    forward = tangent_normal @ normal > 0
    e1a, e1b = jnp.where(forward, a, b), jnp.where(forward, b, a)
    end_a = jnp.where(forward, 0, 1)
    i1 = jnp.argmax(-(vertices @ normal))
    i0, i2 = (i1 + 3) % 4, (i1 + 1) % 4
    # Munk2D's plane i holds the normal of the edge ending at vertex i.
    left = -(outward[i0] @ normal) > -(outward[i1] @ normal)
    first, second = jnp.where(left, i0, i1), jnp.where(left, i1, i2)
    e2a, e2b = vertices[first], vertices[second]
    d1a, d1b, d2a, d2b = cross(e1a, normal), cross(e1b, normal), cross(e2a, normal), cross(e2b, normal)
    tiny = jnp.finfo(vertices.dtype).tiny
    inv1, inv2 = 1 / (d1b - d1a + tiny), 1 / (d2b - d2a + tiny)
    t1 = jnp.clip(jnp.stack((d2b - d1a, d2a - d1a)) * inv1, 0, 1)
    t2 = jnp.clip(jnp.stack((d1a - d2a, d1b - d2a)) * inv2, 0, 1)
    on_segment = normal * WALL_RADIUS + e1a * (1 - t1[:, None]) + e1b * t1[:, None]
    on_quad = e2a * (1 - t2[:, None]) + e2b * t2[:, None]
    gaps = (on_quad - on_segment) @ normal
    hits = (distance - WALL_RADIUS <= 0) & (gaps <= 0)
    identifier = jnp.stack((end_a * 4 + second, (1 - end_a) * 4 + first))
    # Munk2D packs the contacts that pass, in order.
    order = jnp.where(hits[0], jnp.array((0, 1)), jnp.array((1, 0)))
    depth = jnp.where(hits, -gaps, -1.0)[order]
    depth = depth.at[1].set(jnp.where(hits.all(), depth[1], -1.0))
    identifier = jnp.where(depth >= 0, identifier[order], -1)
    return on_quad[order], jnp.broadcast_to(normal, (2, 2)), depth, identifier


def polygon_polygon(a, b):
    # Clip the incident edge against the reference face's two side planes.
    def faces(vertices, other):
        edges = jnp.roll(vertices, -1, axis=0) - vertices
        normals = -perpendicular(edges) / jnp.linalg.norm(edges, axis=-1, keepdims=True)
        depths = jnp.min(other @ normals.T, axis=0) - jnp.sum(vertices * normals, axis=-1)
        face = jnp.argmax(depths)
        return normals, face, depths[face]

    an, ai, ad = faces(a, b)
    bn, bi, bd = faces(b, a)
    flip = bd > ad
    ref, inc = jnp.where(flip, b, a), jnp.where(flip, a, b)
    normals, face = jnp.where(flip, bn, an), jnp.where(flip, bi, ai)
    normal = normals[face]
    edge = jnp.argmin(jnp.where(flip, an, bn) @ normal)
    first, second = inc[edge], inc[(edge + 1) % 4]
    tangent = perpendicular(normal)
    lo, hi = jnp.sort(jnp.array((ref[face] @ tangent, ref[(face + 1) % 4] @ tangent)))
    denom = (second - first) @ tangent
    safe = jnp.where(jnp.abs(denom) > 1e-12, denom, 1)
    ends = jnp.sort(jnp.array(((lo - first @ tangent) / safe, (hi - first @ tangent) / safe)))
    t = jnp.clip(ends, 0, 1)
    points = first + t[:, None] * (second - first)
    depth = (ref[face] - points) @ normal
    overlap = (jnp.maximum(ad, bd) <= 0) & (
        (jnp.abs(denom) > 1e-12) | ((first @ tangent >= lo) & (first @ tangent <= hi))
    )
    depth = jnp.where(overlap, depth, -1)
    depth = depth.at[1].set(jnp.where(jnp.linalg.norm(points[1] - points[0]) > 1e-9, depth[1], -1))
    return (
        points,
        jnp.broadcast_to(jnp.where(flip, -normal, normal), (2, 2)),
        depth,
        face * 8 + edge * 2 + jnp.arange(2),
    )


def contacts(body, geometry, *, circle_agent):
    vertices, cog, inverse, radius, enabled = geometry
    position, angle = body[:, :2], body[:, 4]
    world = position[:, None, None, :] + rotate(vertices, angle[:, None, None])
    points, normals, depths, ids, pairs = [], [], [], [], []
    for a in range(1 if circle_agent else vertices.shape[1]):
        for b in range(vertices.shape[1]):
            if circle_agent:
                point, normal, depth, identifier = circle_polygon(position[0], radius, world[1, b])
                point, normal, depth, identifier = point[None], normal[None], depth[None], identifier[None]
            else:
                polygon = polygon_polygon(world[0, a], world[1, b])
                point, normal, depth, identifier = circle_polygon(position[0], radius, world[1, b])
                circle = (
                    jnp.broadcast_to(point, (2, 2)),
                    jnp.broadcast_to(normal, (2, 2)),
                    jnp.array((depth, -1.0)),
                    jnp.array((identifier, -1)),
                )
                point, normal, depth, identifier = tuple(
                    jnp.where((radius > 0) & (a == 0), c, p) for c, p in zip(circle, polygon, strict=True)
                )
                depth = jnp.where((radius > 0) & (a > 0), -1, depth)
            points.append(point)
            normals.append(normal)
            depths.append(jnp.where(enabled[0, a] & enabled[1, b], depth, -1))
            ids.append(identifier)
            pairs.extend([(0, 1)] * len(depth))
    for index in (1,) if circle_agent else (0, 1):
        for shape in range(vertices.shape[1]):
            for wall in WALLS:
                point, normal, depth, identifier = segment_polygon(*wall, world[index, shape])
                points.append(point)
                normals.append(normal)
                depths.append(jnp.where(enabled[index, shape] & (inverse[index, 0] > 0), depth, -1))
                ids.append(identifier)
                pairs.extend([(2, index)] * 2)
    return tuple(jnp.concatenate(v) for v in (points, normals, depths, ids)) + (tuple(pairs),)


#: Arbiter ages in the contact ids, as ``numba_physics._STALE``.
STALE = 1 << 16


def groups(count, shapes, circle_agent):
    """Each slot's arbiter (shape pair): its first slot and slot count."""
    walls = shapes if circle_agent else 2 * shapes * shapes
    slots = np.arange(count)
    start = np.where(slots >= walls, walls + (slots - walls) // 2 * 2, slots if circle_agent else slots // 2 * 2)
    size = np.where(slots >= walls, 2, 1 if circle_agent else 2)
    return start, size


def _visit_order(tree, bounds, tight, velocity, order, active, walls, circle_agent, shapes):
    """Host callback: ``numba_physics.visit_order`` over a batch.

    The callback thread may run without 64-bit JAX types, so integers pass as int32 and float64 values as
    their uint32 words.
    """
    from .numba_physics import visit_order

    def unpack(words):
        return np.ascontiguousarray(words).view(np.float64)[..., 0]

    tree, bounds = np.array(tree, np.int64), unpack(bounds).copy()
    tight, velocity = unpack(tight), unpack(velocity)
    sequence = np.empty(active.shape, np.int32)
    live = np.empty(active.shape[:-1], np.int32)
    visit_order(
        tree.reshape(-1, tree.shape[-1]),
        bounds.reshape(-1, *bounds.shape[-2:]),
        np.asarray(tight, np.float64).reshape(-1, *tight.shape[-2:]),
        np.asarray(velocity, np.float64).reshape(-1, *velocity.shape[-2:]),
        np.asarray(order, np.int64).reshape(-1, order.shape[-1]),
        np.asarray(active).reshape(-1, active.shape[-1]),
        walls,
        circle_agent,
        shapes,
        sequence.reshape(-1, sequence.shape[-1]),
        live.reshape(-1),
    )
    return tree.astype(np.int32), bounds.view(np.uint32).reshape(*bounds.shape, 2), sequence, live


def control(velocity, position, target, single):
    """The agent velocity after one PD substep, in float32 when the actions are float32, as the source computes."""
    if not single:
        return velocity + (100 * (target - position) - 20 * velocity) * 0.01
    f32 = jnp.float32
    step = (f32(100) * (target.astype(f32) - position.astype(f32))) + (-20 * velocity).astype(f32)
    return (velocity.astype(f32) + step * f32(0.01)).astype(velocity.dtype)


def leaf_bounds(body, vertices, radius, enabled):
    """Each shape's bounding box, as Munk2D caches it, in ``numba_broadphase`` leaf order (absent shapes 0)."""
    from .numba_broadphase import LEAVES, SHAPES

    world = body[:, None, None, :2] + rotate(vertices, body[:, 4, None, None])
    boxes = jnp.concatenate((world.min(axis=2), world.max(axis=2)), -1)
    circle = jnp.concatenate((body[0, :2] - radius, body[0, :2] + radius))
    boxes = boxes.at[0, 0].set(jnp.where(radius > 0, circle, boxes[0, 0]))
    boxes = jnp.where(enabled[..., None], boxes, 0.0)
    padded = jnp.zeros((2, SHAPES, 4), boxes.dtype).at[:, : boxes.shape[1]].set(boxes)
    return padded.reshape(LEAVES, 4)


@partial(jax.jit, static_argnames=("circle_agent", "single"))
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None))
def advance(body, bias, cached, identifiers, tree, tree_bounds, order, geometry, target, circle_agent, single):
    """Step one environment's 10 substeps, as ``numba_physics.advance_batch`` steps it.

    The broadphase that orders the solver (``numba_broadphase``) runs on the host through a callback.
    """
    from .numba_broadphase import SHAPES

    vertices, cog, inverse, radius, enabled = geometry
    count = cached.shape[0]
    shapes = vertices.shape[1]
    walls = shapes if circle_agent else 2 * shapes * shapes
    start, size = groups(count, shapes, circle_agent)
    mate = np.minimum(start + 1, count - 1)
    pair = size == 2

    def substep(_, carry):
        body, bias, cached, identifiers, tree, tree_bounds, total = carry
        velocity = body[:, 2:4]
        velocity = jnp.stack((control(velocity[0], body[0, :2], target, single), velocity[1]))
        center = body[:, :2] + rotate(cog, body[:, 4]) + (velocity + bias[:, :2]) * 0.01
        angle = body[:, 4] + (body[:, 5] + bias[:, 2]) * 0.01
        position = center - rotate(cog, angle)
        body = jnp.concatenate((position, body[:, 2:4], angle[:, None], body[:, 5:6]), 1)
        moving = jnp.repeat(velocity, SHAPES, axis=0)
        # Pymunk applies Space.damping to dynamic bodies after integrating positions.
        dynamic = inverse[:, 0] > 0
        velocity = jnp.where(dynamic[:, None], 0.0, velocity)
        omega = jnp.where(dynamic, 0.0, body[:, 5])
        point, normal, depth, current_ids, pairs = contacts(body, geometry, circle_agent=circle_agent)
        centers = jnp.concatenate((center, jnp.zeros((1, 2))))
        inv = jnp.concatenate((inverse, jnp.zeros((1, 2))))
        first = jnp.array([pair[0] for pair in pairs])
        second = jnp.array([pair[1] for pair in pairs])
        ra, rb = point - centers[first], point - centers[second]
        ca, cb = cross(ra, normal), cross(rb, normal)
        mass = inv[first, 0] + inv[second, 0] + ca**2 * inv[first, 1] + cb**2 * inv[second, 1]
        mass = jnp.maximum(mass, 1e-12)
        active = depth >= 0
        bias_target = BIAS * jnp.maximum(depth - SLOP, 0) / 0.01
        # Each contact starts from the impulse of its arbiter's cached contact with the same id; an arbiter
        # that had no contacts for a step or two keeps its contacts but does not apply them first.
        old_first, old_second = identifiers[start], jnp.where(pair, identifiers[mate], -1)
        impulses = jnp.where(
            active & pair & (old_second >= 0) & (old_second % STALE == current_ids),
            cached[mate],
            jnp.where(active & (old_first >= 0) & (old_first % STALE == current_ids), cached[start], 0.0),
        )
        fresh = (old_first < STALE) & (old_second < STALE)
        new_tree, words, sequence, live = jax.pure_callback(
            partial(_visit_order, walls=walls, circle_agent=circle_agent, shapes=shapes),
            (
                jax.ShapeDtypeStruct(tree.shape, jnp.int32),
                jax.ShapeDtypeStruct((*tree_bounds.shape, 2), jnp.uint32),
                jax.ShapeDtypeStruct((count,), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.int32),
            ),
            tree.astype(jnp.int32),
            jax.lax.bitcast_convert_type(tree_bounds, jnp.uint32),
            jax.lax.bitcast_convert_type(leaf_bounds(body, vertices, radius, enabled), jnp.uint32),
            jax.lax.bitcast_convert_type(moving, jnp.uint32),
            order.astype(jnp.int32),
            active,
            vmap_method="broadcast_all",
        )
        tree = new_tree.astype(tree.dtype)
        tree_bounds = jax.lax.bitcast_convert_type(words, tree_bounds.dtype)
        values = jnp.stack(
            (
                jnp.stack((velocity[0, 0], velocity[0, 1], omega[0])),
                jnp.stack((velocity[1, 0], velocity[1, 1], omega[1])),
                jnp.zeros(3, velocity.dtype),
            )
        )

        def apply(values, impulse, i):
            a, b = first[i], second[i]
            x, y = impulse * normal[i, 0], impulse * normal[i, 1]
            values = values.at[a].add(jnp.stack((-x * inv[a, 0], -y * inv[a, 0], -impulse * ca[i] * inv[a, 1])))
            return values.at[b].add(jnp.stack((x * inv[b, 0], y * inv[b, 0], impulse * cb[i] * inv[b, 1])))

        def normal_speed(values, i):
            a, b = first[i], second[i]
            x = values[b, 0] - rb[i, 1] * values[b, 2] - values[a, 0] + ra[i, 1] * values[a, 2]
            y = values[b, 1] + rb[i, 0] * values[b, 2] - values[a, 1] - ra[i, 0] * values[a, 2]
            return x * normal[i, 0] + y * normal[i, 1]

        # The solver visits the touching shape pairs in the order the broadphase reports them.
        for j in range(count):
            i = sequence[j]
            values = apply(values, jnp.where((j < live) & fresh[i], impulses[i], 0.0), i)

        def iteration(_, carry):
            values, push, impulses, corrections = carry
            for j in range(count):
                i, on = sequence[j], j < live
                impulse = jnp.where(on, jnp.maximum(impulses[i] - normal_speed(values, i) / mass[i], 0), impulses[i])
                correction = jnp.where(
                    on,
                    jnp.maximum(corrections[i] + (bias_target[i] - normal_speed(push, i)) / mass[i], 0),
                    corrections[i],
                )
                values = apply(values, impulse - impulses[i], i)
                push = apply(push, correction - corrections[i], i)
                impulses, corrections = impulses.at[i].set(impulse), corrections.at[i].set(correction)
            return values, push, impulses, corrections

        values, push, impulses, _ = jax.lax.fori_loop(
            0, 10, iteration, (values, jnp.zeros_like(values), impulses, jnp.zeros_like(impulses))
        )
        touching = active[start] | (pair & active[mate])
        aged = jnp.where((identifiers >= 0) & (identifiers < 2 * STALE), identifiers + STALE, -1)
        cached = jnp.where(touching, jnp.where(active, impulses, 0.0), jnp.where(aged >= 0, cached, 0.0))
        identifiers = jnp.where(touching, jnp.where(active, current_ids, -1), aged)
        body = jnp.concatenate((body[:, :2], values[:2, :2], body[:, 4:5], values[:2, 2:3]), 1)
        return body, push[:2], cached, identifiers, tree, tree_bounds, total + active.sum(dtype=jnp.int32)

    body, bias, cached, identifiers, tree, tree_bounds, total = jax.lax.fori_loop(
        0, 10, substep, (body, bias, cached, identifiers, tree, tree_bounds, jnp.int32(0))
    )
    point, normal, depth, _, _ = contacts(body, geometry, circle_agent=circle_agent)
    active = depth >= 0
    order = jnp.argsort(~active, stable=True)
    lines = jnp.concatenate(
        (
            point + normal * (depth[:, None] - 2),
            point + normal * 2,
            jnp.broadcast_to(jnp.array((231.0, 76.0, 60.0, 255.0)), (count, 4)),
        ),
        1,
    )
    lines = jnp.where(active[:, None], lines, 0)[order]
    lines = jnp.pad(lines, ((0, 90 - count), (0, 0)))
    return body, bias, cached, identifiers, tree, tree_bounds, total, lines, active.sum(dtype=jnp.int32)
