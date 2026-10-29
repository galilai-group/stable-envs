"""PushT contact solver adapted from fracapuano/envX at d26c817.

Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
See licenses/envX-APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.
"""

from functools import partial

import jax
import jax.numpy as jnp


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
    return point, -normal, depth, face * 3 + jnp.where(t == 0, 1, jnp.where(t == 1, 2, 0))


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
    wall_normals = jnp.array(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)))
    planes = jnp.array((7.0, -504.0, 7.0, -504.0))
    for index in (1,) if circle_agent else (0, 1):
        for shape in range(vertices.shape[1]):
            verts = world[index, shape]
            penetration = planes[:, None] - wall_normals @ verts.T
            depth, selected = jax.lax.top_k(penetration, 2)
            points.append(verts[selected].reshape(-1, 2))
            normals.append(jnp.repeat(wall_normals, 2, axis=0))
            depths.append(jnp.where(enabled[index, shape] & (inverse[index, 0] > 0), depth.reshape(-1), -1))
            ids.append(selected.reshape(-1))
            pairs.extend([(2, index)] * 8)
    return tuple(jnp.concatenate(v) for v in (points, normals, depths, ids)) + (tuple(pairs),)


@partial(jax.jit, static_argnames=("circle_agent",))
@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, None))
def advance(body, bias, cached, identifiers, geometry, target, circle_agent):
    vertices, cog, inverse, radius, enabled = geometry
    count = cached.shape[0]

    def substep(_, carry):
        body, bias, cached, identifiers, total = carry
        velocity = body[:, 2:4]
        velocity = jnp.stack((velocity[0] + (100 * (target - body[0, :2]) - 20 * velocity[0]) * 0.01, velocity[1]))
        center = body[:, :2] + rotate(cog, body[:, 4]) + (velocity + bias[:, :2]) * 0.01
        angle = body[:, 4] + (body[:, 5] + bias[:, 2]) * 0.01
        position = center - rotate(cog, angle)
        body = jnp.concatenate((position, body[:, 2:4], angle[:, None], body[:, 5:6]), 1)
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
        bias_target = (1 - 0.9**0.6) * jnp.maximum(depth - 0.1, 0) / 0.01
        cached = jnp.where(active & (current_ids == identifiers), cached, 0.0)
        zero = jnp.zeros_like(angle[0])
        velocities = (
            (velocity[0, 0], velocity[0, 1], omega[0]),
            (velocity[1, 0], velocity[1, 1], omega[1]),
            (zero, zero, zero),
        )
        bias = ((zero, zero, zero),) * 3

        def apply(values, impulse, i):
            a, b = pairs[i]
            x, y = impulse * normal[i, 0], impulse * normal[i, 1]
            result = list(values)
            result[a] = (
                values[a][0] - x * inv[a, 0],
                values[a][1] - y * inv[a, 0],
                values[a][2] - impulse * ca[i] * inv[a, 1],
            )
            result[b] = (
                values[b][0] + x * inv[b, 0],
                values[b][1] + y * inv[b, 0],
                values[b][2] + impulse * cb[i] * inv[b, 1],
            )
            return tuple(result)

        for i in range(count):
            velocities = apply(velocities, cached[i], i)

        def normal_speed(values, i):
            a, b = pairs[i]
            x = values[b][0] - rb[i, 1] * values[b][2] - values[a][0] + ra[i, 1] * values[a][2]
            y = values[b][1] + rb[i, 0] * values[b][2] - values[a][1] - ra[i, 0] * values[a][2]
            return x * normal[i, 0] + y * normal[i, 1]

        def iteration(_, carry):
            velocities, bias, impulses, bias_impulses = carry
            impulses, bias_impulses = list(impulses), list(bias_impulses)
            for i in range(count):
                impulse = jnp.where(active[i], jnp.maximum(impulses[i] - normal_speed(velocities, i) / mass[i], 0), 0)
                correction = jnp.where(
                    active[i], jnp.maximum(bias_impulses[i] + (bias_target[i] - normal_speed(bias, i)) / mass[i], 0), 0
                )
                velocities = apply(velocities, impulse - impulses[i], i)
                bias = apply(bias, correction - bias_impulses[i], i)
                impulses[i], bias_impulses[i] = impulse, correction
            return velocities, bias, tuple(impulses), tuple(bias_impulses)

        impulses = tuple(cached[i] for i in range(count))
        velocities, bias, impulses, _ = jax.lax.fori_loop(
            0, 10, iteration, (velocities, bias, impulses, (zero,) * count), unroll=2
        )
        velocities = jnp.array(velocities)
        bias, cached = jnp.array(bias), jnp.stack(impulses)
        body = jnp.concatenate((body[:, :2], velocities[:2, :2], body[:, 4:5], velocities[:2, 2:3]), 1)
        return body, bias[:2], cached, current_ids, total + active.sum(dtype=jnp.int32)

    body, bias, cached, identifiers, total = jax.lax.fori_loop(
        0, 10, substep, (body, bias, cached, identifiers, jnp.int32(0))
    )
    point, normal, depth, _, _ = contacts(body, geometry, circle_agent=circle_agent)
    active = depth >= 0
    order = jnp.argsort(~active, stable=True)
    lines = jnp.concatenate(
        (
            point - normal * (depth[:, None] + 2),
            point + normal * 2,
            jnp.broadcast_to(jnp.array((231.0, 76.0, 60.0, 255.0)), (count, 4)),
        ),
        1,
    )
    lines = jnp.where(active[:, None], lines, 0)[order]
    lines = jnp.pad(lines, ((0, 90 - count), (0, 0)))
    return body, bias, cached, identifiers, total, lines, active.sum(dtype=jnp.int32)
