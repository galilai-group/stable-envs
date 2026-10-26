"""PushT kernels adapted from fracapuano/envX at d26c817c1124229c819be574f137cd68bda4380a.

Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
See licenses/envX-APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array
WINDOW_SIZE = 512.0
TEE_CENTER_OF_GRAVITY = (0.0, 45.0)
TEE_AREA = 6300.0
_BAR_VERTICES = ((-60.0, 0.0), (60.0, 0.0), (60.0, 30.0), (-60.0, 30.0))
_STEM_VERTICES = ((-15.0, 30.0), (15.0, 30.0), (15.0, 120.0), (-15.0, 120.0))
_TEE_VERTICES = (
    (-60.0, 0.0),
    (60.0, 0.0),
    (60.0, 30.0),
    (15.0, 30.0),
    (15.0, 120.0),
    (-15.0, 120.0),
    (-15.0, 30.0),
    (-60.0, 30.0),
)


def rotation_matrix(angle: Array) -> Array:
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    return jnp.array(((cosine, -sine), (sine, cosine)), dtype=jnp.float32)


def transform_points(points: Array, position: Array, angle: Array) -> Array:
    return points @ rotation_matrix(angle).T + position


def inverse_transform_points(points: Array, position: Array, angle: Array) -> Array:
    return (points - position) @ rotation_matrix(angle)


def tee_rectangles(position: Array, angle: Array) -> Array:
    local = jnp.asarray((_BAR_VERTICES, _STEM_VERTICES), dtype=jnp.float32)
    return transform_points(local, position, angle)


def tee_vertices(position: Array, angle: Array) -> Array:
    return transform_points(jnp.asarray(_TEE_VERTICES, dtype=jnp.float32), position, angle)


def tee_center_of_mass(position: Array, angle: Array) -> Array:
    cog = jnp.asarray(TEE_CENTER_OF_GRAVITY, dtype=jnp.float32)
    return transform_points(cog, position, angle)


def origin_from_center_of_mass(center: Array, angle: Array) -> Array:
    cog = jnp.asarray(TEE_CENTER_OF_GRAVITY, dtype=jnp.float32)
    return center - cog @ rotation_matrix(angle).T


def _cross_2d(first: Array, second: Array) -> Array:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _points_in_convex_polygon(points: Array, polygon: Array) -> Array:
    edges = jnp.roll(polygon, -1, axis=0) - polygon
    relative = points[:, None, :] - polygon[None, :, :]
    return jnp.all(_cross_2d(edges[None, :, :], relative) >= -1e-05, axis=1)


def _edge_intersections(first: Array, second: Array) -> tuple[Array, Array]:
    first_delta = jnp.roll(first, -1, axis=0) - first
    second_delta = jnp.roll(second, -1, axis=0) - second
    starts_a = jnp.broadcast_to(first[:, None, :], (4, 4, 2))
    starts_b = jnp.broadcast_to(second[None, :, :], (4, 4, 2))
    directions_a = jnp.broadcast_to(first_delta[:, None, :], (4, 4, 2))
    directions_b = jnp.broadcast_to(second_delta[None, :, :], (4, 4, 2))
    offset = starts_b - starts_a
    denominator = _cross_2d(directions_a, directions_b)
    safe_denominator = jnp.where(jnp.abs(denominator) > 1e-07, denominator, 1.0)
    t = _cross_2d(offset, directions_b) / safe_denominator
    u = _cross_2d(offset, directions_a) / safe_denominator
    valid = (jnp.abs(denominator) > 1e-07) & (t >= -1e-06) & (t <= 1.0 + 1e-06) & (u >= -1e-06) & (u <= 1.0 + 1e-06)
    points = starts_a + t[..., None] * directions_a
    return (points.reshape(16, 2), valid.reshape(16))


def convex_intersection_area(first: Array, second: Array) -> Array:
    (edge_points, edge_valid) = _edge_intersections(first, second)
    candidates = jnp.concatenate((first, second, edge_points), axis=0)
    valid = jnp.concatenate(
        (_points_in_convex_polygon(first, second), _points_in_convex_polygon(second, first), edge_valid)
    )
    count = jnp.sum(valid.astype(jnp.int32))
    safe_count = jnp.maximum(count, 1)
    centroid = jnp.sum(jnp.where(valid[:, None], candidates, 0.0), axis=0) / safe_count
    angles = jnp.arctan2(candidates[:, 1] - centroid[1], candidates[:, 0] - centroid[0])
    angles = jnp.where(valid, angles, jnp.inf)
    order = jnp.argsort(angles)
    ordered = candidates[order]
    indices = jnp.arange(candidates.shape[0], dtype=jnp.int32)
    next_indices = jnp.mod(indices + 1, safe_count)
    cross_terms = _cross_2d(ordered, ordered[next_indices])
    twice_area = jnp.sum(jnp.where(indices < count, cross_terms, 0.0))
    return jnp.where(count >= 3, 0.5 * jnp.abs(twice_area), 0.0)


def coverage(block_position: Array, block_angle: Array, goal_position: Array, goal_angle: Array) -> Array:
    block_rectangles = tee_rectangles(block_position, block_angle)
    goal_rectangles = tee_rectangles(goal_position, goal_angle)

    def rectangle_pair(index: Array) -> Array:
        block_index = index // 2
        goal_index = index % 2
        return convex_intersection_area(block_rectangles[block_index], goal_rectangles[goal_index])

    intersection = jnp.sum(jax.vmap(rectangle_pair)(jnp.arange(4)))
    return jnp.clip(intersection / TEE_AREA, 0.0, 1.0)


def point_in_tee(local_points: Array) -> Array:
    x = local_points[..., 0]
    y = local_points[..., 1]
    in_bar = (x >= -60.0) & (x <= 60.0) & (y >= 0.0) & (y <= 30.0)
    in_stem = (x >= -15.0) & (x <= 15.0) & (y >= 30.0) & (y <= 120.0)
    return in_bar | in_stem
