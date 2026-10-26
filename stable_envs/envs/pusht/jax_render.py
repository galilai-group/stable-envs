"""PushT rasterizer adapted from fracapuano/envX at d26c817c1124229c819be574f137cd68bda4380a.

Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
See licenses/envX-APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.
"""

import jax.numpy as jnp

from .simulation.jax_geometry import inverse_transform_points, point_in_tee


def render(row, *, size, with_target, render_action):
    rows, columns = jnp.indices((size, size), dtype=jnp.float32)
    x, y = (columns + 0.5) * 512 / size, 512 - (rows + 0.5) * 512 / size
    points = jnp.stack((x, y), axis=-1)
    image = jnp.full((size, size, 3), 255, dtype=jnp.uint8)

    def paint(mask, color, image):
        return jnp.where(mask[..., None], jnp.asarray(color, dtype=jnp.uint8), image)

    if with_target:
        image = paint(point_in_tee(inverse_transform_points(points, row[5:7], row[7])), (144, 238, 144), image)
    image = paint((x <= 7) | (x >= 504) | (y <= 7) | (y >= 504), (211, 211, 211), image)
    image = paint(jnp.sum((points - row[:2]) ** 2, axis=-1) <= 225, (65, 105, 225), image)
    image = paint(point_in_tee(inverse_transform_points(points, row[2:4], row[4])), (119, 136, 153), image)
    if render_action:
        delta = jnp.abs(points - row[8:10]) * size / 512
        image = paint(
            ((delta[..., 1] <= 1) & (delta[..., 0] <= 4)) | ((delta[..., 0] <= 1) & (delta[..., 1] <= 4)),
            (255, 0, 0),
            image,
        )
    return image
