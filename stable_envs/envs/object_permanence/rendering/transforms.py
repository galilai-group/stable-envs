"""Shared rigid transformations for sampled scene rendering."""

import torch


def row_matrix3(vectors, matrices):
    """Multiply (..., 3) rows by (..., 3, 3) matrices in a fixed order.

    Leading dimensions broadcast. Three explicit products/additions avoid
    batch-size-dependent GEMM rounding when transforming visibility planes;
    exact static-cache validation must see the same bits for the same inputs.
    """
    return (
        vectors[..., 0:1] * matrices[..., 0, :]
        + vectors[..., 1:2] * matrices[..., 1, :]
        + vectors[..., 2:3] * matrices[..., 2, :]
    )


def rotations(quaternions):
    """Convert wxyz quaternions to matrices on their device, without an engine."""
    q = quaternions / torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(-1, 3, 3)
