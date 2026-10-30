"""Geometry fidelity metrics; tolerances belong to the calling acceptance gate."""

from __future__ import annotations

import numpy as np


def mask_iou(actual, reference):
    """Return visible-mask IoU per image; two empty masks return NaN (not a pass).

    Inputs are boolean arrays shaped (..., H, W). Empty pairs must be counted
    separately so long occlusions cannot inflate visible-frame agreement.
    """
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape or a.ndim < 2 or a.dtype != np.bool_ or b.dtype != np.bool_:
        raise ValueError("Expected equal boolean masks shaped (..., H, W)")
    intersection = (a & b).sum(axis=(-2, -1))
    union = (a | b).sum(axis=(-2, -1))
    return np.divide(intersection, union, out=np.full(np.shape(union), np.nan), where=union != 0)


def depth_mse(actual, reference, valid):
    """Return MSE per image over a fixed reference-valid ROI, in squared scene units.

    Depths are finite nonnegative float arrays shaped (..., H, W), with zero
    reserved for no hit. Missing candidate surfaces in the ROI give infinity;
    empty ROIs give NaN. No frame-wise normalization or RGB conversion is used.
    Extra candidate surfaces are checked separately by object-mask IoU.
    """
    a, b, mask = np.asarray(actual), np.asarray(reference), np.asarray(valid)
    if (
        a.shape != b.shape
        or a.shape != mask.shape
        or a.ndim < 2
        or a.dtype.kind != "f"
        or b.dtype.kind != "f"
        or mask.dtype != np.bool_
    ):
        raise ValueError("Expected equal float depths and a boolean ROI shaped (..., H, W)")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0):
        raise ValueError("Depths must be finite and nonnegative")
    if np.any(mask & (b <= 0)):
        raise ValueError("ROI contains invalid reference depth")
    count = mask.sum(axis=(-2, -1))
    squared = np.where(mask, (a.astype(np.float64) - b.astype(np.float64)) ** 2, 0).sum(axis=(-2, -1))
    mse = np.divide(squared, count, out=np.full(np.shape(count), np.nan), where=count != 0)
    return np.where((mask & (a <= 0)).any(axis=(-2, -1)), np.inf, mse)
