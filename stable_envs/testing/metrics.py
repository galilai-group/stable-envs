from __future__ import annotations

import numpy as np


def normalized_mse(ours, source) -> np.ndarray:
    """Return RGB MSE divided by 255 squared for each leading frame index.

    Inputs are equal-shaped uint8 ``(..., H, W, 3)`` arrays.
    """
    a, b = np.asarray(ours), np.asarray(source)
    if a.dtype != np.uint8 or b.dtype != np.uint8:
        raise TypeError("Pixel comparisons require uint8; implicit conversion is forbidden")
    if a.shape != b.shape or a.ndim < 3 or a.shape[-1] != 3 or min(a.shape[-3:]) == 0:
        raise ValueError(f"Expected equal RGB shapes, got {a.shape} and {b.shape}")
    shape = a.shape[:-3]
    count = int(np.prod(shape)) if shape else 1
    errors = np.empty(count, dtype=np.float64)
    for i, index in enumerate(np.ndindex(shape)):
        delta = a[index].astype(np.int64) - b[index].astype(np.int64)
        errors[i] = np.sum(delta * delta, dtype=np.int64) / (delta.size * 255**2)
    return errors.reshape(shape)


def summarize_errors(errors, *, ddof=0, threshold=None) -> dict:
    """Summarize finite nonnegative errors with population (ddof=0) or sample SD (ddof=1).

    Require more observations than ddof. Optional threshold counts use strict >.
    """
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if ddof not in (0, 1) or values.size <= ddof:
        raise ValueError("Need more observations than ddof (0 or 1)")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Errors must be finite and nonnegative")
    result = dict(
        count=int(values.size),
        mean=float(values.mean()),
        sd=float(values.std(ddof=ddof)),
        ddof=ddof,
        maximum=float(values.max()),
        worst_index=int(values.argmax()),
        quantiles={str(q): float(np.quantile(values, q)) for q in (0.5, 0.95, 0.99)},
    )
    if threshold is not None:
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Threshold must be finite and nonnegative")
        result.update(threshold=float(threshold), over_threshold=int((values > threshold).sum()))
    return result


def leaf_difference(actual, expected) -> dict:
    """Compare leaf shape, dtype and values, reporting the first mismatch and the maximum error.

    NaN pairs compare equal, and unequal non-finite values have infinite error.
    """

    def host(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    a, b = host(actual), host(expected)
    result = dict(
        actual_shape=list(a.shape),
        expected_shape=list(b.shape),
        actual_dtype=str(a.dtype),
        expected_dtype=str(b.dtype),
    )
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(result, comparable=False, exact=False, max_abs_error=None, first_mismatch=None)
    if a.dtype.kind not in "buif":
        raise TypeError(f"Unsupported State dtype: {a.dtype}")
    equal = a == b
    if a.dtype.kind == "f":
        equal |= np.isnan(a) & np.isnan(b)
    bad = np.flatnonzero(~equal)
    maximum = 0
    if bad.size:
        left, right = a.reshape(-1)[bad], b.reshape(-1)[bad]
        if a.dtype.kind in "bui":
            maximum = max(abs(int(x) - int(y)) for x, y in zip(left, right, strict=True))
        else:
            with np.errstate(invalid="ignore", over="ignore"):
                delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            maximum = float(delta.max()) if np.isfinite(delta).all() else float("inf")
    return dict(
        result,
        comparable=True,
        exact=not bool(bad.size),
        mismatched=int(bad.size),
        max_abs_error=maximum,
        first_mismatch=[int(i) for i in np.unravel_index(int(bad[0]), a.shape)] if bad.size else None,
    )
