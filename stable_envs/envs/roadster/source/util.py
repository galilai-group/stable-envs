"""Small helpers shared by several modules: XML number formatting, fonts and rotation maths."""

from __future__ import annotations

import numpy as np
from PIL import ImageFont


def vec(a) -> str:
    """Format a sequence of numbers as a MuJoCo XML attribute value."""
    return " ".join(f"{float(x):.5g}" for x in a)


def font(size: int):
    """Pillow's default font at `size` (falls back to the fixed-size font on Pillow < 10.1)."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def rot_err(R: np.ndarray, Rd: np.ndarray) -> np.ndarray:
    """Body-frame axis-angle of Rd^T R (log map, defined up to a 180 deg flip)."""
    E = Rd.T @ R
    ang = np.arccos(np.clip((np.trace(E) - 1.0) / 2.0, -1.0, 1.0))
    if ang < 1e-6:
        return np.zeros(3)
    if ang > np.pi - 1e-3:  # near-inverted: axis from the symmetric part
        M = E + np.eye(3)
        axis = M[:, np.argmax(np.linalg.norm(M, axis=0))]
    else:
        axis = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    return axis / np.linalg.norm(axis) * ang


def rz(psi: float) -> np.ndarray:
    """Rotation matrix about z."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rx(phi: float) -> np.ndarray:
    """Rotation matrix about x."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
