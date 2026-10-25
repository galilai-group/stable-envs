"""Audited source Curve/FONT cases; the pixel visibility versions stay unchanged."""

from .object_permanence_orthographic import load_orthographic_geometry
from .object_permanence_recordings import load_ray_geometry

CONVERTED_TASKS = ("g04", "g13", "g14", "g29", "g31", "g46")
FONT_TASKS = ("g13", "g46")


def load_converted_geometry(task):
    if task not in CONVERTED_TASKS:
        raise ValueError("unknown converted geometry case")
    return (load_orthographic_geometry if task in ("g13", "g14", "g46") else load_ray_geometry)(task)
