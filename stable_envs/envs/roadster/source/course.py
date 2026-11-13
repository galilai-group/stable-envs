"""Course layout: run-up road -> half-pipe loop -> flight rings -> elevated landing pad."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Ring:
    center: tuple[float, float, float]  # ring lies in the y-z plane, car flies through along +x
    radius: float = 4.0


@dataclass
class Course:
    start_x: float = -130.0
    road_width: float = 8.0
    # Half-pipe loop: a 180 deg circular arc in the x-z plane, tangent to the road at x=0.
    # The car runs up the inside and leaves the top inverted, heading back toward -x.
    loop_radius: float = 10.0
    loop_arc_deg: float = 180.0
    loop_segments: int = 48
    ramp_width: float = 6.0
    rings: list[Ring] = field(
        default_factory=lambda: [
            Ring((60.0, 0.0, 20.0)),
            Ring((110.0, 7.0, 17.0)),
            Ring((160.0, -5.0, 12.0)),
        ]
    )
    pad_center: tuple[float, float] = (205.0, 0.0)
    pad_size: tuple[float, float] = (20.0, 15.0)  # half-extents x, y
    pad_height: float = 4.0  # top surface z

    def ramp_polyline(self) -> np.ndarray:
        """(N+1, 2) array of (x, z) vertices along the loop's riding surface."""
        phi = np.deg2rad(np.linspace(0.0, self.loop_arc_deg, self.loop_segments + 1))
        R = self.loop_radius
        return np.stack([R * np.sin(phi), R * (1.0 - np.cos(phi))], axis=1)

    @property
    def loop_top(self) -> float:
        return 2.0 * self.loop_radius

    @property
    def pad_top(self) -> np.ndarray:
        return np.array([self.pad_center[0], self.pad_center[1], self.pad_height])

    def targets(self) -> list[np.ndarray]:
        """Waypoints the pilot must hit in order: ring centers, then a hover point above the pad."""
        pts = [np.asarray(r.center, dtype=float) for r in self.rings]
        pts.append(self.pad_top + np.array([0.0, 0.0, 1.0]))
        return pts
