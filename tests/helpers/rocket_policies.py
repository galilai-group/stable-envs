"""Scripted Rocket policies for kernel fidelity tests; the landing policies read the env state closed loop."""

from __future__ import annotations

import numpy as np

OPTIONS = {
    "free": {"variation": ["environment"]},
    "burn": {"variation": ["environment"]},
    "smooth": {"variation": ["environment"]},
    "land": None,
    "land_off": {"variation": ["environment.start_horizontal_offset"]},
    "firm": {"variation": ["environment"]},
    "pad": None,
    "hover": {"variation": ["environment"]},
    "climb": {"variation": ["environment"]},
}


def policy(kind, t, env):
    """Scripted actions; the landing policies read ``env`` (the source or a kernel state view) closed loop."""
    if kind == "free":
        return np.zeros(7)
    if kind == "smooth":
        ign = 1.0 if (80 <= t < 140 or 200 <= t < 260) else 0.0
        return np.array(
            [
                0.5 * np.sin(0.05 * t),
                0.5 * np.sin(0.07 * t + 1),
                0.3 * np.sin(0.03 * t),
                ign,
                0.5 + 0.5 * np.sin(0.1 * t),
                0.5 * np.sin(0.2 * t),
                0.5 * np.cos(0.15 * t),
            ]
        )
    z, vz = env.lin_pos[2], env.ground_lin_vel[2]
    roll, pitch = env.ang_pos[0], env.ang_pos[1]
    wx, wy = env.ang_vel[0], env.ang_vel[1]
    h = z - 2.575
    if kind in ("land", "land_off", "firm", "pad"):
        # firm touchdowns (4.5 m/s) crash or touch without landing; then the episode goes on with contact
        vt = -max(4.5 if kind in ("firm", "pad") else 1.0, np.sqrt(2 * 6.0 * max(h, 0.0)))
        armed = h < 1.15 * vz * vz / (2 * 30.0) + 3.0 and vz < -5.0
        ign = 1.0 if (armed or (env.ignition_state and vz < vt)) and vz < vt else 0.0
        g1 = float(np.clip(3.0 * roll + 1.0 * wx + 0.05 * np.sin(0.3 * t), -1, 1))
        g2 = float(np.clip(3.0 * pitch + 1.0 * wy + 0.05 * np.cos(0.2 * t), -1, 1))
        return np.array([0.0, 0.0, 0.0, ign, 1.0, g1, g2])
    if kind == "hover":
        # bang-bang altitude hold at minimum throttle until the 30 s limit truncates the episode
        ign = 1.0 if vz < -0.5 * np.sin(0.02 * t) else 0.0
        g1 = float(np.clip(3.0 * roll + 1.0 * wx, -1, 1))
        g2 = float(np.clip(3.0 * pitch + 1.0 * wy, -1, 1))
        return np.array([0.1 * np.sin(0.03 * t), 0.1 * np.cos(0.02 * t), 0.0, ign, 0.0, g1, g2])
    if kind == "climb":
        # full thrust up through the 500 m ceiling
        g1 = float(np.clip(3.0 * roll + 1.0 * wx, -1, 1))
        g2 = float(np.clip(3.0 * pitch + 1.0 * wy, -1, 1))
        return np.array([0.0, 0.0, 0.0, 1.0, 1.0, g1, g2])
    # suicide burn with attitude hold
    ign = 1.0 if (vz < -3.0 and h < 1.2 * vz * vz / (2 * 36.0) + 8.0) else 0.0
    g1 = float(np.clip(3.0 * roll + 1.0 * wx, -1, 1))
    g2 = float(np.clip(3.0 * pitch + 1.0 * wy, -1, 1))
    return np.array([0.1 * np.sin(0.04 * t), 0.0, 0.0, ign, 1.0, g1, g2])
