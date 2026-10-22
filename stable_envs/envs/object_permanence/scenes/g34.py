"""MuJoCo model and authored trajectory for the nominal Blender G34 scene.

Source: hokindeng/object-permanence at 824baddcdad087810d35638bf8f273b884f74975,
G34_occ_small_ball_behind_screen, build/animate_occ_small_ball_behind_screen.
This port implements the mathematical motion independently; it imports no Blender
code and does not load the frozen test trajectory. See the alignment contract.
"""

from __future__ import annotations

from importlib.resources import files

import numpy as np

FPS = 24
FRAME_COUNT = 120
SUBSTEPS = 4
OBJECT_NAMES = (
    "large_floor_base",
    "rear_backdrop_panel",
    "straight_gray_track",
    "wide_opaque_screen",
    "small_orange_ball",
)


def make_model():
    """Compile the packaged scene, independently for simulation and rendering."""
    import mujoco

    return mujoco.MjModel.from_xml_string(
        files(__package__.rsplit(".", 1)[0]).joinpath("assets/g34/scene.xml").read_text()
    )


def authored_pose(elapsed_steps):
    """Return source-frame ``elapsed_steps + 1`` position and wxyz quaternion.

    A frame is 1/24 s. The source advances from x=-2.9 to 2.9 in 119
    intervals and rotates about world Y by -0.2 radians per source frame.
    """
    steps = np.asarray(elapsed_steps, dtype=np.float64)
    position = np.stack((-2.9 + 5.8 * steps / 119.0, np.zeros_like(steps), np.full_like(steps, 0.194)), -1)
    half_angle = -0.1 * (steps + 1)
    quaternion = np.stack((np.cos(half_angle), np.zeros_like(steps), np.sin(half_angle), np.zeros_like(steps)), -1)
    return position, quaternion


def set_pose(model, data, position, quaternion):
    """Install a mocap pose and recompute derived geometry without advancing time."""
    import mujoco

    data.mocap_pos[0] = position
    data.mocap_quat[0] = quaternion
    mujoco.mj_forward(model, data)
