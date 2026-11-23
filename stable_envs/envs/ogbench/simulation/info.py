"""The OGBench info of Cube/Scene, computed from State with the arithmetic of ``compute_ob_info``."""

from __future__ import annotations

import numpy as np
from tensordict import TensorDict

from ...._core.state_schema import get_field
from ...._core.tensordicts import host_arrays_tensordict


class InfoLayout:
    """The model indices that ``ManipSpaceEnv.compute_ob_info`` reads, taken from one OGBench object."""

    def __init__(self, env):
        model = env.model
        self.scene = env.family == "Scene"
        self.arm = np.asarray(env._arm_joint_ids)
        self.gripper = int(env._gripper_opening_joint_id)
        self.cubes = [int(model.joint(f"object_joint_{i}").qposadr[0]) for i in range(env._num_cubes)]
        self.target_mocaps = np.asarray(env._cube_target_mocap_ids)
        self.buttons, self.slides = [], {}
        if self.scene:
            self.buttons = [
                (int(model.joint(name).qposadr[0]), int(model.joint(name).dofadr[0]))
                for name in (f"buttonbox_joint_{i}" for i in range(env._num_buttons))
            ]
            for name in ("drawer", "window"):
                joint = model.joint(f"{name}_slide")
                self.slides[name] = (int(joint.qposadr[0]), int(joint.dofadr[0]))


def _yaw_of_matrix(rows):
    from ogbench.manipspace import lie

    return np.array([[lie.SO3.from_matrix(row.reshape(3, 3)).compute_yaw_radians()] for row in rows])


def _yaw_of_quaternion(rows):
    from ogbench.manipspace import lie

    return np.array([[lie.SO3(wxyz=row).compute_yaw_radians()] for row in rows])


def manipulation_info(layout: InfoLayout, state, device) -> TensorDict:
    """``compute_ob_info`` of each batch row of ``state``, then ``success`` and the reset's ``goal``.

    The State holds the site poses and the gripper pad force that the upstream step left in ``MjData``, so no
    MuJoCo call runs. Yaw angles and the contact norm are computed one row at a time, with the upstream calls.
    """
    host = state.cpu()

    def get(name):
        return get_field(host, name).numpy()

    qpos, qvel = get("physics/qpos"), get("physics/qvel")
    site_xpos, site_xmat = get("controller/site_xpos"), get("controller/site_xmat")
    n = qpos.shape[0]
    site_xpos, site_xmat = site_xpos.reshape(n, -1, 3), site_xmat.reshape(n, -1, 9)
    force = get("output/gripper_force").reshape(n, 6)
    g = layout.gripper
    info = {
        "proprio/joint_pos": qpos[:, layout.arm],
        "proprio/joint_vel": qvel[:, layout.arm],
        "proprio/effector_pos": site_xpos[:, 0],
        "proprio/effector_yaw": _yaw_of_matrix(site_xmat[:, 0]),
        "proprio/gripper_opening": np.clip(qpos[:, [g]] / 0.8, 0, 1),
        "proprio/gripper_vel": qvel[:, [g]],
        "proprio/gripper_contact": np.array([[np.clip(np.linalg.norm(row) / 50, 0, 1)] for row in force]),
    }
    for i, adr in enumerate(layout.cubes):
        info[f"privileged/block_{i}_pos"] = qpos[:, adr : adr + 3]
        info[f"privileged/block_{i}_quat"] = qpos[:, adr + 3 : adr + 7]
        info[f"privileged/block_{i}_yaw"] = _yaw_of_quaternion(qpos[:, adr + 3 : adr + 7])
    collecting = bool(get("task/mode").any())
    if collecting:
        # The data-collection keys; rows in task mode give the values of their target fields too.
        rows = np.arange(n)
        block = get("task/target/block").reshape(n)
        mocap = layout.target_mocaps[block]
        target = {
            "privileged/target_task": get("task/target/task").reshape(n),
            "privileged/target_block": block,
            "privileged/target_block_pos": get("physics/mocap_pos").reshape(n, -1, 3)[rows, mocap],
            "privileged/target_block_yaw": _yaw_of_quaternion(
                get("physics/mocap_quat").reshape(n, -1, 4)[rows, mocap]
            ),
        }
        if not layout.scene:
            info.update(target)
    if layout.scene:
        buttons = get("scene/cur_button_states").reshape(n, -1)
        for i, (adr, dof) in enumerate(layout.buttons):
            info[f"privileged/button_{i}_state"] = buttons[:, i]
            info[f"privileged/button_{i}_pos"] = qpos[:, adr : adr + 1]
            info[f"privileged/button_{i}_vel"] = qvel[:, dof : dof + 1]
        # controller/site_xpos holds the pinch, drawer and window sites, then the two targets; site_xmat the first three.
        for k, name in enumerate(("drawer", "window"), start=1):
            adr, dof = layout.slides[name]
            info[f"privileged/{name}_pos"] = qpos[:, adr : adr + 1]
            info[f"privileged/{name}_vel"] = qvel[:, dof : dof + 1]
            info[f"privileged/{name}_handle_pos"] = site_xpos[:, k]
            info[f"privileged/{name}_handle_yaw"] = _yaw_of_matrix(site_xmat[:, k])
        if collecting:
            button = get("task/target/button").reshape(n)
            info.update(target)
            info.update(
                {
                    "privileged/target_button": button,
                    "privileged/target_button_state": get("scene/target_button_states").reshape(n, -1)[rows, button],
                    # site_xpos rows 3 and 4 are the drawer and window targets, then the button tops.
                    "privileged/target_button_top_pos": site_xpos[rows, 5 + button],
                    "privileged/target_drawer_pos": get("scene/target_drawer_pos").reshape(n, 1),
                    "privileged/target_drawer_handle_pos": site_xpos[:, 3],
                    "privileged/target_window_pos": get("scene/target_window_pos").reshape(n, 1),
                    "privileged/target_window_handle_pos": site_xpos[:, 4],
                }
            )
        info["prev_button_states"] = get("scene/prev_button_states").reshape(n, -1)
        info["button_states"] = buttons
    info.update(
        {
            "prev_qpos": get("output/prev_qpos"),
            "prev_qvel": get("output/prev_qvel"),
            "qpos": qpos,
            "qvel": qvel,
            "control": get("physics/ctrl"),
            "time": get("physics/time").reshape(n, 1),
            "success": get("task/success").reshape(n),
            "goal": get("task/goal_observation"),
        }
    )
    return host_arrays_tensordict(info, n, device)
