"""The OGBench task-mode info of Cube/Scene, computed from State with the arithmetic of ``compute_ob_info``."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ...._core.state_schema import get_field


class InfoLayout:
    """The model indices that ``ManipSpaceEnv.compute_ob_info`` reads, taken from one OGBench object."""

    def __init__(self, env):
        model = env.model
        self.scene = env.family == "Scene"
        self.arm = np.asarray(env._arm_joint_ids)
        self.gripper = int(env._gripper_opening_joint_id)
        self.cubes = [int(model.joint(f"object_joint_{i}").qposadr[0]) for i in range(env._num_cubes)]
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

    def get(name):
        return get_field(state, name).detach().cpu().numpy()

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
    return TensorDict(
        {key: torch.from_numpy(np.array(value, order="C")) for key, value in info.items()},
        batch_size=[n],
    ).to(device)
