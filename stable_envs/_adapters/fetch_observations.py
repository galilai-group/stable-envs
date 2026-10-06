import numpy as np
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv
from gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
from gymnasium_robotics.envs.fetch.slide import MujocoFetchSlideEnv
from gymnasium_robotics.utils import rotations


class _CachedObservations:
    """Cache topology and Jacobian workspace per simulator instance.

    Read positions, velocities and timestep afresh. Replacing the model invalidates
    topology; the upstream observation builder assembles the returned fields."""
    def generate_mujoco_observations(self):
        model, data = self.model, self.data
        if getattr(self, "_observation_model", None) is not model:
            # Upstream robot_get_obs filters in model order and the caller uses
            # only its final two scalar joints. Derive those addresses once.
            names = [n for n in self._model_names.joint_names if n.startswith("robot")][-2:]
            joints = [self._model_names.joint_name2id[n] for n in names]
            if len(joints) != 2 or any(int(model.jnt_type[j]) not in (2, 3) for j in joints):
                raise ValueError("Fetch observation cache requires two scalar gripper joints")
            self._finger_qpos = model.jnt_qposadr[joints].copy()
            self._finger_qvel = model.jnt_dofadr[joints].copy()
            self._grip_site = self._model_names.site_name2id["robot0:grip"]
            self._object_site = self._model_names.site_name2id["object0"] if self.has_object else None
            self._grip_jac = np.zeros((3, model.nv))
            self._object_jacp = np.zeros((3, model.nv))
            self._object_jacr = np.zeros((3, model.nv))
            self._observation_model = model

        # Read mutable model parameters and data every call. Reset/set_state and
        # mass/timestep variations must never leave cached simulation values.
        dt = self.n_substeps * model.opt.timestep
        grip_pos = data.site_xpos[self._grip_site]
        self._grip_jac.fill(0)
        self._mujoco.mj_jacSite(model, data, self._grip_jac, None, self._grip_site)
        grip_velp = (self._grip_jac @ data.qvel) * dt
        gripper_state = data.qpos[self._finger_qpos]
        gripper_vel = data.qvel[self._finger_qvel] * dt

        if self.has_object:
            object_pos = data.site_xpos[self._object_site]
            object_rot = rotations.mat2euler(data.site_xmat[self._object_site].reshape(3, 3))
            self._object_jacp.fill(0)
            self._object_jacr.fill(0)
            self._mujoco.mj_jacSite(model, data, self._object_jacp, self._object_jacr, self._object_site)
            object_velp = (self._object_jacp @ data.qvel) * dt
            object_velr = (self._object_jacr @ data.qvel) * dt
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        return (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        )


class FetchPickAndPlace(_CachedObservations, MujocoFetchPickAndPlaceEnv):
    pass


class FetchReach(_CachedObservations, MujocoFetchReachEnv):
    pass


class FetchPush(_CachedObservations, MujocoFetchPushEnv):
    pass


class FetchSlide(_CachedObservations, MujocoFetchSlideEnv):
    pass


TASK_CLASSES = {"PickAndPlace": FetchPickAndPlace, "Reach": FetchReach, "Push": FetchPush, "Slide": FetchSlide}
