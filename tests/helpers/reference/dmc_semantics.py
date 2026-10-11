"""Independent pinned-SWM State/task oracle for all twelve DMC defaults.

Capture source fields directly; never borrow the candidate's serializer or schema.
The small model-field map is the only family-specific State handling.
"""

import importlib

import numpy as np
import torch
from tensordict import TensorDict

WRAPPERS = {
    "acrobot": "Acrobot",
    "pendulum": "Pendulum",
    "cartpole": "Cartpole",
    "reacher": "Reacher",
    "cheetah": "Cheetah",
    "hopper": "Hopper",
    "walker": "Walker",
    "ball_in_cup": "BallInCup",
    "finger": "Finger",
    "manipulator": "Manipulator",
    "quadruped": "Quadruped",
    "humanoid": "Humanoid",
}
EXTRA_INFO = {
    "cheetah": ("speed", "torso_height", "bfoot_height", "ffoot_height", "angmomentum"),
    "hopper": ("speed", "height", "angmomentum"),
    "walker": (
        "speed",
        "torso_height",
        "torso_upright",
        "left_foot_height",
        "right_foot_height",
        "left_thigh_height",
        "right_thigh_height",
    ),
    "ball_in_cup": (),
    "finger": ("hinge_velocity", "target_position", "tip_position"),
    "manipulator": (),
    "quadruped": ("torso_velocity",),
    "humanoid": ("speed", "torso_upright", "head_height"),
}
# (State path, named source model field, source indexing)
MODEL_FIELDS = {
    "reacher": ((("task", "target_pos"), "geom_pos", "target"),),
    "finger": ((("model", "target_site_xz"), "site_pos", ("target", ["x", "z"])),),
    "manipulator": (
        (("model", "target_body_xz"), "body_pos", ("target_ball", ["x", "z"])),
        (("model", "target_body_qwqy"), "body_quat", ("target_ball", ["qw", "qy"])),
    ),
}
ALIGNED = {"ball_in_cup", "finger", "manipulator", "quadruped"}
_GRAVITY = ("gravity.x", "gravity.y", "gravity.z")
# Physics leaves stable-envs State carries under ("physics", *leaf), with how SWM holds each one:
# gravity and friction are read from SWM's compiled model, masses from the values SWM recompiled with.
PHYSICS_LEAVES = {
    "acrobot": _GRAVITY,
    "pendulum": _GRAVITY,
    "cartpole": (*_GRAVITY, "agent.cart_mass"),
    "ball_in_cup": (*_GRAVITY, "agent.density", "ball.density"),
    "finger": ("agent.proximal_density", "agent.fingertip_density", "spinner.density", "spinner.friction"),
    "manipulator": ("agent.upper_arm_density", "agent.hand_density"),
    "reacher": ("agent.arm_density", "agent.finger_density"),
    **{
        domain: (*_GRAVITY, "agent.torso_density", f"agent.{foot}_density", "floor.friction")
        for domain, foot in (
            ("cheetah", "back_foot"),
            ("hopper", "foot"),
            ("walker", "left_foot"),
            ("humanoid", "right_lower_arm"),
            ("quadruped", "foot_back_left"),
        )
    },
}
_FLOOR = {"cheetah": "ground"}


def swm_physics(wrapper, domain, leaf):
    """The value SWM actually ran for one physics leaf."""
    physics = wrapper.env.physics
    if leaf.startswith("gravity."):
        return float(physics.model.opt.gravity["xyz".index(leaf[-1])])
    if leaf == "floor.friction":
        return float(physics.named.model.geom_friction[_FLOOR.get(domain, "floor"), 0])
    if leaf == "spinner.friction":
        return float(np.asarray(physics.named.model.dof_frictionloss["hinge"]).reshape(-1)[0])
    group, name = leaf.split(".")
    return float(np.asarray(wrapper.variation_space[group][name].value).reshape(-1)[0])


class DMCReference:
    def __init__(self, domain, num_envs=1, options=None, wrapper=None):
        cls = wrapper or getattr(
            importlib.import_module(f"stable_worldmodel.envs.dmcontrol.{domain}"),
            WRAPPERS[domain] + "DMControlWrapper",
        )
        self.domain, self.num_envs, self.options = domain, num_envs, options
        self.envs = []
        try:
            for lane in range(num_envs):
                self.envs.append(cls(seed=lane))
        except BaseException:
            self.close()
            raise

    def reset(self, seed=None):
        for lane, wrapper in enumerate(self.envs):
            wrapper.reset(seed=None if seed is None else seed + lane, options=self.options)
        return self.get_state()

    def get_state(self):
        rows = []
        for wrapper in self.envs:
            env, physics = wrapper.env._env, wrapper.env.physics
            algorithm, keys, position, has_gauss, cached = env.task.random.get_state()
            assert algorithm == "MT19937"
            row = TensorDict(
                {
                    "qpos": physics.data.qpos.copy(),
                    "qvel": physics.data.qvel.copy(),
                    "time": np.array([physics.data.time], np.float64),
                    "warmstart": physics.data.qacc_warmstart.copy(),
                    "task": {"cumulative_reward": np.array([wrapper._cumulative_reward], np.float64)},
                    "lifecycle": {
                        "step_count": np.array([env._step_count], np.int64),
                        "reset_next_step": np.array([env._reset_next_step], bool),
                    },
                    "rng": {
                        "keys": keys.astype(np.int64),
                        "position": np.array([position], np.int64),
                        "has_gauss": np.array([has_gauss], bool),
                        "cached_gaussian": np.array([cached], np.float64),
                    },
                },
                [],
            ).clone()
            for key, field, index in MODEL_FIELDS.get(self.domain, ()):
                row[key] = torch.tensor(getattr(physics.named.model, field)[index].copy())
            if self.domain == "quadruped":
                row["activation"] = torch.tensor(physics.data.act.copy())
            for leaf in PHYSICS_LEAVES.get(self.domain, ()):
                value = swm_physics(wrapper, self.domain, leaf)
                row[("physics", *leaf.split("."))] = torch.tensor([value], dtype=torch.float64)
            rows.append(row)
        return torch.stack(rows)

    def set_state(self, state):
        for wrapper, row in zip(self.envs, state.cpu().unbind(0), strict=True):
            env, physics = wrapper.env._env, wrapper.env.physics
            for leaf in PHYSICS_LEAVES.get(self.domain, ()):
                value = float(row[("physics", *leaf.split("."))])
                if leaf.startswith("gravity."):
                    gravity = physics.model.opt.gravity.copy()
                    gravity["xyz".index(leaf[-1])] = value
                    wrapper.set_gravity(gravity)
                elif value != swm_physics(wrapper, self.domain, leaf):
                    raise NotImplementedError(f"SWM changes {leaf} by recompiling; reset it through options instead")
            for key, field, index in MODEL_FIELDS.get(self.domain, ()):
                getattr(physics.named.model, field)[index] = row[key].numpy()
            if self.domain in ALIGNED:
                physics.data.qpos[:] = row["qpos"].numpy()
                physics.data.qvel[:] = row["qvel"].numpy()
                physics.data.time = row["time"].item()
                if self.domain == "quadruped":
                    physics.data.act[:] = row["activation"].numpy()
                physics.forward()
            else:
                wrapper.set_state(row["qpos"].numpy(), row["qvel"].numpy())
                physics.data.time = row["time"].item()
            physics.data.qacc_warmstart[:] = row["warmstart"].numpy()
            env._step_count = row["lifecycle", "step_count"].item()
            env._reset_next_step = row["lifecycle", "reset_next_step"].item()
            wrapper._cumulative_reward = row["task", "cumulative_reward"].item()
            env.task.random.set_state(
                (
                    "MT19937",
                    row["rng", "keys"].numpy().astype(np.uint32),
                    row["rng", "position"].item(),
                    int(row["rng", "has_gauss"].item()),
                    row["rng", "cached_gaussian"].item(),
                )
            )

    def step(self, actions):
        outputs = [env.step(np.asarray(action, np.float64)) for env, action in zip(self.envs, actions, strict=True)]
        infos = [row[4] for row in outputs]
        result = TensorDict(
            {
                "state": self.get_state(),
                "reward": torch.tensor([row[1] for row in outputs], dtype=torch.float64),
                "terminated": torch.tensor([row[2] for row in outputs], dtype=torch.bool),
                "truncated": torch.tensor([row[3] for row in outputs], dtype=torch.bool),
                "observation": torch.as_tensor(np.stack([row[0] for row in outputs])),
                "score": torch.tensor([row["score"] for row in infos], dtype=torch.float64),
            },
            [self.num_envs],
        )
        if self.domain in EXTRA_INFO:
            result["success"] = torch.tensor([row["success"] for row in infos], dtype=torch.float64)
            result["info"] = TensorDict(
                {
                    key: torch.as_tensor(np.stack([row[key] for row in infos]))
                    for key in ("qpos", "qvel", *EXTRA_INFO[self.domain])
                },
                [self.num_envs],
            )
        return result

    @property
    def step_limit(self):
        limits = {int(env.env._env._step_limit) for env in self.envs}
        assert len(limits) == 1
        return limits.pop()

    @property
    def action_dim(self):
        return self.envs[0].action_space.shape[0]

    def warmstarts(self):
        return np.stack([env.env.physics.data.qacc_warmstart.copy() for env in self.envs])

    def controls(self):
        return np.stack([env.env.physics.data.ctrl.copy() for env in self.envs])

    def task_caches(self):
        return [{k: v for k, v in env.env.task.__dict__.items() if k != "_random"} for env in self.envs]

    def close(self):
        for env in self.envs:
            env.close()


class ReacherReference(DMCReference):
    def __init__(self, num_envs=1, options=None):
        super().__init__("reacher", num_envs, options)


class DMCHumanoidReference(DMCReference):
    def __init__(self, num_envs=1, options=None):
        super().__init__("humanoid", num_envs, options)
