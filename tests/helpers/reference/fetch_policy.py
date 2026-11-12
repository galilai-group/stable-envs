"""A scripted Fetch controller, run closed loop on gymnasium-robotics and replayed on a batched backend."""

from __future__ import annotations

import numpy as np


def scripted_actions(task, grip, obj, goal, phase, count):
    """Proportional actions for each environment; ``phase`` and ``count`` hold each one's progress.

    Reach moves to the goal. PickAndPlace moves above the block, descends with the gripper open, closes it and
    carries the block to the goal. Push moves behind the block and pushes it to the goal. Slide moves behind the
    puck and strikes it toward the goal at the speed that slides it about the remaining distance.
    """
    n = len(grip)
    action = np.zeros((n, 4), np.float32)
    if task == "Reach":
        action[:, :3] = 10 * (goal - grip)
        return np.clip(action, -1, 1)
    for i in range(n):
        if task == "PickAndPlace":
            if phase[i] == 0:
                target, gripper = obj[i] + [0, 0, 0.06], 1.0
                if np.linalg.norm(target - grip[i]) < 0.01:
                    phase[i] = 1
            elif phase[i] == 1:
                target, gripper = obj[i], 1.0
                if np.linalg.norm(obj[i] - grip[i]) < 0.008:
                    phase[i] = 2
            elif phase[i] == 2:
                target, gripper = obj[i], -1.0
                count[i] += 1
                if count[i] >= 4:
                    phase[i] = 3
            else:
                target, gripper = goal[i], -1.0
            action[i, :3], action[i, 3] = 10 * (target - grip[i]), gripper
            continue
        direction = goal[i][:2] - obj[i][:2]
        distance = np.linalg.norm(direction)
        direction = direction / max(distance, 1e-6)
        behind = obj[i].copy()
        behind[:2] -= direction * 0.06
        if task == "Slide":
            behind[2] += 0.01
        action[i, 3] = -1.0
        if phase[i] == 0:
            target = behind + [0, 0, 0.06]
            if np.linalg.norm(target - grip[i]) < 0.01:
                phase[i] = 1
        elif phase[i] == 1:
            target = behind
            if np.linalg.norm(target - grip[i]) < 0.008:
                phase[i] = 2
        if phase[i] < 2 or task == "Push":
            if phase[i] >= 2:
                target = obj[i].copy()
                target[:2] += direction * 0.05
            action[i, :3] = (10 if phase[i] < 2 else 6) * (target - grip[i])
        elif phase[i] == 2:
            # The strike speed is fixed when it starts: v = sqrt(2 * mu * g * distance) for the puck's friction.
            if count[i] == 0:
                count[i] = int(np.sqrt(2 * 0.98 * distance) * 1000)
            action[i, :2] = direction * min(count[i] / 1000.0 / 1.25, 1.0)
            action[i, 2] = 10 * (behind[2] - grip[i][2])
            if np.linalg.norm(grip[i][:2] - behind[:2]) > 0.1:
                phase[i] = 3
        else:
            action[i, :3] = [0, 0, 1.0]
    return np.clip(action, -1, 1)


def source_rollouts(task, seeds, steps):
    """Run the controller closed loop on ``Fetch{task}-v4``; return actions, qpos and final success per seed."""
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    actions, qpos, success = [], [], []
    for seed in seeds:
        env = gym.make(f"Fetch{task}-v4", max_episode_steps=steps)
        obs, _ = env.reset(seed=int(seed))
        phase, count = np.zeros(1, int), np.zeros(1, int)
        rows, trajectory = [], []
        for _ in range(steps):
            o = obs["observation"]
            obj = o[None, 3:6] if task != "Reach" else o[None, 0:3]
            a = scripted_actions(task, o[None, 0:3], obj, obs["desired_goal"][None], phase, count)[0]
            obs, _, _, _, info = env.step(a)
            rows.append(a)
            trajectory.append(env.unwrapped.data.qpos.copy())
        actions.append(rows)
        qpos.append(trajectory)
        success.append(info["is_success"])
        env.close()
    return np.array(actions), np.array(qpos), np.array(success)
