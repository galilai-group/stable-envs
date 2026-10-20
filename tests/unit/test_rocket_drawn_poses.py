"""CPU checks of ``drawn_poses`` and ``cull_ground`` in ``envs.rocket.legacy``."""

import numpy as np

from stable_envs.envs.rocket import legacy as RL


def test_drawn_poses_put_the_body_and_the_links_at_t():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    state = np.concatenate([[0, 0, 0], q, [0, 0, -24.0], [1.0, 2.0, 100.0]])  # 24 m/s straight down
    xp, xm = RL.drawn_poses(state[None])
    assert np.allclose(xp[0, 1], [1, 2, 100])
    assert np.allclose(xp[0, 3], [1, 2, 100 - 2.0])  # booster
    assert np.allclose(xp[0, 11], [1, 2, 100 - 3.4])  # flame
    assert np.allclose(xp[0, 12], RL.PAD_POS) and np.allclose(xp[0, 0], 0)
    for g in (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12):
        assert np.allclose(xm[0, g], np.eye(3)), g
    leg2 = RL.quat_to_mat(RL.quat_from_euler(np.array([0, 0, 4.188])))
    assert np.allclose(xm[0, 9], leg2)


def test_cull_ground_moves_the_ground_only_when_the_eye_is_below_it():
    xp = np.zeros((2, RL.N_GEOMS, 3))
    xm = np.tile(np.eye(3), (2, RL.N_GEOMS, 1, 1))
    RL.cull_ground(xp, xm, np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -0.5]]))
    assert np.allclose(xp[0, 0], 0)
    assert np.allclose(xp[1, 0], [0, 0, -RL.GROUND_CULL_DROP])
    assert np.allclose(xp[:, 1:], 0)
