"""Rocket geometry is drawn where PyFlyt's TinyRenderer draws it.

Bullet's cached link frames lag the base by one physics substep, and the
camera follows the cached frame. ``getCameraImage`` places the link visuals by forward kinematics
from the current base pose and culls back faces, so the one-sided ground is not drawn from below.
These tests compare the link poses with pybullet and check raw MSE against the official frames.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.regression, pytest.mark.reference, pytest.mark.swm]

if not torch.cuda.is_available():
    pytest.skip("Rocket pixel regression requires a CUDA device", allow_module_level=True)

pytest.importorskip("PyFlyt")
pybullet = pytest.importorskip("pybullet")

import stable_envs as se  # noqa: E402
from stable_envs.envs.rocket import legacy as RL  # noqa: E402
from stable_envs.testing import conformance as C  # noqa: E402

SIZE, N = 480, 32
#: pybullet reports link positions in float32; the one-substep lag is 0.08 m or more in these draws.
POSE_TOL = 1e-4
#: Measured 4.8 to 9.1 over four 64-state draws at 480 px; frames with a fin or the body more
#: than one pixel off measure above 1,000.
MSE_MAX = 25.0


@pytest.fixture(scope="module")
def setup():
    cls = se.registry.REGISTRY["PyFlytRocketLanding"]
    env = cls(num_envs=1, size=SIZE, device="cuda")
    ref = C.reference_env(cls, SIZE)
    states = C.sample_render_rows(env, N, rng=np.random.default_rng(1000))
    yield env, cls, ref, states
    env.close()


def test_drawn_links_follow_the_current_base_and_the_camera_the_cached_frame(setup):
    _, _, ref, states = setup
    u = ref.unwrapped
    client, rid = u.env._client, u.env.drones[0].Id
    links = list(range(pybullet.getNumJoints(rid, physicsClientId=client)))
    style = RL._default_style_row()
    for st in states[:8]:
        _, unreachable = RL._reference_frame_flagged(ref, st, style)
        if unreachable:
            continue
        cached = pybullet.getLinkStates(rid, links, physicsClientId=client)
        fk = pybullet.getLinkStates(rid, links, computeForwardKinematics=1, physicsClientId=client)
        drawn, _ = RL.drawn_poses(st[None].astype(np.float64))
        assert np.abs(np.array([s[4] for s in fk]) - drawn[0, 2:12]).max() < POSE_TOL
        _, _, pos_prev, q_prev, _, _ = RL.lagged_pose(st[None])
        local, _ = RL.link_local_frames()
        lagged = pos_prev[0] + local[1:] @ RL.quat_to_mat(q_prev)[0].T
        assert np.abs(np.array([s[4] for s in cached]) - lagged).max() < POSE_TOL


def test_raw_mse_against_pyflyt(setup):
    env, cls, ref, states = setup
    official, unreachable = cls.reference_render(ref, states, return_flags=True)
    keep = ~np.asarray(unreachable, bool)
    assert keep.sum() >= N - 2
    ours = env._render(states).cpu().numpy()
    d = ours[keep].astype(np.float64) - official[keep].astype(np.float64)
    per_frame = (d**2).mean((1, 2, 3))
    assert per_frame.mean() <= MSE_MAX, f"raw MSE {per_frame.mean():.2f}, per frame {np.round(per_frame, 1)}"
    assert per_frame.max() <= 4 * MSE_MAX, f"worst frame raw MSE {per_frame.max():.2f}"


def test_ground_is_not_drawn_from_below(setup):
    env, cls, ref, _ = setup
    rows = []
    for tilt, x in ((0.0, 3.0), (-0.2, -4.0), (-0.3, 0.5)):
        q = RL.quat_mul(RL.quat_from_rotvec(np.array([0.0, tilt, 0.0])), np.array([1.0, 0.0, 0.0, 0.0]))
        rows.append(np.concatenate([[0.0, 0.0, 0.0], q, [0.0, 0.0, 0.0], [x, 1.0, 2.75]]))
    states = np.array(rows, np.float32)
    eye, _ = RL.camera_pose(*RL.lagged_pose(states)[2:4])
    assert (eye[:, 2] < 0).all()
    official, unreachable = cls.reference_render(ref, states, return_flags=True)
    assert not np.asarray(unreachable).any()
    ours = env._render(states).cpu().numpy()
    per_frame = ((ours.astype(np.float64) - official.astype(np.float64)) ** 2).mean((1, 2, 3))
    # Drawing the ground from below measures above 9,000; axis-aligned fin edges leave about 35 here.
    assert per_frame.max() <= 4 * MSE_MAX, f"raw MSE per frame {np.round(per_frame, 1)}"
    agreement = C.pixel_agreement(ours, official)
    assert agreement["interior_exact_frac"] == 1.0, agreement
