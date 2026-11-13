"""Rocket ground checker texels match PyFlyt's TinyRenderer when a ground corner is near the camera plane.

A ground corner with clip w near zero projects to very large screen coordinates. TinyRenderer
takes double barycentrics of each ground triangle there, so the CUDA State render must select the
same checker texel. The CPU State render is TinyRenderer itself and is the reference.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.regression, pytest.mark.reference]

if not torch.cuda.is_available():
    pytest.skip("Rocket pixel regression requires a CUDA device", allow_module_level=True)

pytest.importorskip("PyFlyt")
pybullet = pytest.importorskip("pybullet")

import stable_envs as se  # noqa: E402
from stable_envs.envs.rocket import legacy as RL  # noqa: E402
from stable_envs.envs.rocket import render as RR  # noqa: E402

SIZE = 224
#: (x, y, z, yaw in degrees) of an almost upright rocket whose camera looks down at the far ground.
POSES = (
    (0.0, 0.0, 20.0, 15.0),
    (0.0, 0.0, 40.0, 345.0),
    (0.0, 0.0, 20.0, 105.0),
    (0.0, 0.0, 25.0, 345.0),
    (0.0, 0.0, 25.0, 255.0),
    (20.0, -10.0, 20.0, 247.5),
    (0.0, 0.0, 40.0, 75.0),
)
#: Measured 3.6 to 4.4 per frame; float32 barycentrics on the ground measured 83 to 588.
MSE_MAX = 15.0
CORNERS = np.array([[100, -100, 0, 1], [100, 100, 0, 1], [-100, 100, 0, 1], [-100, -100, 0, 1]], np.float64)


def _links():
    rows = []
    for x, y, z, yaw in POSES:
        q = RL.quat_from_euler(np.array([[0.02, -0.03, np.radians(yaw)]]))[0]
        rows.append([x, y, z, *q])
    return np.array(rows)


def _corner_w(link):
    view, proj = RR.camera_matrices(link, pybullet)
    mvp = np.asarray(proj).reshape(4, 4).T @ np.asarray(view).reshape(4, 4).T
    return (CORNERS @ mvp.T)[:, 3]


@pytest.fixture(scope="module")
def envs():
    cls = se.registry.REGISTRY["PyFlytRocketLanding"]
    cpu = cls(num_envs=1, size=SIZE, device="cpu")
    gpu = cls(num_envs=1, size=SIZE, device="cuda")
    cpu.reset(seed=0)
    yield cpu, gpu
    cpu.close()
    gpu.close()


def test_ground_texels_match_tinyrenderer_near_the_camera_plane(envs):
    cpu, gpu = envs
    links = _links()
    assert all(np.abs(_corner_w(link)).min() < 0.2 for link in links)
    state = cpu.to_render_state().expand(len(links)).clone()
    state["renderer", "camera_pose"] = torch.as_tensor(links, dtype=torch.float64)
    pose = state["renderer", "source_pose"].clone()
    pose[:, 1] = torch.as_tensor(links, dtype=torch.float64)
    state["renderer", "source_pose"] = pose
    official = cpu.render(state).numpy().astype(np.float64)
    ours = gpu.render(state.to("cuda")).cpu().numpy().astype(np.float64)
    per_frame = ((ours - official) ** 2).mean((1, 2, 3))
    assert per_frame.max() <= MSE_MAX, f"raw MSE per frame {np.round(per_frame, 1)}"
