"""The Numba CPU backend of the DM Control renderers against the Torch backend.

Torch's CPU square root can round through MKL, so the two backends may differ by one level on rare pixels.
With an IEEE square root in Torch, the ray casts agree bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("numba")
mujoco = pytest.importorskip("mujoco")

from stable_envs.render import cpu3d  # noqa: E402
from stable_envs.render import raycast3d as R3  # noqa: E402
from stable_envs.render.cpu_raster import ray_module, rays3d  # noqa: E402

KINDS = (R3.SPHERE, R3.CAPSULE, R3.CYLINDER, R3.BOX, R3.ELLIPSOID)


@pytest.fixture
def ieee_sqrt(monkeypatch):
    monkeypatch.setattr(torch, "sqrt", lambda x: torch.from_numpy(np.sqrt(x.numpy())))


def assert_close_frames(a: np.ndarray, b: np.ndarray) -> None:
    """At most one level apart, on at most one pixel in ten thousand."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    assert d.max() <= 1, f"max difference {d.max()}"
    assert (d > 0).mean() <= 1e-4, f"{int((d > 0).sum())} of {d.size} pixels differ"


def _rays(kind: int, rows: int = 3, count: int = 4000):
    rng = np.random.default_rng(kind)
    sizes = rng.uniform(0.1, 0.5, (rows, 1, 3)).astype(np.float32)
    pos = rng.normal(size=(rows, 1, 3)).astype(np.float32)
    rot = np.linalg.qr(rng.normal(size=(rows, 1, 3, 3)))[0].astype(np.float32)
    env = np.sort(rng.integers(0, rows, count))
    origin = (pos[env, 0] + rng.normal(size=(count, 3)) * 3).astype(np.float32)
    aim = pos[env, 0] + rng.normal(size=(count, 3)) * 0.25
    direction = aim - origin
    direction = (direction / np.linalg.norm(direction, axis=1, keepdims=True)).astype(np.float32)
    return sizes, pos, rot, env, origin, direction


@pytest.mark.parametrize("kind", KINDS)
def test_numba_casts_match_torch_intersections(ieee_sqrt, kind):
    sizes, pos, rot, env, o, d = _rays(kind)
    q = len(env)
    t = np.empty(q, np.float32)
    n = np.empty((q, 3), np.float32)
    i = np.empty(q, np.int64)
    kinds = np.array([kind], np.int64)
    rays3d.cast_rays(kinds, sizes, pos, rot, env, o, d, rays3d.packets(env, rays3d.PACKET), t, n, i)
    T = torch.from_numpy
    et, en = R3.intersect(kind, T(sizes[env, 0]), T(pos[env, 0]), T(rot[env, 0]), T(o), T(d))
    hit = et < R3.INF
    assert hit.float().mean() > 0.05 and (~hit).any()
    assert torch.equal(T(t), et)
    assert torch.equal(T(n)[hit], en[hit])
    lim = np.random.default_rng(1).uniform(0, 6, q).astype(np.float32)
    blocked = np.empty(q, np.bool_)
    skip = np.full(q, -1, np.int64)
    rays3d.blocked_rays(kinds, sizes, pos, rot, env, o, d, lim, skip, rays3d.packets(env, rays3d.PACKET), blocked)
    assert torch.equal(T(blocked), (et > 0) & (et < T(lim)))


def _scene(*, tilt=0, shadows=True, specular=True, reflectance=0.0, size=48):
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco><asset>
      <texture name="floor" type="2d" builtin="checker" width="16" height="16"
               rgb1="0.2 0.3 0.4" rgb2="0.7 0.8 0.9"/>
      <texture type="skybox" builtin="gradient" width="16" height="96"
               rgb1="0.1 0.2 0.3" rgb2="0.8 0.9 1"/>
      <material name="floor" texture="floor" specular="0.3" reflectance="{reflectance}"/>
      <material name="body" rgba="0.7 0.2 0.1 1" specular="0.8" shininess="0.4"/>
    </asset><worldbody>
      <camera pos="0 -4 3" xyaxes="1 0 0 0 0.6 0.8"/>
      <light pos="-1 -1 4" dir="0 0 -1" cutoff="80" exponent="3"/>
      <light pos="1 -1 4" dir="0 0.2 -1" cutoff="60" castshadow="false"/>
      <geom type="plane" size="4 4 0.1" euler="{tilt} 0 0" material="floor"/>
      <body pos="0 0 0.8"><freejoint/>
        <geom type="sphere" size="0.25" material="body"/>
        <geom type="capsule" size="0.12 0.3" pos="0.65 0 0" material="body"/>
        <geom type="box" size="0.2 0.15 0.3" pos="-0.65 0 0" rgba="0.1 0.7 0.3 1"/>
        <geom type="cylinder" size="0.1 0.2" pos="0 0.6 0" material="body"/>
        <geom type="ellipsoid" size="0.1 0.2 0.15" pos="0 -0.6 0.2" rgba="0.3 0.3 0.9 1"/>
      </body>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    scene = cpu3d.Scene(model, 0, size, [1, 2, 3, 4, 5], use_shadows=shadows, enable_specular=specular)
    poses = []
    for i in range(7):
        data.qpos[0] = (i - 3) * 0.2
        data.qpos[2] = 0.6 + (i % 3) * 0.15
        data.qpos[3:7] = [np.cos(0.2 * i), 0, np.sin(0.2 * i), 0]
        mujoco.mj_forward(model, data)
        poses.append(
            (
                data.cam_xpos[0].copy(),
                data.geom_xpos[1:].copy(),
                data.geom_xmat[1:].reshape(-1, 3, 3).copy(),
                data.light_xpos.copy(),
                data.light_xdir.copy(),
            )
        )
    args = [torch.tensor(np.stack([p[k] for p in poses]), dtype=torch.float32) for k in range(5)]
    material = torch.tensor(model.mat_rgba[:, :3].copy()).expand(7, -1, -1).clone()
    material[:, 1, 0] = torch.linspace(0.2, 0.8, 7)
    lights = torch.tensor(model.light_diffuse.copy()).expand(7, -1, -1).clone()
    return scene, (*args, material, lights)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"specular": False},
        {"reflectance": 0.4},
        {"tilt": 25, "shadows": False, "reflectance": 0.3},
    ],
)
def test_numba_scene_matches_torch(options):
    scene, args = _scene(**options)
    plane_mat = None
    if options.get("tilt"):
        angle = torch.linspace(0, 0.2, 7)
        c, s, z, o = torch.cos(angle), torch.sin(angle), torch.zeros(7), torch.ones(7)
        plane_mat = torch.stack((c, z, s, z, o, z, -s, z, c), -1).reshape(7, 3, 3) @ torch.tensor(
            scene.plane_mat, dtype=torch.float32
        )
    expected = cpu3d.render(scene, *args, backend="torch", plane_mat=plane_mat).numpy()
    actual = cpu3d.render(scene, *args, backend="numba", plane_mat=plane_mat).numpy()
    assert len(np.unique(expected.reshape(-1, 3), axis=0)) > 100
    assert_close_frames(actual, expected)
    one = cpu3d.render(
        scene, *(a[3:4] for a in args), backend="numba", plane_mat=None if plane_mat is None else plane_mat[3:4]
    )
    assert np.array_equal(one.numpy()[0], actual[3]), "a Numba frame depends on its batch"


def test_backend_selection():
    assert ray_module("auto", "cpu") is rays3d
    assert ray_module("numba", "cpu") is rays3d
    assert ray_module("torch", "cpu") is None
    assert ray_module("auto", "cuda") is None
    with pytest.raises(ValueError, match="backend"):
        ray_module("triton", "cpu")


DMC = (
    "DMCCheetahRun",
    "DMCHopperHop",
    "DMCWalkerWalk",
    "DMCHumanoidWalk",
    "DMCQuadrupedWalk",
    "DMCSwimmerSwimmer6",
    "DMCFingerSpin",
    "DMCBallInCupCatch",
    "DMCManipulatorBringBall",
    "DMCReacher",
    "DMCCartpoleSwingup",
    "DMCPointMassEasy",
)


@pytest.mark.gl
@pytest.mark.parametrize("name", DMC)
def test_dmc_numba_frames_match_torch_frames(name):
    import stable_envs as se
    from stable_envs.testing import conformance as C

    frames = {}
    for backend in ("torch", "numba"):
        env = se.make(name, num_envs=4, size=64, device="cpu", backend=backend)
        try:
            rows = torch.as_tensor(C.sample_render_rows(env, 4, rng=np.random.default_rng(3)))
            style = env.sample_style(4, rng=np.random.default_rng(4))
            frames[backend] = (env._render(rows).numpy(), env._render(rows, style=style).numpy())
        finally:
            env.close()
    for a, b in zip(frames["numba"], frames["torch"]):
        assert_close_frames(a, b)


@pytest.mark.gl
@pytest.mark.parametrize("name", ["DMCReacher", "DMCBallInCupCatch"])
def test_dmc_numba_supersampled_frames_match_torch_frames(name):
    import stable_envs as se
    from stable_envs.testing import conformance as C

    frames = {}
    for backend in ("torch", "numba"):
        env = se.make(name, num_envs=3, size=48, supersample=2, device="cpu", backend=backend)
        try:
            rows = torch.as_tensor(C.sample_render_rows(env, 3, rng=np.random.default_rng(5)))
            frames[backend] = env._render(rows).numpy()
        finally:
            env.close()
    assert_close_frames(frames["numba"], frames["torch"])
