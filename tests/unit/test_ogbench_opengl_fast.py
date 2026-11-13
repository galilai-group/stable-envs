"""The opengl_fast backend: simplified arm meshes and a smaller shadow map, close to backend="opengl"."""

import os

import numpy as np
import pytest
import torch

import stable_envs as se

FAMILIES = [
    pytest.param("OGBCube", marks=pytest.mark.family("ogbench")),
    pytest.param("OGBScene", marks=pytest.mark.family("ogbench")),
]


@pytest.mark.parametrize("name", FAMILIES)
def test_cluster_meshes_simplifies_only_the_arm(name):
    import mujoco

    from stable_envs.envs.ogbench import manipulation_opengl as gl
    from stable_envs.envs.ogbench.simulation.compatibility import CompatibleCube, CompatibleScene

    family = name[3:]
    env = (CompatibleCube if family == "Cube" else CompatibleScene)(seed=0)
    env._mjcf_model = env.modify_mjcf_model(env.build_mjcf_model())
    env.compile_model_and_data()
    model = env.model
    before = model.mesh_facenum.copy()
    gl.cluster_meshes(model, gl.FAST[family][1])
    meshes = (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & (model.geom_group < 3)
    drawn = {int(i) for i in model.geom_dataid[meshes]}
    arm = [i for i in drawn if model.mesh(i).name.startswith("ur5e/")]
    assert arm and all(model.mesh_facenum[i] < before[i] for i in arm)
    assert sum(model.mesh_facenum[i] for i in arm) < 0.4 * sum(before[i] for i in arm)
    others = [i for i in range(model.nmesh) if i not in arm]
    np.testing.assert_array_equal(model.mesh_facenum[others], before[others])
    for i in arm:
        fa, fn, vn = int(model.mesh_faceadr[i]), int(model.mesh_facenum[i]), int(model.mesh_vertnum[i])
        face = model.mesh_face[fa : fa + fn]
        assert face.min() >= 0 and face.max() < vn
        assert np.all((face[:, 0] != face[:, 1]) & (face[:, 1] != face[:, 2]) & (face[:, 0] != face[:, 2]))


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.egl
def test_opengl_fast_draws_close_to_opengl(name):
    selection = os.environ.get("SE_FIDELITY_TEST_FAMILY", "")
    if selection and not name.startswith(selection):
        pytest.skip("Different pinned family profile")
    env = se.make(name, num_envs=4, device="cpu", seed=5, backend="opengl_fast")
    try:
        env.reset(seed=11)
        for _ in range(6):
            env.step(env.action_spec.sample(env.B, np.random.default_rng(0)), frames=None)
        state = env.to_render_state()
        fast = env.render(state, backend="opengl_fast")
        exact = env.render(state, backend="opengl")
        live = env.render()
    finally:
        env.close()
    assert torch.equal(live, fast)
    assert not torch.equal(fast, exact)
    mse = ((fast.double() - exact.double()) ** 2).mean(dim=(1, 2, 3))
    assert float(mse.mean()) <= 5.0
