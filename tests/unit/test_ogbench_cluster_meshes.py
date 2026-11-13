"""Arm mesh clustering of the private render models: only the ur5e meshes lose faces."""

import numpy as np
import pytest

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
    gl.cluster_meshes(model, 0.003)
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
