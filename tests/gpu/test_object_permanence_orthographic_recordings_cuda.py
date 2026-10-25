"""V2 source fidelity and bit-exact cache reuse for every orthographic recording."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("orthographic rendering requires CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT_V2
from tests.helpers.reference.object_permanence_orthographic import ORTHOGRAPHIC_TASKS
from tests.helpers.reference.object_permanence_recordings import FIXTURES, RECORDINGS


@pytest.mark.parametrize("task", ORTHOGRAPHIC_TASKS)
@pytest.mark.parametrize("cache_static", [True, False])
def test_all_frame_orthographic_source_depth_and_every_object_mask(task, cache_static):
    from scripts.testing.check_object_permanence_rendering import check

    result = check(
        RECORDINGS,
        FIXTURES / f"semantics/object_permanence/{task}/opaque_pixel_rays_v2/nominal",
        task,
        visibility_contract=CONTRACT_V2,
        cache_static=cache_static,
    )
    assert result["frames"] == 120 and result["certification_eligible"]
    assert result["passed"], result["failures"]


@pytest.mark.parametrize("task", ORTHOGRAPHIC_TASKS)
@pytest.mark.parametrize("size", [90, 180])
def test_orthographic_cache_fields_chunks_output_isolation_and_all_frame_parity(task, size):
    with RigidTrajectory(RECORDINGS, task + "-nominal") as clip:
        with (
            RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT_V2) as cached,
            RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT_V2, cache_static=False) as full,
        ):
            frames = np.arange(120)[::-1].copy()
            expected = full.render_many(frames)
            actual = cached.render_many(frames)
            assert all(torch.equal(actual[k], expected[k]) for k in actual)
            indices = np.array([44, 0, 119, 44, 67])
            for chunk in (4, 1, 3, 7, 4):
                cached.chunk_size = chunk
                selected = cached.render_many(indices)
                assert all(torch.equal(selected[k], expected[k][119 - indices]) for k in selected)
                assert cached.renderer._ray_coverage.builds == 1
                assert cached.renderer._ray_coverage.fallbacks == 0
            selected["ids"].zero_()
            assert torch.equal(cached.render(indices, mode="ids"), expected["ids"][119 - indices])


def test_cold_orthographic_replay_needs_neither_blender_nor_mujoco():
    import subprocess
    import sys

    code = """
import sys
sys.modules['mujoco'] = None
sys.modules['bpy'] = None
import torch
from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT_V2
from tests.helpers.reference.object_permanence_recordings import RECORDINGS
for task in ('g09','g11','g75'):
    with RigidTrajectory(RECORDINGS, task+'-nominal') as clip:
        with RigidSceneRenderer(clip, size=90, visibility_contract=CONTRACT_V2) as renderer:
            fields = renderer.render_many([0,44,119])
            assert fields['rgb'].shape == (3,90,160,3)
            assert fields['depth'].dtype == torch.float32
            assert fields['ids'].dtype == torch.int16
            assert all(value.is_cuda for value in fields.values())
assert sys.modules['mujoco'] is None and sys.modules['bpy'] is None
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=90)
