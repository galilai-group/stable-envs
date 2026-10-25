"""Converted source geometry uses the existing shared renderer and cache contract."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("recorded rendering requires CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT
from tests.helpers.reference.object_permanence_recordings import RECORDINGS


@pytest.mark.parametrize("task", ["g04", "g29", "g31"])
@pytest.mark.parametrize("size", [90, 180])
def test_converted_perspective_cache_reuse_across_chunk_sizes(task, size):
    # Orthographic converted cases also run the existing V2 all-frame/chunk gate.
    with RigidTrajectory(RECORDINGS, task + "-nominal") as clip:
        with (
            RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT) as cached,
            RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT, cache_static=False) as full,
        ):
            expected = full.render_many(np.arange(120))
            for chunk in (4, 1, 3, 7, 4):
                cached.chunk_size = chunk
                indices = np.array([44, 0, 119, 44, 67])
                actual = cached.render_many(indices)
                assert all(torch.equal(actual[key], expected[key][indices]) for key in actual)
                assert cached.renderer._ray_coverage.builds == 1
                assert cached.renderer._ray_coverage.fallbacks == 0


def test_curve_and_font_recordings_render_without_source_engines_or_fonts():
    import subprocess
    import sys

    code = """
import sys
sys.modules['bpy']=None
sys.modules['mujoco']=None
from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from tests.helpers.reference.object_permanence_recordings import RECORDINGS
for task,version in [('g04','v1'),('g13','v2'),('g46','v2')]:
    with RigidTrajectory(RECORDINGS,task+'-nominal') as clip:
        with RigidSceneRenderer(clip,size=90,visibility_contract='stable-envs/opaque-pixel-rays-'+version) as renderer:
            output=renderer.render_many([0,44,119])
            assert output['rgb'].shape==(3,90,160,3)
            assert all(value.is_cuda for value in output.values())
assert sys.modules['bpy'] is None and sys.modules['mujoco'] is None
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=90)
