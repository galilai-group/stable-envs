"""Every source alpha layer survives shared replay and exact cache reuse."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("alpha recordings need CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from scripts.export.object_permanence import ALPHA_TASKS
from scripts.testing.check_object_permanence_alpha import check
from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from tests.helpers.reference.object_permanence_recordings import FIXTURES, RECORDINGS


@pytest.mark.parametrize("task", sorted(ALPHA_TASKS))
@pytest.mark.parametrize("cached", [False, True])
def test_alpha_every_frame_depth_and_per_layer_and_overlapping_masks(task, cached):
    result = check(
        RECORDINGS,
        FIXTURES / f"semantics/object_permanence/{task.lower()}/transparent_pixel_rays_v1/nominal",
        task,
        cache_static=cached,
    )
    assert result["passed"] and result["frames"] == 120, result["failures"]


@pytest.mark.parametrize("task", sorted(ALPHA_TASKS))
@pytest.mark.parametrize("size", [90, 180])
def test_alpha_cache_all_fields_across_chunks_and_shuffled_repeated_frames(task, size):
    with RigidTrajectory(RECORDINGS, task.lower() + "-nominal") as clip:
        with (
            RigidSceneRenderer(clip, size=size, visibility_contract="stable-envs/transparent-pixel-rays-v1") as cached,
            RigidSceneRenderer(
                clip, size=size, visibility_contract="stable-envs/transparent-pixel-rays-v1", cache_static=False
            ) as full,
        ):
            modes = ("rgb", "depth", "ids", "masks", "weights", "layer_depth", "layer_ids")
            # Compare all frames in bounded groups rather than retaining a dense
            # [120,G,H,W] weight tensor for large multi-object scenes.
            for start in range(0, 120, 4):
                indices = np.arange(start, min(start + 4, 120))
                expected = full.render_many(indices, modes=modes)
                actual = cached.render_many(indices, modes=modes)
                assert all(torch.equal(actual[k], expected[k]) for k in modes)
            indices = np.array([44, 0, 119, 44, 67, 30, 90])
            expected = full.render_many(indices, modes=modes)
            for chunk in (1, 3, 7, 4):
                cached.chunk_size = chunk
                actual = cached.render_many(indices, modes=modes)
                assert all(torch.equal(actual[k], expected[k]) for k in modes)
                assert cached.renderer._layer_cache.builds == 1
                assert cached.renderer._layer_cache.fallbacks == 0
            full.renderer.shade_empty_layers = True
            baseline = full.render_many(indices, modes=modes)
            assert all(torch.equal(baseline[k], expected[k]) for k in modes)
            large = np.resize(indices, 32)
            expected = full.render_many(large)
            for chunk in (16, 32, 4):
                cached.chunk_size = chunk
                actual = cached.render_many(large)
                assert all(torch.equal(actual[k], expected[k]) for k in actual)
                assert cached.renderer._layer_cache.builds == 1


def test_cold_alpha_replay_uses_no_source_engine():
    import subprocess
    import sys

    code = r"""
import sys
sys.modules['bpy']=None;sys.modules['mujoco']=None
from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from tests.helpers.reference.object_permanence_recordings import RECORDINGS
for task in ('g90','g54'):
 with RigidTrajectory(RECORDINGS,task+'-nominal') as clip:
  with RigidSceneRenderer(clip,size=90,visibility_contract='stable-envs/transparent-pixel-rays-v1') as renderer:
   output=renderer.render_many([0,44,119],modes=('rgb','masks'))
   assert all(v.is_cuda for v in output.values())
assert sys.modules['bpy'] is None and sys.modules['mujoco'] is None
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=90)
