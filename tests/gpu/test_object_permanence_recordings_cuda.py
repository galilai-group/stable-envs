"""Representative source fidelity and cache behavior, without a per-scene matrix."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("recording rendering requires CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from scripts.object_permanence_member import Member
from scripts.testing.check_object_permanence_member import cache_check
from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from stable_envs.testing.geometry import depth_mse, mask_iou
from tests.helpers.reference.object_permanence_recordings import (
    DEPTH_MSE_MAX,
    FIXTURES,
    MASK_IOU_MIN,
    RECORDINGS,
    load_geometry_directory,
)


@pytest.mark.parametrize(
    "task,oracle,contract",
    [
        ("G62", "g62/nominal", None),
        ("G13", "g13/opaque_pixel_rays_v2/nominal", "stable-envs/opaque-pixel-rays-v2"),
        ("G146", "g146/opaque_pixel_rays_v1/nominal", "stable-envs/opaque-pixel-rays-v1"),
    ],
)
def test_source_fidelity_and_cross_batch_cache(task, oracle, contract):
    directory = FIXTURES / "semantics/object_permanence" / oracle
    reference, arrays = load_geometry_directory(directory)
    frames = arrays["geometry_frames"] - 1
    assert len(frames) == 120
    with RigidTrajectory(RECORDINGS, f"{task.lower()}-nominal") as clip:
        assert clip.manifest["source"]["script_sha256"] == reference["source_script_sha256"]
        for cached in (False, True):
            with RigidSceneRenderer(
                clip, visibility_contract=contract, geometry_precision="float64", cache_static=cached
            ) as renderer:
                fields = renderer.render_many(frames, modes=("depth", "ids"))
            depth, ids = fields["depth"].cpu().numpy(), fields["ids"].cpu().numpy()
            for index in range(len(frames)):
                expected = arrays["object_ids"][index]
                roi = (expected > 0) | (ids[index] > 0)
                assert depth_mse(depth[index], arrays["depth"][index], roi) <= DEPTH_MSE_MAX
                for label in reference["object_ids"].values():
                    iou = float(mask_iou(ids[index] == label, expected == label))
                    assert np.isnan(iou) or iou >= MASK_IOU_MIN
    results = cache_check(RECORDINGS, Member(task), contract)
    assert all(r["passed"] and r["bit_identical"] for r in results), results


@pytest.mark.parametrize("task", ["g34", "g62"])
def test_multi_outputs_preserve_single_mode_buffers_and_borrowed_data(task):
    with RigidTrajectory(RECORDINGS, f"{task}-nominal") as clip:
        with RigidSceneRenderer(clip, size=90) as renderer:
            frames = np.array([119, 0, 59, 44, 0])
            fields = renderer.render_many(frames)
            for mode in fields:
                assert torch.equal(fields[mode], renderer.render(frames, mode=mode))
            out = {mode: torch.empty_like(value) for mode, value in fields.items()}
            assert renderer.render_many(frames, out=out) is out
            assert all(torch.equal(out[m], fields[m]) for m in fields)
            bad = {m: torch.full_like(v, 17) for m, v in out.items()}
            bad["depth"] = bad["depth"].cpu()
            with pytest.raises(ValueError, match="out"):
                renderer.render_many(frames, out=bad)
            assert all(torch.all(value == 17) for value in bad.values())
        assert not clip.closed
