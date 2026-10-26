"""Additional G02 members use the same all-frame certification as member zero."""

import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("member rendering requires CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from scripts.object_permanence_member import Member
from scripts.testing.check_object_permanence_member import check
from tests.helpers.reference.object_permanence_recordings import FIXTURES, RECORDINGS


@pytest.mark.parametrize("member", [1, 2])
def test_all_member_frames_source_masks_depth_and_cross_batch_cache(member):
    result = check(
        RECORDINGS, FIXTURES / "semantics/object_permanence", Member("G02", member), "stable-envs/opaque-pixel-rays-v1"
    )
    assert result["passed"], result
    assert all(r["frames"] == 120 for r in result["fidelity"] + result["cache"])
    assert all(r["static_builds"] == 1 and r["fallbacks"] == 0 for r in result["cache"])


@pytest.mark.parametrize(
    "task,member,contract",
    [
        ("G02", 0, "stable-envs/opaque-pixel-rays-v1"),
        ("G09", 0, "stable-envs/opaque-pixel-rays-v2"),
        ("G90", 0, "stable-envs/transparent-pixel-rays-v1"),
        ("G01", 1, "stable-envs/sampled-optical-rays-v1"),
        ("G03", 1, "stable-envs/sampled-optical-rays-v1"),
    ],
)
def test_existing_policy_and_member_routes_keep_full_frame_certification(task, member, contract):
    result = check(RECORDINGS, FIXTURES / "semantics/object_permanence", Member(task, member), contract)
    assert result["passed"], result
    assert all(r["frames"] == 120 for r in result["fidelity"] + result["cache"])
