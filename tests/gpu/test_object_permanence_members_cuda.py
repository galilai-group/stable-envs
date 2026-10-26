"""Audited additional members share complete-frame source and cache certification."""

import pytest
import torch

pytestmark = pytest.mark.gpu
if not torch.cuda.is_available():
    pytest.skip("member rendering requires CUDA", allow_module_level=True)
pytest.importorskip("nvdiffrast.torch")

from scripts.object_permanence_member import Member
from scripts.testing.check_object_permanence_member import check
from tests.helpers.reference.object_permanence_recordings import FIXTURES, RECORDINGS


@pytest.mark.parametrize(
    "task,member,version",
    [
        ("G02", 1, "v1"),
        ("G02", 2, "v1"),
        ("G04", 1, "v1"),
        ("G05", 2, "v1"),
        ("G05", 4, "v1"),
        ("G05", 5, "v1"),
        ("G05", 6, "v1"),
        ("G06", 1, "v1"),
        ("G06", 2, "v1"),
        ("G07", 1, "v1"),
        ("G08", 1, "v2"),
        ("G09", 1, "v2"),
        ("G09", 2, "v2"),
    ],
)
def test_all_member_frames_source_masks_depth_and_cross_batch_cache(task, member, version):
    result = check(
        RECORDINGS,
        FIXTURES / "semantics/object_permanence",
        Member(task, member),
        f"stable-envs/opaque-pixel-rays-{version}",
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
