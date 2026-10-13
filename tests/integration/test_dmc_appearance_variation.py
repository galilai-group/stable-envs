"""DM Control leaves that move pixels, rendered against the pinned stable_worldmodel wrappers."""

import os
import sys

import pytest

pytest.importorskip("dm_control")
pytestmark = [
    pytest.mark.swm,
    pytest.mark.nightly,
    pytest.mark.skipif(sys.platform == "darwin", reason="DM Control frames are certified on Linux GL"),
]

from tests.helpers.reference.gates.dmc_appearance import (  # noqa: E402
    CASES,
    FLOOR_CASES,
    FLOOR_COLORS,
    _mse,
    floor_gate,
    frames,
    leaf_gate,
    run_gate,
)

#: The contract's worst-frame bar for DM Control source pixels.
MSE_MAX = 0.001


@pytest.mark.parametrize("cell", CASES)
def test_leaf_frames_match_swm(cell):
    evidence = run_gate([cell], os.environ.get("SE_DMC_APPEARANCE_DEVICE", "cpu"))[cell]
    default = evidence["registered_default_mse"]
    for case in evidence["cases"]:
        assert case["mse"]["worst"] <= MSE_MAX, case
        # The leaf adds no residual beyond the cell's own at swm's registered default.
        assert case["mse"]["mean"] <= 1.5 * default["mean"] + 1e-4, (case, default)


_BRIGHT_HUMANOID_FLOOR_ON_CUDA = "mujoco_warp's floor shading residual on Humanoid's bright floor exceeds the bar"


@pytest.mark.parametrize(("cell", "case"), [(c, i) for c, cases in FLOOR_CASES.items() for i in range(len(cases))])
def test_floor_leaf_frames_match_swm(cell, case):
    device = os.environ.get("SE_DMC_APPEARANCE_DEVICE", "cpu")
    cfg = FLOOR_CASES[cell][case]
    if "floor.color" in cfg:
        # The recolour is the substituted texture; what remains against swm is the sampler's.
        got = floor_gate(cell, cfg, device=device)
        assert got["max_abs_vs_substituted"] <= 1, got
        if device == "cuda" and cell == "DMCHumanoidWalk" and cfg["floor.color"] == FLOOR_COLORS[0]:
            pytest.xfail(_BRIGHT_HUMANOID_FLOOR_ON_CUDA)
        assert got["mse"]["worst"] <= MSE_MAX, got
        return
    theirs, ours, _ = frames(cell, {}, device=device)
    default = _mse(ours, theirs)
    got = leaf_gate(cell, cfg, device=device)
    assert got["mse"]["worst"] <= MSE_MAX, got
    assert got["mse"]["mean"] <= 1.5 * default["mean"] + 1e-4, (got, default)
