"""DM Control leaves that move pixels, rendered against the pinned stable_worldmodel wrappers."""

import os
import sys

import pytest

pytest.importorskip("dm_control")
pytest.importorskip("stable_worldmodel")
pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="DM Control frames are certified on Linux GL")

from tests.helpers.reference.gates.dmc_appearance import CASES, run_gate  # noqa: E402

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
