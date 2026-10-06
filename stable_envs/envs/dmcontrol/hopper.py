from __future__ import annotations

from ._cells import _AT, _AT_MEASURED_WITH, _AT_VERSIONS, trackcom_cell  # noqa: F401
from ._continuation import DMCContinuationSemantics

VISUAL_FIDELITY: dict[str, dict] = {
    "DMCHopperHop": {
        "mean_abs": 4.35,
        "within8_frac": 0.905,
        "ssim": 0.684,
        "interior_exact_frac": None,
        "interior_frac": 0.0000,
        "all_exact_frac": 0.307,
        "skybox_share": 0.893,
        "floor_share": 0.082,
        "draw_spread": {"mean_abs": [4.327, 4.364], "within8_frac": [0.9044, 0.9051], "ssim": [0.6826, 0.6866]},
        "mean_abs_max": 5.7,
        "within8_min": 0.86,
        "ssim_min": 0.64,
        "measured_at": _AT,
        "reference_versions": _AT_VERSIONS,
        "measured_with": _AT_MEASURED_WITH,
    },
}

DMCHopperHop = trackcom_cell(
    "DMCHopperHop",
    "hopper",
    "hop",
    0,
    VISUAL_FIDELITY["DMCHopperHop"],
    bases=(DMCContinuationSemantics,),
)
