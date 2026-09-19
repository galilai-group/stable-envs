"""DM Control ``walker`` — the visual tier.

The most balanced of the tier — skybox 48%, floor 45% — so both named residual mechanisms carry
roughly half each.

These are NOT the official pixels. What the difference measures is declared below and the two
mechanisms that carry it are named in ``VISUAL_NOTE``.
"""

from __future__ import annotations

from ._cells import _AT, _AT_MEASURED_WITH, _AT_VERSIONS, trackcom_cell  # noqa: F401

VISUAL_FIDELITY: dict[str, dict] = {
    "DMCWalkerWalk": {
        "mean_abs": 8.07,
        "within8_frac": 0.700,
        "ssim": 0.826,
        "interior_exact_frac": 0.0046,
        "interior_frac": 0.0010,
        "all_exact_frac": 0.205,
        "skybox_share": 0.478,
        "floor_share": 0.448,
        "draw_spread": {"mean_abs": [7.636, 8.302], "within8_frac": [0.6887, 0.7199], "ssim": [0.8170, 0.8306]},
        "mean_abs_max": 10.5,
        "within8_min": 0.65,
        "ssim_min": 0.77,
        "measured_at": _AT,
        "reference_versions": _AT_VERSIONS,
        "measured_with": _AT_MEASURED_WITH,
    },
}

DMCWalkerWalk = trackcom_cell("DMCWalkerWalk", "walker", "walk", 0, VISUAL_FIDELITY["DMCWalkerWalk"])
