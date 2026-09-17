"""DM Control ``cheetah`` — the visual tier.

The skybox is 83% of this frame and MJWarp samples it with one unfiltered tap, which is what
holds SSIM down here more than anywhere else in the tier. ``enable_specular=False`` is a wash on
this domain (5.016 -> 5.008) and is NOT set; see ``swimmer``.

These are NOT the official pixels. What the difference measures is declared below and the two
mechanisms that carry it are named in ``VISUAL_NOTE``.
"""
from __future__ import annotations

from ._cells import _AT, _AT_MEASURED_WITH, _AT_VERSIONS, trackcom_cell  # noqa: F401

VISUAL_FIDELITY: dict[str, dict] = {
    "DMCCheetahRun": {
        "mean_abs": 5.01, "within8_frac": 0.882, "ssim": 0.661,
        "interior_exact_frac": None, "interior_frac": 0.0000, "all_exact_frac": 0.290,
        "skybox_share": 0.831, "floor_share": 0.127,
        "draw_spread": {"mean_abs": [4.986, 5.032], "within8_frac": [0.8813, 0.8826],
                        "ssim": [0.6597, 0.6615]},
        "mean_abs_max": 6.5, "within8_min": 0.84, "ssim_min": 0.62, "measured_at": _AT,
        "reference_versions": _AT_VERSIONS, "measured_with": _AT_MEASURED_WITH},
}

DMCCheetahRun = trackcom_cell(
    "DMCCheetahRun", "cheetah", "run", 0,
    VISUAL_FIDELITY["DMCCheetahRun"])
