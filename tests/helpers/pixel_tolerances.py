"""Per-class pixel tolerances for the tests that compare frames with the source renderer.

A class with no entry in ``AGREEMENT`` is compared pixel by pixel against ``EXACT`` plus its
entry in ``EXACT_OVERRIDES``. A class in ``AGREEMENT`` is compared by mean error, the
within-8/255 fraction and SSIM; it also declares ``visual_fidelity`` with its measurements.
"""

from __future__ import annotations

#: Default exact-pixel criterion. Interior pixels (no anti-aliasing ambiguity) must be exact; the
#: all-pixel floor is lower because aliased reference rasterizers disagree with any other rasterizer
#: at edges.
EXACT = {"interior_exact_min": 1.0, "all_exact_min": 0.90, "mean_abs_max": 4.0}

_FETCH = tuple(f"Fetch{s}{v}" for s in ("Reach", "Push", "Slide", "PickAndPlace") for v in ("", "Dense", "Dict"))
_GAUSSIAN_DOT_NAVIGATION = ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones")

#: Lowered exact-pixel floors, each with its cause.
EXACT_OVERRIDES: dict[str, dict[str, float]] = {
    # MJWarp shades the robot meshes per pixel; MuJoCo shades them per vertex.
    **{n: {"interior_exact_min": 0.99} for n in _FETCH},
    # torch's CPU float32 exp differs from the reference's by 1 ulp; that moves a pixel by 1/255
    # where the dot colour equals the background.
    **{n: {"interior_exact_min": 0.999995} for n in _GAUSSIAN_DOT_NAVIGATION},
    # Seven movers cover 11.7% of the frame, so curved-silhouette rims lower the all-pixel fraction.
    "DMCFingerTurnEasy": {"all_exact_min": 0.89},
}

#: Agreement limits: a ``mean_abs`` ceiling, a ``within8_frac`` floor and an ``ssim`` floor. Each
#: leaves a margin over the measurement in the class's ``visual_fidelity``. For OGBCube and OGBScene
#: the ceiling is 1.3 times the measured mean error, rounded to 0.1, and each floor is at least 0.04
#: below the worst of eight draws, rounded down to 0.05. PyFlytRocketLanding uses the same rule over
#: four draws, with the ceiling rounded up to 0.1.
AGREEMENT: dict[str, dict[str, float]] = {
    "DMCCheetahRun": {"mean_abs_max": 1.7, "within8_min": 0.92, "ssim_min": 0.95},
    "DMCHopperHop": {"mean_abs_max": 1.2, "within8_min": 0.93, "ssim_min": 0.95},
    "DMCHumanoidWalk": {"mean_abs_max": 2.9, "within8_min": 0.92, "ssim_min": 0.94},
    "DMCQuadrupedWalk": {"mean_abs_max": 1.5, "within8_min": 0.94, "ssim_min": 0.95},
    "DMCSwimmerSwimmer6": {"mean_abs_max": 3.3, "within8_min": 0.92, "ssim_min": 0.95},
    "DMCWalkerWalk": {"mean_abs_max": 2.3, "within8_min": 0.91, "ssim_min": 0.95},
    "OGBCube": {"mean_abs_max": 4.3, "within8_min": 0.85, "ssim_min": 0.85},
    "OGBScene": {"mean_abs_max": 8.7, "within8_min": 0.80, "ssim_min": 0.85},
    "PyFlytRocketLanding": {"mean_abs_max": 0.1, "within8_min": 0.95, "ssim_min": 0.95},
}


def exact_floors(name: str) -> dict[str, float]:
    """Return the exact-pixel criterion for the class ``name``."""
    return {**EXACT, **EXACT_OVERRIDES.get(name, {})}


def agreement_limits(name: str) -> dict[str, float] | None:
    """Return the agreement limits for the class ``name``, or None when it is compared exactly."""
    return AGREEMENT.get(name)
