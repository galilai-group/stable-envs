from __future__ import annotations

from ._cells import _AT, _AT_MEASURED_WITH, _AT_VERSIONS, trackcom_cell  # noqa: F401
from ._humanoid import DMCHumanoidSemantics

_FROZEN_STAND_VISUAL_FIDELITY: dict[str, dict] = {
    "DMCHumanoidStand": {
        "mean_abs": 6.26,
        "within8_frac": 0.735,
        "ssim": 0.868,
        "interior_exact_frac": 0.00018,
        "interior_frac": 0.0297,
        "all_exact_frac": 0.023,
        "skybox_share": 0.000,
        "floor_share": 0.959,
        "draw_spread": {"mean_abs": [6.203, 6.300], "within8_frac": [0.7308, 0.7442], "ssim": [0.8660, 0.8707]},
        "mean_abs_max": 8.2,
        "within8_min": 0.69,
        "ssim_min": 0.82,
        "measured_at": _AT,
        "reference_versions": _AT_VERSIONS,
        "measured_with": _AT_MEASURED_WITH,
    },
}

VISUAL_FIDELITY: dict[str, dict] = {"DMCHumanoidWalk": _FROZEN_STAND_VISUAL_FIDELITY["DMCHumanoidStand"]}

DMCHumanoidWalk = trackcom_cell(
    "DMCHumanoidWalk",
    "humanoid",
    "walk",
    0,
    VISUAL_FIDELITY["DMCHumanoidWalk"],
    bases=(DMCHumanoidSemantics,),
    # Bound temporary CPU ray arrays by rendering four worlds per chunk.
    CPU_RENDER_CHUNK=4,
)
DMCHumanoidWalk.name = "dmc-humanoid-walk"
DMCHumanoidWalk.reference_id = "dm_control/humanoid-walk"
DMCHumanoidWalk.__doc__ = """Historical SWM Humanoid Walk using authoritative CPU physics.

The public identity names Humanoid(move_speed=1, pure_state=False), selected by
swm/HumanoidDMControl-v0. Full State includes physical, RNG and lifecycle fields.
CUDA selects State/render placement; simulation remains delegated to CPU.

VISUAL TIER: rendered with MJWarp on CUDA or cpu3d on CPU, so pixels approximate the source
rasterizer; see visual_note and visual_fidelity.
"""
