from __future__ import annotations

from ._cells import trackcom_cell
from ._continuation import DMCContinuationSemantics

VISUAL_FIDELITY: dict[str, dict] = {
    "DMCHopperHop": {
        "mean_abs": 0.89,
        "within8_frac": 0.976,
        "ssim": 0.990,
        "interior_exact_frac": None,
        "interior_frac": 0.0000,
        "all_exact_frac": 0.836,
        "skybox_share": 0.893,
        "floor_share": 0.082,
        "draw_spread": {"mean_abs": [0.887, 0.904], "within8_frac": [0.9754, 0.9773], "ssim": [0.9896, 0.9905]},
        "mean_abs_max": 1.2,
        "within8_min": 0.93,
        "ssim_min": 0.95,
        "measured_at": (
            "84 px, 64 states x 4 independent draws from sample_states (the domain's own initialize_episode "
            "plus a 0-40 step random-action rollout), default appearance (material self rgba .7 .5 .3), "
            "RTX 3090 / driver 575.57.08 / MUJOCO_GL=egl, mujoco 3.12.0 + mujoco-warp 3.12.0, against "
            "dm_control's physics.render at camera 0"
        ),
        "reference_versions": {"mujoco": "3.12.0"},
        "measured_with": {"mujoco_warp": "3.12.0"},
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
