from __future__ import annotations

from ._cells import trackcom_cell
from ._continuation import DMCContinuationSemantics

VISUAL_FIDELITY: dict[str, dict] = {
    "DMCWalkerWalk": {
        "mean_abs": 1.75,
        "within8_frac": 0.953,
        "ssim": 0.991,
        "interior_exact_frac": 0.9955,
        "interior_frac": 0.0087,
        "all_exact_frac": 0.716,
        "skybox_share": 0.478,
        "floor_share": 0.448,
        "draw_spread": {"mean_abs": [1.655, 1.811], "within8_frac": [0.9519, 0.9556], "ssim": [0.9900, 0.9915]},
        "mean_abs_max": 2.3,
        "within8_min": 0.91,
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

DMCWalkerWalk = trackcom_cell(
    "DMCWalkerWalk",
    "walker",
    "walk",
    0,
    VISUAL_FIDELITY["DMCWalkerWalk"],
    bases=(DMCContinuationSemantics,),
)
