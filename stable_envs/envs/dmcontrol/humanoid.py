from __future__ import annotations

from ._cells import trackcom_cell
from ._humanoid import DMCHumanoidSemantics

VISUAL_FIDELITY: dict[str, dict] = {
    "DMCHumanoidWalk": {
        "mean_abs": 2.23,
        "within8_frac": 0.967,
        "ssim": 0.984,
        "interior_exact_frac": 0.9977,
        "interior_frac": 0.0311,
        "all_exact_frac": 0.274,
        "skybox_share": 0.000,
        "floor_share": 0.959,
        "draw_spread": {"mean_abs": [2.206, 2.270], "within8_frac": [0.9657, 0.9680], "ssim": [0.9840, 0.9845]},
        "mean_abs_max": 2.9,
        "within8_min": 0.92,
        "ssim_min": 0.94,
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
