from __future__ import annotations

from ._cells import trackcom_cell
from ._humanoid import DMCHumanoidSemantics

VISUAL_FIDELITY: dict[str, dict] = {
    "DMCHumanoidWalk": {
        "mse_mean": 24.0855,
        "mse_p95": 30.3416,
        "mse_max": 31.173,
        "measured_at": (
            "84 px, camera 0, per-frame MSE over pixels and channels on the 0-255 scale, over the 64 render "
            "rows of sample_render_rows(default_rng(20260811)), default appearance (material self rgba .7 .5 "
            ".3), NVIDIA H200 / driver 580.173.02 / MUJOCO_GL=egl, torch 2.15.0+cu132, mujoco 3.17.0 + "
            "mujoco-warp 3.17.0 + warp-lang 1.17.0, against dm_control's physics.render at camera 0"
        ),
        "reference_versions": {"mujoco": "3.17.0"},
        "measured_with": {"mujoco_warp": "3.17.0"},
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
DMCHumanoidWalk.__doc__ = """stable-worldmodel Humanoid Walk with CPU dm_control physics.

Simulates Humanoid(move_speed=1, pure_state=False), the task swm/HumanoidDMControl-v0
selects. Full State includes physical, RNG and lifecycle fields. CUDA selects State
and render placement; simulation runs on CPU.

Rendered with MJWarp on CUDA or cpu3d on CPU, so pixels approximate the source rasterizer;
see visual_note and visual_fidelity.
"""
