from __future__ import annotations

from ._cells import trackcom_cell
from ._continuation import DMCContinuationSemantics

VISUAL_FIDELITY: dict[str, dict] = {
    "DMCHopperHop": {
        "mse_mean": 11.1461,
        "mse_p95": 15.102,
        "mse_max": 16.9639,
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

DMCHopperHop = trackcom_cell(
    "DMCHopperHop",
    "hopper",
    "hop",
    0,
    VISUAL_FIDELITY["DMCHopperHop"],
    bases=(DMCContinuationSemantics,),
)
