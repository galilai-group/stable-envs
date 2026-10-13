from __future__ import annotations

from ._aligned import DMCAlignedSemantics
from ._cells import trackcom_cell

# This measurement uses the swimmer-specific render flags below.
_SWIMMER_AT = (
    "84 px, 64 states x 4 independent draws from sample_states, WORST of the four, "
    "default appearance, enable_specular=False, one RTX 3090 / driver 575.57.08 / MUJOCO_GL=egl, "
    "mujoco 3.12.0 + mujoco-warp 3.12.0, "
    "against dm_control's physics.render at camera 0"
)


# Reference pins describe the oracle; measured_with identifies the candidate backend.
_SWIMMER_AT_VERSIONS = {"mujoco": "3.12.0"}
_SWIMMER_AT_MEASURED_WITH = {"mujoco_warp": "3.12.0"}

# Disabling specular is specific to this cell; do not apply it to the whole family.
SPECULAR_NOTE = (
    "This domain renders with enable_specular=False, which is measured to be BETTER "
    "than the default on all four declared metrics as well as 1.30x faster (see "
    "SPECULAR_NOTE in stable_envs/envs/dmcontrol.py for the per-domain table and for "
    "why hopper does not get it)."
)

TRACKCOM_RENDER_FLAGS: dict[str, dict] = {
    "DMCSwimmerSwimmer6": {"enable_specular": False},
}

VISUAL_FIDELITY: dict[str, dict] = {
    "DMCSwimmerSwimmer6": {
        "mean_abs": 2.53,
        "within8_frac": 0.963,
        "ssim": 0.994,
        "interior_exact_frac": 0.510,
        "interior_mean_abs": 1.48,
        "interior_frac": 0.0045,
        "all_exact_frac": 0.178,
        "skybox_share": 0.000,
        "floor_share": 0.981,
        # Episode draws change the target geom and target light as well as the pose.
        "draw_spread": {
            "mean_abs": [2.370, 2.533],
            "within8_frac": [0.963, 0.970],
            "ssim": [0.994, 0.994],
            "interior_exact_frac": [0.510, 0.614],
        },
        "mean_abs_max": 3.3,
        "within8_min": 0.92,
        "ssim_min": 0.95,
        "measured_at": _SWIMMER_AT,
        "reference_versions": _SWIMMER_AT_VERSIONS,
        "measured_with": _SWIMMER_AT_MEASURED_WITH,
    },
}

DMCSwimmerSwimmer6 = trackcom_cell(
    "DMCSwimmerSwimmer6",
    "swimmer",
    "swimmer6",
    0,
    VISUAL_FIDELITY["DMCSwimmerSwimmer6"],
    render_flags=TRACKCOM_RENDER_FLAGS["DMCSwimmerSwimmer6"],
    note_suffix=SPECULAR_NOTE,
    bases=(DMCAlignedSemantics,),
    SIMULATOR="swimmer-swimmer6",
)
