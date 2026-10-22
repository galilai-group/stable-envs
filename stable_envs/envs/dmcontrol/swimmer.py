from __future__ import annotations

from ._aligned import DMCAlignedSemantics
from ._cells import trackcom_cell

# This measurement uses the swimmer-specific render flags below.
_SWIMMER_AT = (
    "84 px, camera 0, per-frame MSE over pixels and channels on the 0-255 scale, over the 64 render "
    "rows of sample_render_rows(default_rng(20260811)), default appearance (enable_specular=False), "
    "NVIDIA GeForce RTX 3090 / driver 575.57.08 / MUJOCO_GL=egl, torch 2.14.1+cu126, mujoco 3.12.0 + "
    "mujoco-warp 3.12.0 + warp-lang 1.17.0, against dm_control's physics.render at camera 0"
)


# Reference pins describe the oracle; measured_with identifies the candidate backend.
_SWIMMER_AT_VERSIONS = {"mujoco": "3.12.0"}
_SWIMMER_AT_MEASURED_WITH = {"mujoco_warp": "3.12.0"}

# Disabling specular is specific to this cell; do not apply it to the whole family.
SPECULAR_NOTE = (
    "This domain renders with enable_specular=False, which is 1.30x faster than the default. "
    "visual_fidelity is measured with this flag."
)

TRACKCOM_RENDER_FLAGS: dict[str, dict] = {
    "DMCSwimmerSwimmer6": {"enable_specular": False},
}

VISUAL_FIDELITY: dict[str, dict] = {
    "DMCSwimmerSwimmer6": {
        "mse_mean": 27.5853,
        "mse_p95": 51.7046,
        "mse_max": 58.1057,
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
