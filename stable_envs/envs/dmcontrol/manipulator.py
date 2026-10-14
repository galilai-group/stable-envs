from __future__ import annotations

from ...render import raycast3d as R3
from ._aligned import DMCAlignedSemantics
from ._cells import parity_cell

DEFAULT_SIZE = 224


# Size of the registered wrapper target alternatives.
_REGISTERED_SMALL_TARGET = 0.015

# Keep the compiled model target as a third choice and the default.
# Wrapper box/sphere choices use a smaller size and must not replace the source default.
SHAPE_KNOB_SPECS: dict[str, dict] = {
    "DMCManipulatorBringBall": {
        "target_ball": (
            (R3.BOX, (_REGISTERED_SMALL_TARGET,) * 3),  # target.shape = 0
            (R3.SPHERE, (_REGISTERED_SMALL_TARGET, 0.0, 0.0)),  # target.shape = 1
            (R3.SPHERE, (0.02, 0.0, 0.0)),
        ),  # dm_control's own, the default
    },
}

# Candidates only: the conformance check must verify sphere-only rotation before excluding them.
RENDER_INVARIANT_DIMS: dict[str, tuple[str, ...]] = {
    "DMCManipulatorBringBall": ("ball_y", "target_ball_quat_qw", "target_ball_quat_qy"),
}

STATE_DIMS_NOTE: dict[str, str] = {
    "DMCManipulatorBringBall": (
        "Three of fifteen dimensions (ball_y, target_ball_quat_qw, target_ball_quat_qy) are excluded "
        "from check_state_dims_matter as render-invariant. All three orient an untextured sphere "
        "(`ball`, size .022, material `self`; `target_ball`, size .02, material `target`), and an "
        "analytic per-pixel intersector returns the same hit point, normal and colour for every "
        "rotation of a sphere. The class only offers the three (RENDER_INVARIANT_DIMS); "
        "check_state_dims_matter grants each exclusion only after verifying on the compiled model that "
        "perturbing it changes nothing but the geom_xmat of sphere geoms, and reports it in "
        "excluded_dims. Under the default appearance the three move 0 pixels here at 224 px and 84 px, "
        "while the official renderer moves 896 px (max 55/255) for ball_y and 734 px (max 33/255) for "
        "each quaternion slot, because MuJoCo draws a sphere as a tessellated mesh shaded per vertex. "
        "That residual is the one-level rim on curved silhouettes this family already declares: "
        "all-pixel exactness after each perturbation is 0.98275-0.98277 against a 0.98280 baseline, "
        "and interior-pixel agreement stays 1.000000. Under target_ball.shape = 0 the target is a box, "
        "so validation refuses both quaternion dimensions; they then move 201 px (max 145/255) at "
        "224 px and 28 px at 84 px. ball_y stays excluded because `ball` is a sphere in both styles."
    ),
}


DMCManipulatorBringBall = parity_cell(
    "DMCManipulatorBringBall",
    "manipulator",
    "bring_ball",
    default_size=DEFAULT_SIZE,
    SHAPE_KNOBS=SHAPE_KNOB_SPECS["DMCManipulatorBringBall"],
    RENDER_INVARIANT_DIMS=RENDER_INVARIANT_DIMS["DMCManipulatorBringBall"],
    state_dims_note=STATE_DIMS_NOTE["DMCManipulatorBringBall"],
    bases=(DMCAlignedSemantics,),
)
