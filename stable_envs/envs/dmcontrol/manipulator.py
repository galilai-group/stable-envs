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
        "Three of fifteen dimensions -- ball_y, target_ball_quat_qw and target_ball_quat_qy -- are "
        "EXCLUDED from check_state_dims_matter as render-invariant, under the validated contract that "
        "check documents. They are not dead and they are not undrawn: they are geometrically incapable "
        "of changing THIS renderer's image, and the reason is a property of the solid rather than of "
        "the implementation. All three orient an UNTEXTURED SPHERE -- `ball` is sphere size=.022 on "
        "material `self`, `target_ball` is sphere size=.02 on material `target`, and neither material "
        "carries a texture -- and a sphere is a surface of revolution about every axis through its "
        "centre, so an analytic per-pixel intersector returns the same hit point, the same normal and "
        "therefore the same colour for every rotation of it. Exactly, not approximately. "
        "NOTHING IS EXCLUDED BY NAME: this class only OFFERS the three (RENDER_INVARIANT_DIMS), and "
        "check_state_dims_matter grants the exclusion only after verifying against the compiled model "
        "that perturbing the dimension changes nothing but the geom_xmat of SPHERE geoms -- geom_xpos, "
        "geom_size, geom_type, every non-sphere orientation and every material, light and camera array "
        "bit-identical. A dimension the renderer merely failed to draw moves one of those and is "
        "refused. Each granted exclusion is REPORTED in the check's excluded_dims with its reason; none "
        "of them is a silent pass. "
        "MEASURED, both directions. Under the default (sphere) appearance the three move 0 pixels here "
        "at 224 px and at 84 px, while the OFFICIAL renderer moves 896 px at max 55/255 for ball_y and "
        "734 px at max 33/255 for each quaternion slot -- MuJoCo draws a sphere as a TESSELLATED mesh "
        "shaded PER VERTEX, so rotating it moves the Gouraud interpolation and the specular vertex. "
        "That residual is the same class this family already declares as a one-level rim on curved "
        "silhouettes: it is bounded (all-pixel exactness after each perturbation is 0.98275-0.98277 "
        "against a 0.98280 baseline, interior-pixel agreement stays 1.000000), and closing it would "
        "mean reproducing MuJoCo's icosphere tessellation in the mover pass for EVERY DM Control cell "
        "to recover ~800 px at <= 55/255 on one of them. "
        "AND THE EXCLUSION EVAPORATES WHEN THE GEOMETRY CHANGES, which is what makes it a statement "
        "about spheres rather than about rotation: under target_ball.shape = 0 the target is a BOX, the "
        "validation refuses both quaternion dimensions because they now rotate a non-sphere, and they "
        "measure 201 px (max 145/255) at 224 px and 28 px at 84 px. ball_y stays excluded there, "
        "correctly -- it orients the `ball` prop, which is a sphere in both styles. "
        "tests/unit/test_dmcontrol_episode_quat.py asserts every clause of this."
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
