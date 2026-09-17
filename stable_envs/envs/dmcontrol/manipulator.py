"""DM Control ``manipulator`` — the parity tier, and the three declarations only this cell carries.

``bring_ball`` is the one fixed-camera domain with a per-episode ORIENTATION and a per-episode BODY,
and it turned out to need nothing from the renderer that a per-episode position does not already need
(``_episode_slots``); a per-episode body resolves to the geoms hanging off it (``_hideable``). Its
camera is ``fovy = 4``, nearly orthographic from 16 m, and its one model light is DIRECTIONAL with
``castshadow = 1`` — both of which the existing bake and cast handled unchanged.

Three things are declared here and nowhere else in the family, and all three are PRESERVED Step-8
facts rather than choices this migration made: the registered default size (224 px, which is
``stable_worldmodel``'s ``DMControlWrapper.render(width=224, height=224)``), the one ``SHAPE_KNOBS``
entry, and the task-specific state exclusions ``check_state_dims_matter`` is OFFERED.
"""
from __future__ import annotations

from ...render import raycast3d as R3
from ._cells import parity_cell

#: Cells whose registered SWM default is rendered at something other than the family's 84 px, so the
#: size the conformance suite builds them at is the size they are certified at. ``manipulator`` is
#: ``stable_worldmodel``'s ``DMControlWrapper.render(width=224, height=224)``.
DEFAULT_SIZE = 224


#: ``_SMALL_TARGET`` in ``stable_worldmodel.envs.dmcontrol.manipulator``: the edge/radius its
#: ``target.shape`` knob writes for both of its alternatives.
_SWM_SMALL_TARGET = 0.015

#: Per-cell ``SHAPE_KNOBS``. ONE cell, because one cell's registered wrapper offers a shape knob.
#: ``stable_worldmodel``'s ``target.shape`` is ``Discrete(2)`` over a 0.015 box and a 0.015 sphere, and
#: neither is the compiled model's own target (``sphere size=.02``, from ``manipulator.xml``'s ``ghost``
#: class) — so the leaf carries THREE alternatives and its default is index 2, the model's own. That is
#: what keeps the DEFAULT STYLE a render of the compiled model, which is the appearance every parity
#: number here is measured under, while still making the registered SWM default reachable, which is the
#: Gate-F obligation. Note what that is NOT: ``check_default_style_is_official`` compares
#: ``render(states)`` against ``render(states, style=default_style())`` and never touches the official
#: renderer, so it cannot see a default that drifted off the compiled model. The ``len(cur) != 1``
#: assertion in ``make_variation_space`` is what covers that, and it is why the assertion exists.
SHAPE_KNOB_SPECS: dict[str, dict] = {
    "DMCManipulatorBringBall": {
        "target_ball": ((R3.BOX, (_SWM_SMALL_TARGET,) * 3),            # swm target.shape = 0
                        (R3.SPHERE, (_SWM_SMALL_TARGET, 0.0, 0.0)),    # swm target.shape = 1
                        (R3.SPHERE, (0.02, 0.0, 0.0))),                # dm_control's own, the default
    },
}

#: somewhere a reviewer can approve or refuse it.
#: Candidate render-invariant dimensions per cell, OFFERED to ``check_state_dims_matter``. A name here
#: excludes nothing: the check validates each one against the compiled model and refuses any whose
#: perturbation moves anything but the orientation of SPHERE geoms. See ``STATE_DIMS_NOTE``.
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
        "tests/unit/test_dmcontrol_episode_quat.py asserts every clause of this."),
}


DMCManipulatorBringBall = parity_cell(
    "DMCManipulatorBringBall", "manipulator", "bring_ball",
    default_size=DEFAULT_SIZE,
    SHAPE_KNOBS=SHAPE_KNOB_SPECS["DMCManipulatorBringBall"],
    RENDER_INVARIANT_DIMS=RENDER_INVARIANT_DIMS["DMCManipulatorBringBall"],
    state_dims_note=STATE_DIMS_NOTE["DMCManipulatorBringBall"],
)
