"""DM Control ``swimmer`` — the visual tier, and the one domain that turns MJWarp's specular off.

Two things are declared here and nowhere else: its own provenance string (its numbers are measured
under a non-default render context and over four draws rather than three, so it cannot borrow
``_AT``), and the CPU refusal. Swimmer has no CPU certification; without an explicit
refusal it would be drawn on CPU — its bodies are ``mjGEOM_ELLIPSOID`` and ``raycast3d._ellipsoid``
can intersect them — with no certification behind the frames. The refusal is therefore BY NAME, with the reason, and the GPU path is unaffected.
"""

from __future__ import annotations

from ._cells import trackcom_cell

#: Conditions every ``visual_fidelity`` below was measured under. One string, so a reader cannot get a
#: number without the conditions that produced it.
#: swimmer's entry is measured under its own render context (``enable_specular=False``) and over four
#: draws rather than three, so it carries its own provenance string rather than borrowing ``_AT``.
_SWIMMER_AT = (
    "84 px, 64 states x 4 independent draws from sample_states, WORST of the four, "
    "default appearance, enable_specular=False, one H200, mujoco 3.11.0 / warp 1.16.0, "
    "against dm_control's physics.render at camera 0"
)


#: The versions ``_SWIMMER_AT`` names, as fields. ``reference_versions`` holds the REFERENCE
#: side only (mujoco); ``measured_with`` holds OUR side (warp) and is recorded, never compared.
_SWIMMER_AT_VERSIONS = {"mujoco": "3.11.0"}
_SWIMMER_AT_MEASURED_WITH = {"warp": "1.16.0"}

#: batch 1024 against dm_control's own renderer, one draw of 64 states per cell:
#:
#:     domain         speed    mean abs        within-8         SSIM
#:     swimmer6       1.30x    4.16 -> 2.31    0.854 -> 0.955   0.9869 -> 0.9935   ship
#:     walker-walk    1.19x    5.85 -> 5.78    0.762 -> 0.763   0.9335 -> 0.9336   marginal
#:     cheetah-run    1.16x    5.016 -> 5.008  0.882 -> 0.884   0.66098 -> 0.66093 a wash
#:     humanoid-stand 1.16x    4.35 -> 4.23    0.866 -> 0.878   0.9364 -> 0.9404   SSIM up, mixed at 84 px
#:     hopper-hop     1.15x    4.347 -> 4.355  0.9053 -> 0.9051 0.6821 -> 0.6818   A LOSS
#:
#: IT IS NOT A FAMILY-WIDE SWITCH, and hopper is the reason it is written down: the same flag makes
#: that domain measurably worse on all three metrics. The mechanism is that dm_control's GL
#: fixed-function specular and MJWarp's are different lobes, so removing the term moves a frame
#: toward the reference only where MJWarp was adding a highlight the reference does not have. That
#: happens on the two floor-dominated domains (swimmer floor_share 0.981, humanoid 0.959) and not on
#: the skybox-dominated ones — the same camera-and-scene dependence the RoboCasa investigation found
#: between its agentview and wrist cameras. Swimmer is the one
#: domain where the gain is large (mean |diff| down 37%) rather than in the third decimal, so it is
#: the one domain that sets the flag.
SPECULAR_NOTE = (
    "This domain renders with enable_specular=False, which is measured to be BETTER "
    "than the default on all four declared metrics as well as 1.30x faster (see "
    "SPECULAR_NOTE in stable_envs/envs/dmcontrol.py for the per-domain table and for "
    "why hopper does not get it)."
)

#: The one domain that overrides ``create_render_context``. Keyed by class name so the table above and
#: the switch cannot drift apart.
TRACKCOM_RENDER_FLAGS: dict[str, dict] = {
    "DMCSwimmerSwimmer6": {"enable_specular": False},
}

#: The reason ``device="cpu"`` is refused, carried by the cell (``CPU_UNCERTIFIED_REASON``). The
#: package re-exports the collected mapping as ``CPU_UNCERTIFIED`` for the scripts that read it.
CPU_UNCERTIFIED_REASON = "no CPU certification (certification deferred)"

VISUAL_FIDELITY: dict[str, dict] = {
    # swimmer is the one domain that renders with ``enable_specular=False`` (see SPECULAR_NOTE), so
    # its entry is the only one measured under a non-default render context. Everything here is the
    # WORST of 4 independent 64-state draws, natively at 84 px with no resampling
    # , and every number is BETTER than the shipped
    # specular-on measurement it replaces: mean 5.48 -> 3.45, within-8 0.813 -> 0.917,
    # SSIM 0.980 -> 0.986, interior mean 6.22 -> 0.91, interior-exact 0.464 -> 0.515,
    # all-exact 0.123 -> 0.148. The thresholds keep the family's own margin convention (ceiling
    # 1.37x the measurement, floors 0.93x / 0.95x of it) rather than being tightened onto the
    # measurement, so they are far below the old ones without being fragile.
    "DMCSwimmerSwimmer6": {
        "mean_abs": 3.45,
        "within8_frac": 0.917,
        "ssim": 0.986,
        "interior_exact_frac": 0.515,
        "interior_mean_abs": 0.91,
        "interior_frac": 0.0059,
        "all_exact_frac": 0.148,
        "skybox_share": 0.000,
        "floor_share": 0.981,
        # the widest spread of the five, and it is the widened state doing it: swimmer draws its
        # target geom AND its target light per episode, so two draws differ in the scene and not only
        # in the pose. The ceiling is set from the worst of the four rather than from the mean.
        "draw_spread": {
            "mean_abs": [3.223, 3.452],
            "within8_frac": [0.917, 0.921],
            "ssim": [0.986, 0.986],
            "interior_exact_frac": [0.515, 0.657],
        },
        "mean_abs_max": 4.8,
        "within8_min": 0.85,
        "ssim_min": 0.94,
        "measured_at": _SWIMMER_AT,
        "reference_versions": _SWIMMER_AT_VERSIONS,
        "measured_with": _SWIMMER_AT_MEASURED_WITH,
    },
    # quadruped at the registered default (camera 2, 224 px). Three residuals, per surface off the
    # segmentation buffer (tests/fixtures/regression/join/step7/dmc-quadruped/findings.md S5): the floor near the
    # body -- the body's REFLECTION in the reflectance-0.2 floor and its shadow -- carries ~half of
    # it; the far checker and the one-tap skybox the rest; the ellipsoid itself is 1.2 % of the frame
    # at mean ~4/255. Thresholds by the family's margin convention (ceiling ~1.3x, floors ~0.93x).
}

DMCSwimmerSwimmer6 = trackcom_cell(
    "DMCSwimmerSwimmer6",
    "swimmer",
    "swimmer6",
    0,
    VISUAL_FIDELITY["DMCSwimmerSwimmer6"],
    render_flags=TRACKCOM_RENDER_FLAGS["DMCSwimmerSwimmer6"],
    note_suffix=SPECULAR_NOTE,
    CPU_UNCERTIFIED_REASON=CPU_UNCERTIFIED_REASON,
)
