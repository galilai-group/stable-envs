"""DM Control ``quadruped`` — the visual tier, at camera 2, on ``stable_worldmodel``'s own model.

The one cell in the family whose MODEL is not ``suite.load``'s and whose camera is not 0. Both are
These facts and both are preserved exactly: ``swm/QuadrupedDMControl-v0`` registers quadruped-walk
at camera 2 (``y``, on the torso, trackcom) with SWM's default appearance, and the certified
configuration is the registered default (Step 2 S4.2), so this is the model the cell draws and the
reference renders. The suite's own appearance is one ``material.self`` knob away.

The torso is an ELLIPSOID, drawn by ``raycast3d._ellipsoid`` on CPU.
"""

from __future__ import annotations

import numpy as np

from ._cells import _AT_VERSIONS, trackcom_cell

#: Quadruped (Step 7): the family's convention -- three independent MJWarp draws through the shipped
#: env -- at the REGISTERED default (camera 2, 224 px, SWM's appearance; Step 2 S4.2), under the
#: declared 3.11.0 pair, on a 3090 (job 6394987, gpu2107; the draws of job 6394721 on gpu2104 were
#: overwritten by it and are not what the numbers rest on). The CPU backend
#: measured the same picture locally first (3.42 / 0.902 / 0.957 on one draw, Apple M5) and is
#: certified against this entry, not its own.
_QUADRUPED_AT = (
    "224 px, camera 2 ('y', trackcom), 64 states x 3 independent draws from sample_states "
    "(SWM's model and default appearance: material self rgba .6 .3 .3, foot_back_left "
    "density 1000), RTX 3090 / driver 575.57.08 / MUJOCO_GL=egl, mujoco 3.11.0 + "
    "mujoco-warp 3.11.0, against dm_control's physics.render at camera 2 "
    "(scratch/dmcontrol/visual_calib.py --env DMCQuadrupedWalk --size 224 --draws 3)"
)
_QUADRUPED_AT_MEASURED_WITH = {"mujoco_warp": "3.11.0"}

#: MEASURED per domain, then the thresholds the conformance suite asserts. The thresholds are set from
#: the SPREAD over independent state draws rather than from one, which is why they are not the
#: measurements: ``scratch/dmcontrol/visual_calib.py`` re-draws three times and the ceiling sits clear


def _swm_quadruped_env(seed: int):
    """The env ``swm/QuadrupedDMControl-v0`` registers, built the way ``stable_worldmodel`` builds it.

    SWM's ``QuadrupedDMControlWrapper`` does not call ``suite.load``: it takes
    ``quadruped.make_model(floor_size=20 * 0.5)`` (the suite's ``walk`` model), round-trips it
    through ``dm_control.mjcf`` applying its variation space's DEFAULT values at every reset, and
    drives ``quadruped.Move(desired_speed=0.5)`` (the suite's ``walk`` task) under
    ``action_scale.Wrapper``. Measured against ``suite.load("quadruped", "walk")``
    (``examples/results/step7/dmc-quadruped/phase0/swm_model_roundtrip.json``): geometry, cameras,
    lights, texture bytes and ``qpos0`` identical, the material/texture tables permuted, and TWO
    defaults that differ from the suite -- material ``self`` is rgba (0.6, 0.3, 0.3, 1) where the suite
    has (0.7, 0.5, 0.3, 1) (mean 1.34/255 between the two on the same qpos), and ``foot_back_left``
    has density 1000 where the suite's ``body`` class gives 500 (a mass, so the visited distribution).
    The certified configuration is the registered default (Step 2 S4.2), so this is the model the cell
    draws and the reference renders; the suite's own appearance is one ``material.self`` knob away.
    The lines below are SWM's ``modify_mjcf_model`` at ``init_value``, transcribed.
    """
    import os
    import tempfile

    from dm_control import mjcf
    from dm_control.rl import control
    from dm_control.suite import quadruped
    from dm_control.suite.wrappers import action_scale

    speed = 0.5  # _TASK_SPEEDS["walk"]
    xml = quadruped.make_model(floor_size=20 * speed).replace(b'file="./common/', b'file="common/')
    model = mjcf.from_xml_string(
        xml, model_dir=os.path.dirname(quadruped.__file__), assets=quadruped.common.ASSETS or {}
    )
    grid = model.find("texture", "grid")
    grid.rgb1 = np.array([0.1, 0.2, 0.3])
    grid.rgb2 = np.array([0.2, 0.3, 0.4])
    model.find("material", "self").rgba = np.array([0.6, 0.3, 0.3, 1.0], np.float32)
    floor = model.find("geom", "floor")
    fr = (
        np.array([1.0, 0.005, 0.0001], np.float32)
        if floor.friction is None
        else np.asarray(floor.friction, np.float32).copy()
    )
    fr[0] = 1.0
    floor.friction = fr
    model.find("geom", "torso").density = 1000.0
    model.find("geom", "foot_back_left").density = 1000.0
    model.find("light", "light").diffuse = 0.7 * np.ones(3, np.float32)
    # The export exists only to be compiled: ``Physics.from_xml_path`` reads the XML and its assets
    # into the MjModel at once, so the directory is scoped to the call and removed on exit (SWM keeps
    # its ``TemporaryDirectory`` on the wrapper for its lifetime; a ``mkdtemp`` here leaked one
    # directory per ``sample_states``/``reference_env`` call -- review of 9188c2c).
    with tempfile.TemporaryDirectory(prefix="se_swm_quadruped_") as td:
        mjcf.export_with_assets(model, td, out_file_name="quadruped.xml")
        physics = quadruped.Physics.from_xml_path(os.path.join(td, "quadruped.xml"))
    task = quadruped.Move(desired_speed=speed, random=seed)
    env = control.Environment(physics, task, time_limit=20, control_timestep=0.02)
    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


class _SWMQuadrupedModel:
    """Mixin for the one trackcom cell whose model is not ``suite.load``'s: SWM's env through the
    ``_DMCModel._load_env`` hook, and SWM's ``render()`` default size. Nothing else is overridden
    (asserted by ``tests/unit/test_cpu3d.py``), so ``_bundle``/``sample_states``/``reference_env``
    stay the base class's and cannot drift from it."""

    default_size = 224

    @classmethod
    def _load_env(cls, seed: int):
        return _swm_quadruped_env(seed)


#: The quadruped's third residual, named as the other trackcom cells name theirs. Not an
#: approximation this cell introduced: no backend here draws reflections, and that is a statement of
#: the renderer's semantics on every trackcom cell (the others' floors reflect too; the quadruped's
#: camera is 4.5 m from a body over an 80 % floor, so it shows).
QUADRUPED_NOTE = (
    "(3) the floor material has reflectance 0.2, so dm_control draws the body's MIRROR IMAGE in the "
    "floor beside its shadow; MJWarp and cpu3d draw no reflection (the shadow is a 0.3 any-hit factor "
    "against a 2048 shadow map). Measured off the segmentation buffer at the registered default "
    "(camera 2, 224 px): the floor within the body's box and its mirror is 35 % of the frame and "
    "carries 57 % of the residual; the far checker 29 %, the skybox 6.5 %, the body 7 %, the "
    "ellipsoid torso 1.2 %. (4) physics.render draws three SITES that neither backend draws -- the "
    "two pupil spheres (material eye) and the red torso site -- up to ~50 of the 50,176 px of a 224 "
    "px frame (0.01-0.1 %), max 186/255, 0.5 % of the residual; the eye material is therefore not "
    "an appearance knob here. The cell draws SWM's default appearance (material self .6 .3 .3); the "
    "suite's own (.7 .5 .3) is the material.self knob."
)


VISUAL_FIDELITY: dict[str, dict] = {
    # quadruped at the registered SWM default (camera 2, 224 px). Three residuals, per surface off the
    # segmentation buffer (examples/results/step7/dmc-quadruped/findings.md S5): the floor near the
    # body -- the body's REFLECTION in the reflectance-0.2 floor and its shadow -- carries ~half of
    # it; the far checker and the one-tap skybox the rest; the ellipsoid itself is 1.2 % of the frame
    # at mean ~4/255. Thresholds by the family's margin convention (ceiling ~1.3x, floors ~0.93x).
    "DMCQuadrupedWalk": {
        "mean_abs": 3.52,
        "within8_frac": 0.897,
        "ssim": 0.955,
        "interior_exact_frac": 0.206,
        "interior_frac": 0.0284,
        "all_exact_frac": 0.171,
        "skybox_share": 0.148,
        "floor_share": 0.806,
        "draw_spread": {"mean_abs": [3.520, 3.527], "within8_frac": [0.8964, 0.8987], "ssim": [0.9536, 0.9557]},
        "mean_abs_max": 4.6,
        "within8_min": 0.85,
        "ssim_min": 0.90,
        "measured_at": _QUADRUPED_AT,
        "reference_versions": _AT_VERSIONS,
        "measured_with": _QUADRUPED_AT_MEASURED_WITH,
    },
}

DMCQuadrupedWalk = trackcom_cell(
    "DMCQuadrupedWalk",
    "quadruped",
    "walk",
    2,
    VISUAL_FIDELITY["DMCQuadrupedWalk"],
    note_suffix=QUADRUPED_NOTE,
    bases=(_SWMQuadrupedModel,),
)
