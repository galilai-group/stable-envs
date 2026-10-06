from __future__ import annotations

import numpy as np

from ._aligned import DMCAlignedSemantics
from ._cells import _AT_VERSIONS, trackcom_cell

_QUADRUPED_AT = (
    "224 px, camera 2 ('y', trackcom), 64 states x 3 independent draws from sample_states "
    "(certified model and default appearance: material self rgba .6 .3 .3, foot_back_left "
    "density 1000), RTX 3090 / driver 575.57.08 / MUJOCO_GL=egl, mujoco 3.11.0 + "
    "mujoco-warp 3.11.0, against dm_control's physics.render at camera 2"
)
_QUADRUPED_AT_MEASURED_WITH = {"mujoco_warp": "3.11.0"}


def _transcribed_quadruped_env(seed: int):
    """Build the registered quadruped model and walk task with scaled actions.

    Apply registered material, density, friction, and lighting values before
    compilation; vanilla suite.load does not supply this configuration.
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
    # Compilation consumes the XML and assets before the temporary directory is removed.
    with tempfile.TemporaryDirectory(prefix="se_quadruped_mjcf_") as td:
        mjcf.export_with_assets(model, td, out_file_name="quadruped.xml")
        physics = quadruped.Physics.from_xml_path(os.path.join(td, "quadruped.xml"))
    task = quadruped.Move(desired_speed=speed, random=seed)
    env = control.Environment(physics, task, time_limit=20, control_timestep=0.02)
    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


class _TranscribedQuadrupedModel:
    """Supply the registered model through _load_env, retaining shared rendering and sampling."""

    default_size = 224

    @classmethod
    def _load_env(cls, seed: int):
        return _transcribed_quadruped_env(seed)


QUADRUPED_NOTE = (
    "(3) the floor material has reflectance 0.2, so dm_control draws the body's MIRROR IMAGE in the "
    "floor beside its shadow; MJWarp and cpu3d draw no reflection (the shadow is a 0.3 any-hit factor "
    "against a 2048 shadow map). Measured off the segmentation buffer at the registered default "
    "(camera 2, 224 px): the floor within the body's box and its mirror is 35 % of the frame and "
    "carries 57 % of the residual; the far checker 29 %, the skybox 6.5 %, the body 7 %, the "
    "ellipsoid torso 1.2 %. (4) physics.render draws three SITES that neither backend draws -- the "
    "two pupil spheres (material eye) and the red torso site -- up to ~50 of the 50,176 px of a 224 "
    "px frame (0.01-0.1 %), max 186/255, 0.5 % of the residual; the eye material is therefore not "
    "an appearance knob here. The cell draws the registered default appearance (material self .6 .3 .3); the "
    "suite's own (.7 .5 .3) is the material.self knob."
)


VISUAL_FIDELITY: dict[str, dict] = {
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
    bases=(DMCAlignedSemantics, _TranscribedQuadrupedModel),
)
