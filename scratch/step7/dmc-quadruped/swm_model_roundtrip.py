"""Does the SWM wrapper's model equal the suite's quadruped-walk model?

``stable_worldmodel.envs.dmcontrol.quadruped.QuadrupedDMControlWrapper`` does not call ``suite.load``:
it builds ``quadruped.make_model(floor_size=20*0.5)``, round-trips the XML through ``dm_control.mjcf``
(``from_xml_string`` -> ``modify_mjcf_model`` -> ``export_with_assets``) and compiles
``quadruped.Physics.from_xml_path`` with ``quadruped.Move(desired_speed=0.5)``. This replicates those
lines (stable_worldmodel itself is not importable in this venv) and diffs the compiled ``MjModel``
against ``suite.load("quadruped", "walk")`` -- geometry, cameras, lights, materials, texture bytes --
once with the mjcf round trip alone and once with the wrapper's DEFAULT variation values applied
(``modify_mjcf_model`` with every ``init_value``), which is what a fresh SWM env renders.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")


def swm_physics(apply_defaults: bool):
    from dm_control import mjcf
    from dm_control.suite import quadruped
    xml = quadruped.make_model(floor_size=20 * 0.5)
    xml = xml.replace(b'file="./common/', b'file="common/')
    suite_dir = os.path.dirname(quadruped.__file__)
    model = mjcf.from_xml_string(xml, model_dir=suite_dir, assets=quadruped.common.ASSETS or {})
    if apply_defaults:                                   # modify_mjcf_model with init_values
        grid = model.find("texture", "grid")
        grid.rgb1 = np.array([0.1, 0.2, 0.3])
        grid.rgb2 = np.array([0.2, 0.3, 0.4])
        model.find("material", "self").rgba = np.array([0.6, 0.3, 0.3, 1.0], np.float32)
        floor = model.find("geom", "floor")
        fr = np.array([1.0, 0.005, 0.0001], np.float32) if floor.friction is None else np.asarray(floor.friction, np.float32).copy()
        fr[0] = 1.0
        floor.friction = fr
        model.find("geom", "torso").density = 1000.0
        model.find("geom", "foot_back_left").density = 1000.0
        model.find("light", "light").diffuse = 0.7 * np.ones(3, np.float32)
    td = tempfile.TemporaryDirectory()
    mjcf.export_with_assets(model, td.name, out_file_name="quadruped.xml")
    physics = quadruped.Physics.from_xml_path(os.path.join(td.name, "quadruped.xml"))
    return physics, td


def digest(m) -> dict:
    import mujoco
    h = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]  # noqa: E731
    names = lambda obj, n: [mujoco.mj_id2name(m, obj, i) for i in range(n)]  # noqa: E731
    return {
        "ngeom": int(m.ngeom), "nsite": int(m.nsite), "ncam": int(m.ncam), "nlight": int(m.nlight),
        "nmat": int(m.nmat), "ntex": int(m.ntex), "nhfield": int(m.nhfield), "nmesh": int(m.nmesh),
        "nq": int(m.nq), "nbody": int(m.nbody),
        "geom_names": names(mujoco.mjtObj.mjOBJ_GEOM, m.ngeom),
        "geom_type": h(np.asarray(m.geom_type)), "geom_size": h(np.asarray(m.geom_size)),
        "geom_pos": h(np.asarray(m.geom_pos)), "geom_quat": h(np.asarray(m.geom_quat)),
        "geom_group": h(np.asarray(m.geom_group)), "geom_matid": h(np.asarray(m.geom_matid)),
        "geom_rgba": h(np.asarray(m.geom_rgba)), "geom_bodyid": h(np.asarray(m.geom_bodyid)),
        "body_pos": h(np.asarray(m.body_pos)), "body_quat": h(np.asarray(m.body_quat)),
        "body_mass": h(np.asarray(m.body_mass)), "body_ipos": h(np.asarray(m.body_ipos)),
        "site_names": names(mujoco.mjtObj.mjOBJ_SITE, m.nsite), "site_group": h(np.asarray(m.site_group)),
        "site_rgba": h(np.asarray(m.site_rgba)), "site_matid": h(np.asarray(m.site_matid)),
        "cam_names": names(mujoco.mjtObj.mjOBJ_CAMERA, m.ncam), "cam_mode": [int(x) for x in m.cam_mode],
        "cam_pos": h(np.asarray(m.cam_pos)), "cam_quat": h(np.asarray(m.cam_quat)), "cam_fovy": [float(x) for x in m.cam_fovy],
        "light_mode": [int(x) for x in m.light_mode], "light_pos": h(np.asarray(m.light_pos)),
        "light_diffuse": np.asarray(m.light_diffuse).round(4).tolist(), "light_castshadow": [int(x) for x in m.light_castshadow],
        "mat_names": names(mujoco.mjtObj.mjOBJ_MATERIAL, m.nmat), "mat_rgba": np.asarray(m.mat_rgba).round(4).tolist(),
        "mat_texid": h(np.asarray(m.mat_texid)), "mat_texrepeat": h(np.asarray(m.mat_texrepeat)),
        "mat_reflectance": [float(x) for x in m.mat_reflectance],
        "tex_data": h(np.asarray(m.tex_data)), "tex_wh": [[int(w), int(hh)] for w, hh in zip(m.tex_width, m.tex_height)],
        "qpos0": h(np.asarray(m.qpos0)), "opt_gravity": [float(x) for x in m.opt.gravity],
        "vis_offwh": [int(m.vis.global_.offwidth), int(m.vis.global_.offheight)],
    }


def diff(a: dict, b: dict) -> dict:
    return {k: [a[k], b[k]] for k in a if a[k] != b[k]}


def main():
    import mujoco
    from dm_control import suite
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    ref = suite.load("quadruped", "walk", task_kwargs={"random": 0})
    d_suite = digest(ref.physics.model.ptr)
    p1, td1 = swm_physics(apply_defaults=False)
    d_rt = digest(p1.model.ptr)
    p2, td2 = swm_physics(apply_defaults=True)
    d_def = digest(p2.model.ptr)
    res = {"mujoco": mujoco.__version__,
           "suite_walk_vs_swm_roundtrip_no_variation": diff(d_suite, d_rt),
           "suite_walk_vs_swm_with_default_variation_values": diff(d_suite, d_def),
           "suite_digest": d_suite}
    # the frame the two defaults produce, same qpos, camera 2, 84 px
    q = np.asarray(ref.physics.data.qpos, np.float64).copy()
    frames = {}
    for name, ph in (("suite", ref.physics), ("swm_default", p2)):
        with ph.reset_context():
            ph.data.qpos[:] = q
            ph.data.qvel[:] = 0
        frames[name] = ph.render(height=84, width=84, camera_id=2).copy()
    d = np.abs(frames["suite"].astype(np.int16) - frames["swm_default"].astype(np.int16))
    res["frame_suite_vs_swm_default_cam2_84px"] = {"mean_abs": float(d.mean()), "exact_frac": float((d == 0).all(-1).mean()),
                                                   "max": int(d.max())}
    json.dump(res, open(os.path.join(out, "swm_model_roundtrip.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "suite_digest"}, indent=1))


if __name__ == "__main__":
    main()
