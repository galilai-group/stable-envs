"""Model surgery that turns a Fetch MJCF into something MJWarp draws the same picture from.

Four edits, each forced by a MEASURED property of MJWarp rather than by taste:

1. **A model camera at the free camera's pose.** The official renderer drives an ``mjvCamera`` of
   type FREE (distance 2.5, azimuth 132, elevation -14, lookat (1.3, 0.75, 0.55)); MJWarp renders
   MODEL cameras only. The pose is not derived from those four numbers — it is read off MuJoCo's own
   ``mjvScene`` as the AVERAGE of the two GL eyes, which is what mono rendering uses.
2. **The target site becomes a geom.** ``mujoco_warp``'s render kernel has no site path at all
   (``grep site render.py`` is empty), and ``target0`` — the red goal sphere — is a site. It is 0.03%
   of the frame and 255/255 wrong where it lands.
3. **One material per geom.** MJWarp resolves a geom's colour as "material's rgba if the geom HAS a
   material, else the geom's own rgba"; MuJoCo's rule is "the geom's own rgba unless it is the
   compiler default (0.5,0.5,0.5,1), in which case the material's". Fetch trips this on 11 of 24
   geoms — ``robot0:arm_mat`` is white and is worn by both the white and the blue arm links — so
   MJWarp paints the whole arm white. Cloning one material per geom, carrying the RESOLVED colour and
   the original specular/shininess, fixes it without touching the shading path.
4. **The four translucent mocap markers are hidden.** MJWarp never reads alpha, so a marker at
   alpha 0.1 is drawn opaque saturated green.

The spec is edited before compiling only where it must be (a camera, a geom, N materials); everything
else is a numpy write on the compiled model, so geom order is preserved and every index is looked up
by name.
"""
from __future__ import annotations

import numpy as np
import mujoco

#: geom group nothing in a Fetch model uses; ``enabled_geom_groups`` excludes it.
HIDDEN_GROUP = 5
#: MuJoCo's defaults for a geom/site with no material (mjv_initGeom).
DEFAULT_SPECULAR, DEFAULT_SHININESS, DEFAULT_EMISSION = 0.5, 0.5, 0.0


def free_camera_pose(m, d, cam):
    """World (pos, forward, up) of an ``mjvCamera``, measured off MuJoCo's own scene.

    Mono rendering uses the AVERAGE of the two ``mjvGLCamera`` eyes, which are separated by
    ``vis.global_.ipd`` = 0.068 — picking eye 0 is a 34 mm lateral error that looks like nothing and
    moves pixels. Same finding as the OGBench port, re-measured here.
    """
    scn = mujoco.MjvScene(m, maxgeom=1000)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), mujoco.MjvPerturb(), cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scn)
    e0, e1 = scn.camera[0], scn.camera[1]
    pos = 0.5 * (np.array(e0.pos, np.float64) + np.array(e1.pos, np.float64))
    fwd = np.array(e0.forward, np.float64)
    up = np.array(e0.up, np.float64)
    return pos, fwd / np.linalg.norm(fwd), up / np.linalg.norm(up)


def quat_from_forward_up(fwd, up):
    """MuJoCo camera frame: +x right, +y up, -z forward. Returns (w,x,y,z)."""
    z = -np.asarray(fwd, np.float64)
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def effective_rgba(m, g):
    """MuJoCo's own resolution order: the geom's rgba wins unless it is the compiler default."""
    rgba = np.asarray(m.geom_rgba[g], np.float64)
    mid = int(m.geom_matid[g])
    if mid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        return np.asarray(m.mat_rgba[mid], np.float64)
    return rgba


def build(xml_path: str, cam_pos, cam_quat, fovy: float, target_size: float = 0.02,
          target_rgba=(1.0, 0.0, 0.0, 1.0)):
    """Compile the surgical model. Returns ``(model, info)`` with the name->id maps a renderer needs."""
    spec = mujoco.MjSpec.from_file(xml_path)
    base = spec.compile()                              # to read the RESOLVED colours off
    n = base.ngeom + 1                                 # +1 for the target geom we are about to add

    for i in range(n):
        spec.add_material(name=f"se:mat{i}")
    cam = spec.worldbody.add_camera()
    cam.name = "se:cam"
    cam.pos = np.asarray(cam_pos, np.float64)
    cam.quat = np.asarray(cam_quat, np.float64)
    cam.fovy = float(fovy)
    tg = spec.worldbody.add_geom()
    tg.name = "se:target"
    tg.type = mujoco.mjtGeom.mjGEOM_SPHERE
    tg.size = np.array([target_size, target_size, target_size])
    tg.rgba = np.asarray(target_rgba, np.float32)
    tg.contype, tg.conaffinity = 0, 0
    tg.mass = 0.0
    m = spec.compile()

    name = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    bname = lambda i: mujoco.mj_id2name(base, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    # map every geom of the surgical model back to the base model. Names are unique where they exist;
    # the unnamed ones (the four mocap markers, the table box in some scenes) are matched by their
    # order inside their body, which is what the compiler preserves.
    b_by_body: dict[int, list[int]] = {}
    for g in range(base.ngeom):
        b_by_body.setdefault(int(base.geom_bodyid[g]), []).append(g)
    s_by_body: dict[int, list[int]] = {}
    for g in range(m.ngeom):
        if name(g) == "se:target":
            continue
        s_by_body.setdefault(int(m.geom_bodyid[g]), []).append(g)
    back = {}
    for body, gs in s_by_body.items():
        bs = b_by_body[body]
        assert len(gs) == len(bs), f"geom count changed in body {body}: {len(gs)} vs {len(bs)}"
        for a, b in zip(gs, bs):
            assert name(a) == bname(b), f"geom {a} is {name(a)!r}, base {b} is {bname(b)!r}"
            back[a] = b

    target = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "se:target")
    mat0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "se:mat0")
    for g in range(m.ngeom):
        k = mat0 + g
        if g == target:
            rgba = np.asarray(target_rgba, np.float64)
            spec_, shin, emis = DEFAULT_SPECULAR, DEFAULT_SHININESS, DEFAULT_EMISSION
        else:
            b = back[g]
            rgba = effective_rgba(base, b)
            mid = int(base.geom_matid[b])
            if mid >= 0:
                spec_ = float(base.mat_specular[mid])
                shin = float(base.mat_shininess[mid])
                emis = float(base.mat_emission[mid])
            else:
                spec_, shin, emis = DEFAULT_SPECULAR, DEFAULT_SHININESS, DEFAULT_EMISSION
        m.mat_rgba[k] = rgba
        m.mat_specular[k] = spec_
        m.mat_shininess[k] = shin
        m.mat_emission[k] = emis
        m.mat_reflectance[k] = 0.0
        m.mat_texid[k] = -1
        m.geom_matid[g] = k
        m.geom_rgba[g] = rgba

    # EVERY site is hidden, and that is not tidying. ``mujoco.Renderer`` draws sites in groups 0-2 and
    # MJWarp draws none at all, so any site left visible is in the plate and in nothing else. Measured:
    # with the movers hidden, ``object0``'s site — a red sphere that lives INSIDE the block and is
    # invisible in the official frame — became visible and made the "static" plate depend on qpos[15:17]
    # (1123 pixels over 8 states). ``target0``'s site is worse: it stays at its model position while the
    # env moves the goal by writing ``model.site_pos``, so leaving it drew a SECOND red sphere at
    # (0.8, 0.75, 0.5). The one visible site is reproduced as ``se:target`` above.
    m.site_group[:] = HIDDEN_GROUP

    # This model is never simulated — it exists to be rendered — and MJWarp's ``put_model`` refuses a
    # non-zero collision margin under MULTICCD. Margin is a contact parameter that reaches no pixel,
    # and the L1 rung of the ladder (this model, MuJoCo, against the official frame) is what proves it.
    m.geom_margin[:] = 0.0
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0

    info = dict(
        target_geom=target, mat0=mat0, back=back,
        cam=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "se:cam"),
        translucent=[g for g in range(m.ngeom)
                     if g != target and float(effective_rgba(base, back[g])[3]) < 1.0],
        geom_names={g: (name(g) or f"geom{g}") for g in range(m.ngeom)},
        target_site_body=int(base.site_bodyid[
            mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_SITE, "target0")]),
    )
    return m, info
