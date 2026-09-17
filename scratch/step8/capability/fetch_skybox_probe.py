"""Fetch skybox probe: is the skybox in the MuJoCo-baked plate, and does an in-place texel
substitution + mjr_uploadTexture reproduce what swm's background.color produces?

swm's own implementation (stable_worldmodel/envs/gymnasium_robotics/fetch.py:400-430) does NOT
recompile: it writes the skybox texture's texels flat to (bg*255).astype(uint8) and calls
mjr_uploadTexture on the viewer context. The "full recompile" arm of this probe therefore compiles
a model whose MJCF skybox texture is rgb1=rgb2=bg (a gradient between equal colours is a flat
texture), which is the compiled equivalent, and diffs it against the in-place substitution.
"""
import json, os, sys, hashlib
import numpy as np, mujoco

ASSETS = "/Users/samib/se8-fetch-venv/lib/python3.12/site-packages/gymnasium_robotics/envs/assets/fetch"
SHARED = os.path.join(ASSETS, "shared.xml")
CAM = dict(distance=2.5, azimuth=132.0, elevation=-14.0, lookat=[1.3, 0.75, 0.55])
RES = 480
PROBES = [(0.9, 0.2, 0.2), (0.1, 0.1, 0.1), (0.05, 0.6, 0.95), (1.0, 1.0, 1.0)]


def quantise(v):
    return np.clip((np.asarray(v, np.float64) * 255.0).astype(np.int64), 0, 255).astype(np.uint8)


def skybox_id(m):
    for t in range(m.ntex):
        if m.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
            return t
    return -1


def texview(m, t):
    adr = int(m.tex_adr[t]); nch = int(m.tex_nchannel[t])
    n = int(m.tex_width[t]) * int(m.tex_height[t]) * nch
    return np.asarray(m.tex_data[adr:adr + n]).reshape(-1, nch)


def shot(m, d, r):
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera=-1)
    c = r.scene.camera  # free camera set below via update_scene default? use explicit MjvCamera
    return r.render().copy()


def render(m, d, r, cam):
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera=cam)
    return r.render().copy()


def make_cam():
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.distance = CAM["distance"]; c.azimuth = CAM["azimuth"]; c.elevation = CAM["elevation"]
    c.lookat[:] = CAM["lookat"]
    return c


def build(path, sky=None):
    """Compile the scene; if `sky` is given, rewrite the MJCF skybox rgb1/rgb2 first (recompile arm)."""
    xml = open(path).read()
    if sky is not None:
        shared = open(SHARED).read()
        old = '<texture type="skybox" builtin="gradient" rgb1="0.44 0.85 0.56" rgb2="0.46 0.87 0.58" width="32" height="32"></texture>'
        assert old in shared
        s = " ".join(repr(float(x)) for x in sky)
        new = f'<texture type="skybox" builtin="gradient" rgb1="{s}" rgb2="{s}" width="32" height="32"></texture>'
        tmp = os.path.join(ASSETS, "_probe_shared.xml")
        open(tmp, "w").write(shared.replace(old, new))
        xml = xml.replace('"shared.xml"', '"_probe_shared.xml"')
        m = mujoco.MjModel.from_xml_string(xml, {})
        # from_xml_string cannot resolve includes without assets; write a sibling file instead
    return m


def main():
    out = {"probe": "fetch_skybox_in_baked_plate", "res": RES,
           "mujoco": mujoco.__version__, "numpy": np.__version__, "scenes": {}}
    for scene in ("push", "pick_and_place", "reach", "slide"):
        path = os.path.join(ASSETS, f"{scene}.xml")
        m = mujoco.MjModel.from_xml_path(path)
        d = mujoco.MjData(m)
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), RES)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), RES)
        t = skybox_id(m)
        r = mujoco.Renderer(m, height=RES, width=RES)
        cam = make_cam()
        base = render(m, d, r, cam)
        rec = {"skybox_tex_id": int(t), "tex_shape": [int(m.tex_width[t]), int(m.tex_height[t]),
                                                      int(m.tex_nchannel[t])],
               "plate_default_sha256": hashlib.sha256(base.tobytes()).hexdigest(), "values": []}
        for bg in PROBES:
            # --- arm 1: in-place texel substitution + mjr_uploadTexture on the plate context
            v = texview(m, t)
            v[:, :3] = quantise(bg)[:3]
            r._gl_context.make_current()
            mujoco.mjr_uploadTexture(m, r._mjr_context, t)
            sub = render(m, d, r, cam)
            sub_tex = v[:, :3].copy()
            # --- arm 2: recompiled model with the MJCF skybox flattened to the same colour
            m2 = mujoco.MjModel.from_xml_path(path)
            g = m2.spec if False else None
            # rewrite through MjSpec so include resolution keeps working
            spec = mujoco.MjSpec.from_file(path)
            tx = [x for x in spec.textures if x.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX][0]
            tx.rgb1 = list(map(float, bg)); tx.rgb2 = list(map(float, bg))
            m2 = spec.compile()
            m2.vis.global_.offwidth = max(int(m2.vis.global_.offwidth), RES)
            m2.vis.global_.offheight = max(int(m2.vis.global_.offheight), RES)
            d2 = mujoco.MjData(m2)
            r2 = mujoco.Renderer(m2, height=RES, width=RES)
            recomp = render(m2, d2, r2, make_cam())
            t2 = skybox_id(m2)
            rec_tex = texview(m2, t2)[:, :3]
            px_vs_default = int((sub != base).any(-1).sum())
            px_vs_recompile = int((sub != recomp).any(-1).sum())
            rec["values"].append({
                "bg": list(bg),
                "tex_bytes_differing_sub_vs_recompile": int((sub_tex != rec_tex).sum()),
                "tex_total_bytes": int(sub_tex.size),
                "plate_px_moved_vs_default": px_vs_default,
                "plate_px_total": RES * RES,
                "plate_px_differing_sub_vs_recompile": px_vs_recompile,
                "plate_maxdiff_sub_vs_recompile": int(np.abs(sub.astype(int) - recomp.astype(int)).max()),
            })
            r2.close()
            # restore default texels for the next probe value
            m3 = mujoco.MjModel.from_xml_path(path)
            v[:, :3] = texview(m3, skybox_id(m3))[:, :3]
            r._gl_context.make_current()
            mujoco.mjr_uploadTexture(m, r._mjr_context, t)
        restored = render(m, d, r, cam)
        rec["plate_restored_sha256"] = hashlib.sha256(restored.tobytes()).hexdigest()
        rec["default_restore_is_byte_identical"] = bool((restored == base).all())
        r.close()
        out["scenes"][scene] = rec
        print(scene, json.dumps(rec["values"][0]), rec["default_restore_is_byte_identical"], flush=True)
    print(json.dumps(out, indent=2))
    open(sys.argv[1], "w").write(json.dumps(out, indent=2) + "\n")


main()
