"""Step 6G cpu-3d, Phase 1: model census + same-host official renderer for the four Fetch scenes.

Runs on an Oscar BATCH node in se-s3b-fetch311 (gymnasium_robotics, mujoco 3.11 -- the Step 3B pin).
No GPU: MJWarp is never constructed. What is read is the surgical model stable_envs.envs.fetch
compiles (one material per geom, se:cam, se:target), the mesh table, and MuJoCo's own renderer on
this host for the official-fps denominator and the plate-bake cost. Writes JSON under --out.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
GEOM = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder",
        6: "box", 7: "mesh", 8: "sdf"}


def census(scene: str, states_dir: Path, out_dir: Path) -> dict:
    import mujoco
    import stable_envs  # noqa: F401  (registers; fetch imports need warp on the path only by find_spec)
    from stable_envs.envs import fetch as F
    b = F._build_bundle(scene)
    m = b.model
    info = b.info
    r: dict = {"scene": scene, "mujoco": mujoco.__version__, "nq": int(m.nq), "ngeom": int(m.ngeom),
               "nmesh": int(m.nmesh), "ntex": int(m.ntex), "nmat": int(m.nmat), "nlight": int(m.nlight),
               "ncam": int(m.ncam), "nsite": int(m.nsite)}
    movers = set(int(x) for x in b.movers)
    geoms, tri_mover, tri_static, meshes = [], 0, 0, {}
    for g in range(m.ngeom):
        t = int(m.geom_type[g])
        vis = int(m.geom_group[g]) in (0, 1, 2)
        ntri = 0
        if t == 7:
            did = int(m.geom_dataid[g])
            ntri = int(m.mesh_facenum[did])
            meshes[did] = {"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, did),
                           "verts": int(m.mesh_vertnum[did]), "faces": ntri}
        if vis:
            if g in movers:
                tri_mover += ntri
            else:
                tri_static += ntri
        geoms.append({"id": g, "name": info["names"].get(g), "type": GEOM[t], "visible": vis,
                      "mover": g in movers, "body": int(m.geom_bodyid[g]), "tris": ntri,
                      "size": [round(float(x), 4) for x in m.geom_size[g]],
                      "rgba": [round(float(x), 3) for x in m.geom_rgba[g]]})
    r["geoms"] = geoms
    r["meshes"] = meshes
    vis = [x for x in geoms if x["visible"]]
    by = {}
    for x in vis:
        k = ("mover" if x["mover"] else "static") + ":" + x["type"]
        by[k] = by.get(k, 0) + 1
    r["visible_geoms_by_type"] = by
    r["triangles_visible_movers"] = tri_mover
    r["triangles_visible_static"] = tri_static
    r["n_movers"] = len([x for x in vis if x["mover"]])
    r["n_static"] = len([x for x in vis if not x["mover"]])
    cam = info["cam"]
    r["camera"] = {"id": cam, "fovy": float(m.cam_fovy[cam]), "pos": [float(x) for x in m.cam_pos[cam]],
                   "quat": [float(x) for x in m.cam_quat[cam]], "mode": int(m.cam_mode[cam]),
                   "defaults": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in b.cam_defaults.items()}}
    r["lights"] = [{"id": i, "mode": int(m.light_mode[i]), "castshadow": bool(m.light_castshadow[i]),
                    "type": int(m.light_type[i]) if hasattr(m, "light_type") else None,
                    "directional": bool(m.light_directional[i]) if hasattr(m, "light_directional") else None,
                    "pos": [float(x) for x in m.light_pos[i]], "dir": [float(x) for x in m.light_dir[i]],
                    "attenuation": [float(x) for x in m.light_attenuation[i]], "cutoff": float(m.light_cutoff[i]),
                    "exponent": float(m.light_exponent[i]), "ambient": [float(x) for x in m.light_ambient[i]],
                    "diffuse": [float(x) for x in m.light_diffuse[i]], "specular": [float(x) for x in m.light_specular[i]]}
                   for i in range(m.nlight)]
    hl = m.vis.headlight
    r["headlight"] = {"active": int(hl.active), "ambient": [float(x) for x in hl.ambient],
                      "diffuse": [float(x) for x in hl.diffuse], "specular": [float(x) for x in hl.specular]}
    r["vis"] = {"znear": float(m.vis.map.znear), "zfar": float(m.vis.map.zfar), "extent": float(m.stat.extent),
                "offsamples": int(m.vis.quality.offsamples), "shadowsize": int(m.vis.quality.shadowsize)}
    r["textures"] = [{"id": i, "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, i), "type": int(m.tex_type[i]),
                      "w": int(m.tex_width[i]), "h": int(m.tex_height[i])} for i in range(m.ntex)]
    mat_used = sorted(set(int(m.geom_matid[g]) for g in range(m.ngeom) if int(m.geom_group[g]) in (0, 1, 2)))
    r["materials_used"] = [{"id": i, "rgba": [round(float(x), 3) for x in m.mat_rgba[i]],
                            "reflectance": float(m.mat_reflectance[i]), "specular": float(m.mat_specular[i]),
                            "shininess": float(m.mat_shininess[i]), "emission": float(m.mat_emission[i]),
                            "texid": [int(x) for x in np.atleast_1d(m.mat_texid[i])]} for i in mat_used]
    # FK on this host
    d = mujoco.MjData(m)
    cname = f"Fetch{scene}"
    st = np.load(states_dir / f"{cname}.npz")["states"].astype(np.float64)
    d.qpos[:] = st[0, :m.nq]
    for fn, name in ((mujoco.mj_kinematics, "mj_kinematics"), (mujoco.mj_forward, "mj_forward")):
        fn(m, d); t0 = time.perf_counter()
        for _ in range(500):
            fn(m, d)
        r[name + "_us"] = (time.perf_counter() - t0) / 500 * 1e6
    # mover world extent over the corpus (for screen-bbox reasoning)
    ext = []
    for row in st:
        d.qpos[:] = row[:m.nq]; mujoco.mj_kinematics(m, d)
        xp = np.asarray(d.geom_xpos)[sorted(movers)]
        ext.append(np.concatenate([xp.min(0), xp.max(0)]))
    ext = np.asarray(ext)
    r["mover_aabb_extent_median_m"] = np.median(ext[:, 3:] - ext[:, :3], 0).round(3).tolist()
    return r


def official_and_plate(scene: str, states_dir: Path, out_dir: Path) -> dict:
    """The official renderer at 480 px on THIS host, and the plate bake through mujoco.Renderer."""
    import mujoco
    from stable_envs.envs import fetch as F
    cname = f"Fetch{scene}"
    cls = F.REGISTRY[cname] if hasattr(F, "REGISTRY") else __import__("stable_envs").core.env.REGISTRY[cname]
    st = np.load(states_dir / f"{cname}.npz")["states"].astype(np.float64)
    rec = {}
    try:
        from OpenGL import GL
    except Exception:
        GL = None
    ref = cls.reference_env(480)
    fr = cls.reference_render(ref, st[:2])
    if GL is not None:
        try:
            rec["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
        except Exception as e:
            rec["gl_renderer"] = f"unreadable: {e}"
    n = 30
    t0 = time.perf_counter()
    cls.reference_render(ref, st[:n])
    dt = (time.perf_counter() - t0) / n
    rec["official_fps_from_state_480"] = 1.0 / dt
    # render-only (the state already set): the warm number Step 3 quotes as official_cpu
    u = ref.unwrapped
    for _ in range(5):
        u.render()
    t0 = time.perf_counter()
    for _ in range(n):
        u.render()
    rec["official_fps_render_only_480"] = n / (time.perf_counter() - t0)
    np.save(out_dir / f"{cname}_official_480.npy", fr[0])
    # decode bar on the real frame
    import io
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(fr[0]).save(buf, format="JPEG", quality=90); blob = buf.getvalue()
    nn = 200
    for _ in range(20):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    t0 = time.perf_counter()
    for _ in range(nn):
        np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
    rec["decode_fps_per_core_480"] = nn / (time.perf_counter() - t0)
    rec["jpeg_bytes_480"] = len(blob)
    # plate bake on this host (software GL), and the mover pixel share from official - plate
    try:
        env = cls(batch=1, device="cpu")
        t0 = time.perf_counter()
        plate = env._bake_plate(480, None).cpu().numpy()
        rec["plate_bake_s_480"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        env._bake_plate(480, None)
        rec["plate_bake_s_480_warm"] = time.perf_counter() - t0
        diff = np.abs(fr.astype(np.int16) - plate[None].astype(np.int16)).max(-1) > 0
        rec["mover_px_share_official_minus_plate"] = float(diff.mean())
        ys, xs = np.nonzero(diff[0])
        rec["mover_bbox_share_state0"] = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) / 480 ** 2)
        rec["plate_vs_official_static_exact"] = float((~diff).mean())
    except Exception as e:
        rec["plate_error"] = f"{type(e).__name__}: {e}"[:400]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--states-dir", type=Path, default=ROOT / "examples/results/step4/states")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    res = {"host": host, "cpu": platform.processor() or platform.machine(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "mujoco_gl": os.environ.get("MUJOCO_GL"), "scenes": {}}
    for scene in ("Reach", "Push", "Slide", "PickAndPlace"):
        r = census(scene, a.states_dir, a.out)
        try:
            r["official"] = official_and_plate(scene, a.states_dir, a.out)
        except Exception as e:
            r["official"] = {"error": f"{type(e).__name__}: {e}"[:400]}
        res["scenes"][scene] = r
        print(f"Fetch{scene}: movers={r['n_movers']} static={r['n_static']} tris mover/static={r['triangles_visible_movers']}/{r['triangles_visible_static']} "
              f"types={r['visible_geoms_by_type']} fk={r['mj_kinematics_us']:.1f}us official={r['official']}", flush=True)
        (a.out / f"census_fetch_{host}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
