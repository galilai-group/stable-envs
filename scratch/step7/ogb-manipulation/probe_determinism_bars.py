"""Step 7 / W5 phase 0 -- render determinism, the hidden Scene bits, and the two bars.

Three things `probe_purity_fps.py` could only flag:

1. **Determinism of the official renderer itself.**  `N` renders of one unchanged
   `MjData`; how many pixels ever move, and by how much.  An official reference that is
   not bit-stable frame to frame bounds what any parity claim can mean.
2. **The Scene's two hidden bits.**  `SceneEnv.set_state` takes `button_state_0/1` as
   REQUIRED keyword arguments; they drive `geom_rgba` of the two buttons and `mat_rgba`
   of the drawer/window handles.  Flip them at fixed `qpos` and measure the pixel share.
3. **The bars**, measured properly: median over repeated blocks rather than one pass,
   one torch/BLAS thread, plus the JPEG-q90 one-core decode fps of the env's own frame.
"""
from __future__ import annotations

import argparse, io, json, os, platform, socket, statistics, time
from pathlib import Path

import numpy as np


def main():
    import gymnasium as gym
    import mujoco
    import stable_worldmodel  # noqa: F401
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", nargs="+", default=["swm/OGBCube-v0", "swm/OGBScene-v0"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--blocks", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for eid in a.envs:
        env = gym.make(eid); u = env.unwrapped
        env.reset(seed=3)
        m, d = u.model, u.data
        h, w = u._render_height, u._render_width
        rec = {"env_id": eid, "h": h, "w": w, "host": socket.gethostname(),
               "platform": platform.platform(), "mujoco": mujoco.__version__,
               "mujoco_gl": os.environ.get("MUJOCO_GL"),
               "offsamples_msaa": int(m.vis.quality.offsamples),
               "shadowsize": int(m.vis.quality.shadowsize)}

        # --- 1. determinism of N renders of one unchanged state --------------
        frames = np.stack([u.render().copy() for _ in range(32)])
        base = frames[0]
        ever = (frames != base[None]).any(0).any(-1)
        rec["determinism"] = {
            "n_renders": int(frames.shape[0]),
            "frac_pixels_ever_differing": float(ever.mean()),
            "max_abs_diff": int(np.abs(frames.astype(np.int16) - base[None].astype(np.int16)).max()),
            "n_distinct_frames": int(len({f.tobytes() for f in frames})),
        }

        # --- 2. hidden bits ---------------------------------------------------
        nb = getattr(u, "_num_buttons", 0)
        rec["num_hidden_button_bits"] = int(nb)
        if nb:
            qpos, qvel = d.qpos.copy(), d.qvel.copy()
            shots = {}
            for bits in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                u.set_state(qpos, qvel, **{f"button_state_{i}": bits[i] for i in range(nb)})
                shots[bits] = u.render().copy()
            ref = shots[(0, 0)]
            rec["button_bit_pixel_share"] = {
                f"{b[0]}{b[1]}": float((shots[b] != ref).any(-1).mean()) for b in shots}
            rec["button_bits_change_frame"] = bool(any(v > 0 for v in rec["button_bit_pixel_share"].values()))
            np.savez_compressed(out / f"button_bits_{eid.split('/')[-1]}.npz",
                                **{f"b{b[0]}{b[1]}": shots[b] for b in shots})
            u.set_state(qpos, qvel, **{f"button_state_{i}": 0 for i in range(nb)})

        # --- 3. the bars ------------------------------------------------------
        env.reset(seed=3); m, d = u.model, u.data
        for _ in range(15):
            u.render()
        blk = []
        for _ in range(a.blocks):
            t0 = time.perf_counter()
            for _ in range(a.reps):
                u.render()
            blk.append(a.reps / (time.perf_counter() - t0))
        rec["official_fps_render_only_blocks"] = blk
        rec["official_fps_render_only"] = statistics.median(blk)
        rec["official_fps_render_only_max"] = max(blk)

        frame = u.render().copy()
        buf = io.BytesIO(); Image.fromarray(frame).save(buf, format="JPEG", quality=90)
        blob = buf.getvalue()
        for _ in range(30):
            np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
        dblk = []
        for _ in range(a.blocks):
            t0 = time.perf_counter()
            for _ in range(300):
                np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
            dblk.append(300 / (time.perf_counter() - t0))
        rec["decode_fps_per_core_blocks"] = dblk
        rec["decode_fps_per_core"] = statistics.median(dblk)
        rec["jpeg_q90_bytes"] = len(blob)
        rec["bar_S_cpu_ge_5_abs_fps"] = 5.0 * rec["official_fps_render_only"]
        rec["bar_cores_cpu_ge_1_abs_fps"] = rec["decode_fps_per_core"]
        rec["binding_bar_abs_fps"] = max(rec["bar_S_cpu_ge_5_abs_fps"], rec["bar_cores_cpu_ge_1_abs_fps"])
        rec["binding_bar_ns_per_pixel"] = 1e9 / rec["binding_bar_abs_fps"] / (h * w)

        # --- 4. projected triangle size at this resolution --------------------
        rec["projected_tri"] = projected_tri(u, m, d, h, w)

        tag = eid.split("/")[-1].replace("-v0", "")
        (out / f"bars_{tag}.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(f"{eid} {h}x{w}: official {rec['official_fps_render_only']:.1f} f/s "
              f"decode {rec['decode_fps_per_core']:.0f} f/s  S>=5 bar {rec['bar_S_cpu_ge_5_abs_fps']:.0f} fps  "
              f"cores>=1 bar {rec['bar_cores_cpu_ge_1_abs_fps']:.0f} fps  "
              f"det: {rec['determinism']['frac_pixels_ever_differing']*100:.4f}% px move, "
              f"{rec['determinism']['n_distinct_frames']} distinct/32  "
              f"tri_px_median {rec['projected_tri'].get('median_edge_px')}", flush=True)
        env.close()


def projected_tri(u, m, d, h, w):
    """Median projected triangle edge, in pixels, for the VISIBLE mesh set.

    Transforms each visible mesh's vertices by its geom frame, projects with the
    registered camera's fovy, and reports the per-triangle bounding-box edge in pixels.
    This is the number the Step-6 Fetch CPU feasibility ticket reported as 0.50 px at
    480 px; it is what decides whether a rasteriser is setup-bound or fill-bound.
    """
    import mujoco
    cam = "front_pixels"
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    if cid < 0:
        return {"error": f"no camera {cam}"}
    fovy = float(m.cam_fovy[cid])
    f = (h / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
    cpos = np.array(d.cam_xpos[cid]); cmat = np.array(d.cam_xmat[cid]).reshape(3, 3)
    gg = [i for i, v in enumerate(u._scene_option.geomgroup) if v]
    edges, tris, areas = [], 0, []
    for i in range(m.ngeom):
        if int(m.geom_type[i]) != 7 or int(m.geom_group[i]) not in gg or m.geom_rgba[i][3] <= 0:
            continue
        mi = int(m.geom_dataid[i])
        va, nv = int(m.mesh_vertadr[mi]), int(m.mesh_vertnum[mi])
        fa, nf = int(m.mesh_faceadr[mi]), int(m.mesh_facenum[mi])
        verts = m.mesh_vert[va:va + nv].reshape(-1, 3)
        faces = m.mesh_face[fa:fa + nf].reshape(-1, 3)
        R = np.array(d.geom_xmat[i]).reshape(3, 3); p = np.array(d.geom_xpos[i])
        wv = verts @ R.T + p
        # camera frame: MuJoCo camera looks down -z of cam_xmat
        cv = (wv - cpos) @ cmat
        z = -cv[:, 2]
        ok = z > 1e-4
        if not ok.all():
            z = np.where(ok, z, np.nan)
        px = f * cv[:, 0] / z + w / 2.0
        py = -f * cv[:, 1] / z + h / 2.0
        P = np.stack([px, py], 1)
        t = P[faces]
        bb = np.nanmax(t, 1) - np.nanmin(t, 1)
        e = np.nanmax(bb, 1)
        e = e[np.isfinite(e)]
        edges.append(e)
        # signed area (for a fill estimate)
        ar = 0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                          - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1]))
        ar = ar[np.isfinite(ar)]
        areas.append(ar)
        tris += nf
    if not edges:
        return {"error": "no visible mesh geoms"}
    e = np.concatenate(edges); ar = np.concatenate(areas)
    return {"camera": cam, "fovy": fovy, "visible_tri": int(tris),
            "median_edge_px": float(np.median(e)), "mean_edge_px": float(e.mean()),
            "p90_edge_px": float(np.percentile(e, 90)), "max_edge_px": float(e.max()),
            "frac_edge_lt_1px": float((e < 1.0).mean()),
            "median_area_px2": float(np.median(ar)), "mean_area_px2": float(ar.mean()),
            "sum_area_px2": float(ar.sum()), "frame_px": int(h * w),
            "overdraw_if_all_front_facing": float(ar.sum() / (h * w))}


if __name__ == "__main__":
    main()
