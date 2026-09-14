"""CPU3D-FETCH-1: dump the ACTUAL Fetch mover mesh set, per frozen state, in screen space.

For each of the first N frozen states of a Fetch scene: MuJoCo FK on the surgical model, every visible
mover mesh's triangles transformed to world, then to the ``se:cam`` camera's pixel space at 480 px
(the same frustum MJWarp and the official renderer use), with per-vertex normals rotated to world.
Written as one flat float32 binary the span rasteriser reads: per state, ``ntri`` triangles of
``3 x (x_px, y_px, depth, nx, ny, nz)``. Boxes/spheres (the fingers, block, goal) are not meshes and
are left out -- they are 3 primitives against 31,720 triangles.
"""
import argparse, json, math, struct
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser(); ap.add_argument("--scene", default="Reach"); ap.add_argument("--n", type=int, default=16)
ap.add_argument("--res", type=int, default=480); ap.add_argument("--out", required=True)
a = ap.parse_args()
import mujoco
import stable_envs  # noqa: F401
from stable_envs.envs import fetch as F
b = F._build_bundle(a.scene)
m, info = b.model, b.info
d = mujoco.MjData(m)
st = np.load(ROOT / f"examples/results/step4/states/Fetch{a.scene}.npz")["states"][:a.n].astype(np.float64)
cam = info["cam"]
movers = [g for g in b.movers if int(m.geom_type[g]) == 7]
meshes = {}
for g in movers:
    did = int(m.geom_dataid[g])
    va, vn = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did]); fa, fn = int(m.mesh_faceadr[did]), int(m.mesh_facenum[did])
    vert = np.asarray(m.mesh_vert[va:va + vn], np.float64).reshape(-1, 3)
    face = np.asarray(m.mesh_face[fa:fa + fn], np.int64).reshape(-1, 3)
    na, nn = int(m.mesh_normaladr[did]), int(m.mesh_normalnum[did])
    nrm = np.asarray(m.mesh_normal[na:na + nn], np.float64).reshape(-1, 3)
    fnrm = np.asarray(m.mesh_facenormal[fa:fa + fn], np.int64).reshape(-1, 3)
    meshes[g] = (vert, face, nrm, fnrm)
ntri = sum(len(v[1]) for v in meshes.values())
f = math.tan(0.5 * math.radians(float(m.cam_fovy[cam])))
out = open(a.out, "wb"); out.write(struct.pack("iii", a.n, ntri, a.res))
stats = []
for i in range(a.n):
    d.qpos[:] = st[i, :m.nq]; d.mocap_pos[info["goal_mocap"]] = st[i, m.nq:m.nq + 3]
    mujoco.mj_kinematics(m, d); mujoco.mj_camlight(m, d)
    cp = np.asarray(d.cam_xpos[cam]); cR = np.asarray(d.cam_xmat[cam]).reshape(3, 3)
    tris = []
    for g, (vert, face, nrm, fnrm) in meshes.items():
        R = np.asarray(d.geom_xmat[g]).reshape(3, 3); p = np.asarray(d.geom_xpos[g])
        vw = vert @ R.T + p; nw = nrm @ R.T
        loc = (vw - cp) @ cR                       # camera coords: right, up, back
        depth = -loc[:, 2]
        px = (loc[:, 0] / depth / f + 1) * 0.5 * a.res - 0.5
        py = (-loc[:, 1] / depth / f + 1) * 0.5 * a.res - 0.5
        v6 = np.stack([px, py, depth], 1)
        tri = np.concatenate([v6[face], nw[fnrm]], -1)          # (F,3,6)
        tris.append(tri)
    T = np.concatenate(tris).astype(np.float32)
    out.write(T.tobytes())
    inside = ((T[..., 0] >= 0) & (T[..., 0] < a.res) & (T[..., 1] >= 0) & (T[..., 1] < a.res)).any(1)
    e = np.linalg.norm(T[:, 1, :2] - T[:, 0, :2], axis=1)
    stats.append({"tris_in_view": int(inside.sum()), "median_edge_px": float(np.median(e[inside])) if inside.any() else None})
out.close()
json.dump({"scene": a.scene, "n": a.n, "ntri": ntri, "meshes": len(meshes), "res": a.res, "per_state": stats}, open(a.out + ".json", "w"), indent=1)
print(a.scene, "ntri", ntri, "meshes", len(meshes), stats[0])
