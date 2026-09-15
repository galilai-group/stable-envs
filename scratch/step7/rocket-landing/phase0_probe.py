"""Step 7 / W6 rocket-landing — Phase 0 feasibility gate on REAL frames.

Runs the four gates of docs/research/when_does_this_pay_off.md against swm/PFRocketLanding-v0
(PyFlyt 0.29 / PyBullet 3.2.7 TinyRenderer, 480x480) and writes numbers, frames and previews under
examples/results/step7/rocket-landing/phase0/. One process: PyFlyt's numba JIT (~57 s) is paid once.

Usage: /Users/samib/se7-rocket-venv/bin/python scratch/step7/rocket-landing/phase0_probe.py
"""
from __future__ import annotations

import io
import json
import os
import platform
import sys
import time
from importlib.metadata import version as pkg_version

import numpy as np

T0 = time.time()
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "phase0")
os.makedirs(OUT, exist_ok=True)
FLAME_LINK = 9
IGNITION_COLOR = [1.0, 0.3, 0.3, 0.8]
TINY_DEFAULTS = dict(lightDirection=[-5, 200, -40], lightColor=[1, 1, 1], lightDistance=2.0,
                     lightAmbientCoeff=0.6, lightDiffuseCoeff=0.35, lightSpecularCoeff=0.05, shadow=0)

import gymnasium as gym  # noqa: E402
import pybullet as p  # noqa: E402
from PIL import Image  # noqa: E402

t = time.time()
import stable_worldmodel  # noqa: E402,F401  (registers swm/* ids)

IMPORT_S = time.time() - t
R: dict = {"versions": {k: pkg_version(k) for k in ["PyFlyt", "pybullet", "cvxpy", "gymnasium", "numpy",
                                                    "torch", "Pillow", "stable-worldmodel", "numba"]},
           "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
           "swm_import_s": round(IMPORT_S, 2), "notes": []}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def save_png(name, arr):
    Image.fromarray(np.ascontiguousarray(arr)).save(os.path.join(OUT, name))


def ndiff(a, b):
    return int(np.any(a != b, axis=-1).sum())


class Env:
    """Thin handle over the swm env: poses, colours, frames, census."""

    def __init__(self):
        t = time.time()
        self.g = gym.make("swm/PFRocketLanding-v0", render_mode="rgb_array")
        self.u = self.g.unwrapped
        self.make_s = time.time() - t

    def reset(self, seed, variation=("all",)):
        t = time.time()
        opts = {"variation": list(variation)} if variation else {}
        obs, info = self.g.reset(seed=seed, options=opts)
        self.reset_s = time.time() - t
        self.c = self.u.env._client
        self.rid = self.u.env.drones[0].Id
        self.pad = self.u.landing_pad_id
        self.plane = self.u.env.planeId
        return obs, info

    def frame(self):
        return self.u.render()

    def raw(self, res=480, **kw):
        cam = self.u.env.drones[0].camera
        _, _, rgba, depth, seg = p.getCameraImage(res, res, viewMatrix=cam.view_mat,
                                                  projectionMatrix=cam.proj_mat, physicsClientId=self.c, **kw)
        return (np.asarray(rgba, np.uint8).reshape(res, res, -1)[..., :3],
                np.asarray(depth, np.float32).reshape(res, res), np.asarray(seg).reshape(res, res))

    def pose(self):
        pos, q = p.getBasePositionAndOrientation(self.rid, physicsClientId=self.c)
        return np.array(pos, np.float64), np.array(q, np.float64)

    def set_pose(self, pos, q):
        p.resetBasePositionAndOrientation(self.rid, list(pos), list(q), physicsClientId=self.c)

    def vel(self):
        lv, av = p.getBaseVelocity(self.rid, physicsClientId=self.c)
        return np.array(lv), np.array(av)

    def set_vel(self, lv, av):
        p.resetBaseVelocity(self.rid, list(lv), list(av), physicsClientId=self.c)

    def colours(self):
        return {"rocket": [tuple(round(x, 4) for x in v[7]) for v in
                           p.getVisualShapeData(self.rid, physicsClientId=self.c)],
                "pad": [tuple(round(x, 4) for x in v[7]) for v in p.getVisualShapeData(self.pad, physicsClientId=self.c)]}

    def variation(self):
        vs = self.u.variation_space
        out = {}
        for grp in ["rocket", "pad", "environment"]:
            for k, leaf in vs[grp].spaces.items():
                v = leaf.value
                out[f"{grp}.{k}"] = np.asarray(v).tolist()
        return out

    def census(self):
        rows = []
        for b in range(p.getNumBodies(physicsClientId=self.c)):
            bid = p.getBodyUniqueId(b, physicsClientId=self.c)
            info = p.getBodyInfo(bid, physicsClientId=self.c)
            for v in p.getVisualShapeData(bid, p.VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS, physicsClientId=self.c):
                dyn = p.getDynamicsInfo(bid, v[1], physicsClientId=self.c)
                rows.append({"body": info[1].decode(), "link": int(v[1]), "geom_type": int(v[2]),
                             "dims": [round(float(x), 4) for x in v[3]], "mesh_file": v[4].decode(),
                             "local_pos": [round(float(x), 4) for x in v[5]],
                             "local_orn": [round(float(x), 4) for x in v[6]],
                             "rgba": [round(float(x), 4) for x in v[7]],
                             "texture_id": int(v[8]) if len(v) > 8 else None,
                             "mass": round(float(dyn[0]), 4)})
        return rows

    def camera(self):
        cam = self.u.env.drones[0].camera
        return {"link": int(cam.camera_id), "fov_deg": float(cam.camera_FOV_degrees), "aspect": 1.0,
                "near": 0.1, "far": 255.0, "angle_deg": float(cam.camera_angle_degrees),
                "use_gimbal": bool(cam.use_gimbal), "tracking": bool(cam.is_tracking_camera),
                "offset_in_link_frame": [float(x) for x in cam.camera_position_offset],
                "proj_mat": [round(float(x), 6) for x in cam.proj_mat],
                "view_mat_now": [round(float(x), 6) for x in cam.view_mat]}

    def step(self, a):
        return self.g.step(np.asarray(a, np.float64))

    def close(self):
        self.g.close()


def quat_from_euler(r, pt, y):
    return np.array(p.getQuaternionFromEuler([r, pt, y]))


# ----------------------------------------------------------------------------- 0. construct
log("constructing")
E = Env()
obs, info = E.reset(seed=0)
R["timing"] = {"gym_make_s": round(E.make_s, 3), "first_reset_s_incl_numba_jit": round(E.reset_s, 2)}
f0 = E.frame()
R["frame"] = {"shape": list(f0.shape), "dtype": str(f0.dtype), "bytes": int(f0.nbytes),
              "obs_shape": list(obs.shape), "obs_dtype": str(obs.dtype), "obs_bytes": int(obs.nbytes),
              "goal_shape": list(info["goal"].shape)}
R["camera"] = E.camera()
R["connection"] = p.getConnectionInfo(physicsClientId=E.c)
save_png("preview_reset_seed0_variation_all.png", f0)
save_png("preview_goal_seed0.png", info["goal"])
log("first frame", f0.shape, "reset", round(E.reset_s, 1), "s")

# ----------------------------------------------------------------------------- 1. G3 decoder constancy
log("G3: census across resets")
G3 = {"resets": []}
geom_keys = ("body", "link", "geom_type", "dims", "mesh_file", "local_pos", "local_orn", "texture_id", "mass")
census_ref = None
for seed, var in [(0, ("all",)), (1, ("all",)), (2, ("all",)), (3, ()), (4, ("rocket.body_color",)), (0, ("all",))]:
    E.reset(seed, var)
    cen = E.census()
    fr = E.frame()
    geom = [{k: r[k] for k in geom_keys} for r in cen]
    if census_ref is None:
        census_ref = geom
        R["scene_census"] = cen
    same_geom = geom == census_ref
    G3["resets"].append({"seed": seed, "variation": list(var), "reset_s": round(E.reset_s, 3),
                         "n_bodies": int(p.getNumBodies(physicsClientId=E.c)), "n_visual_shapes": len(cen),
                         "geometry_identical_to_first": bool(same_geom),
                         "rgba_per_shape": [r["rgba"] for r in cen],
                         "sampled_variation": E.variation(),
                         "background_px_top_left": fr[0, 0].tolist(),
                         "pad_pose": [round(float(x), 4) for x in
                                      p.getBasePositionAndOrientation(E.pad, physicsClientId=E.c)[0]]})
    log(f"  seed={seed} var={var} geometry_identical={same_geom} bg={fr[0,0].tolist()} pad_rgba={cen[-1]['rgba']}")
    if seed == 3:
        save_png("preview_reset_seed3_no_variation.png", fr)
G3["what_changes_per_reset"] = "rgba only (see rgba_per_shape); geometry/mass/mesh/texture rows identical"
G3["sky_color_read_anywhere"] = False
G3["triangle_estimate"] = {"cylinder_32_steps_each": "3 rocket cylinders + pad = 4 x ~128",
                           "boxes_7_x_12": 84, "ground_quad": 2, "total_approx": 4 * 128 + 84 + 2}
R["G3"] = G3

# ----------------------------------------------------------------------------- 2. lighting law
log("lighting law ablations")
E.reset(0)
base, depth, seg = E.raw()
L = {}
expl, _, _ = E.raw(**TINY_DEFAULTS)
L["explicit_tiny_defaults_vs_default_diff_px"] = ndiff(base, expl)
tiny, _, _ = E.raw(renderer=p.ER_TINY_RENDERER)
L["explicit_ER_TINY_RENDERER_diff_px"] = ndiff(base, tiny)
sh, _, _ = E.raw(shadow=1)
L["shadow_on_diff_px"] = ndiff(base, sh)
amb, _, _ = E.raw(lightAmbientCoeff=1.0, lightDiffuseCoeff=0.0, lightSpecularCoeff=0.0)
L["ambient_only_diff_px"] = ndiff(base, amb)
nospec, _, _ = E.raw(lightSpecularCoeff=0.0)
L["specular_off_diff_px"] = ndiff(base, nospec)
# flat-colour check: ambient-only rocket body pixels vs int(255*c)
body_mask = seg == E.rid
rgba_body = E.colours()["rocket"][0]
if body_mask.any():
    vals, cnt = np.unique(amb[body_mask].reshape(-1, 3), axis=0, return_counts=True)
    top = vals[np.argsort(-cnt)[:5]].tolist()
    L["ambient_only_body_top_colours"] = top
    L["body_material_rgb_x255"] = [rgba_body[i] * 255 for i in range(3)]
    L["body_material_rgb_x255_trunc"] = [int(rgba_body[i] * 255) for i in range(3)]
    L["body_material_rgb_x255_trunc_x0.6"] = [int(0.6 * int(rgba_body[i] * 255)) for i in range(3)]
    vals2, cnt2 = np.unique(base[body_mask].reshape(-1, 3), axis=0, return_counts=True)
    L["default_body_top_colours"] = vals2[np.argsort(-cnt2)[:6]].tolist()
    L["default_body_distinct_colours"] = int(len(vals2))
# is the light world-fixed? render the same pose from a camera rotated 180 deg about the rocket
cam = E.u.env.drones[0].camera
pos, q = E.pose()
Rm = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
eye_a = pos + Rm @ np.array([-1.0, 0.0, 3.0])
eye_b = pos + Rm @ np.array([+1.0, 0.0, 3.0])
up = Rm @ np.array([0.0, 0.0, 1.0])
va = p.computeViewMatrix(list(eye_a), list(pos), list(up))
vb = p.computeViewMatrix(list(eye_b), list(pos), list(up))
_, _, ra, _, sa = p.getCameraImage(480, 480, viewMatrix=va, projectionMatrix=cam.proj_mat, physicsClientId=E.c)
_, _, rb, _, sb = p.getCameraImage(480, 480, viewMatrix=vb, projectionMatrix=cam.proj_mat, physicsClientId=E.c)
ra = np.asarray(ra, np.uint8).reshape(480, 480, 4)[..., :3]
rb = np.asarray(rb, np.uint8).reshape(480, 480, 4)[..., :3]
sa = np.asarray(sa).reshape(480, 480) == E.rid
sb = np.asarray(sb).reshape(480, 480) == E.rid
L["body_mean_rgb_camera_at_-x"] = ra[sa].mean(0).round(2).tolist() if sa.any() else None
L["body_mean_rgb_camera_at_+x"] = rb[sb].mean(0).round(2).tolist() if sb.any() else None
L["body_mean_rgb_camera_at_-x_left_half_vs_right_half"] = [
    ra[:, :240][sa[:, :240]].mean().round(2).item() if sa[:, :240].any() else None,
    ra[:, 240:][sa[:, 240:]].mean().round(2).item() if sa[:, 240:].any() else None]
L["body_mean_rgb_camera_at_+x_left_half_vs_right_half"] = [
    rb[:, :240][sb[:, :240]].mean().round(2).item() if sb[:, :240].any() else None,
    rb[:, 240:][sb[:, 240:]].mean().round(2).item() if sb[:, 240:].any() else None]
save_png("preview_light_cam_minus_x.png", ra)
save_png("preview_light_cam_plus_x.png", rb)
save_png("preview_ambient_only.png", amb)
L["source"] = ("bullet3 examples/SharedMemory/plugins/tinyRendererPlugin/TinyRendererVisualShapeConverter.cpp: "
               "m_lightDirection=(-5,200,-40), m_lightColor=(1,1,1), m_lightDistance=2, ambient 0.6, diffuse 0.35, "
               "specular 0.05, m_hasShadow=false, clearColor=(255,255,255,255), cylinder numSteps=32, box = fixed cube; "
               "examples/TinyRenderer/TinyRenderer.cpp Shader::fragment: color=diffuse(uv)*colorRGBA; "
               "out_i=int(A*c_i + shadow*(D*max(0,n.l) + S*max(0,r.z)^spec(uv))*c_i*L_i), clamp 255, per-vertex "
               "normals barycentrically interpolated, light_dir used UNTRANSFORMED (world-fixed), no gamma")
R["lighting"] = L
log("  lighting:", {k: v for k, v in L.items() if k.endswith("diff_px")})

# ----------------------------------------------------------------------------- 3. G1 purity
log("G1: purity")
G1 = {}
rng = np.random.default_rng(0)


def rollout(env, seed, n, ignite, rng):
    env.reset(seed)
    for _ in range(n):
        a = rng.uniform(-1, 1, size=7)
        a[3] = 1.0 if ignite else 0.0
        a[4] = rng.uniform(0.2, 1.0) if ignite else 0.0
        env.step(a)


E.reset(1)
fa = E.frame()
fb = E.frame()
G1["same_state_twice_diff_px"] = ndiff(fa, fb)

rollout(E, 1, 12, False, rng)
pose_A = E.pose()
vel_A = E.vel()
frame_A = E.frame()
obs_A = E.u.state.copy()
col_A = E.colours()
G1["history_A"] = {"pos": pose_A[0].round(4).tolist(), "quat": pose_A[1].round(5).tolist(),
                   "lin_vel": vel_A[0].round(3).tolist(), "ang_vel": vel_A[1].round(3).tolist()}

# different history B (same seed => same colours), teleport to pose_A
rollout(E, 1, 25, False, rng)
G1["history_B_pose_before_teleport"] = E.pose()[0].round(3).tolist()
assert E.colours() == col_A, "same seed must give same colours"
E.set_pose(*pose_A)
frame_B = E.frame()
G1["different_history_same_pose_diff_px"] = ndiff(frame_A, frame_B)
G1["view_mat_after_teleport_matches_pose"] = bool(np.allclose(
    np.array(p.getLinkState(E.rid, 0, physicsClientId=E.c)[0]), pose_A[0], atol=1e-6))

# velocity perturbation
lv, av = E.vel()
E.set_vel(lv * 100 + 7.0, av * 100 + 3.0)
frame_V = E.frame()
G1["velocity_x100_diff_px"] = ndiff(frame_A, frame_V)
E.set_vel(lv, av)

# ignition history: same pose, but the flame link was recoloured by a past ignition transition
rollout(E, 1, 6, True, rng)
for _ in range(4):
    a = np.zeros(7)
    E.step(a)
G1["flame_rgba_after_ignition_history"] = E.colours()["rocket"][FLAME_LINK + 1]
G1["ignition_state_now"] = float(E.u.ignition_state)
E.set_pose(*pose_A)
frame_I = E.frame()
d = np.any(frame_I != frame_A, axis=-1)
ys, xs = np.nonzero(d)
G1["ignition_history_same_pose_diff_px"] = int(d.sum())
G1["ignition_history_diff_bbox_yx"] = [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())] if d.any() else None
save_png("preview_g1_pose_A.png", frame_A)
save_png("preview_g1_pose_A_after_ignition_history.png", frame_I)
vis = np.zeros_like(frame_A)
vis[d] = 255
save_png("preview_g1_ignition_history_diff_mask.png", vis)
# is the default flame (alpha 0.1) visible at all?
E.reset(1)
E.set_pose(*pose_A)
f_flame = E.frame()
p.changeVisualShape(E.rid, FLAME_LINK, rgbaColor=[1.0, 0.8, 0.0, 0.0], physicsClientId=E.c)
f_noflame = E.frame()
G1["default_flame_alpha0.1_vs_alpha0_diff_px"] = ndiff(f_flame, f_noflame)
p.changeVisualShape(E.rid, FLAME_LINK, rgbaColor=IGNITION_COLOR, physicsClientId=E.c)
f_ign = E.frame()
G1["ignition_colour_vs_default_flame_diff_px"] = ndiff(f_flame, f_ign)
G1["ignition_colour_vs_alpha0_diff_px"] = ndiff(f_noflame, f_ign)
G1["restored_pose_A_diff_px_vs_frame_A"] = ndiff(f_flame, frame_A)

# fresh instance, colours applied by hand through changeVisualShape (rocket) and the pad URDF path
log("G1: fresh instance")
E2 = Env()
E2.reset(1)
E2.set_pose(*pose_A)
frame_F = E2.frame()
G1["fresh_instance_same_seed_same_pose_diff_px"] = ndiff(frame_A, frame_F)
# a NON-variation reset recoloured by hand to the seed-1 colours: is the frame f(pose, colours) only?
E2.reset(7, ())
for i, rgba in enumerate(col_A["rocket"]):
    p.changeVisualShape(E2.rid, i - 1, rgbaColor=list(rgba), physicsClientId=E2.c)
p.changeVisualShape(E2.pad, -1, rgbaColor=list(col_A["pad"][0]), physicsClientId=E2.c)
E2.set_pose(*pose_A)
frame_H = E2.frame()
G1["hand_recoloured_default_reset_same_pose_diff_px"] = ndiff(frame_A, frame_H)
E2.close()
R["G1"] = G1
log("  G1:", {k: v for k, v in G1.items() if k.endswith("diff_px")})

# ----------------------------------------------------------------------------- 4. G2 compactness
fb_ = int(f0.nbytes)
G2 = {"frame_bytes": fb_,
      "render_state_per_step": {"pose_f32_7": 28, "pose_f64_7": 56},
      "render_state_per_episode": {"colours_5x3_uint8": 15, "flame_recoloured_flag_bits": 1, "bytes": 16},
      "ratio_per_step_f32": 28 / fb_, "ratio_per_step_f64": 56 / fb_,
      "ratio_per_step_amortised_200_step_episode_f32": (28 + 16 / 200) / fb_,
      "env_obs_bytes_30_f64": 240, "ratio_env_obs": 240 / fb_,
      "goal_frame_per_episode_bytes": fb_, "goal_is_f_of": "(pad colour, rocket colours, fixed landed pose)"}
R["G2"] = G2

# ----------------------------------------------------------------------------- 5. G4 economics
log("G4: render fps / step share / decode")
G4 = {}
E.reset(2)
for _ in range(5):
    E.frame()
n = 60
t = time.time()
for _ in range(n):
    E.frame()
G4["official_render_480_fps"] = round(n / (time.time() - t), 2)
cam = E.u.env.drones[0].camera
t = time.time()
for _ in range(n):
    p.getCameraImage(128, 128, viewMatrix=cam.view_mat, projectionMatrix=cam.proj_mat, physicsClientId=E.c)
G4["tiny_render_128_fps"] = round(n / (time.time() - t), 2)
t = time.time()
for _ in range(n):
    cam.view_mat
G4["view_mat_only_us"] = round((time.time() - t) / n * 1e6, 1)

# step timing, with the drone camera (render_mode=rgb_array => 128^2 capture per agent step) and without
a0 = np.zeros(7)
E.reset(2)
ns = 100
t = time.time()
for _ in range(ns):
    E.step(a0)
G4["step_s_with_drone_camera"] = round((time.time() - t) / ns, 5)
gn = gym.make("swm/PFRocketLanding-v0", render_mode=None)
gn.reset(seed=2, options={"variation": ["all"]})
t = time.time()
for _ in range(ns):
    gn.step(a0)
G4["step_s_no_camera"] = round((time.time() - t) / ns, 5)
gn.close()
r480 = 1.0 / G4["official_render_480_fps"]
G4["render_share_of_step_if_480_rendered_per_step"] = round(r480 / (G4["step_s_no_camera"] + r480), 4)
G4["drone_camera_share_of_step"] = round((G4["step_s_with_drone_camera"] - G4["step_s_no_camera"]) /
                                         G4["step_s_with_drone_camera"], 4)

# decode: JPEG q90 of the env's own frames, one core, from memory
E.reset(2)
frames_for_jpeg = []
for i in range(16):
    E.set_pose(np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(2, 450)]),
               quat_from_euler(rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(-3.14, 3.14)))
    frames_for_jpeg.append(E.frame())
bufs = []
for fr in frames_for_jpeg:
    b = io.BytesIO()
    Image.fromarray(fr).save(b, format="JPEG", quality=90)
    bufs.append(b.getvalue())
G4["jpeg_q90_bytes_mean"] = int(np.mean([len(b) for b in bufs]))
for b in bufs[:4]:
    np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
nd = 200
t = time.time()
for i in range(nd):
    np.asarray(Image.open(io.BytesIO(bufs[i % len(bufs)])).convert("RGB"))
G4["jpeg_q90_decode_480_fps_one_core"] = round(nd / (time.time() - t), 1)
G4["official_render_vs_decode_cores"] = round(G4["official_render_480_fps"] / G4["jpeg_q90_decode_480_fps_one_core"], 3)
G4["setup"] = {"numba_jit_per_process_s": R["timing"]["first_reset_s_incl_numba_jit"],
               "second_reset_s": round(E.reset_s, 3), "gym_make_s": R["timing"]["gym_make_s"],
               "note": "the JIT is PyFlyt physics, not rendering; a state->frame decoder pays none of it"}
R["G4"] = G4
log("  G4:", G4)

# ----------------------------------------------------------------------------- 6. corpus + coverage
log("corpus: 48 synthetic descent poses + 32 rollout poses")
E.reset(5)
col = E.colours()
states, frames, cover = [], [], []
heights = np.concatenate([np.geomspace(450, 2.0, 40), [1.5, 3.0, 10.0, 30.0, 100.0, 200.0, 250.0, 260.0]])
for h in heights:
    pos = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), h])
    q = quat_from_euler(rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(-3.14, 3.14))
    E.set_pose(pos, q)
    rgb, depth, seg = E.raw()
    states.append(np.concatenate([pos, q]))
    frames.append(rgb)
    cover.append({"h": round(float(h), 2), "rocket_px": int((seg == E.rid).sum()), "pad_px": int((seg == E.pad).sum()),
                  "ground_px": int((seg == E.plane).sum()), "bg_px": int((seg < 0).sum()),
                  "distinct_colours": int(len(np.unique(rgb.reshape(-1, 3), axis=0)))})
# a real rollout (random actions, ignition on half the time) for natural attitudes
E.reset(5)
roll_states, roll_frames = [], []
for i in range(32):
    a = rng.uniform(-1, 1, size=7)
    a[3] = float(rng.random() < 0.5)
    a[4] = rng.uniform(0, 1)
    E.step(a)
    pos, q = E.pose()
    roll_states.append(np.concatenate([pos, q]))
    roll_frames.append(E.frame())
flame_after = E.colours()["rocket"][FLAME_LINK + 1]
np.savez_compressed(os.path.join(OUT, "frames_descent.npz"), states=np.array(states, np.float32),
                    frames=np.array(frames, np.uint8),
                    colours_rocket=np.array(col["rocket"], np.float32), colours_pad=np.array(col["pad"], np.float32),
                    heights=np.array(heights, np.float32))
np.savez_compressed(os.path.join(OUT, "frames_rollout.npz"), states=np.array(roll_states, np.float32),
                    frames=np.array(roll_frames, np.uint8), colours_rocket=np.array(col["rocket"], np.float32),
                    colours_pad=np.array(col["pad"], np.float32), flame_rgba_at_end=np.array(flame_after, np.float32))
np.savez_compressed(os.path.join(OUT, "frames_g1_pairs.npz"), frame_A=frame_A, frame_B=frame_B, frame_V=frame_V,
                    frame_I=frame_I, frame_F=frame_F, frame_H=frame_H, pose_A=np.concatenate(pose_A))
for i in [0, 10, 20, 30, 39, 40]:
    save_png(f"preview_descent_h{cover[i]['h']:.0f}.png", frames[i])
save_png("preview_rollout_last.png", roll_frames[-1])
R["coverage"] = {"per_height": cover,
                 "frames_with_ground_visible": int(sum(c["ground_px"] > 0 for c in cover)),
                 "n": len(cover),
                 "rocket_px_frac_mean": round(float(np.mean([c["rocket_px"] for c in cover])) / 480 / 480, 4),
                 "distinct_colours_mean": round(float(np.mean([c["distinct_colours"] for c in cover])), 1),
                 "far_plane_m": 255.0, "start_height_m": [0.7 * 500, 0.95 * 500]}
R["rollout"] = {"flame_rgba_after_32_random_steps": flame_after}
R["files"] = sorted(os.listdir(OUT))
R["total_probe_s"] = round(time.time() - T0, 1)
E.close()
with open(os.path.join(OUT, "phase0.json"), "w") as fh:
    json.dump(R, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
log("done ->", os.path.join(OUT, "phase0.json"))
