"""What shading scalar does MuJoCo apply to the OGBench floor, and what do we apply?

The floor's material colour is (1,1,1) and all of its colour comes from a texture, so its rendered
value is ``round(255 * clamp(S * tex))`` with S the shading scalar. Sampling our own texture at the
same pixels therefore recovers MuJoCo's S by division -- no fitting, and no need to disturb the
reference. The remaining OGBench floor residual is mean 3.4-4.0/255 with 99.9 % of pixels within 8,
which is what a ~0.02 error in S would cost, so this says whether the constants or the filter own it.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from common import emit, save

import gymnasium, mujoco, torch, ogbench  # noqa
from stable_envs.render import nvdiff as ND

SIZE = 224
env = gymnasium.make("pointmaze-medium-v0", render_mode="rgb_array", ob_type="pixels",
                     width=SIZE, height=SIZE)
env.reset(options={"task_id": 1})
u = env.unwrapped
m, d = u.model, u.data
m.vis.quality.offsamples = 0
cam_mjv = u.custom_camera
mujoco.mj_forward(m, d)

ren = mujoco.Renderer(m, height=SIZE, width=SIZE)
ren.update_scene(d, camera=cam_mjv)
ref = ren.render().copy()
ren.enable_segmentation_rendering(); ren.update_scene(d, camera=cam_mjv)
seg = ren.render()[..., 0].copy(); ren.disable_segmentation_rendering()
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

br = ND.BatchRenderer(m, res=SIZE, device="cuda", tex_policy="exact")
cam, _ = br.resolve_from_mujoco(m, d, camera=cam_mjv)
ours = br.render_qpos(torch.as_tensor(d.qpos[None], dtype=torch.float32), cam=cam)[0].cpu().numpy()

# our texture alone: zero the lights on OUR side only, leaving the reference untouched
L = br.lights
br.lights = ND.Lights(L.pos, L.dir, L.directional, np.zeros_like(L.ambient),
                      np.zeros_like(L.diffuse), np.zeros_like(L.specular), L.attenuation,
                      L.cutoff, L.exponent, L.headlight)
sc = br.scene
sc.emission = torch.ones_like(sc.emission)
sc.mat_rgb = torch.ones_like(sc.mat_rgb)
texonly = br.render_qpos(torch.as_tensor(d.qpos[None], dtype=torch.float32), cam=cam)[0].cpu().numpy()

sel = (seg == floor) & (texonly.min(-1) > 8) & (ref.max(-1) < 250)
t = texonly[sel].astype(np.float64)
r = ref[sel].astype(np.float64)
o = ours[sel].astype(np.float64)
S_ref = (r / np.maximum(t, 1e-9))
S_ours = (o / np.maximum(t, 1e-9))
emit("S", {"npx": int(sel.sum()),
           "S_ref_median": float(np.median(S_ref)), "S_ref_p10": float(np.quantile(S_ref, .1)),
           "S_ref_p90": float(np.quantile(S_ref, .9)),
           "S_ours_median": float(np.median(S_ours)),
           "ratio_median": float(np.median(S_ref / np.maximum(S_ours, 1e-9))),
           "A+D+Sp_ours": 0.1 + 1.4 + 0.3})
save("S.json", {"ok": True})
