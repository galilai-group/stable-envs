"""Step 7 / W5 phase 0 -- localise the MJWarp fault on `swm/OGBScene-v0`.

Jobs 6395461 and 6395736 both put OGBScene through `MJ.BatchRenderer` and both died with
`Warp CUDA error 700: an illegal memory access` -- the second in a FRESH process, with the
error appearing first at `wp_cuda_context_synchronize` (i.e. during the warm-up frame, before
the graph capture), so it is a genuine OGBScene fault and not contamination from OGBCube.

`BatchRenderer` exposes the megakernel's feature switches, so the fault can be bisected instead
of guessed at. The standing hypothesis is the **CUBE texture**: `OGBCube` has one 2-D texture on a
PLANE plus a skybox and builds fine; `OGBScene` adds `T_btn`, a 512x512 **CUBE** texture worn by
two MESH geoms. `mjwarp.texture_gaps` returns empty for it because MESH is in the UV-derivable set,
but a cube map is a different sampler from a 2-D one, so an empty `texture_gaps` is not evidence
that this texture is safe.

Each configuration runs in its own subprocess: an illegal memory access poisons the CUDA context,
so a second configuration measured in the same process would be meaningless.
"""
from __future__ import annotations

import argparse, json, subprocess, sys
from pathlib import Path

CHILD = r'''
import json, sys, os
sys.path.insert(0, os.environ["WS"])
import numpy as np, torch, gymnasium as gym, mujoco, stable_worldmodel  # noqa
from stable_envs.render import mjwarp as MJ
cfg = json.loads(sys.argv[1])
env = gym.make(cfg["env"]); u = env.unwrapped; env.reset(seed=0)
m, d = u.model, u.data
res = int(u._render_height)
cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")
if cfg.get("drop_cube_textures"):
    # Detach every CUBE texture from its material, leaving the material's flat rgba.
    n = 0
    T = np.asarray(m.mat_texid)
    for j in range(m.nmat):
        for k in range(T.shape[1]):
            t = int(T[j, k])
            if t >= 0 and int(m.tex_type[t]) == 1:      # mjTEXTURE_CUBE
                m.mat_texid[j, k] = -1; n += 1
    print("detached_cube_textures", n)
with MJ.gl_rounding():
    br = MJ.BatchRenderer(m, batch=int(cfg["batch"]), res=res, camera=int(cam),
                          textures=cfg["textures"], shadows=cfg["shadows"],
                          skybox=cfg["skybox"], capture=cfg["capture"])
q = torch.as_tensor(np.repeat(d.qpos.copy()[None], int(cfg["batch"]), 0).astype(np.float32), device="cuda")
f = br.render(q)
torch.cuda.synchronize()
a = f.cpu().numpy()
print("RESULT", json.dumps({"ok": True, "shape": list(a.shape), "mean": float(a.mean()),
                            "std": float(a.std()), "nonconstant": bool(a.std() > 1e-6)}))
'''


def run(ws, env, **cfg):
    cfg = {"env": env, "batch": 2, "textures": True, "shadows": True, "skybox": True,
           "capture": True, **cfg}
    p = subprocess.run([sys.executable, "-c", CHILD, json.dumps(cfg)],
                       capture_output=True, text=True, timeout=1800,
                       env={**__import__("os").environ, "WS": ws})
    out = p.stdout.strip().splitlines()
    res = {"cfg": {k: v for k, v in cfg.items() if k != "env"}, "rc": p.returncode}
    for line in out:
        if line.startswith("RESULT "):
            res.update(json.loads(line[7:]))
        elif line.startswith("detached_cube_textures"):
            res["detached_cube_textures"] = int(line.split()[1])
    if p.returncode != 0:
        res["ok"] = False
        err = [l for l in p.stderr.splitlines() if "Error" in l or "error" in l]
        res["error"] = (err[-1] if err else p.stderr.strip().splitlines()[-1:] or [""])[:300] \
            if err else (p.stderr.strip()[-300:])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="swm/OGBScene-v0")
    ap.add_argument("--ws", default=str(Path(__file__).resolve().parents[3]))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cfg in [
        ("baseline", {}),
        ("no_capture", {"capture": False}),
        ("no_textures", {"textures": False}),
        ("no_skybox", {"skybox": False}),
        ("no_shadows", {"shadows": False}),
        ("drop_cube_textures", {"drop_cube_textures": True}),
    ]:
        r = run(a.ws, a.env, **cfg)
        r["name"] = name
        rows.append(r)
        print(f"{name:20s} ok={r.get('ok')} {r.get('error', '')[:150]}", flush=True)
    tag = a.env.split("/")[-1].replace("-v0", "")
    (out / f"mjwarp_bisect_{tag}.json").write_text(json.dumps({"env": a.env, "rows": rows},
                                                             indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
