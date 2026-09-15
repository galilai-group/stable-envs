"""Step 7 / W5 M1 -- attribute OGBCube's MJWarp-vs-MuJoCo interior gap TERM BY TERM.

Phase 0 measured raw MJWarp on `swm/OGBCube-v0` at **0.0101 interior-exact** against MuJoCo's own
EGL frames, with only `gl_rounding` applied. Fetch's uncorrected figure was 0.791 and its corrected
one 0.9988, so this is two orders of magnitude further away and none of the four known Fetch
corrections applies (`texture_gaps` empty, no material/rgba conflict, no visible sites, one
translucent geom owning 0.62 %). Writing a cell before knowing WHY would be guessing.

The method is `cpu3d`'s M1 recipe run on the GPU side: the terms this scene has and the Fetch scene
does not are a `castshadow` DIRECTIONAL light, a SPOT light (cutoff 100, exponent 10) and a
HEADLIGHT that owns 98.7 % of pixels. Each is neutralised **in the MjModel**, and then BOTH
renderers are driven from that same modified model, so the comparison isolates the term rather than
changing one side's picture. A term MJWarp implements correctly leaves agreement unchanged when it
is removed; a term MJWarp gets wrong makes agreement JUMP when it is removed. That jump is the
attribution.

Three things are measured alongside, because each can masquerade as a shading bug:

* **Per-class, from MuJoCo's own segmentation buffer.** 85.4 % of this frame is the static plate
  (floor + skybox), which the shipped cell would bake through MuJoCo exactly and never ask MJWarp to
  draw. A whole-frame number that is dragged down by MJWarp's floor says nothing about the mover
  pass that would actually be used, so floor / sky / mover are reported separately.
* **SIGNED.** The single largest Fetch term was invisible until the difference was looked at signed
  (`{0: 86.8 %, -1: 13.1 %}` is one float quantised two ways, not a shading hypothesis).
* **An affine fit per channel.** If MJWarp's frame is official x a + b, the disagreement is one
  global lighting scale, not geometry -- a different repair from a per-pixel one.
"""
from __future__ import annotations

import argparse, json, os, platform, socket, sys
from pathlib import Path

import numpy as np


def seg_classes(u, m, mv_ids):
    """(H,W) int: 0 = sky, 1 = floor/static, 2 = mover, from MuJoCo's segmentation buffer."""
    seg = u.render(segmentation=True)
    gid = seg[..., 0]
    out = np.zeros(gid.shape, np.int8)
    out[gid >= 0] = 1
    out[np.isin(gid, list(mv_ids))] = 2
    return out


def movers(m):
    out = []
    for i in range(m.ngeom):
        b = int(m.geom_bodyid[i]); moving = False
        while b > 0:
            if int(m.body_dofnum[b]) > 0 or int(m.body_mocapid[b]) >= 0:
                moving = True; break
            b = int(m.body_parentid[b])
        if moving:
            out.append(i)
    return out


def compare(mjw, off, im, cls):
    """interior-exact / mean|d| overall and per segmentation class."""
    d = np.abs(mjw.astype(np.int16) - off.astype(np.int16)).max(-1)
    di, ci = d[:, :-1, :-1], cls[:, :-1, :-1]
    res = {"all_exact_frac": float((d == 0).mean()),
           "all_within8_frac": float((d <= 8).mean()),
           "mean_abs": float(d.mean()),
           "interior_frac": float(im.mean()),
           "interior_exact_frac": float((di[im] == 0).mean()) if im.any() else None,
           "interior_mean_abs": float(di[im].mean()) if im.any() else None}
    for k, name in ((0, "sky"), (1, "static"), (2, "mover")):
        sel = im & (ci == k)
        res[f"interior_exact_{name}"] = float((di[sel] == 0).mean()) if sel.any() else None
        res[f"interior_mean_abs_{name}"] = float(di[sel].mean()) if sel.any() else None
        res[f"interior_share_{name}"] = float(sel.sum() / max(im.sum(), 1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="swm/OGBCube-v0")
    ap.add_argument("--states", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import gymnasium as gym, mujoco, torch
    import stable_worldmodel  # noqa: F401
    from stable_envs.render import mjwarp as MJ
    from stable_envs.testing.conformance import interior_mask

    env = gym.make(a.env); u = env.unwrapped
    env.reset(seed=0); m, d = u.model, u.data
    res_px = int(u._render_height)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")
    mv = set(movers(m))

    rec = {"env": a.env, "res": res_px, "host": socket.gethostname(),
           "platform": platform.platform(), "mujoco": mujoco.__version__,
           "gpu": torch.cuda.get_device_name(0), "mujoco_gl": os.environ.get("MUJOCO_GL"),
           "states": a.states, "terms": []}

    # a pool of real states
    pool = []
    for s in range(a.states):
        env.reset(seed=700 + s); m, d = u.model, u.data
        for _ in range(int(np.random.default_rng(s).integers(0, 20))):
            env.step(env.action_space.sample())
        pool.append(d.qpos.copy())
    qpos = np.stack(pool).astype(np.float32)

    def render_both(tag, mutate=None, **brkw):
        env.reset(seed=0)
        mm, dd = u.model, u.data
        saved = {}
        if mutate:
            saved = mutate(mm)
        offs, clss = [], []
        for q in pool:
            dd.qpos[:] = q; mujoco.mj_forward(mm, dd)
            offs.append(u.render().copy())
            clss.append(seg_classes(u, mm, mv))
        offs = np.stack(offs); clss = np.stack(clss)
        with MJ.gl_rounding():
            br = MJ.BatchRenderer(mm, batch=len(pool), res=res_px, camera=int(cam), **brkw)
            fr = br.render(torch.as_tensor(qpos, device="cuda")).cpu().numpy()
        del br; torch.cuda.empty_cache()
        im = interior_mask(offs)
        r = compare(fr, offs, im, clss)
        r["term"] = tag
        # signed histogram on interior pixels, and an affine fit per channel
        sd = (fr.astype(np.int16) - offs.astype(np.int16))
        sel = np.repeat(im[:, :, :, None], 3, 3)
        v = sd[:, :-1, :-1, :][sel]
        vals, cnt = np.unique(np.clip(v, -20, 20), return_counts=True)
        top = sorted(zip(vals.tolist(), (cnt / cnt.sum()).tolist()), key=lambda t: -t[1])[:6]
        r["signed_interior_top"] = [[int(k), round(float(p), 5)] for k, p in top]
        x = offs[:, :-1, :-1, :][sel].astype(np.float64)
        y = fr[:, :-1, :-1, :][sel].astype(np.float64)
        if x.size > 10:
            A = np.polyfit(x, y, 1)
            r["affine_fit_slope"], r["affine_fit_intercept"] = float(A[0]), float(A[1])
            r["affine_residual_mean_abs"] = float(np.abs(y - (A[0] * x + A[1])).mean())
        if saved:
            for k, val in saved.items():
                tgt = mm
                for part in k.split(".")[:-1]:
                    tgt = getattr(tgt, part)
                attr = k.split(".")[-1]
                cur = getattr(tgt, attr)
                if isinstance(cur, np.ndarray):
                    cur[:] = val
                else:
                    setattr(tgt, attr, val)
        return r, fr, offs

    def mut_shadows(mm):
        s = {"light_castshadow": mm.light_castshadow.copy()}
        mm.light_castshadow[:] = 0
        return s

    def mut_headlight(mm):
        s = {"vis.headlight.active": int(mm.vis.headlight.active)}
        mm.vis.headlight.active = 0
        return s

    def mut_spot(mm):
        s = {"light_active": mm.light_active.copy()}
        for i in range(mm.nlight):
            if int(mm.light_type[i]) == 0:       # mjLIGHT_SPOT
                mm.light_active[i] = 0
        return s

    def mut_specular(mm):
        s = {"mat_specular": mm.mat_specular.copy(), "light_specular": mm.light_specular.copy()}
        mm.mat_specular[:] = 0.0; mm.light_specular[:] = 0.0
        return s

    def mut_msaa(mm):
        s = {"vis.quality.offsamples": int(mm.vis.quality.offsamples)}
        mm.vis.quality.offsamples = 0
        u._renderer.close(); u._initialize_renderer()
        return s

    def mut_all_lights(mm):
        s = {"light_castshadow": mm.light_castshadow.copy(),
             "light_active": mm.light_active.copy(),
             "vis.headlight.active": int(mm.vis.headlight.active),
             "mat_specular": mm.mat_specular.copy()}
        mm.light_castshadow[:] = 0
        mm.mat_specular[:] = 0.0
        return s

    plan = [
        ("baseline", None, {}),
        ("no_shadows", mut_shadows, {"shadows": False}),
        ("no_headlight", mut_headlight, {}),
        ("no_spot", mut_spot, {}),
        ("no_specular", mut_specular, {}),
        ("no_msaa", mut_msaa, {}),
        ("no_textures", None, {"textures": False}),
        ("no_skybox", None, {"skybox": False}),
        ("no_shadow_no_specular", mut_all_lights, {"shadows": False}),
    ]
    first = None
    for tag, mut, brkw in plan:
        try:
            r, fr, offs = render_both(tag, mut, **brkw)
            if tag == "baseline":
                first = (fr[:2], offs[:2])
            rec["terms"].append(r)
            print(f"{tag:24s} interior {r['interior_exact_frac']:.4f}  "
                  f"mover {r['interior_exact_mover']}  static {r['interior_exact_static']}  "
                  f"sky {r['interior_exact_sky']}  mean|d| {r['interior_mean_abs']:.2f}  "
                  f"slope {r.get('affine_fit_slope', 0):.3f} b {r.get('affine_fit_intercept', 0):.2f}",
                  flush=True)
        except Exception as e:
            rec["terms"].append({"term": tag, "error": f"{type(e).__name__}: {e}"[:300]})
            print(f"{tag:24s} FAILED {type(e).__name__}: {e}"[:200], flush=True)
            torch.cuda.empty_cache()

    if first is not None:
        np.savez_compressed(out / "m1_baseline_frames.npz", mjwarp=first[0], official=first[1])
    (out / "m1_attribution.json").write_text(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
