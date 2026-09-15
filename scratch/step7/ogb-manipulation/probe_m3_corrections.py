"""M3 design gate: can OGBCube's spot-light term be corrected in the MODEL, as Fetch corrects its?

M1 attributed the whole interior gap to one light: neutralising the spot on both sides took interior
from 0.0082 to 0.4568 and mean |d| from 8.78 to 1.54. The mechanism is that `mujoco_warp` honours a
spot cutoff of 100 deg -- cone plus `pow(cos_theta, 10)` -- while MuJoCo's GL path DISCARDS it,
because `GL_SPOT_CUTOFF` accepts [0,90] or the sentinel 180 and leaves anything else at 180 = "not a
spotlight" (measured: cutoff 100 renders bit-identically to 180, max abs 0).

That points at a correction of exactly the shape `stable_envs/envs/fetch.py` already applies four
times -- surgery on the MjModel before `put_model`, not renderer code:

    a spot light whose cutoff is outside OpenGL's legal range is re-typed as a POINT light,
    because that is what MuJoCo's own renderer does with it.

In `mujoco_warp`'s shader the spot cone and exponent live behind `if lighttype == 0`, so a POINT
light (type 2) takes `L = normalize(lightpos - hitpoint)` with `attenuation = 1/(1,0,0).(1,d,d^2) = 1`
and no angular term -- which is what GL does at cutoff 180. This measures whether that is true in
pixels, and what is left afterwards.

`no_spot` from M1 is the WRONG comparison to stop at: it disabled the light on both sides, which
agrees by removing the term. This keeps the light lit on both sides and asks whether the two
renderers then draw the same picture.
"""
from __future__ import annotations

import argparse, json, os, platform, socket, sys
from pathlib import Path

import numpy as np

GL_SPOT_MAX = 90.0          # OpenGL: GL_SPOT_CUTOFF accepts [0, 90] or the sentinel 180


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
    res = int(u._render_height)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")

    pool = []
    for s in range(a.states):
        env.reset(seed=700 + s); m, d = u.model, u.data
        for _ in range(int(np.random.default_rng(s).integers(0, 20))):
            env.step(env.action_space.sample())
        pool.append(d.qpos.copy())
    qpos = np.stack(pool).astype(np.float32)

    rec = {"env": a.env, "res": res, "host": socket.gethostname(),
           "gpu": torch.cuda.get_device_name(0), "platform": platform.platform(),
           "mujoco_gl": os.environ.get("MUJOCO_GL"), "states": a.states, "rows": []}

    def run(tag, mutate=None):
        """Render both sides from one model. `mutate` changes ONLY what MJWarp reads."""
        env.reset(seed=0)
        mm, dd = u.model, u.data
        # official frames first, from the UNMUTATED model: the reference must not move
        offs = []
        for q in pool:
            dd.qpos[:] = q; mujoco.mj_forward(mm, dd)
            offs.append(u.render().copy())
        offs = np.stack(offs)
        undo = mutate(mm) if mutate else {}
        with MJ.gl_rounding():
            br = MJ.BatchRenderer(mm, batch=len(pool), res=res, camera=int(cam))
            fr = br.render(torch.as_tensor(qpos, device="cuda")).cpu().numpy()
        del br; torch.cuda.empty_cache()
        for k, v in undo.items():
            getattr(mm, k)[:] = v
        im = interior_mask(offs)
        dd_ = np.abs(fr.astype(np.int16) - offs.astype(np.int16)).max(-1)
        di = dd_[:, :-1, :-1]
        sd = (fr.astype(np.int16) - offs.astype(np.int16))
        sel = np.repeat(im[:, :, :, None], 3, 3)
        v = sd[:, :-1, :-1, :][sel]
        vals, cnt = np.unique(np.clip(v, -20, 20), return_counts=True)
        top = sorted(zip(vals.tolist(), (cnt / cnt.sum()).tolist()), key=lambda t: -t[1])[:5]
        r = {"correction": tag,
             "interior_exact_frac": float((di[im] == 0).mean()),
             "interior_mean_abs": float(di[im].mean()),
             "all_exact_frac": float((dd_ == 0).mean()),
             "all_within8_frac": float((dd_ <= 8).mean()),
             "mean_abs": float(dd_.mean()),
             "signed_interior_top": [[int(k), round(float(p), 5)] for k, p in top]}
        rec["rows"].append(r)
        print(f"{tag:34s} interior {r['interior_exact_frac']:.4f}  mean|d| {r['interior_mean_abs']:6.3f}  "
              f"all {r['all_exact_frac']:.4f}  within8 {r['all_within8_frac']:.4f}  "
              f"signed {r['signed_interior_top'][:3]}", flush=True)
        return fr, offs

    def retype_spot(mm):
        """MuJoCo discards an out-of-range spot cutoff; make MJWarp do the same by re-typing."""
        undo = {"light_type": mm.light_type.copy()}
        n = 0
        for i in range(mm.nlight):
            if int(mm.light_type[i]) == 0 and not (0.0 <= float(mm.light_cutoff[i]) <= GL_SPOT_MAX):
                mm.light_type[i] = 2           # mjLIGHT_POINT
                n += 1
        rec.setdefault("retyped_lights", n)
        return undo

    def retype_and_widen(mm):
        undo = retype_spot(mm)
        undo["light_cutoff"] = mm.light_cutoff.copy()
        for i in range(mm.nlight):
            if not (0.0 <= float(mm.light_cutoff[i]) <= GL_SPOT_MAX):
                mm.light_cutoff[i] = 180.0
        return undo

    def retype_and_kill_headlight(mm):
        undo = retype_spot(mm)
        return undo

    run("baseline")
    fr, offs = run("spot retyped to POINT", retype_spot)
    run("spot retyped + cutoff=180", retype_and_widen)

    np.savez_compressed(out / "m3_corrections_frames.npz", mjwarp=fr[:2], official=offs[:2])
    (out / "m3_corrections.json").write_text(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
