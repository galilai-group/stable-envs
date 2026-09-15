"""M1b — the one authorised further attribution pass: what is left after the spot's angular term?

M3's gate applied the spot correction (`exponent = 0` on the MJWarp model only) and reached
**0.3507** interior-exact, up 38x from 0.0091 but still short of parity — and, tellingly, BELOW
M1's `no_spot` row (0.4568), where the light was removed from both sides. So the spot still carries
disagreement beyond the cone and the exponent, and M1's second term (the headlight, independently
worth +0.0372) has never been measured on top of the correction.

Same recipe as M1, unchanged: every term is neutralised **in the MjModel**, both renderers are then
driven from that same model, the host's render-noise floor is measured so "no change" is judged
against it rather than against zero, and every claim carries a verdict flag.

**Per-class reporting is the point of this pass, not a garnish.** The interior mask on this scene is
99.2 % floor, and the shipped cell bakes that floor through MuJoCo itself — exactly as
`stable_envs/envs/fetch.py` does, where a provably static plate made 91 % of the frame exact BY
CONSTRUCTION and the declared 0.9988 interior came from the mover pass on top. So the number that
predicts this cell's tier is `interior_exact_mover`, not the whole-frame figure, and both are
reported at every step with the mover pixel count beside them.
"""
from __future__ import annotations

import argparse, json, os, platform, socket, sys
from pathlib import Path

import numpy as np

GL_SPOT_MAX = 90.0


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
    env.reset(seed=0); env.action_space.seed(0)
    m, d = u.model, u.data
    res = int(u._render_height)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front_pixels")
    mv = set(movers(m))

    pool = []
    for s in range(a.states):
        env.reset(seed=700 + s); env.action_space.seed(700 + s)
        m, d = u.model, u.data
        for _ in range(int(np.random.default_rng(s).integers(0, 20))):
            env.step(env.action_space.sample())
        pool.append(d.qpos.copy())
    qpos = np.stack(pool).astype(np.float32)

    rec = {"env": a.env, "res": res, "host": socket.gethostname(),
           "gpu": torch.cuda.get_device_name(0), "platform": platform.platform(),
           "mujoco_gl": os.environ.get("MUJOCO_GL"), "states": a.states, "rows": []}

    def spot_fix(mm):
        """The M3 correction, applied under every row of this pass."""
        n = 0
        for i in range(mm.nlight):
            if int(mm.light_type[i]) == 0 and not (0.0 <= float(mm.light_cutoff[i]) <= GL_SPOT_MAX):
                mm.light_exponent[i] = 0.0
                n += 1
        return n

    def run(tag, mutate=None, apply_spot_fix=True):
        env.reset(seed=0)
        mm, dd = u.model, u.data
        undo_ref = mutate(mm) if mutate else {}     # BOTH sides see this
        offs, clss = [], []
        for q in pool:
            dd.qpos[:] = q; mujoco.mj_forward(mm, dd)
            offs.append(u.render().copy())
            seg = u.render(segmentation=True)[..., 0]
            c = np.zeros(seg.shape, np.int8); c[seg >= 0] = 1
            c[np.isin(seg, list(mv))] = 2
            clss.append(c)
        offs = np.stack(offs); clss = np.stack(clss)
        noise = u.render().copy()
        nd = np.abs(noise.astype(np.int16) - offs[-1].astype(np.int16)).max()
        saved_exp = mm.light_exponent.copy()
        nfix = spot_fix(mm) if apply_spot_fix else 0   # MJWarp side only
        with MJ.gl_rounding():
            br = MJ.BatchRenderer(mm, batch=len(pool), res=res, camera=int(cam))
            fr = br.render(torch.as_tensor(qpos, device="cuda")).cpu().numpy()
        del br; torch.cuda.empty_cache()
        mm.light_exponent[:] = saved_exp
        for k, v in undo_ref.items():
            getattr(mm, k)[:] = v
        im = interior_mask(offs)
        dif = np.abs(fr.astype(np.int16) - offs.astype(np.int16)).max(-1)
        di, ci = dif[:, :-1, :-1], clss[:, :-1, :-1]
        r = {"term": tag, "spot_fix_applied": bool(nfix), "host_noise_max_abs": int(nd),
             "interior_exact_frac": float((di[im] == 0).mean()),
             "interior_mean_abs": float(di[im].mean()),
             "all_exact_frac": float((dif == 0).mean()),
             "all_within8_frac": float((dif <= 8).mean())}
        for k, name in ((2, "mover"), (1, "static"), (0, "sky")):
            sel = im & (ci == k)
            r[f"interior_exact_{name}"] = float((di[sel] == 0).mean()) if sel.any() else None
            r[f"interior_mean_abs_{name}"] = float(di[sel].mean()) if sel.any() else None
            r[f"interior_n_{name}"] = int(sel.sum())
        # the number the cell's tier actually rests on: mover pixels, plate assumed exact
        movpx = clss == 2
        r["mover_px_share"] = float(movpx.mean())
        r["mover_all_exact_frac"] = float((dif[movpx] == 0).mean()) if movpx.any() else None
        r["mover_mean_abs"] = float(dif[movpx].mean()) if movpx.any() else None
        r["composited_all_exact_frac"] = float(((dif == 0) | ~movpx).mean())
        sd = (fr.astype(np.int16) - offs.astype(np.int16))
        sel = np.repeat(im[:, :, :, None], 3, 3)
        v = sd[:, :-1, :-1, :][sel]
        vals, cnt = np.unique(np.clip(v, -20, 20), return_counts=True)
        r["signed_interior_top"] = [[int(k), round(float(p), 5)] for k, p in
                                    sorted(zip(vals.tolist(), (cnt / cnt.sum()).tolist()),
                                           key=lambda t: -t[1])[:5]]
        rec["rows"].append(r)
        print(f"{tag:32s} int {r['interior_exact_frac']:.4f}  mover {r['interior_exact_mover']:.4f} "
              f"(n={r['interior_n_mover']})  moverALL {r['mover_all_exact_frac']:.4f}  "
              f"comp {r['composited_all_exact_frac']:.4f}  m|d| {r['interior_mean_abs']:6.3f}  "
              f"noise {int(nd)}", flush=True)
        return r

    def off_headlight(mm):
        u_ = {}
        rec.setdefault("_hl", int(mm.vis.headlight.active))
        mm.vis.headlight.active = 0
        return u_

    def restore_headlight(mm):
        mm.vis.headlight.active = rec.get("_hl", 1)
        return {}

    def off_spot(mm):
        u_ = {"light_active": mm.light_active.copy()}
        for i in range(mm.nlight):
            if int(mm.light_type[i]) == 0:
                mm.light_active[i] = 0
        return u_

    def off_global(mm):
        u_ = {"light_active": mm.light_active.copy()}
        for i in range(mm.nlight):
            if int(mm.light_type[i]) == 1:
                mm.light_active[i] = 0
        return u_

    def zero_ambient(mm):
        rec.setdefault("_amb", [float(x) for x in mm.vis.headlight.ambient])
        mm.vis.headlight.ambient[:] = 0.0
        return {}

    def restore_ambient(mm):
        mm.vis.headlight.ambient[:] = rec.get("_amb", [0.1, 0.1, 0.1])
        return {}

    run("no correction (M3 baseline)", None, apply_spot_fix=False)
    run("spot fix only")
    r_hl = run("spot fix + headlight off", off_headlight); run("restore", restore_headlight)
    run("spot fix + spot light off", off_spot)
    run("spot fix + global light off", off_global)
    r_am = run("spot fix + ambient=0", zero_ambient); run("restore", restore_ambient)

    base = [r for r in rec["rows"] if r["term"] == "spot fix only"][0]
    rec["verdict_spot_fix_helps"] = bool(
        base["interior_exact_frac"] > rec["rows"][0]["interior_exact_frac"] * 5)
    rec["verdict_mover_is_the_tier_driver"] = (
        "the interior mask is 99.2 % floor; the shipped cell bakes the floor through MuJoCo, so "
        "interior_exact_mover and composited_all_exact_frac are the numbers that predict the tier")
    (out / "m1b_residual.json").write_text(json.dumps(rec, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
