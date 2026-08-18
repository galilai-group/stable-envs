"""Close the agent-pixel shading gap. Two hypotheses, scored against MuJoCo on agent pixels only.

The whole-frame number is dominated by a static plate; the number that matters is agreement on the
agent's 20-150 pixels. With the shading law fitted on FLOOR pixels those come out 8.9% exact / 59.8%
within 1, mean 3.3, worst 48 -- i.e. most agent pixels are already right and a tail is badly wrong.

Two candidate mechanisms, and they predict different things:

  H1  the specular lobe is mis-fitted. The floor only ever shows ``n.l = 1`` and ``n.h`` within a few
      degrees of 1, so an exponent fitted there is an extrapolation everywhere on a sphere, where the
      normal sweeps a whole hemisphere. If this is it, refitting (A, D, Sp, p) ON AGENT PIXELS closes it
      and the residual is unstructured.
  H2  Gouraud. MuJoCo lights per vertex on a tessellated sphere/capsule
      (``numslices=28``, ``numstacks=16``) and interpolates. If this is it, no per-pixel law can close it
      and the fix is to ray-cast the triangles and interpolate the vertex shades.

H1 is checked first because it is cheap and because a sphere 12 px across is covered by ~224
front-facing quads -- sub-pixel triangles, where Gouraud and per-pixel agree by construction. That
argues against H2 on its own; the fit says whether it is right.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import raycast as RC  # noqa: E402
from shade import normals  # noqa: E402
from general_check import make2, agent_geoms, cast_subset  # noqa: E402
from analytic_rgb import base_rgb  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
INF = RC.INF
L = np.array([0.0, 0.0, 1.0])


def collect(loco, size, img=224, nstates=40, marker=1.0, seed=0):
    """Agent-pixel observations: (n.l, n.h, base, obs, is_agent_interior) pooled over states."""
    import mujoco
    env, u = make2(loco, size, img)
    m, d = u.model, u.data
    if marker != 1.0:
        ag0, _ = agent_geoms(m)
        for g in ag0:
            m.geom_size[g] = m.geom_size[g] * marker
    m.vis.quality.offsamples = 0
    r = mujoco.Renderer(m, width=img, height=img)
    r.update_scene(d, camera=u.custom_camera)
    cam = RC.avgcam(RC.glcam(r.scene, 0), RC.glcam(r.scene, 1))
    ag, names = agent_geoms(m)
    static = [g for g in range(m.ngeom) if g not in ag]
    _, bg_t, o, dr = cast_subset(m, d, cam, img, img, static)
    V = -dr / np.linalg.norm(dr, axis=-1, keepdims=True)
    hv = L + V
    hv = hv / np.linalg.norm(hv, axis=-1, keepdims=True)

    rows = {k: [] for k in ("ndl", "ndh", "base", "obs", "gid")}
    rng = np.random.default_rng(seed)
    for _ in range(nstates):
        for _ in range(9):
            env.step(u.action_space.sample())
        r.update_scene(d, camera=u.custom_camera)
        obs = r.render().astype(np.float64)
        gid, t, _, _ = cast_subset(m, d, cam, img, img, ag)
        vis = (t < INF / 2) & (t < bg_t)
        if vis.sum() < 4:
            continue
        p = o + t[..., None] * dr
        n = normals(m, d, np.where(vis, gid, -1), p)
        # agent-INTERIOR: the pixel and its 4 neighbours are all agent, so coverage is not in play
        e = vis.copy()
        e[1:] &= vis[:-1]
        e[:-1] &= vis[1:]
        e[:, 1:] &= vis[:, :-1]
        e[:, :-1] &= vis[:, 1:]
        base = np.zeros((img, img, 3))
        for g in np.unique(gid[vis]):
            base[vis & (gid == g)] = base_rgb(m, int(g))
        rows["ndl"].append(np.clip(n @ L, 0, None)[e])
        rows["ndh"].append(np.clip((n * hv).sum(-1), 0, None)[e])
        rows["base"].append(base[e])
        rows["obs"].append(obs[e])
        rows["gid"].append(gid[e])
    env.close()
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rows.items()}


def fit(ndl, ndh, S, exps=range(1, 129)):
    best = None
    for p in exps:
        A = np.stack([np.ones_like(ndl), ndl, ndh ** p], 1)
        coef, *_ = np.linalg.lstsq(A, S, rcond=None)
        rms = float(np.sqrt(((A @ coef - S) ** 2).mean()))
        if best is None or rms < best[0]:
            best = (rms, p, coef)
    return best


def score(ndl, ndh, base, obs, coef, p):
    S = coef[0] + coef[1] * ndl + coef[2] * ndh ** p
    pred = np.clip(np.round(np.clip(base * S[:, None], 0, 1) * 255.0), 0, 255)
    dd = np.abs(pred - obs).max(-1)
    return dict(n=int(len(dd)), exact=float((dd == 0).mean()), le1=float((dd <= 1).mean()),
                le2=float((dd <= 2).mean()), mean=float(dd.mean()), max=float(dd.max()))


def run(loco, size, marker=1.0):
    r = collect(loco, size, marker=marker)
    out = dict(env=f"{loco}maze-{size}", marker=marker, n_agent_interior_px=int(len(r["ndl"])))
    if len(r["ndl"]) < 50:
        out["skipped"] = "too few agent interior pixels"
        return out
    base, obs = r["base"], r["obs"]
    ok = (base > 1e-6).all(-1) & ((obs < 254.5).all(-1))
    S = (obs[ok] / (base[ok] * 255.0))
    # is the shading a pure multiplier here too? (channels must agree)
    out["channel_spread_of_S"] = float(np.abs(S - S.mean(1, keepdims=True)).max())
    Sm = S.mean(1)
    ndl, ndh = r["ndl"][ok], r["ndh"][ok]
    rms, p, coef = fit(ndl, ndh, Sm)
    out["agent_fit"] = dict(ambient=round(float(coef[0]), 6), diffuse=round(float(coef[1]), 6),
                            specular=round(float(coef[2]), 6), exponent=int(p), rms_S=round(rms, 6))
    out["score_agent_fit"] = score(r["ndl"], r["ndh"], base, obs, coef, p)
    # the floor-fitted law, for contrast
    out["score_floor_fit"] = score(r["ndl"], r["ndh"], base, obs,
                                   np.array([0.098039, 1.3907, 0.26]), 60)
    # theory-from-the-light-table, for contrast
    out["score_light_table"] = score(r["ndl"], r["ndh"], base, obs,
                                     np.array([0.1, 1.4, 0.3]), 64)
    # residual structure: is what is left a function of n.h (H2) or noise (H1)?
    Sp = coef[0] + coef[1] * ndl + coef[2] * ndh ** p
    res = Sm - Sp
    q = np.clip((ndh * 20).astype(int), 0, 20)
    out["resid_by_ndh_bin"] = {int(b): [int((q == b).sum()), round(float(res[q == b].mean()), 5),
                                        round(float(res[q == b].std()), 5)]
                               for b in np.unique(q) if (q == b).sum() > 20}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "agent_shading.log"), "w")
    allr = []
    for loco, size, mk in [("point", "medium", 1.0), ("point", "medium", 3.0),
                           ("point", "large", 1.0), ("ant", "large", 3.0),
                           ("humanoid", "large", 4.0)]:
        try:
            r = run(loco, size, mk)
        except Exception:  # noqa: BLE001
            import traceback
            log.write(f"FAIL {loco}-{size}\n{traceback.format_exc()}\n")
            log.flush()
            continue
        allr.append(r)
        log.write(json.dumps(r, indent=1) + "\n")
        log.flush()
        print(json.dumps(r, indent=1), flush=True)
    log.close()
    with open(os.path.join(OUT, "agent_shading.json"), "w") as f:
        json.dump(allr, f, indent=1)


if __name__ == "__main__":
    main()
