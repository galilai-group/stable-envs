"""Can MJWarp draw Meta-World, once the state gate is closed?

``docs/research/metaworld.md`` rejected the family on two gates. ``probe_state_gate.py`` closes the
second exactly on 50/50 tasks, so this asks whether the first one — geometry — is really a gate, or
whether it is the *Fetch* situation seen from the wrong angle.

The metaworld doc's geometry objection has two halves and they are not the same claim:

  * 17-29 % of the frame is MESH. That was decisive against ``raycast3d``, which has no mesh path.
    It is not a claim about MJWarp, which ships a BVH mesh intersector with texcoord interpolation.
  * 5-10 textured non-mesh geoms per task sample one texel under MJWarp. That is only a fidelity
    problem for geoms that MOVE: a STATIC one is baked through MuJoCo itself and is exact by
    construction, which is exactly why the Fetch port reports 0.992 interior while full-scene MJWarp
    on the same scene reports 0.798.

So the deciding number is the *moving undrawable* share, not the textured-geom count, and the
plate has to be shown to hold. Measured here, per task, through the model camera ``corner``:

  1. model facts, including whether any texture is ``mjTEXTURE_CUBE`` — the hard CUDA-700 fault that
     killed Franka Kitchen (``docs/research/kitchen.md`` Gate 1);
  2. the frame budget, split mover/static x drawable/undrawable, over states the env reaches;
  3. the shadow pass share and the plate's non-mover interior exactness with shadows on and off —
     Franka Kitchen's Gate 2, applied to this family;
  4. whether the movers-hidden plate is a constant of the appearance at all.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from scratch.metaworld.probe_metaworld import make                          # noqa: E402
from stable_envs.render import mjwarp as MW                                 # noqa: E402
from stable_envs.testing.conformance import interior_mask                   # noqa: E402

SIZE = 224
NSTATE = 12
OUT = "/mnt/vast/home/randall.balestriero/se_logs/metaworld"


def states(env, u, m, d, n=NSTATE, steps=14, seed=0):
    """States the env actually reaches, each one FORWARDED — ``probe_state_gate.py`` measures that a
    frame taken straight after ``reset`` is stale on 28/50 tasks and is not a function of any state.
    One ``step`` forwards, so a rollout state is clean; ``reset`` alone is not."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        env.reset()
        for _ in range(steps):
            env.step(rng.uniform(-1, 1, size=u.action_space.shape).astype(np.float32))
        out.append((np.array(d.qpos, copy=True),
                    {a: np.array(getattr(m, a), copy=True) for a in
                     ("body_pos", "body_quat", "site_pos", "geom_pos", "geom_size")}))
    return out


def probe(task: str) -> dict:
    env = make(task, SIZE)
    u = env.unwrapped
    m, d = u.model, u.data
    env.reset()
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "corner")

    rec: dict = {"task": task, "ngeom": int(m.ngeom), "nmesh": int(m.nmesh),
                 "ntex": int(m.ntex), "nlight": int(m.nlight)}

    # --- 1. model facts -------------------------------------------------------------------------
    ttype = np.asarray(m.tex_type, np.int64)
    rec["tex_types"] = {int(t): int((ttype == t).sum()) for t in np.unique(ttype)}
    rec["has_cube_texture"] = bool((ttype == int(mujoco.mjtTexture.mjTEXTURE_CUBE)).any())
    rec["n_castshadow_lights"] = int(np.asarray(m.light_castshadow, np.int64).sum())

    vis = [g for g in range(m.ngeom) if int(m.geom_group[g]) in MW.DEFAULT_GROUPS]
    alpha = MW.effective_alpha(m, "geom")
    gaps = {g for g, _, _ in MW.texture_gaps(m, MW.DEFAULT_GROUPS)}
    translucent = {g for g in vis if 0.0 < alpha[g] < 1.0}
    invisible = {g for g in vis if alpha[g] <= 0.0}
    undrawable = (gaps | translucent) - invisible
    # a mover is anything not welded to the world, plus anything whose body a reset writes
    movers = [g for g in vis if int(m.body_weldid[m.geom_bodyid[g]]) != 0
              or int(m.body_mocapid[m.geom_bodyid[g]]) >= 0]

    pool = states(env, u, m, d)

    # per-episode placement is written into the MODEL, so a body a reset MOVES is a mover even when
    # it is welded to the world -- measure which body rows actually change over the pool
    ref = pool[0][1]
    moved_bodies = {b for b in range(m.nbody)
                    if any(not np.array_equal(ref["body_pos"][b], s[1]["body_pos"][b])
                           or not np.array_equal(ref["body_quat"][b], s[1]["body_quat"][b])
                           for s in pool)}
    movers = sorted(set(movers) | {g for g in vis if int(m.geom_bodyid[g]) in moved_bodies})
    static = [g for g in vis if g not in movers]
    rec["n_movers"], rec["n_static"] = len(movers), len(static)
    rec["n_undrawable_movers"] = len(set(movers) & undrawable)

    r = mujoco.Renderer(m, height=SIZE, width=SIZE)
    opt = mujoco.MjvOption()

    def apply(s):
        q, ma = s
        for a, v in ma.items():
            getattr(m, a)[:] = v
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

    def shot(hide=None, shadows=True, sites=True):
        keep = np.array(m.geom_group, copy=True)
        if hide:
            m.geom_group[np.asarray(hide, np.int64)] = MW.HIDDEN_GROUP
        o = mujoco.MjvOption()
        if not sites:
            o.sitegroup[:] = 0
        r.update_scene(d, camera=cam, scene_option=o)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadows)
        img = r.render().copy()
        m.geom_group[:] = keep
        return img

    # --- 2. frame budget ------------------------------------------------------------------------
    keys = ("moving_undrawable", "moving_ok", "static_undrawable", "static_ok", "site",
            "background")
    cov = dict.fromkeys(keys, 0.0)
    sets = {"moving_undrawable": [g for g in movers if g in undrawable],
            "moving_ok": [g for g in movers if g not in undrawable],
            "static_undrawable": [g for g in static if g in undrawable],
            "static_ok": [g for g in static if g not in undrawable]}
    mesh_share = 0.0
    for s in pool:
        apply(s)
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=cam, scene_option=opt)
        seg = r.render()
        r.disable_segmentation_rendering()
        oid, otype = seg[..., 0].astype(np.int64), seg[..., 1].astype(np.int64)
        gpix = otype == int(mujoco.mjtObj.mjOBJ_GEOM)
        spix = otype == int(mujoco.mjtObj.mjOBJ_SITE)
        tot = float(oid.size)
        for k, sel in sets.items():
            a = np.asarray(sel, np.int64) if sel else np.zeros(0, np.int64)
            cov[k] += float((np.isin(oid, a) & gpix).sum() / tot) / len(pool)
        cov["site"] += float(spix.sum() / tot) / len(pool)
        cov["background"] += float((~(gpix | spix)).sum() / tot) / len(pool)
        gt = np.zeros_like(oid)
        gt[gpix] = np.asarray(m.geom_type)[oid[gpix]]
        mesh_share += float((gpix & (gt == 7)).sum() / tot) / len(pool)
    rec["coverage"] = {k: round(v, 5) for k, v in cov.items()}
    rec["frac_mesh_pixels"] = round(mesh_share, 5)
    rec["undrawable_movers"] = {
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"):
            ("alpha=%.2f" % alpha[g] if g in translucent else "textured type=%d"
             % int(m.geom_type[g]))
        for g in sorted(set(movers) & undrawable)}

    # --- 3. shadows, and the plate --------------------------------------------------------------
    apply(pool[0])
    a, b = shot(), shot(shadows=False)
    e = np.abs(a.astype(np.int16) - b.astype(np.int16))
    rec["shadow_pass"] = {"frac_changed": round(float((e.max(-1) > 0).mean()), 5),
                          "max": int(e.max())}

    # is the movers-hidden plate a constant of the appearance?
    plates_on, plates_off, movermasks, refs = [], [], [], []
    for s in pool:
        apply(s)
        plates_on.append(shot(hide=movers, sites=False))
        plates_off.append(shot(hide=movers, sites=False, shadows=False))
        refs.append(shot())
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=cam, scene_option=opt)
        seg = r.render()
        r.disable_segmentation_rendering()
        oid, otype = seg[..., 0].astype(np.int64), seg[..., 1].astype(np.int64)
        mm = np.isin(oid, np.asarray(movers, np.int64)) & (
            otype == int(mujoco.mjtObj.mjOBJ_GEOM))
        movermasks.append(mm | (otype == int(mujoco.mjtObj.mjOBJ_SITE)))

    def plate_quality(plates, tag):
        p0 = plates[0]
        drift = max(int(np.abs(p.astype(np.int16) - p0.astype(np.int16)).max()) for p in plates)
        ex, inex, worst = [], [], 1.0
        for p, ref, mm in zip(plates, refs, movermasks):
            keep = ~mm
            dif = np.abs(p.astype(np.int16) - ref.astype(np.int16)).max(-1)
            # interior_mask is (N, H-1, W-1) -- everything it indexes must be cropped to match
            im = interior_mask(ref[None])[0] & keep[:-1, :-1]
            ex.append(float((dif[keep] == 0).mean()) if keep.any() else 1.0)
            if im.any():
                v = float((dif[:-1, :-1][im] == 0).mean())
                inex.append(v)
                worst = min(worst, v)
        return {f"plate_drift_max_{tag}": drift,
                f"nonmover_exact_{tag}": round(float(np.mean(ex)), 6),
                f"nonmover_interior_exact_{tag}": round(float(np.mean(inex)), 6) if inex else None,
                f"nonmover_interior_worst_{tag}": round(worst, 6) if inex else None}

    rec.update(plate_quality(plates_on, "shadows_on"))
    rec.update(plate_quality(plates_off, "shadows_off"))
    rec["mover_silhouette"] = round(float(np.mean([mm.mean() for mm in movermasks])), 5)
    env.close()
    return rec


if __name__ == "__main__":
    tasks = sys.argv[1:]
    if not tasks or tasks == ["ALL"]:
        from metaworld.env_dict import ALL_V3_ENVIRONMENTS
        tasks = sorted(ALL_V3_ENVIRONMENTS)
    os.makedirs(OUT, exist_ok=True)
    out = []
    for t in tasks:
        try:
            rr = probe(t)
        except Exception as ex:                                             # noqa: BLE001
            import traceback
            traceback.print_exc()
            rr = {"task": t, "error": f"{type(ex).__name__}: {ex}"}
        out.append(rr)
        print(json.dumps(rr), flush=True)
    p = os.path.join(OUT, "probe_backend.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p)
