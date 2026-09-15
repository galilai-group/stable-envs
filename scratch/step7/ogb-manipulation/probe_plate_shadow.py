"""Step 7 / W5 phase 0 -- is the Fetch plate-bake precondition satisfied here?

`stable_envs/envs/fetch.py` bakes a static plate because every Fetch material has
``reflectance=0`` and the one light is ``castshadow="false"``, so with the movers hidden
the remaining pixels are a constant.  This measures the same precondition for the two
OGBench manipulation scenes, which have a DIRECTIONAL light with ``castshadow=1``:

* hide every mover (anything on a body with dofs) and render at several states;
* if those frames differ, the plate is not static and the Fetch pattern's cheap half
  does not exist here.  Report the pixel share that moves and where.

Also reports the mover pixel share (official minus movers-hidden), the site census,
and the alpha census -- the three other Fetch-pattern preconditions.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np


def movers(m):
    """Geom ids whose body (or an ancestor) has degrees of freedom, plus mocap bodies."""
    out = []
    for i in range(m.ngeom):
        b = int(m.geom_bodyid[i])
        moving = False
        while b > 0:
            if int(m.body_dofnum[b]) > 0 or int(m.body_mocapid[b]) >= 0:
                moving = True
                break
            b = int(m.body_parentid[b])
        if moving:
            out.append(i)
    return out


def main():
    import gymnasium as gym
    import mujoco
    import stable_worldmodel  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", nargs="+", default=["swm/OGBCube-v0", "swm/OGBScene-v0"])
    ap.add_argument("--states", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    for eid in a.envs:
        env = gym.make(eid); u = env.unwrapped
        env.reset(seed=5); m, d = u.model, u.data
        h, w = u._render_height, u._render_width
        rec = {"env_id": eid, "h": h, "w": w}

        mv = movers(m)
        rec["n_mover_geoms"] = len(mv)
        rec["n_static_geoms"] = int(m.ngeom) - len(mv)
        gg = [i for i, v in enumerate(u._scene_option.geomgroup) if v]
        vis_mv = [i for i in mv if int(m.geom_group[i]) in gg and m.geom_rgba[i][3] > 0]
        rec["n_visible_mover_geoms"] = len(vis_mv)
        rec["visible_mover_tri"] = int(sum(int(m.mesh_facenum[int(m.geom_dataid[i])])
                                           for i in vis_mv if int(m.geom_type[i]) == 7))

        # collect a pool of states
        pool = []
        for s in range(a.states):
            env.reset(seed=200 + s); m, d = u.model, u.data
            for _ in range(np.random.default_rng(s).integers(0, 20)):
                env.step(env.action_space.sample())
            pool.append((d.qpos.copy(), d.qvel.copy(), d.mocap_pos.copy(), d.mocap_quat.copy()))

        env.reset(seed=5); m, d = u.model, u.data
        nb = getattr(u, "_num_buttons", 0)
        kw = (lambda: {f"button_state_{i}": 0 for i in range(nb)}) if nb else (lambda: {})

        # official frames
        offs = []
        for q, v, mp, mq in pool:
            u.set_state(q, v, **kw())
            d.mocap_pos[:] = mp; d.mocap_quat[:] = mq
            mujoco.mj_forward(m, d)
            offs.append(u.render().copy())
        offs = np.stack(offs)

        # plate: hide every visible mover
        saved = m.geom_rgba.copy()
        for i in vis_mv:
            m.geom_rgba[i, 3] = 0.0
        plates = []
        for q, v, mp, mq in pool:
            d.qpos[:] = q; d.qvel[:] = v
            d.mocap_pos[:] = mp; d.mocap_quat[:] = mq
            mujoco.mj_forward(m, d)
            plates.append(u.render().copy())
        plates = np.stack(plates)
        m.geom_rgba[:] = saved
        mujoco.mj_forward(m, d)

        base = plates[0]
        moved = (plates != base[None]).any(0).any(-1)
        rec["plate_static"] = bool(not moved.any())
        rec["plate_frac_pixels_moving"] = float(moved.mean())
        rec["plate_max_abs_diff"] = int(np.abs(plates.astype(np.int16) - base[None].astype(np.int16)).max())
        rec["plate_n_distinct"] = int(len({p.tobytes() for p in plates}))

        # how much of the frame is the mover pass?
        movd = (offs != plates).any(-1)
        rec["mover_px_share_mean"] = float(movd.mean())
        rec["static_px_share_mean"] = float(1.0 - movd.mean())
        # the share that WOULD have been free if the plate were static
        rec["free_px_share_if_plate_static"] = float(((~movd) & (~moved[None])).mean())

        # shadow check: hide movers, toggle castshadow off, compare
        li = [i for i in range(m.nlight) if int(m.light_castshadow[i])]
        rec["lights_casting_shadow"] = len(li)
        if li:
            for i in vis_mv:
                m.geom_rgba[i, 3] = 0.0
            d.qpos[:] = pool[0][0]; d.qvel[:] = pool[0][1]
            mujoco.mj_forward(m, d); f_sh = u.render().copy()
            for i in li:
                m.light_castshadow[i] = 0
            f_no = u.render().copy()
            for i in li:
                m.light_castshadow[i] = 1
            m.geom_rgba[:] = saved
            mujoco.mj_forward(m, d)
            rec["shadow_px_share_on_plate"] = float((f_sh != f_no).any(-1).mean())

        # sites
        rec["sites"] = [{"id": i,
                         "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or f"site_{i}",
                         "group": int(m.site_group[i]), "rgba": m.site_rgba[i].tolist(),
                         "type": int(m.site_type[i])} for i in range(m.nsite)]
        sg = [i for i, v in enumerate(u._scene_option.sitegroup) if v]
        rec["n_visible_sites"] = sum(1 for s in rec["sites"]
                                     if s["group"] in sg and s["rgba"][3] > 0)

        # alpha census on rendered geoms
        al = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}",
               float(m.geom_rgba[i][3]),
               float(m.mat_rgba[int(m.geom_matid[i])][3]) if int(m.geom_matid[i]) >= 0 else None)
              for i in range(m.ngeom) if int(m.geom_group[i]) in gg]
        rec["translucent_rendered_geoms"] = [
            {"geom": n, "geom_alpha": g, "mat_alpha": mm}
            for n, g, mm in al if (0.0 < g < 1.0) or (mm is not None and 0.0 < mm < 1.0)]
        rec["n_translucent_rendered_geoms"] = len(rec["translucent_rendered_geoms"])

        tag = eid.split("/")[-1].replace("-v0", "")
        np.savez_compressed(out / f"plate_{tag}.npz", official=offs[:2], plate=plates[:2])
        (out / f"plate_{tag}.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(f"{eid}: plate_static={rec['plate_static']} plate_moving={rec['plate_frac_pixels_moving']*100:.2f}% "
              f"(max {rec['plate_max_abs_diff']}) mover_share={rec['mover_px_share_mean']*100:.1f}% "
              f"shadow_on_plate={rec.get('shadow_px_share_on_plate', 0)*100:.2f}% "
              f"translucent={rec['n_translucent_rendered_geoms']} visible_sites={rec['n_visible_sites']}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
