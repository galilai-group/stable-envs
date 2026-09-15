"""Q11 for Sami: what does baking OGBScene's button cube-map to its mean colour actually cost?

`swm/OGBScene-v0` kills MJWarp because its `btn_metal` material wears a 512x512 **CUBE** map on two
button MESH geoms (bisect job 6396447). Option (b) is to replace that texture with its mean colour
through the `bake_primitive_textures` recipe, which turns a crash into a declared visual-tier
residual. This measures the residual so the decision is made on numbers.

The measurement is MuJoCo against MuJoCo -- official frames with the texture, official frames with
the mean colour -- so it is the cost of the APPROXIMATION alone, with no MJWarp term mixed in, and
it needs no GPU. The two button geoms are located through the segmentation buffer, so "share of
frame" and "on the buttons" are measured rather than estimated.

Nothing here modifies `bake_primitive_textures`: it writes the mean colour into a throwaway model to
observe the consequence. `bake_primitive_textures` in its shipped form would in fact SKIP this
material twice over (it refuses non-2-D textures, and refuses materials worn by MESH geoms), which
is itself part of the answer: option (b) is not a no-op call, it is a change to that helper.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np


def main():
    import gymnasium as gym
    import mujoco
    import stable_worldmodel  # noqa: F401
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from stable_envs.render.mjwarp import texture_mean_rgb
    from stable_envs.testing.conformance import interior_mask

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--states", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    env = gym.make("swm/OGBScene-v0"); u = env.unwrapped
    env.reset(seed=0); m, d = u.model, u.data
    h, w = u._render_height, u._render_width

    # locate the cube-textured material and the geoms wearing it
    T = np.asarray(m.mat_texid)
    cube_mats, cube_texs = [], []
    for j in range(m.nmat):
        for t in np.atleast_1d(T[j]):
            t = int(t)
            if t >= 0 and int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_CUBE):
                cube_mats.append(j); cube_texs.append(t)
    cube_mats = sorted(set(cube_mats)); cube_texs = sorted(set(cube_texs))
    wearers = [g for g in range(m.ngeom) if int(m.geom_matid[g]) in cube_mats]
    rec = {"env": "swm/OGBScene-v0", "h": h, "w": w, "states": a.states,
           "cube_materials": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, j) for j in cube_mats],
           "cube_textures": [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t),
                              "w": int(m.tex_width[t]), "h": int(m.tex_height[t]),
                              "mean_rgb": texture_mean_rgb(m, t).tolist()} for t in cube_texs],
           "wearer_geoms": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in wearers],
           "wearer_geom_ids": wearers,
           "bake_primitive_textures_would_bake_it": False,
           "bake_primitive_textures_reason": (
               "shipped helper refuses it twice: tex_type != mjTEXTURE_2D, and the material is worn "
               "by MESH geoms which it classifies as 'mixed' (MJWarp can UV a mesh). Option (b) is "
               "therefore a change to that helper, not a call to it.")}

    nb = getattr(u, "_num_buttons", 0)
    kw = {f"button_state_{i}": 0 for i in range(nb)} if nb else {}

    pool = []
    for s in range(a.states):
        env.reset(seed=800 + s); m, d = u.model, u.data
        for _ in range(int(np.random.default_rng(s).integers(0, 20))):
            env.step(env.action_space.sample())
        pool.append((d.qpos.copy(), d.qvel.copy()))

    env.reset(seed=0); m, d = u.model, u.data
    T = np.asarray(m.mat_texid)

    def shoot():
        fr, seg = [], []
        for q, v in pool:
            u.set_state(q, v, **kw) if nb else u.set_state(q, v)
            fr.append(u.render().copy())
            seg.append(u.render(segmentation=True).copy()[..., 0])
        return np.stack(fr), np.stack(seg)

    with_tex, segs = shoot()
    # bake: mean colour into mat_rgba, detach every cube texture role
    saved_texid = np.array(m.mat_texid, copy=True)
    saved_rgba = np.array(m.mat_rgba, copy=True)
    for j in cube_mats:
        tex = None
        for t in np.atleast_1d(np.asarray(m.mat_texid)[j]):
            if int(t) >= 0 and int(m.tex_type[int(t)]) == int(mujoco.mjtTexture.mjTEXTURE_CUBE):
                tex = int(t); break
        if tex is not None:
            m.mat_rgba[j, :3] = texture_mean_rgb(m, tex)
        if np.ndim(m.mat_texid[j]):
            m.mat_texid[j, :] = -1
        else:
            m.mat_texid[j] = -1
    baked, _ = shoot()
    m.mat_texid[:] = saved_texid
    m.mat_rgba[:] = saved_rgba

    dif = np.abs(with_tex.astype(np.int16) - baked.astype(np.int16)).max(-1)
    any_px = dif > 0
    btn = np.isin(segs, wearers)
    rec["residual"] = {
        "frac_frame_differing": float(any_px.mean()),
        "n_pixels_differing_per_frame": float(any_px.sum() / len(pool)),
        "max_abs": int(dif.max()),
        "mean_abs_over_frame": float(dif.mean()),
        "mean_abs_over_differing": float(dif[any_px].mean()) if any_px.any() else 0.0,
        "frac_frame_gt8": float((dif > 8).mean()),
        "frac_frame_gt32": float((dif > 32).mean()),
        "button_px_share_of_frame": float(btn.mean()),
        "n_button_px_per_frame": float(btn.sum() / len(pool)),
        "mean_abs_on_buttons": float(dif[btn].mean()) if btn.any() else None,
        "max_abs_on_buttons": int(dif[btn].max()) if btn.any() else None,
        "frac_of_differing_that_is_button": float((any_px & btn).sum() / max(any_px.sum(), 1)),
        "frac_differing_off_buttons": float((any_px & ~btn).mean()),
    }
    im = interior_mask(with_tex)
    di = dif[:, :-1, :-1]
    rec["residual"]["interior_exact_frac_bake_vs_official"] = float((di[im] == 0).mean())
    rec["residual"]["interior_mean_abs"] = float(di[im].mean())

    np.savez_compressed(out / "q11_button_bake_frames.npz",
                        with_texture=with_tex[:2], baked=baked[:2])
    (out / "q11_button_bake.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    r = rec["residual"]
    print(f"cube materials {rec['cube_materials']} on geoms {rec['wearer_geoms']}")
    print(f"mean_rgb {[t['mean_rgb'] for t in rec['cube_textures']]}")
    print(f"buttons are {r['button_px_share_of_frame']*100:.3f}% of the frame "
          f"({r['n_button_px_per_frame']:.0f} px/frame)")
    print(f"residual: {r['frac_frame_differing']*100:.3f}% of pixels differ "
          f"({r['n_pixels_differing_per_frame']:.0f} px/frame), max {r['max_abs']}, "
          f"mean|d| over frame {r['mean_abs_over_frame']:.4f}, over differing {r['mean_abs_over_differing']:.2f}")
    print(f"  >8: {r['frac_frame_gt8']*100:.4f}%   >32: {r['frac_frame_gt32']*100:.4f}%")
    print(f"  on buttons: mean|d| {r['mean_abs_on_buttons']}, max {r['max_abs_on_buttons']}")
    print(f"  share of differing pixels that ARE buttons: {r['frac_of_differing_that_is_button']*100:.1f}%")
    print(f"  interior_exact(bake vs official) {r['interior_exact_frac_bake_vs_official']:.5f}")


if __name__ == "__main__":
    main()
