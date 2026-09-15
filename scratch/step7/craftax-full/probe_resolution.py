"""``check_resolution_consistency`` and the dead-dimension risk, on the OFFICIAL full renderer.

W1's ruling (Sami's, decision A) applies here unchanged: an applicability declaration may only be
made from an **oracle experiment on the official renderer**, never from our own implementation. This
runs that experiment before there is an implementation, so phase 1 knows what it is aiming at.

Protocol: craftax's own renderer at S and 2S (2S textures built through
``load_all_textures_given_size``, patched into the module's cache so every pixel is still craftax's),
un-jitted, each state's own ``state_rng``, then a box downsample of the 2S frame and the check's own
``exact_frac`` statistic. Split by band **and by region**, because the world rows and the strip fail
for different reasons.

Also measures what would be dead in a naive state pool: ``light_map`` on the overworld,
``light_level`` underground, the PRNG key outside night-on-level-0.

Writes ``phase0/resolution_and_dead_dims.json``.
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "examples", "results", "step7", "craftax-full", "phase0")
POOL = os.path.join(HERE, "_pool.pkl")
S = 10


def box_downsample(img, k):
    h, w, c = img.shape
    return img.reshape(h // k, k, w // k, k, c).mean(axis=(1, 3))


def main():
    import jax
    import jax.numpy as jnp
    from craftax.craftax import constants as C
    import craftax.craftax.renderer as R

    rep = {"S": S, "twoS": 2 * S}

    tex_s = C.load_all_textures_given_size(S)
    tex_2s = C.load_all_textures_given_size(2 * S)
    orig = R.load_all_textures
    R.load_all_textures = lambda: {S: tex_s, 2 * S: tex_2s}
    try:
        r_s = R.make_craftax_pixel_renderer(S)
        r_2s = R.make_craftax_pixel_renderer(2 * S)
    finally:
        R.load_all_textures = orig

    pool = pickle.load(open(POOL, "rb"))
    states = pool["rollout"][:40] + pool["level_shifted"]
    obs_h = C.OBS_DIM[0]
    world_rows = obs_h * S

    def stat(states_sub):
        exact, n, mx, tot = 0, 0, 0, 0.0
        we, wn, se, sn = 0, 0, 0, 0
        for st in states_sub:
            a = np.round(np.asarray(r_s(st))).clip(0, 255).astype(np.uint8)
            b = np.round(box_downsample(np.asarray(r_2s(st)).astype(np.float64), 2)
                         ).clip(0, 255).astype(np.uint8)
            eq = (a == b)
            exact += int(eq.sum()); n += eq.size
            d = np.abs(a.astype(int) - b.astype(int))
            mx = max(mx, int(d.max())); tot += float(d.sum())
            we += int(eq[:world_rows].sum()); wn += eq[:world_rows].size
            se += int(eq[world_rows:].sum()); sn += eq[world_rows:].size
        return {"n_states": len(states_sub), "exact_frac": exact / max(n, 1),
                "world_exact_frac": we / max(wn, 1), "strip_exact_frac": se / max(sn, 1),
                "mean_abs": tot / max(n, 1), "max_abs": mx}

    def band(st):
        lvl = int(np.asarray(st.player_level))
        light = float(np.asarray(st.light_level))
        day = light if lvl == 0 else 1.0
        return "day" if day == 1.0 else ("dusk" if day >= 0.5 else "night")

    groups = {"all": states}
    for b in ("day", "dusk", "night"):
        g = [s for s in states if band(s) == b]
        if g:
            groups[b] = g
    groups["level0"] = [s for s in states if int(np.asarray(s.player_level)) == 0]
    groups["level_gt0"] = [s for s in states if int(np.asarray(s.player_level)) > 0]
    rep["official_resolution_consistency"] = {k: stat(v) for k, v in groups.items()}
    rep["strip_geometry_scales_by_two"] = {
        "int(0.8*S)": int(0.8 * S), "int(0.8*2S)": int(0.8 * 2 * S),
        "int(0.4*S)": int(0.4 * S), "int(0.4*2S)": int(0.4 * 2 * S),
        "left_space_S": (S - int(0.8 * S)) // 2, "left_space_2S": (2 * S - int(0.8 * 2 * S)) // 2,
        "scales_exactly": (int(0.8 * 2 * S) == 2 * int(0.8 * S)
                           and int(0.4 * 2 * S) == 2 * int(0.4 * S)
                           and (2 * S - int(0.8 * 2 * S)) // 2 == 2 * ((S - int(0.8 * S)) // 2)),
        "note": "unlike Classic at 16->32 (int(0.8*16)=12 but int(0.8*32)=25), full Craftax's "
                "10->20 geometry does scale by two -- but the ICONS are loaded by resampling the "
                "source PNG at each size, which is not 2x replication of the S icon. Measured above.",
    }

    # ---------------------------------------------------------------- dead-dimension risk
    render = r_s
    dead = {}
    lvl0 = [s for s in states if int(np.asarray(s.player_level)) == 0]
    lmvals = {}
    for s in states[:20]:
        lvl = int(np.asarray(s.player_level))
        pp = np.asarray(s.player_position)
        pad = C.MAX_OBS_DIM + 2
        v = np.pad(np.asarray(s.light_map)[lvl], pad, constant_values=0.0)
        tl = pp - np.array(C.OBS_DIM) // 2 + pad
        w = v[tl[0]:tl[0] + C.OBS_DIM[0], tl[1]:tl[1] + C.OBS_DIM[1]]
        lmvals.setdefault(lvl, []).append(float(w.min()))
    dead["light_map_window_min_by_level"] = {
        str(k): {"min": min(v), "max": max(v), "all_one": all(x == 1.0 for x in v)}
        for k, v in lmvals.items()}

    base = lvl0[0]
    day = base.replace(light_level=jnp.float32(1.0))
    night = base.replace(light_level=jnp.float32(0.1))
    k2 = jax.random.PRNGKey(999)
    dead["state_rng_moves_pixels"] = {
        "level0_day": int((np.asarray(render(day.replace(state_rng=k2)))
                           != np.asarray(render(day))).sum()),
        "level0_night": int((np.asarray(render(night.replace(state_rng=k2)))
                             != np.asarray(render(night))).sum()),
    }
    for lvl in (1, 4, 8):
        s = pool["level_shifted"][lvl]
        sn = s.replace(light_level=jnp.float32(0.1))
        dead["state_rng_moves_pixels"][f"level{lvl}_light0.1"] = int(
            (np.asarray(render(sn.replace(state_rng=k2))) != np.asarray(render(sn))).sum())
        dead.setdefault("light_level_moves_pixels", {})[f"level{lvl}"] = int(
            (np.asarray(render(s.replace(light_level=jnp.float32(0.0))))
             != np.asarray(render(s.replace(light_level=jnp.float32(1.0))))).sum())
    dead["light_level_moves_pixels"] = dead.get("light_level_moves_pixels", {})
    dead["light_level_moves_pixels"]["level0"] = int(
        (np.asarray(render(day)) != np.asarray(render(night))).sum())
    dead["conclusion"] = (
        "The four PRNG-key slots reach a pixel only where BOTH player_level == 0 and "
        "light_level < 0.5; light_level reaches a pixel only on level 0; the per-tile light_map "
        "slots reach a pixel only where the level is not fully lit. A pool that is not stratified "
        "over (level, band) therefore reports dead dimensions that are properties of the reference.")
    rep["dead_dim_risk"] = dead

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "resolution_and_dead_dims.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    sys.exit(main())
