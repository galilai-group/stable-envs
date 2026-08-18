"""What does each fixed-camera procgen game actually put in a frame?

The seven games whose `game_reset` sets `center_agent = false` (or leaves it false outside
MemoryMode) share maze/heist/miner's fixed full-world camera, so the scrolling path is not needed
and the shipped renderer's two raster rules already cover them. What is NOT known without measuring
is the *state contract*: how many entity slots a frame can hold, which Entity fields vary (so are
state) and which are constants of the type, and whether the grid reaches a pixel at all.

Everything here is harvested from real rollouts of the official env, because a synthetic prior over
the alphabet would size the slot tables for states no episode reaches.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import state_io  # noqa: E402

GAMES = ["bigfish", "bossfight", "chaser", "dodgeball", "leaper", "plunder", "starpilot"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recon.json")

# fields that are pure dynamics and can never reach a pixel (they are not read by draw_entity)
DYNAMIC = {"vx", "vy", "will_erase", "collides_with_entities", "collision_margin", "vrot",
           "fire_time", "spawn_time", "life_time", "expire_time", "friction", "smart_step",
           "avoids_collisions", "auto_erase", "health", "theta", "grow_rate", "alpha_decay",
           "climber_spawn_x"}
# what draw_entity reads: get_object_rect(x, y, rx, ry, use_abs_coords), rotation, is_reflected,
# image_type, image_theme, alpha, and render_z for the pass it lands in.
RENDER = ["x", "y", "rx", "ry", "type", "image_type", "image_theme", "render_z", "rotation",
          "is_reflected", "alpha", "use_abs_coords"]


def roll(game, num=16, steps=400, warm=64, seed=0):
    from procgen import ProcgenGym3Env
    env = ProcgenGym3Env(num=num, env_name=game, distribution_mode="hard", rand_seed=seed,
                         start_level=0, num_levels=0, use_backgrounds=True)
    rng = np.random.default_rng(seed)
    out = []
    for t in range(steps):
        env.act(rng.integers(0, 15, size=num).astype(np.int32))
        if t < warm:
            continue
        for blob in env.get_state():
            out.append(state_io.parse(blob))
    return out


def summarise(game, states):
    rep = {"game": game, "n_states": len(states)}
    b0 = states[0]
    rep["main_wh"] = [b0["main_width"], b0["main_height"]]
    rep["grid_wh"] = [b0["grid_w"], b0["grid_h"]]
    rep["center_agent"] = b0["center_agent"]
    rep["blob_bytes"] = int(np.mean([len(state_io.dump(s)) for s in states[:8]]))
    rep["rng_text_bytes"] = state_io.rng_text_bytes(b0)

    # ---- grid
    gvals = set()
    for s in states:
        gvals |= set(np.unique(s["grid"]).tolist())
    rep["grid_values"] = sorted(int(v) for v in gvals)
    rep["grid_all_space"] = rep["grid_values"] == [100]

    # ---- background
    rep["bg_index_range"] = [int(min(s["background_index"] for s in states)),
                             int(max(s["background_index"] for s in states))]
    rep["bg_tile_ratio"] = sorted({round(float(s["bg_tile_ratio"]), 6) for s in states})
    rep["bg_pct_x"] = sorted({round(float(s["bg_pct_x"]), 4) for s in states})[:6]
    rep["use_procgen_background"] = sorted({int(s["use_procgen_background"]) for s in states})

    # ---- entities
    counts = [len(s["entities"]) for s in states]
    rep["n_entities"] = {"min": min(counts), "max": max(counts), "mean": float(np.mean(counts)),
                         "p99": float(np.percentile(counts, 99))}
    # per (type, render_z): how many at once, and which render fields vary
    per = defaultdict(lambda: {"max_at_once": 0, "vals": defaultdict(set)})
    for s in states:
        seen = defaultdict(int)
        for e in s["entities"]:
            k = (int(e["type"]), int(e["render_z"]))
            seen[k] += 1
            d = per[k]["vals"]
            for f in RENDER:
                v = e[f]
                d[f].add(round(float(v), 6) if isinstance(v, float) else int(v))
        for k, c in seen.items():
            per[k]["max_at_once"] = max(per[k]["max_at_once"], c)
    ent = []
    for (ty, z), d in sorted(per.items()):
        # `type` and `render_z` are the KEY here, so the observed-value sets below are written under
        # `seen_*` rather than over the key — the first draft overwrote them with lists.
        row = {"type": ty, "render_z": z, "max_at_once": d["max_at_once"]}
        for f in RENDER:
            if f in ("type", "render_z"):
                continue
            vs = d["vals"][f]
            row[f] = (sorted(vs) if len(vs) <= 8 else
                      {"n": len(vs), "min": min(vs), "max": max(vs)})
        ent.append(row)
    rep["entities_by_type"] = ent
    # varying-but-non-render fields, as a check that nothing render-relevant was missed
    rep["dynamic_fields_ignored"] = sorted(DYNAMIC)
    return rep


def main():
    reps = {}
    for g in GAMES:
        try:
            st = roll(g)
            reps[g] = summarise(g, st)
            e = reps[g]["n_entities"]
            print(f"{g:10s} grid={reps[g]['grid_wh']} all_space={reps[g]['grid_all_space']} "
                  f"gridvals={reps[g]['grid_values'][:8]} ents {e['min']}-{e['max']} "
                  f"(p99 {e['p99']:.0f}) bg={reps[g]['bg_index_range']} "
                  f"tile_ratio={reps[g]['bg_tile_ratio']}", flush=True)
            for r in reps[g]["entities_by_type"]:
                print(f"    type={r['type']:3d} z={r['render_z']:2d} max={r['max_at_once']:3d} "
                      f"img_type={r['image_type']} theme={r['image_theme']} "
                      f"rx={r['rx']} ry={r['ry']} rot={r['rotation']} refl={r['is_reflected']} "
                      f"alpha={r['alpha']} abs={r['use_abs_coords']}", flush=True)
        except Exception as exc:                                        # noqa: BLE001
            reps[g] = {"game": g, "error": f"{type(exc).__name__}: {exc}"}
            print(f"{g:10s} FAILED {exc}", flush=True)
    with open(OUT, "w") as fh:
        json.dump(reps, fh, indent=1, default=str)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
