"""Facts the torch renderer's SHAPE depends on. Each one is a constant that reads plausibly and
could be wrong.

  1. how many grid cells can cover one device column / row (the fused layer count)
  2. whether the order of heist's z=0 entities is ever observable (do two of them share a pixel?)
  3. the alphabet of grid values, entity types, rotations, reflections and background indices that
     real rollouts actually produce
  4. background image dimensions, which decide what the atlas costs
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qtblit as Q  # noqa: E402
import refrender as R  # noqa: E402
import state_io as S  # noqa: E402

DIMS = {"maze": 25, "heist": 13, "miner": 20}


def cover_counts(mw, mh, size=64):
    unit, view_dim, x_off, y_off, _ = Q.prepare(mw, mh, False, (0, 0), 0.0, size)
    colc = np.zeros(size, int)
    rowc = np.zeros(size, int)
    for x in range(mw):
        rx, ry, rw, rh = Q.screen_rect(x, 1, 1, 1, unit, view_dim, x_off, y_off, eps=0.02)
        l, _, w, _ = Q.to_rect(rx, ry, rw, rh)
        for c in range(max(int(l), 0), min(int(l) + int(w), size)):
            colc[c] += 1
    for y in range(mh):
        rx, ry, rw, rh = Q.screen_rect(0, y + 1, 1, 1, unit, view_dim, x_off, y_off, eps=0.02)
        _, t, _, h = Q.to_rect(rx, ry, rw, rh)
        for r in range(max(int(t), 0), min(int(t) + int(h), size)):
            rowc[r] += 1
    return int(colc.max()), int(rowc.max()), int(colc.min()), int(rowc.min())


def main():
    from procgen import ProcgenGym3Env
    print("max cells covering one device column/row (and min), at several sizes:")
    for g, d in DIMS.items():
        for size in (64, 128, 256, 512):
            print(f"   {g:6s} {d}x{d} @ {size}: {cover_counts(d, d, size)}")

    rng = np.random.default_rng(0)
    for game in ("maze", "heist", "miner"):
        seen = defaultdict(set)
        order_changes = 0
        nf = 0
        for level in range(30):
            env = ProcgenGym3Env(num=1, env_name=game, distribution_mode="hard", rand_seed=0,
                                 start_level=level, num_levels=1)
            for step in range(10):
                env.act(rng.integers(0, 15, size=1).astype(np.int32))
                st = S.parse(env.get_state()[0])
                seen["grid"].update(np.unique(st["grid"]).tolist())
                seen["bg"].add(st["background_index"])
                seen["nent"].add(len(st["entities"]))
                for e in st["entities"]:
                    seen["type"].add(e["type"])
                    seen[f"rot_t{e['type']}"].add(round(e["rotation"], 4) == 0.0)
                    seen[f"refl_t{e['type']}"].add(e["is_reflected"])
                    seen[f"rxry_t{e['type']}"].add((round(e["rx"], 5), round(e["ry"], 5)))
                    seen[f"z_t{e['type']}"].add(e["render_z"])
                if game == "heist":
                    a = R.render(game, st["grid"], st["entities"], st["background_index"],
                                 st["bg_pct_x"], st["has_keys"])
                    ents = list(st["entities"])
                    z0 = [e for e in ents if e["render_z"] == 0]
                    other = [e for e in ents if e["render_z"] != 0]
                    b = R.render(game, st["grid"], other[:1] + z0[::-1] + other[1:],
                                 st["background_index"], st["bg_pct_x"], st["has_keys"])
                    order_changes += int(not np.array_equal(a, b))
                    nf += 1
        print(f"\n{game}: grid values {sorted(seen['grid'])}, bg indices "
              f"{min(seen['bg'])}..{max(seen['bg'])} ({len(seen['bg'])} seen), "
              f"n_entities {sorted(seen['nent'])}")
        for k in sorted(seen):
            if k.startswith(("rot_", "refl_", "rxry_", "z_", "type")):
                print(f"    {k}: {sorted(seen[k])}")
        if game == "heist":
            print(f"    z=0 entity order observable in {order_changes}/{nf} frames")

    print("\nbackground image sizes:")
    for name, lst in (("topdown", R.TOPDOWN_BG), ("platform+space", R.PLATFORM_BG + R.SPACE_BG)):
        sizes = [R.background(p).shape[:2] for p in lst]
        tot = sum(h * w * 3 for h, w in sizes)
        print(f"   {name}: {len(lst)} images, max {max(sizes)}, total {tot/1e6:.1f} MB raw")


if __name__ == "__main__":
    main()
