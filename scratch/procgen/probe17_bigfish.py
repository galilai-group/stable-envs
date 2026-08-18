"""bigfish: what a fixed-width state has to be able to hold, and what the fish sprites cost.

bigfish is the first procgen game in this repo whose entities are not a fixed cast. maze has one
agent, heist has 3+3+1+3, miner has 2 — all constants of the game. bigfish spawns a fish with
probability 1/10 per step and erases it when it leaves the world, so the count is a random variable
and a fixed-width state has to CAP it. The cap cannot be read off the source; this measures it.

Three more things the renderer's shape depends on:

  * fish carry a per-entity radius (`FISH_MIN_R = .25` to `FISH_MAX_R = 2`, sampled) and a per-entity
    theme (3 sprites, `choose_random_theme`), so `_EntSlot`'s constant `rx`/`ry`/`sprite` do not
    cover them. If the three sprites differ in pixel size the blit's source extent is per
    environment as well;
  * fish overlap each other constantly, so unlike heist and chaser the z=0 draw ORDER is observable
    and the slot assignment has to be the entity list order rather than a sort;
  * `water_backgrounds` is a 7-image list; whether they are square decides whether `bg_pct_x` is
    part of the render state (it is for miner and is not for maze/heist).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import state_io as S  # noqa: E402

OUT = os.path.join(HERE, "bigfish_facts.json")
WATER_BG = [f"water_backgrounds/{n}" for n in
            ["water1.png", "water2.png", "water3.png", "water4.png",
             "underwater1.png", "underwater2.png", "underwater3.png"]]
FISH_SPRITES = ["misc_assets/fishTile_074.png", "misc_assets/fishTile_078.png",
                "misc_assets/fishTile_080.png"]
AGENT_SPRITE = "misc_assets/fishTile_072.png"
FISH = 2


def main():
    import importlib.util
    from PIL import Image
    from procgen import ProcgenGym3Env
    spec = importlib.util.find_spec("procgen")
    root = os.path.join(os.path.dirname(spec.origin), "data", "assets")

    rep = {}
    # ---- sprites and backgrounds
    rep["sprite_sizes"] = {}
    for p in [AGENT_SPRITE] + FISH_SPRITES:
        with Image.open(os.path.join(root, p)) as im:
            rep["sprite_sizes"][p] = list(im.size)
    rep["fish_sprites_same_size"] = len({tuple(rep["sprite_sizes"][p]) for p in FISH_SPRITES}) == 1
    rep["bg_sizes"] = {}
    for p in WATER_BG:
        with Image.open(os.path.join(root, p)) as im:
            rep["bg_sizes"][p] = list(im.size)
    rep["all_backgrounds_square"] = all(w == h for w, h in rep["bg_sizes"].values())
    print("sprite sizes:", rep["sprite_sizes"])
    print("the three FISH sprites share a size:", rep["fish_sprites_same_size"])
    print("water backgrounds:", rep["bg_sizes"])
    print("all square:", rep["all_backgrounds_square"], flush=True)

    # ---- how many fish at once, over long episodes rather than a warm-up window
    rng = np.random.default_rng(0)
    num = 32
    env = ProcgenGym3Env(num=num, env_name="bigfish", distribution_mode="hard", rand_seed=0,
                         start_level=0, num_levels=0, use_backgrounds=True)
    hist = np.zeros(128, np.int64)
    pct_x = set()
    bgs = set()
    ry_over_rx = set()
    reflected_vs_dir = {"both": 0}
    n_steps = 2000
    for t in range(n_steps):
        env.act(rng.integers(0, 15, size=num).astype(np.int32))
        if t % 5:
            continue
        for blob in env.get_state():
            st = S.parse(blob)
            fish = [e for e in st["entities"] if e["type"] == FISH]
            hist[min(len(fish), 127)] += 1
            pct_x.add(round(float(st["bg_pct_x"]), 5))
            bgs.add(int(st["background_index"]))
            for e in fish:
                ry_over_rx.add((int(e["image_theme"]), round(e["rx"] / e["ry"], 5)))
    nz = np.nonzero(hist)[0]
    rep["fish_count"] = {"max": int(nz.max()), "p99": int(np.searchsorted(np.cumsum(hist) /
                                                                         hist.sum(), 0.99)),
                         "histogram_tail": {int(k): int(hist[k]) for k in nz[-8:]},
                         "samples": int(hist.sum()), "steps": n_steps, "envs": num}
    rep["bg_index_seen"] = sorted(bgs)
    rep["bg_pct_x_seen"] = sorted(pct_x)[:8]
    rep["aspect_ratio_per_theme"] = sorted(ry_over_rx)
    print("fish at once:", rep["fish_count"], flush=True)
    print("bg indices:", rep["bg_index_seen"], "bg_pct_x:", rep["bg_pct_x_seen"], flush=True)
    print("rx/ry per theme (should equal the sprite aspect ratio):",
          rep["aspect_ratio_per_theme"], flush=True)

    # ---- is the z=0 order observable? (fish overlap, so this should say YES, unlike heist/chaser)
    changed = 0
    frames = 0
    overlap = 0
    e2 = ProcgenGym3Env(num=1, env_name="bigfish", distribution_mode="hard", rand_seed=0,
                        start_level=0, num_levels=0, use_backgrounds=True)
    for t in range(600):
        e2.act(rng.integers(0, 15, size=1).astype(np.int32))
        if t < 100 or t % 3:
            continue
        blob = e2.get_state()[0]
        st = S.parse(blob)
        _, obs, _ = e2.observe()
        a = np.asarray(obs["rgb"][0]).copy()
        z0 = [e for e in st["entities"] if e["render_z"] == 0]
        rest = [e for e in st["entities"] if e["render_z"] != 0]
        ov = any(abs(z0[i]["x"] - z0[j]["x"]) < z0[i]["rx"] + z0[j]["rx"]
                 and abs(z0[i]["y"] - z0[j]["y"]) < z0[i]["ry"] + z0[j]["ry"]
                 for i in range(len(z0)) for j in range(i + 1, len(z0)))
        overlap += int(ov)
        st2 = dict(st)
        st2["entities"] = rest + z0[::-1]
        e2.set_state([S.dump(st2)])
        _, obs2, _ = e2.observe()
        changed += int(not (a == np.asarray(obs2["rgb"][0])).all())
        frames += 1
        e2.set_state([blob])
    rep["z0_order"] = {"frames": frames, "changed_when_reversed": changed,
                       "frames_with_world_overlap": overlap}
    print("z=0 order:", rep["z0_order"], flush=True)

    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
