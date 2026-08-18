"""Two facts chaser's SHAPE depends on, before any of it is written.

1. **Do the four enemy sprites share a source size?** The ENEMY slot's sprite is state (it is
   `image_for_type`'s `cur_time` fold), so if `enemyWalking_1b` / `enemyFlying_1..3` differ in pixel
   dimensions then the blit's source extent is per environment rather than a constant, and
   `_axis_map` has to be fed a tensor. Measured rather than assumed.

2. **Is a fixed z=0 slot order safe?** heist's z=0 entities were measured never to share a device
   pixel, so their draw order is unobservable. chaser's are not obviously in that position: an ENEMY
   (r=0.5) can stand on a LARGE_ORB (r=0.4), and two enemies can cross. The test is the one
   `probe09_facts.py` used — render every real state with the z=0 entity list reversed and count the
   frames that change.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import state_io as S  # noqa: E402

OUT = os.path.join(HERE, "chaser_order.json")
ASSET_ROOT = None
SPRITES = ["misc_assets/enemyFloating_1b.png", "misc_assets/yellowCrystal.png",
           "misc_assets/enemyWalking_1b.png", "misc_assets/enemySpikey_1b.png",
           "misc_assets/tileStone_slope.png", "misc_assets/enemyFlying_1.png",
           "misc_assets/enemyFlying_2.png", "misc_assets/enemyFlying_3.png"]


def main():
    import importlib.util
    from PIL import Image
    from procgen import ProcgenGym3Env
    spec = importlib.util.find_spec("procgen")
    root = os.path.join(os.path.dirname(spec.origin), "data", "assets")

    rep = {"sprite_sizes": {}}
    for p in SPRITES:
        with Image.open(os.path.join(root, p)) as im:
            rep["sprite_sizes"][p] = list(im.size)
    print("sprite sizes (w, h):")
    for k, v in rep["sprite_sizes"].items():
        print(f"   {k:44s} {v}")
    enemy = [rep["sprite_sizes"][p] for p in SPRITES[2:3] + SPRITES[5:8]]
    rep["enemy_sprites_same_size"] = len({tuple(v) for v in enemy}) == 1
    print("the four ENEMY sprites share a size:", rep["enemy_sprites_same_size"], flush=True)

    # ---- 2. z=0 order sensitivity, on real states
    rng = np.random.default_rng(0)
    changed = 0
    frames = 0
    overlap_frames = 0
    for level in range(30):
        env = ProcgenGym3Env(num=1, env_name="chaser", distribution_mode="hard", rand_seed=0,
                             start_level=level, num_levels=1, use_backgrounds=True)
        for _ in range(12):
            env.act(rng.integers(0, 15, size=1).astype(np.int32))
            blob = env.get_state()[0]
            st = S.parse(blob)
            _, obs, _ = env.observe()
            a = np.asarray(obs["rgb"][0]).copy()

            z0 = [e for e in st["entities"] if e["render_z"] == 0]
            rest = [e for e in st["entities"] if e["render_z"] != 0]
            # do any two z=0 entities' world discs overlap at all? (a cheap necessary condition)
            ov = False
            for i in range(len(z0)):
                for j in range(i + 1, len(z0)):
                    if (abs(z0[i]["x"] - z0[j]["x"]) < z0[i]["rx"] + z0[j]["rx"]
                            and abs(z0[i]["y"] - z0[j]["y"]) < z0[i]["ry"] + z0[j]["ry"]):
                        ov = True
            overlap_frames += int(ov)

            st2 = dict(st)
            st2["entities"] = rest[:1] + z0[::-1] + rest[1:]
            env.set_state([S.dump(st2)])
            _, obs2, _ = env.observe()
            b = np.asarray(obs2["rgb"][0]).copy()
            changed += int(not (a == b).all())
            frames += 1
            env.set_state([blob])                       # put it back so the rollout continues
    rep["z0_order"] = {"frames": frames, "changed_when_reversed": changed,
                       "frames_with_world_overlap": overlap_frames}
    print(f"z=0 order: {changed}/{frames} frames changed when reversed; "
          f"{overlap_frames}/{frames} frames had two z=0 entities overlapping in world space",
          flush=True)

    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
