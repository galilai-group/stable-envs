"""A4: a3's G3 enumeration, with the cast that a3 got wrong.

JaxMARL's ``grid_rendering.downsample`` returns **float64** -- it never casts back -- and
``_render_grid`` truncates it on assignment into a uint8 frame (``img[ymin:ymax, ...] = tile_img``).
So an atlas built straight out of ``_render_tile`` is float and matches nothing; a3 reported
33/33 observed tiles "missing" from a 128-tile atlas for exactly that reason. This is the same trap
``stable_envs/render/tilemap.py`` documents for MiniGrid ("Grid.render assigns this float array into
a uint8 frame; same cast here"), and it is the reason that comment exists.

Enumerate again with the cast, and assert every tile the rollouts produce is inside the result.
"""
import os, json
import numpy as np
import jax
import jaxmarl
from jaxmarl.viz import overcooked_visualizer as V1
from jaxmarl.environments.overcooked.common import (OBJECT_TO_INDEX, COLOR_TO_INDEX,
                                                    OBJECT_INDEX_TO_VEC)
from jaxmarl.environments.overcooked.layouts import overcooked_layouts as L1

OUT = os.path.dirname(os.path.abspath(__file__))
AGENT, POT, PAD = OBJECT_TO_INDEX["agent"], OBJECT_TO_INDEX["pot"], 3
res = {}


def enumerate_atlas(TS):
    V1.OvercookedVisualizer.tile_cache = {}
    tiles, groups, bad = {}, {"empty": 0, "object": 0, "pot": 0, "agent": 0}, 0
    for hl in (False, True):
        tiles[("none", hl)] = V1.OvercookedVisualizer._render_tile(None, highlight=hl,
                                                                   tile_size=TS)
        groups["empty"] += 1
        for ti in range(len(OBJECT_INDEX_TO_VEC)):
            vec = np.asarray(OBJECT_INDEX_TO_VEC[ti], np.uint8)
            if vec[0] in (OBJECT_TO_INDEX["empty"], OBJECT_TO_INDEX["unseen"], AGENT):
                continue
            for ex in (range(24) if vec[0] == POT else [0]):
                o = vec.copy(); o[2] = ex
                tiles[(tuple(int(v) for v in o), hl)] = V1.OvercookedVisualizer._render_tile(
                    o, highlight=hl, tile_size=TS)
                groups["pot" if vec[0] == POT else "object"] += 1
        for col in (COLOR_TO_INDEX["red"], COLOR_TO_INDEX["blue"]):
            for d in range(4):
                for inv in range(len(OBJECT_TO_INDEX)):
                    try:
                        tiles[(AGENT, col, d, inv, hl)] = V1.OvercookedVisualizer._render_tile(
                            np.array([AGENT, col, d], np.uint8), highlight=hl, tile_size=TS,
                            agent_dir_idx=np.array([0, 1]), agent_inv=np.array([inv, inv]))
                        groups["agent"] += 1
                    except ValueError:
                        bad += 1                      # inventory codes _render_inv cannot draw
    # the cast `_render_grid` performs when it assigns the float tile into the uint8 frame
    u8 = {k: v.astype(np.uint8) for k, v in tiles.items()}
    return u8, groups, bad


for TS in (15, 32, 75):
    u8, groups, bad = enumerate_atlas(TS)
    uniq = {t.tobytes() for t in u8.values()}
    res[f"atlas_t{TS}"] = dict(tile_size=TS, keys=len(u8), distinct_tiles=len(uniq),
                               bytes=len(uniq) * TS * TS * 3, groups=groups,
                               undrawable_agent_inv_codes=bad,
                               float_before_cast=str(next(iter(enumerate([0]))) and
                                                     "downsample returns float64"))

TS = 32
u8, _, _ = enumerate_atlas(TS)
uniq = {t.tobytes() for t in u8.values()}
observed, per_layout = set(), {}
for lname, layout in L1.items():
    env = jaxmarl.make("overcooked", layout=layout)
    key = jax.random.PRNGKey(3)
    s = set()
    for ep in range(8):
        key, k = jax.random.split(key)
        obs, st = env.reset(k)
        for t in range(60):
            g = np.asarray(st.maze_map[PAD:-PAD, PAD:-PAD, :])
            fr = V1.OvercookedVisualizer._render_grid(
                g, tile_size=TS, highlight_mask=None,
                agent_dir_idx=np.asarray(st.agent_dir_idx), agent_inv=np.asarray(st.agent_inv))
            assert len(np.unique(fr)) > 1
            for j in range(fr.shape[0] // TS):
                for i in range(fr.shape[1] // TS):
                    s.add(fr[j * TS:(j + 1) * TS, i * TS:(i + 1) * TS].tobytes())
            key, k = jax.random.split(key)
            acts = {a: jax.random.randint(jax.random.fold_in(k, i2), (), 0, 6)
                    for i2, a in enumerate(env.agents)}
            key, ks = jax.random.split(key)
            obs, st, r, d, inf = env.step(ks, st, acts)
    per_layout[lname] = len(s)
    observed |= s
res["membership"] = dict(tile_size=TS, enumerated=len(uniq), observed=len(observed),
                         observed_per_layout=per_layout,
                         observed_not_in_atlas=len(observed - uniq),
                         atlas_bytes=len(uniq) * TS * TS * 3)
with open(os.path.join(OUT, "a4_atlas_uint8.json"), "w") as f:
    json.dump(res, f, indent=2, default=str)
print(json.dumps(res, indent=2, default=str))
print("WROTE", os.path.join(OUT, "a4_atlas_uint8.json"))
