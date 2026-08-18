"""A3: G3 enumerated rather than observed, and G4 timed against the official renderers.

a1 counted the tiles a random policy happened to reach (33). That is a lower bound and the wrong
number for "how big is the atlas" -- a random policy rarely fills a pot, so almost none of the 24
cooking-timer tiles appear. This enumerates the alphabet the way ``tilemap.py::_official_atlas``
enumerates MiniGrid's: by calling the official ``_render_tile`` over every reachable cell code, and
then asserting that every tile the rollouts actually produced is a member of it.

G4 is timed here too: JaxMARL v1's NumPy ``_render_grid`` (warm and cold tile cache) and
OvercookedV2's jitted, vmap-able ``_render_state`` on the GPU, against a2's ``tilemap.blit`` numbers.
"""
import os, sys, json, time
import numpy as np
import jax, jax.numpy as jnp

OUT = os.path.dirname(os.path.abspath(__file__))
import jaxmarl
from jaxmarl.viz import overcooked_visualizer as V1
from jaxmarl.viz import overcooked_v2_visualizer as V2
from jaxmarl.environments.overcooked.common import (OBJECT_TO_INDEX, COLOR_TO_INDEX,
                                                    OBJECT_INDEX_TO_VEC)
from jaxmarl.environments.overcooked.layouts import overcooked_layouts as L1
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts as L2

res = {"jax_devices": [str(d) for d in jax.devices()]}
TS = 32
PAD = 3
AGENT = OBJECT_TO_INDEX["agent"]
POT = OBJECT_TO_INDEX["pot"]

# ---------------------------------------------------------------- G3, enumerated
V1.OvercookedVisualizer.tile_cache = {}
tiles, entries, failed = {}, [], []
for hl in (False, True):
    tiles[("none", hl)] = V1.OvercookedVisualizer._render_tile(None, highlight=hl, tile_size=TS)
    entries.append(dict(kind="empty", highlight=hl))
    for ti in range(len(OBJECT_INDEX_TO_VEC)):
        vec = np.asarray(OBJECT_INDEX_TO_VEC[ti], np.uint8)
        if vec[0] in (OBJECT_TO_INDEX["empty"], OBJECT_TO_INDEX["unseen"], AGENT):
            continue
        extras = range(24) if vec[0] == POT else [0]
        for ex in extras:
            o = vec.copy(); o[2] = ex
            try:
                tiles[(tuple(int(v) for v in o), hl)] = V1.OvercookedVisualizer._render_tile(
                    o, highlight=hl, tile_size=TS)
                entries.append(dict(kind="object", obj=[int(v) for v in o], highlight=hl))
            except Exception as e:                                     # noqa: BLE001
                failed.append((int(vec[0]), ex, str(e)))
    for col in (COLOR_TO_INDEX["red"], COLOR_TO_INDEX["blue"]):
        for d in range(4):
            for inv in range(len(OBJECT_TO_INDEX)):
                o = np.array([AGENT, col, d], np.uint8)
                ainv = np.array([inv, inv])
                try:
                    t = V1.OvercookedVisualizer._render_tile(
                        o, highlight=hl, tile_size=TS, agent_dir_idx=np.array([0, 1]),
                        agent_inv=ainv)
                    tiles[(AGENT, col, d, inv, hl)] = t
                    entries.append(dict(kind="agent", color=int(col), dir=d, inv=inv, highlight=hl))
                except Exception as e:                                 # noqa: BLE001
                    failed.append((AGENT, (int(col), d, inv), str(e)))
uniq = {t.tobytes() for t in tiles.values()}
res["G3_enumerated"] = dict(
    tile_size=TS, keys_enumerated=len(tiles), distinct_tiles=len(uniq),
    bytes=len(uniq) * TS * TS * 3,
    unsupported_codes=len(failed), unsupported_examples=failed[:6],
    breakdown=dict(
        object_types=len(OBJECT_TO_INDEX), pot_status_values=24,
        agent_colors=2, agent_dirs=4, agent_inv_values=len(OBJECT_TO_INDEX), highlight=2),
)

# every tile the rollouts produce must be inside that atlas
observed = set()
for lname, layout in L1.items():
    env = jaxmarl.make("overcooked", layout=layout)
    key = jax.random.PRNGKey(3)
    for ep in range(6):
        key, k = jax.random.split(key)
        obs, st = env.reset(k)
        for t in range(50):
            g = np.asarray(st.maze_map[PAD:-PAD, PAD:-PAD, :])
            fr = V1.OvercookedVisualizer._render_grid(
                g, tile_size=TS, highlight_mask=None,
                agent_dir_idx=np.asarray(st.agent_dir_idx), agent_inv=np.asarray(st.agent_inv))
            for j in range(fr.shape[0] // TS):
                for i in range(fr.shape[1] // TS):
                    observed.add(fr[j * TS:(j + 1) * TS, i * TS:(i + 1) * TS].tobytes())
            key, k = jax.random.split(key)
            acts = {a: jax.random.randint(jax.random.fold_in(k, i2), (), 0, 6)
                    for i2, a in enumerate(env.agents)}
            key, ks = jax.random.split(key)
            obs, st, r, d, inf = env.step(ks, st, acts)
res["G3_membership"] = dict(observed_tiles=len(observed),
                            observed_not_in_enumerated_atlas=len(observed - uniq),
                            layouts=len(L1))

# ---------------------------------------------------------------- G4, official renderers
env = jaxmarl.make("overcooked", layout=L1["cramped_room"])
key = jax.random.PRNGKey(0)
obs, st = env.reset(key)
g = np.asarray(st.maze_map[PAD:-PAD, PAD:-PAD, :])
adi, ai = np.asarray(st.agent_dir_idx), np.asarray(st.agent_inv)


def t_render(n, cold):
    t0 = time.perf_counter()
    for _ in range(n):
        if cold:
            V1.OvercookedVisualizer.tile_cache = {}
        V1.OvercookedVisualizer._render_grid(g, tile_size=TS, highlight_mask=None,
                                             agent_dir_idx=adi, agent_inv=ai)
    return n / (time.perf_counter() - t0)


for _ in range(3):
    V1.OvercookedVisualizer._render_grid(g, tile_size=TS, highlight_mask=None,
                                         agent_dir_idx=adi, agent_inv=ai)
warm = t_render(300, False)
cold = t_render(3, True)
res["G4_official_v1"] = dict(frame_shape=[g.shape[0] * TS, g.shape[1] * TS, 3],
                             warm_cache_fps=warm, cold_cache_fps=cold,
                             note="warm = tile cache populated; cold = cache cleared per frame")

v2 = {}
try:
    e2 = jaxmarl.make("overcooked_v2", layout=L2["cramped_room"])
    k2 = jax.random.PRNGKey(0)
    obs2, s2 = e2.reset(k2)
    viz = V2.OvercookedV2Visualizer(tile_size=TS)
    states = []
    s = s2
    for t in range(64):
        states.append(s)
        k2, kk = jax.random.split(k2)
        acts = {a: jax.random.randint(jax.random.fold_in(kk, i), (), 0, 6)
                for i, a in enumerate(e2.agents)}
        k2, ks = jax.random.split(k2)
        obs2, s, r, d, inf = e2.step(ks, s, acts)
    stack = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    rows = []
    for B in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        idx = jnp.arange(B) % 64
        batch = jax.tree.map(lambda x: x[idx], stack)
        f = jax.jit(jax.vmap(viz._render_state, in_axes=(0, None)), static_argnums=(1,))
        out = f(batch, None); out.block_until_ready()
        it = max(3, min(60, int(4e8 / out.size)))
        t0 = time.perf_counter()
        for _ in range(it):
            f(batch, None).block_until_ready()
        rows.append(dict(batch=B, fps=B * it / (time.perf_counter() - t0)))
    best = max(rows, key=lambda r: r["fps"])
    v2 = dict(ok=True, frame_shape=list(np.asarray(viz._render_state(s2, None)).shape),
              rows=rows, best_fps=best["fps"], best_batch=best["batch"])
except Exception as e:                                                # noqa: BLE001
    import traceback; traceback.print_exc()
    v2 = dict(ok=False, error=f"{type(e).__name__}: {e}")
res["G4_official_v2_jax"] = v2

with open(os.path.join(OUT, "a3_atlas_and_speed.json"), "w") as f:
    json.dump(res, f, indent=2, default=str)
print(json.dumps(res, indent=2, default=str)[:6000])
print("WROTE", os.path.join(OUT, "a3_atlas_and_speed.json"))
