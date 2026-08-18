"""A1: the four gates on JaxMARL's Overcooked, and the one question that decides the family.

docs/research/overcooked.md records Overcooked-AI as "go, all four gates pass -- and NOT YET BUILT",
blocked on exactly one thing: `overcooked_ai_py`'s `StateVisualizer` draws the pot's cooking timer as
**font-rendered Roboto text at a fixed 20 px, after the x5 tile scale**, so pot tiles are
tile_size-specific and cannot be composed from the 15x15 source art.

This asks whether a JAX/GPU-native Overcooked draws that timer from something finite instead, and
CONFIRMS IT BY RENDERING rather than by reading. It then runs the gates on that renderer.

It also settles, explicitly, whether the JAX frame is the same picture as `overcooked_ai_py`'s --
it is not, and the evidence is dumped for the comparison script rather than asserted.
"""
import os, sys, json, time, hashlib, itertools, inspect
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
res = {}

# ---------------------------------------------------------------- 0. GPU, and no text anywhere
import jax
import jax.numpy as jnp
res["jax_version"] = jax.__version__
res["jax_devices"] = [str(d) for d in jax.devices()]
res["has_gpu"] = any(d.platform == "gpu" for d in jax.devices())
if not res["has_gpu"]:
    print("WARNING: no GPU device visible to jax --", res["jax_devices"])

import jaxmarl
from jaxmarl.viz import overcooked_visualizer as V1
from jaxmarl.viz import overcooked_v2_visualizer as V2
from jaxmarl.viz import grid_rendering, grid_rendering_v2
from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX, COLOR_TO_INDEX
from jaxmarl.environments.overcooked.layouts import overcooked_layouts as L1
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts as L2

TEXT_TOKENS = ("freetype", "font", "SysFont", "render_text", "Roboto", "pygame", "PIL.ImageDraw",
               "ImageFont", "putText", "cv2")
scan = {}
for mod in (V1, V2, grid_rendering, grid_rendering_v2):
    src = inspect.getsource(mod)
    scan[mod.__name__] = sorted({t for t in TEXT_TOKENS if t in src})
res["text_token_scan"] = scan
res["v1_render_primitives"] = sorted({n for n in dir(grid_rendering) if n.startswith("point_in")
                                      or n in ("fill_coords", "downsample", "rotate_fn",
                                               "highlight_img")})
res["v2_render_state_is_jitted"] = hasattr(V2.OvercookedV2Visualizer._render_state, "lower") or \
    "jit" in str(type(V2.OvercookedV2Visualizer._render_state))
res["v2_pot_progress_bar_source"] = inspect.getsource(
    V2.OvercookedV2Visualizer._render_pot).split("# Render progress bar")[-1].strip()
res["v1_pot_progress_bar_source"] = inspect.getsource(
    V1.OvercookedVisualizer._render_pot).split("# Render progress bar")[-1].strip()

# ---------------------------------------------------------------- 1. the pot timer, by rendering
POT = OBJECT_TO_INDEX["pot"]
BLACK = COLOR_TO_INDEX["black"]


def pot_tile(status, tile_size):
    V1.OvercookedVisualizer.tile_cache = {}
    return V1.OvercookedVisualizer._render_tile(np.array([POT, BLACK, status], np.uint8),
                                                tile_size=tile_size)


pot = {}
for ts in (8, 12, 20, 32, 64, 75):
    tiles = [pot_tile(s, ts) for s in range(24)]
    d = {}
    base = tiles[23]
    rows = set()
    for s in range(24):
        for s2 in range(s + 1, 24):
            if np.array_equal(tiles[s], tiles[s2]):
                d.setdefault(s, []).append(s2)
    # which rows differ between two adjacent cooking states -> where the "timer" lives
    a, b = tiles[10], tiles[11]
    rr = np.nonzero((a != b).any(axis=(1, 2)))[0]
    pot[ts] = dict(
        n_status_values=24,
        n_distinct_tiles=len({t.tobytes() for t in tiles}),
        duplicate_groups=d,
        differing_rows_status10_vs_11=[int(rr.min()), int(rr.max())] if len(rr) else [],
        differing_rows_as_fraction=[float(rr.min()) / ts, float(rr.max() + 1) / ts] if len(rr) else [],
        n_differing_pixels_10_vs_11=int((a != b).any(-1).sum()),
        tile_shape=list(tiles[0].shape),
        distinct_colors=int(len(np.unique(np.stack(tiles).reshape(-1, 3), axis=0))),
    )
res["pot_timer"] = pot
np.save(os.path.join(OUT, "pot_tiles_t32.npy"), np.stack([pot_tile(s, 32) for s in range(24)]))

# ---------------------------------------------------------------- 2. roll out and render, v1
V1.OvercookedVisualizer.tile_cache = {}
PAD = 3          # agent_view_size (5) - 2, the crop `animate` uses


def frame_of(state, tile_size=32):
    g = np.asarray(state.maze_map[PAD:-PAD, PAD:-PAD, :])
    return OvercookedVisualizer_render(g, state, tile_size)


def OvercookedVisualizer_render(g, state, tile_size):
    return V1.OvercookedVisualizer._render_grid(
        g, tile_size=tile_size, highlight_mask=None,
        agent_dir_idx=np.asarray(state.agent_dir_idx), agent_inv=np.asarray(state.agent_inv))


def render_slice(state):
    """Exactly the bytes `_render_grid` reads: the cropped maze_map, agent_dir_idx, agent_inv."""
    g = np.asarray(state.maze_map[PAD:-PAD, PAD:-PAD, :], np.uint8)
    return (g.tobytes(), np.asarray(state.agent_dir_idx).astype(np.int32).tobytes(),
            np.asarray(state.agent_inv).astype(np.int32).tobytes())


rollouts = {}
for lname, layout in L1.items():
    env = jaxmarl.make("overcooked", layout=layout)
    key = jax.random.PRNGKey(0)
    recs = []
    for ep in range(14):
        key, k = jax.random.split(key)
        obs, state = env.reset(k)
        hist = []
        for t in range(60):
            key, k = jax.random.split(key)
            acts = {a: jax.random.randint(jax.random.fold_in(k, i), (), 0, 6)
                    for i, a in enumerate(env.agents)}
            recs.append((render_slice(state), tuple(hist), state))
            hist.append(tuple(int(v) for v in acts.values()))
            key, ks = jax.random.split(key)
            obs, state, rew, done, info = env.step(ks, state, acts)
    rollouts[lname] = (env, recs)
    print(f"{lname}: {len(recs)} states", flush=True)

# ---- G1a: render twice
g1a = {"pairs": 0, "max_abs": 0}
for lname, (env, recs) in rollouts.items():
    for rs, hist, st in recs[:60]:
        a = frame_of(st)
        b = frame_of(st)
        g1a["pairs"] += 1
        g1a["max_abs"] = max(g1a["max_abs"], int(np.abs(a.astype(int) - b.astype(int)).max()))
res["G1_render_twice"] = g1a

# ---- G1b: two different histories to the same render state
g1b = {"layouts": {}}
for lname, (env, recs) in rollouts.items():
    seen = {}
    pairs = 0
    maxabs = 0
    uniform = 0
    for rs, hist, st in recs:
        if rs in seen:
            h0, st0 = seen[rs]
            if h0 != hist:
                a, b = frame_of(st0), frame_of(st)
                if len(np.unique(a)) < 2:
                    uniform += 1
                pairs += 1
                maxabs = max(maxabs, int(np.abs(a.astype(int) - b.astype(int)).max()))
        else:
            seen[rs] = (hist, st)
    g1b["layouts"][lname] = dict(distinct_render_states=len(seen), cross_history_pairs=pairs,
                                 max_abs=maxabs, uniform_reference_frames=uniform)
res["G1_cross_history"] = g1b

# ---- G2: render slice / frame
g2 = {}
for lname, (env, recs) in rollouts.items():
    st = recs[0][2]
    fr = frame_of(st)
    rs = render_slice(st)
    H, W = env.layout["height"], env.layout["width"]
    naive = sum(len(x) for x in rs)
    n_pot = int((np.asarray(st.maze_map[PAD:-PAD, PAD:-PAD, 0]) == POT).sum())
    n_counter = int((np.asarray(st.maze_map[PAD:-PAD, PAD:-PAD, 0]) == OBJECT_TO_INDEX["wall"]).sum())
    compact = 2 * 2 + 2 * 1 + n_pot * 1 + n_counter * 1        # pos+dir(2B/agent) inv(1B) pot(1B) counter(1B)
    g2[lname] = dict(grid=[int(W), int(H)], frame_bytes=int(fr.size),
                     frame_shape=list(fr.shape),
                     render_slice_bytes=int(naive), ratio_naive=naive / fr.size,
                     compact_bytes=int(compact), ratio_compact=compact / fr.size,
                     n_pots=n_pot, n_counters=n_counter,
                     distinct_colors=int(len(np.unique(fr.reshape(-1, 3), axis=0))))
res["G2"] = g2

# ---- G3: enumerate the tile alphabet, and is it shared across layouts
TS = 32
alphabet, per_layout = {}, {}
for lname, (env, recs) in rollouts.items():
    s = set()
    for rs, hist, st in recs:
        fr = frame_of(st, TS)
        h, w = fr.shape[0] // TS, fr.shape[1] // TS
        for j in range(h):
            for i in range(w):
                s.add(fr[j * TS:(j + 1) * TS, i * TS:(i + 1) * TS].tobytes())
    per_layout[lname] = len(s)
    alphabet[lname] = s
union = set().union(*alphabet.values())
shared = set.intersection(*alphabet.values())
res["G3"] = dict(
    tiles_per_layout=per_layout, tiles_union=len(union), tiles_shared_by_all=len(shared),
    union_bytes=len(union) * TS * TS * 3,
    tile_cache_entries=len(V1.OvercookedVisualizer.tile_cache),
    tile_cache_is_class_level=("tile_cache" in V1.OvercookedVisualizer.__dict__),
    reachable_cell_codes=dict(
        obj_types=len(OBJECT_TO_INDEX),
        pot_status_values=24,
        agent_dir=4, agent_inv=len(OBJECT_TO_INDEX),
    ),
)

# ---- dump a real atlas + index for the tilemap.blit parity/bench in the other venv
lname = "cramped_room"
env, recs = rollouts[lname]
tiles = sorted(alphabet[lname])
tidx = {t: i for i, t in enumerate(tiles)}
frames, idxs = [], []
for rs, hist, st in recs[:512]:
    fr = frame_of(st, TS)
    h, w = fr.shape[0] // TS, fr.shape[1] // TS
    ix = np.empty((h, w), np.int32)
    for j in range(h):
        for i in range(w):
            ix[j, i] = tidx[fr[j * TS:(j + 1) * TS, i * TS:(i + 1) * TS].tobytes()]
    frames.append(fr)
    idxs.append(ix)
atlas = np.stack([np.frombuffer(t, np.uint8).reshape(TS, TS, 3) for t in tiles])
np.savez_compressed(os.path.join(OUT, "blit_input_cramped_room.npz"),
                    atlas=atlas, idx=np.stack(idxs), frames=np.stack(frames), tile_size=TS)
res["blit_dump"] = dict(layout=lname, atlas=list(atlas.shape), idx=list(np.stack(idxs).shape),
                        frames=list(np.stack(frames).shape))

# ---- reference frame for the "same picture?" comparison
np.save(os.path.join(OUT, "jaxmarl_v1_cramped_room_t15.npy"), frame_of(recs[0][2], 15))
np.save(os.path.join(OUT, "jaxmarl_v1_cramped_room_t75.npy"), frame_of(recs[0][2], 75))

# ---------------------------------------------------------------- 3. OvercookedV2, the JAX renderer
v2 = {}
try:
    envv2 = jaxmarl.make("overcooked_v2", layout=L2["cramped_room"])
    key = jax.random.PRNGKey(0)
    obs, st2 = envv2.reset(key)
    viz = V2.OvercookedV2Visualizer(tile_size=32)
    f = viz._render_state(st2, None)
    f = np.asarray(f)
    # vmap over a batch of states
    states = []
    s = st2
    for t in range(64):
        key, k = jax.random.split(key)
        acts = {a: jax.random.randint(jax.random.fold_in(k, i), (), 0, 6)
                for i, a in enumerate(envv2.agents)}
        states.append(s)
        key, ks = jax.random.split(key)
        obs, s, r, d, i2 = envv2.step(ks, s, acts)
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    fb = jax.vmap(viz._render_state, in_axes=(0, None))(batch, None)
    fb.block_until_ready()
    v2 = dict(ok=True, frame_shape=list(f.shape), distinct_colors=int(
        len(np.unique(f.reshape(-1, 3), axis=0))),
        vmapped_batch_shape=list(fb.shape),
        n_layouts=len(L2),
        render_twice_max_abs=int(np.abs(np.asarray(viz._render_state(st2, None)).astype(int)
                                        - f.astype(int)).max()))
    np.save(os.path.join(OUT, "jaxmarl_v2_cramped_room_t32.npy"), f)
except Exception as e:                                                # noqa: BLE001
    import traceback; traceback.print_exc()
    v2 = dict(ok=False, error=f"{type(e).__name__}: {e}")
res["overcooked_v2"] = v2

with open(os.path.join(OUT, "a1_gates.json"), "w") as f:
    json.dump(res, f, indent=2, default=str)
print(json.dumps({k: v for k, v in res.items() if k != "pot_timer"}, indent=2, default=str)[:6000])
print("POT:", json.dumps(res["pot_timer"], indent=2, default=str)[:2500])
print("WROTE", os.path.join(OUT, "a1_gates.json"))
