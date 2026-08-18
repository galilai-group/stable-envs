"""Measurement-only probe: run the four gates (G1-G4) on Overcooked-AI.

Writes nothing but stdout + a small npz of example frames under se_logs.
"""
from __future__ import annotations

import os
import sys
import json
import time
import copy
import itertools
import collections

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
    ObjectState,
    SoupState,
    Recipe,
)
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.static import GRAPHICS_DIR, LAYOUTS_DIR

OUT = "/mnt/vast/home/randall.balestriero/se_logs/overcooked"
os.makedirs(OUT, exist_ok=True)


def hr(t):
    print("\n" + "=" * 78)
    print("== " + t)
    print("=" * 78)


def frame(vis, state, grid, **kw):
    """Return HxWx3 uint8 RGB from the visualiser."""
    surf = vis.render_state(state, grid, **kw)
    arr = pygame.surfarray.array3d(surf)  # (W, H, 3)
    return np.transpose(arr, (1, 0, 2)).copy()


# --------------------------------------------------------------------------
# 0. headless sanity: the surface must NOT be blank
# --------------------------------------------------------------------------
hr("0. HEADLESS SANITY (the blank-surface trap)")
mdp = OvercookedGridworld.from_layout_name("cramped_room")
grid = mdp.terrain_mtx
vis = StateVisualizer()
s0 = mdp.get_standard_start_state()
f0 = frame(vis, s0, grid)
print("SDL_VIDEODRIVER      =", os.environ["SDL_VIDEODRIVER"])
print("pygame version       =", pygame.version.ver)
print("frame shape / dtype  =", f0.shape, f0.dtype)
ncol0 = len(np.unique(f0.reshape(-1, 3), axis=0))
print("distinct RGB colours =", ncol0, "   (need > 3)")
assert ncol0 > 3, "BLANK SURFACE -- headless render produced a near-uniform frame"

# two visibly different states -> visibly different frames
sA = s0.deepcopy()
sB = s0.deepcopy()
sB.players[0].update_pos_and_or(sB.players[0].position, Direction.NORTH)
sB.players[1].update_pos_and_or(sB.players[1].position, Direction.WEST)
fA, fB = frame(vis, sA, grid), frame(vis, sB, grid)
ndiff = int((fA != fB).any(axis=2).sum())
print("orientation change   -> %d pixels differ (of %d)" % (ndiff, fA.shape[0] * fA.shape[1]))
assert ndiff > 0, "BLANK/CONSTANT RENDER -- two different states gave identical frames"
# move a player
sC = s0.deepcopy()
p = sC.players[0]
sC.players[0].update_pos_and_or((p.position[0], p.position[1] + 1), p.orientation)
fC = frame(vis, sC, grid)
print("position change      -> %d pixels differ" % int((fA != fC).any(axis=2).sum()))
np.savez_compressed(os.path.join(OUT, "sanity_frames.npz"), fA=fA, fB=fB, fC=fC)

# --------------------------------------------------------------------------
# G3: is the decoder a constant of the family?
# --------------------------------------------------------------------------
hr("G3. SHARED DECODER")
gfiles = sorted(os.listdir(GRAPHICS_DIR))
tot = 0
for f in gfiles:
    n = os.path.getsize(os.path.join(GRAPHICS_DIR, f))
    tot += n
    print("   %-24s %7d B" % (f, n))
print("   graphics dir: %d files, %d bytes total" % (len(gfiles), tot))

sheets = {
    "terrain": StateVisualizer.TERRAINS_IMG,
    "objects": StateVisualizer.OBJECTS_IMG,
    "soups": StateVisualizer.SOUPS_IMG,
    "chefs": StateVisualizer.CHEFS_IMG,
}
tot_frames = 0
for k, v in sheets.items():
    n = len(v.frames_rectangles)
    tot_frames += n
    sizes = set((r.w, r.h) for r in v.frames_rectangles.values())
    print("   sheet %-8s : %3d named frames, sprite sizes %s, sheet px %s"
          % (k, n, sizes, v.image.get_size()))
print("   TOTAL named sprite frames = %d  (+3 loose: arrow/interact/stay)" % tot_frames)

layouts = sorted(f for f in os.listdir(LAYOUTS_DIR) if f.endswith(".layout"))
print("   layouts shipped: %d" % len(layouts))
# are the sheets the SAME objects across two "resets" / two mdps?
mdp2 = OvercookedGridworld.from_layout_name("counter_circuit_o_1order")
vis2 = StateVisualizer()
same = all(
    getattr(StateVisualizer, a) is getattr(vis2.__class__, a)
    for a in ("TERRAINS_IMG", "OBJECTS_IMG", "SOUPS_IMG", "CHEFS_IMG")
)
print("   sprite sheets are class-level singletons shared across layouts/instances:", same)
# layouts fixed?
sizes = {}
for L in layouts:
    try:
        m = OvercookedGridworld.from_layout_name(L[:-7])
        sizes[L[:-7]] = (len(m.terrain_mtx[0]), len(m.terrain_mtx))
    except Exception as e:
        sizes[L[:-7]] = "ERR:%s" % type(e).__name__
n_ok = sum(1 for v in sizes.values() if not isinstance(v, str))
print("   layouts that instantiate: %d/%d" % (n_ok, len(layouts)))
# reset twice, same terrain?
a = OvercookedGridworld.from_layout_name("cramped_room").terrain_mtx
b = OvercookedGridworld.from_layout_name("cramped_room").terrain_mtx
print("   terrain identical across two constructions:", a == b)
gw = sorted(set(v for v in sizes.values() if not isinstance(v, str)))
print("   grid sizes (w,h) across family: min %s max %s, %d distinct" % (gw[0], gw[-1], len(gw)))

# --------------------------------------------------------------------------
# frame structure: is it EXACTLY a grid of tile_size x tile_size tiles?
# --------------------------------------------------------------------------
hr("FRAME STRUCTURE / TILE MODEL")
TS = StateVisualizer.DEFAULT_VALUES["tile_size"]
UTS = StateVisualizer.UNSCALED_TILE_SIZE
print("   UNSCALED_TILE_SIZE = %d ; default tile_size = %d ; scale factor = %g"
      % (UTS, TS, TS / UTS))
h, w = f0.shape[:2]
print("   cramped_room grid %dx%d -> frame %dx%d px  (=%d B RGB)"
      % (len(grid[0]), len(grid), w, h, f0.nbytes))
print("   frame is exact multiple of tile_size:", w % TS == 0 and h % TS == 0)

# is pygame.transform.scale by 5 exactly nearest-neighbour block replication?
vis15 = StateVisualizer(tile_size=UTS)
f15 = frame(vis15, s0, grid)
up = np.repeat(np.repeat(f15, TS // UTS, axis=0), TS // UTS, axis=1)
print("   scale(x%d) == np.repeat block replication  : max abs diff = %d"
      % (TS // UTS, int(np.abs(up.astype(int) - f0.astype(int)).max())))

# --------------------------------------------------------------------------
# G1: purity
# --------------------------------------------------------------------------
hr("G1. PURITY")


def rollout(mdp, n, seed, vis, grid, collect=True):
    rng = np.random.default_rng(seed)
    s = mdp.get_standard_start_state()
    out = []
    for t in range(n):
        ja = tuple(Action.ALL_ACTIONS[i] for i in rng.integers(0, len(Action.ALL_ACTIONS), 2))
        s, _ = mdp.get_state_transition(s, ja)
        if collect:
            out.append(s.deepcopy())
    return out


def sig(state):
    """render-relevant signature: everything the visualiser reads from state."""
    ps = tuple(
        (p.position, p.orientation,
         None if p.held_object is None
         else (p.held_object.name,
               tuple(sorted(i.name for i in p.held_object._ingredients))
               if p.held_object.name == "soup" else ()))
        for p in state.players
    )
    obs = tuple(sorted(
        (o.position, o.name,
         tuple(sorted(i.name for i in o._ingredients)) if o.name == "soup" else (),
         getattr(o, "_cooking_tick", -1))
        for o in state.objects.values()
    ))
    return (ps, obs)


for lay in ["cramped_room", "counter_circuit_o_1order"]:
    m = OvercookedGridworld.from_layout_name(lay)
    g = m.terrain_mtx
    v = StateVisualizer()
    states = rollout(m, 400, 0, v, g)

    # (a) render twice -> does render() mutate what it draws?
    maxdiff_repeat = 0
    for s in states[:64]:
        x = frame(v, s, g)
        y = frame(v, s, g)
        maxdiff_repeat = max(maxdiff_repeat, int(np.abs(x.astype(int) - y.astype(int)).max()))

    # (b) round-trip through to_dict/from_dict (fresh objects, no history)
    maxdiff_rt, n_rt = 0, 0
    for s in states:
        s2 = OvercookedState.from_dict(s.to_dict())
        x = frame(v, s, g)
        y = frame(v, s2, g)
        maxdiff_rt = max(maxdiff_rt, int(np.abs(x.astype(int) - y.astype(int)).max()))
        n_rt += 1

    # (c) irrelevant fields randomised: timestep + order lists (hud off)
    maxdiff_irr = 0
    rng = np.random.default_rng(7)
    for s in states[:100]:
        d = s.to_dict()
        d["timestep"] = int(rng.integers(0, 10 ** 6))
        s3 = OvercookedState.from_dict(d)
        x = frame(v, s, g)
        y = frame(v, s3, g)
        maxdiff_irr = max(maxdiff_irr, int(np.abs(x.astype(int) - y.astype(int)).max()))

    # (d) SAME candidate state reached by DIFFERENT histories
    buckets = collections.defaultdict(list)
    for seed in range(6):
        for s in rollout(m, 300, 100 + seed, v, g):
            buckets[sig(s)].append((seed, s))
    pairs, maxdiff_hist = 0, 0
    for k, lst in buckets.items():
        seeds = {sd for sd, _ in lst}
        if len(seeds) < 2:
            continue
        by_seed = {}
        for sd, s in lst:
            by_seed.setdefault(sd, s)
        ss = list(by_seed.values())
        base = frame(v, ss[0], g)
        for s in ss[1:]:
            maxdiff_hist = max(maxdiff_hist, int(np.abs(base.astype(int) - frame(v, s, g).astype(int)).max()))
            pairs += 1
        if pairs > 4000:
            break
    print("   %-26s render-twice maxdiff=%d over 64 | dict round-trip maxdiff=%d over %d states"
          % (lay, maxdiff_repeat, maxdiff_rt, n_rt))
    print("   %-26s timestep randomised maxdiff=%d over 100 | different-history pairs=%d maxdiff=%d"
          % ("", maxdiff_irr, pairs, maxdiff_hist))

# what does the visualiser read that is NOT in the state?
hr("G1b. WHAT IS NOT IN THE STATE (hud chrome)")
m = OvercookedGridworld.from_layout_name("cramped_room")
g = m.terrain_mtx
v = StateVisualizer()
s = m.get_standard_start_state()
f_nohud = frame(v, s, g)
hud = StateVisualizer.default_hud_data(s, score=0)
f_hud0 = frame(v, s, g, hud_data=hud)
hud2 = StateVisualizer.default_hud_data(s, score=137)
f_hud1 = frame(v, s, g, hud_data=hud2)
print("   render_state(state, grid)                 -> %s   (NO hud: `is_rendering_hud and hud_data`)" % (f_nohud.shape,))
print("   render_state(..., hud_data=default+score) -> %s" % (f_hud0.shape,))
print("   hud height = %d px ; hud content = font text for keys %s"
      % (f_hud0.shape[0] - f_nohud.shape[0], StateVisualizer.DEFAULT_VALUES["hud_data_default_key_order"]))
print("   score 0 vs 137 (state IDENTICAL) -> %d pixels differ  => hud is CHROME, not state"
      % int((f_hud0 != f_hud1).any(axis=2).sum()))
print("   grid part identical under hud change:",
      np.array_equal(f_hud0[-f_nohud.shape[0]:], f_hud1[-f_nohud.shape[0]:]))

# --------------------------------------------------------------------------
# G2: compression
# --------------------------------------------------------------------------
hr("G2. COMPRESSION")
print("   Recipe.MAX_NUM_INGREDIENTS =", Recipe.MAX_NUM_INGREDIENTS,
      "| ALL_INGREDIENTS =", Recipe.ALL_INGREDIENTS)
for lay in ["cramped_room", "counter_circuit_o_1order"]:
    m = OvercookedGridworld.from_layout_name(lay)
    g = m.terrain_mtx
    W, H = len(g[0]), len(g)
    fb = W * TS * H * TS * 3
    n_counter = sum(r.count("X") for r in g)
    n_pot = sum(r.count("P") for r in g)
    n_free = sum(sum(1 for c in r if c == " ") for r in g)
    # render-relevant per-step state, packed:
    #   2 players x (pos idx 1B, orient 2b, held code 1B) = 2 x 2 B = 4 B  (pos<=255 cells)
    #   loose objects: one code per COUNTER cell (dish/onion/tomato/soup(t,o) -> <=1B)
    #   pots: one soup descriptor per POT cell: (n_tomato, n_onion, cooking_tick) -> 1B + 1B
    per_step = 2 * 2 + n_counter * 1 + n_pot * 2
    # generous variant: 1 code + 1 tick byte for EVERY non-wall cell
    per_step_gen = 2 * 2 + (n_counter + n_pot + n_free) * 2
    print("   %-26s grid %dx%d -> frame %dx%d = %d B" % (lay, W, H, W * TS, H * TS, fb))
    print("   %-26s counters=%d pots=%d free=%d | render-state packed = %d B -> ratio 1/%d"
          % ("", n_counter, n_pot, n_free, per_step, fb // per_step))
    print("   %-26s generous (2 B every non-wall cell)  = %d B -> ratio 1/%d"
          % ("", per_step_gen, fb // per_step_gen))
    print("   %-26s decoder-side CONSTANT (terrain codes, per layout) = %d B, amortised over episode"
          % ("", W * H))

# --------------------------------------------------------------------------
# atlas enumeration: how many distinct tiles can the renderer draw?
# --------------------------------------------------------------------------
hr("ATLAS SIZE (distinct reachable tiles)")
TERR = list(StateVisualizer.TILE_TO_FRAME_NAME.keys())
held = ["", "onion", "tomato", "dish", "soup-onion", "soup-tomato"]
dirs = ["NORTH", "SOUTH", "EAST", "WEST"]
colors = StateVisualizer.DEFAULT_VALUES["player_colors"]
n_chef = len(dirs) * len(held) * len(colors)
soup_names = sorted(StateVisualizer.SOUPS_IMG.frames_rectangles.keys())
n_soup_done = sum(1 for n in soup_names if n.startswith("soup_done"))
n_soup_idle = sum(1 for n in soup_names if n.startswith("soup_idle"))
n_soup_cooked = sum(1 for n in soup_names if n.startswith("soup_cooked"))
print("   terrain tiles                       : %d" % len(TERR))
print("   chef composites (dir x held x colour): %d x %d x %d = %d  (always on FLOOR)"
      % (len(dirs), len(held), len(colors), n_chef))
print("   loose objects on a COUNTER          : onion/tomato/dish = 3, + soup_done = %d -> %d"
      % (n_soup_done, 3 + n_soup_done))
print("   POT contents                        : soup_idle=%d + soup_cooked=%d = %d"
      % (n_soup_idle, n_soup_cooked, n_soup_idle + n_soup_cooked))
# cooking timer text
cook_times = sorted({Recipe(("onion",) * i + ("tomato",) * j).time
                     for i in range(0, 4) for j in range(0, 4) if 0 < i + j <= 3})
print("   cooking-timer TEXT values           : _cooking_tick in [0, cook_time]; cook_time(s) = %s"
      % cook_times)
print("     -> %d distinct digit strings (0..%d), drawn RED at a fixed sub-tile offset AFTER scaling"
      % (max(cook_times) + 1, max(cook_times)))
atlas = len(TERR) + n_chef + (3 + n_soup_done) + (n_soup_idle + n_soup_cooked) * (max(cook_times) + 1)
print("   UPPER BOUND on composite tile atlas : %d tiles @ %dx%d (unscaled %dx%d)"
      % (atlas, TS, TS, UTS, UTS))
print("   atlas bytes @ %dx%d RGB             : %.2f MB   (@%dx%d unscaled: %.1f kB)"
      % (TS, TS, atlas * TS * TS * 3 / 1e6, UTS, UTS, atlas * UTS * UTS * 3 / 1e3))

# empirical: distinct 75x75 tiles actually observed in a long rollout
m = OvercookedGridworld.from_layout_name("cramped_room")
g = m.terrain_mtx
v = StateVisualizer()
seen = set()
for seed in range(4):
    for s in rollout(m, 500, 1000 + seed, v, g):
        fr = frame(v, s, g)
        H, W = fr.shape[:2]
        for yy in range(0, H, TS):
            for xx in range(0, W, TS):
                seen.add(fr[yy:yy + TS, xx:xx + TS].tobytes())
print("   empirically observed distinct %dx%d tiles over 2000 random states: %d" % (TS, TS, len(seen)))

# scaled/rotated blits?
hr("SCALED / ROTATED DRAWS")
print("   sprites: blitted 1:1 at 15x15 (MultiFramePygameImage.blit_on_surface, no scale/rot)")
print("   whole grid surface: pygame.transform.scale by tile_size/15 -> verified block-exact above")
print("   pygame.transform.scale used in HUD order icons (15->hud_order_size=%d) : HUD only"
      % StateVisualizer.DEFAULT_VALUES["hud_order_size"])
print("   pygame.transform.rotozoom used ONLY in _render_actions_probs (action_probs=None by default)")
alpha = {}
for k, v_ in sheets.items():
    alpha[k] = (v_.image.get_flags() & pygame.SRCALPHA) != 0 or v_.image.get_alpha() is not None
print("   per-pixel alpha on sheets:", alpha, " -> chef+hat+object composite over terrain")

# --------------------------------------------------------------------------
# G4: economics
# --------------------------------------------------------------------------
hr("G4. ECONOMICS (single core)")
for lay in ["cramped_room", "counter_circuit_o_1order"]:
    m = OvercookedGridworld.from_layout_name(lay)
    g = m.terrain_mtx
    v = StateVisualizer()
    states = rollout(m, 200, 3, v, g)
    fr = frame(v, states[0], g)
    px = fr.shape[0] * fr.shape[1]
    # warmup
    for s in states[:20]:
        v.render_state(s, g)
    N = 400
    t0 = time.perf_counter()
    for i in range(N):
        v.render_state(states[i % len(states)], g)
    t1 = time.perf_counter()
    fps = N / (t1 - t0)
    # including surfarray extraction to numpy (what a data pipeline actually needs)
    t0 = time.perf_counter()
    for i in range(N):
        frame(v, states[i % len(states)], g)
    t2 = time.perf_counter()
    fps_np = N / (t2 - t0)
    print("   %-26s frame %dx%d (%d px, %d B) : render_state %.0f f/s ; +to-numpy %.0f f/s"
          % (lay, fr.shape[1], fr.shape[0], px, fr.nbytes, fps, fps_np))
    # equivalent-area decode baseline, interpolated on the doc's curve
    anchors = [(64 * 64, 11842), (96 * 96, 8341), (128 * 128, 7483), (224 * 224, 6762), (256 * 256, 5486)]
    # interpolate f/s ~ linear in log(px)
    xs = np.log([a[0] for a in anchors])
    ys = np.log([a[1] for a in anchors])
    dec = float(np.exp(np.interp(np.log(px), xs, ys)))
    print("   %-26s JPEG-decode baseline interpolated at %d px = %.0f f/s/core -> regen/decode = %.3fx"
          % ("", px, dec, fps / dec))

# render share of an env step
hr("G4b. RENDER SHARE OF AN ENV STEP")
m = OvercookedGridworld.from_layout_name("cramped_room")
g = m.terrain_mtx
v = StateVisualizer()
s = m.get_standard_start_state()
rng = np.random.default_rng(0)
acts = [tuple(Action.ALL_ACTIONS[i] for i in rng.integers(0, 6, 2)) for _ in range(2000)]
t0 = time.perf_counter()
ss = s
for a in acts:
    ss, _ = m.get_state_transition(ss, a)
t1 = time.perf_counter()
step_s = (t1 - t0) / len(acts)
t0 = time.perf_counter()
for _ in range(400):
    v.render_state(s, g)
t1 = time.perf_counter()
rend_s = (t1 - t0) / 400
print("   mdp.get_state_transition : %.4f ms/step" % (step_s * 1e3))
print("   render_state             : %.4f ms/frame" % (rend_s * 1e3))
print("   render share of a step   : %.1f %%" % (100 * rend_s / (rend_s + step_s)))

print("\nDONE")
