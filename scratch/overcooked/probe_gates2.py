"""Follow-up probe: real decode baseline at the true frame size, systematic
alphabet purity sweep, cooking-timer tile containment, family-wide frame sizes.
"""
from __future__ import annotations

import os, io, time, itertools, collections

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from overcooked_ai_py.mdp.overcooked_mdp import (
    OvercookedGridworld, OvercookedState, PlayerState, ObjectState, SoupState, Recipe)
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.static import LAYOUTS_DIR

TS = StateVisualizer.DEFAULT_VALUES["tile_size"]


def hr(t):
    print("\n" + "=" * 78); print("== " + t); print("=" * 78)


def frame(vis, state, grid, **kw):
    surf = vis.render_state(state, grid, **kw)
    return np.transpose(pygame.surfarray.array3d(surf), (1, 0, 2)).copy()


# --------------------------------------------------------------------------
hr("A. SYSTEMATIC ALPHABET PURITY SWEEP (two independent construction paths)")
m = OvercookedGridworld.from_layout_name("cramped_room")
g = m.terrain_mtx
v = StateVisualizer()
# locate cells
pots, counters, floors = [], [], []
for y, row in enumerate(g):
    for x, c in enumerate(row):
        if c == "P": pots.append((x, y))
        elif c == "X": counters.append((x, y))
        elif c == " ": floors.append((x, y))
print("   cramped_room: pots=%s floors=%s n_counters=%d" % (pots, floors, len(counters)))

held_opts = [None, "onion", "tomato", "dish", ("soup", ("onion",) * 3), ("soup", ("tomato",) * 3)]
ing_opts = [("onion",), ("tomato",), ("onion", "onion"), ("onion", "tomato"),
            ("onion",) * 3, ("tomato",) * 3, ("onion", "onion", "tomato")]
tick_opts = [-1, 0, 1, 5, 9, 10, 19, 20]


def mk_held(pos, spec):
    if spec is None: return None
    if isinstance(spec, tuple):
        return SoupState(pos, [ObjectState(i, pos) for i in spec[1]], cooking_tick=20)
    return ObjectState(spec, pos)


def build(p0dir, p1dir, h0, h1, pot_ing, pot_tick, counter_obj):
    pos0, pos1 = floors[0], floors[1]
    players = [PlayerState(pos0, p0dir, mk_held(pos0, h0)),
               PlayerState(pos1, p1dir, mk_held(pos1, h1))]
    objs = {}
    if pot_ing is not None:
        pp = pots[0]
        objs[pp] = SoupState(pp, [ObjectState(i, pp) for i in pot_ing], cooking_tick=pot_tick)
    if counter_obj is not None:
        cp = counters[3]
        if isinstance(counter_obj, tuple):
            objs[cp] = SoupState(cp, [ObjectState(i, cp) for i in counter_obj], cooking_tick=20)
        else:
            objs[cp] = ObjectState(counter_obj, cp)
    return OvercookedState(players, objs, all_orders=[{"ingredients": ["onion"] * 3}])


rng = np.random.default_rng(0)
n, maxdiff = 0, 0
combos = list(itertools.product(Direction.ALL_DIRECTIONS, Direction.ALL_DIRECTIONS,
                                held_opts, held_opts))
rng.shuffle(combos)
for (d0, d1, h0, h1) in combos[:96]:
    pot_ing = ing_opts[int(rng.integers(0, len(ing_opts)))]
    pot_tick = tick_opts[int(rng.integers(0, len(tick_opts)))]
    if pot_tick >= 0 and len(pot_ing) < 1: pot_tick = -1
    cobj = [None, "onion", "tomato", "dish", ("onion",) * 3][int(rng.integers(0, 5))]
    s_direct = build(d0, d1, h0, h1, pot_ing, pot_tick, cobj)
    s_dict = OvercookedState.from_dict(s_direct.to_dict())
    s_deep = s_direct.deepcopy()
    a = frame(v, s_direct, g); b = frame(v, s_dict, g); c = frame(v, s_deep, g)
    maxdiff = max(maxdiff, int(np.abs(a.astype(int) - b.astype(int)).max()),
                  int(np.abs(a.astype(int) - c.astype(int)).max()))
    n += 1
print("   %d alphabet states x 3 construction paths (direct ctor / from_dict / deepcopy)"
      "  -> max abs pixel diff = %d" % (n, maxdiff))

# same *candidate state* reached by DIFFERENT action histories (deliberate)
hr("B. DIFFERENT HISTORY -> SAME STATE (deliberate paths)")
start = m.get_standard_start_state()
ST = Action.STAY
paths = {
    "A": [(ST, ST), ((0, -1), (0, -1)), ((0, 1), (0, 1))],
    "B": [((0, -1), (0, -1)), ((0, 1), (0, 1)), (ST, ST)],
    "C": [((-1, 0), (1, 0)), ((1, 0), (-1, 0)), (ST, ST)],
    "D": [((1, 0), (-1, 0)), ((-1, 0), (1, 0)), (ST, ST)],
    "E": [((0, 1), (0, 1)), (ST, ST), ((0, -1), (0, -1))],
}
ends = {}
for k, acts in paths.items():
    s = start.deepcopy()
    for a in acts:
        s, _ = m.get_state_transition(s, tuple(a))
    ends[k] = s
groups = collections.defaultdict(list)
for k, s in ends.items():
    groups[repr(s.to_dict()["players"]) + repr(sorted(map(str, s.objects.values())))].append((k, s))
pairs, md = 0, 0
for _, lst in groups.items():
    if len(lst) < 2: continue
    base = frame(v, lst[0][1], g)
    for k, s in lst[1:]:
        md = max(md, int(np.abs(base.astype(int) - frame(v, s, g).astype(int)).max()))
        pairs += 1
print("   deliberate distinct 3-step paths converging: %d pairs, max abs diff = %d" % (pairs, md))

# big collision harvest from many short rollouts
buckets = collections.defaultdict(dict)


def sigk(s):
    ps = tuple((p.position, p.orientation,
                None if p.held_object is None else
                (p.held_object.name, tuple(sorted(i.name for i in p.held_object._ingredients))
                 if p.held_object.name == "soup" else ()))
               for p in s.players)
    obs = tuple(sorted((o.position, o.name,
                        tuple(sorted(i.name for i in o._ingredients)) if o.name == "soup" else (),
                        getattr(o, "_cooking_tick", -1)) for o in s.objects.values()))
    return (ps, obs)


for seed in range(60):
    r = np.random.default_rng(seed)
    s = m.get_standard_start_state()
    for t in range(60):
        ja = tuple(Action.ALL_ACTIONS[i] for i in r.integers(0, 6, 2))
        s, _ = m.get_state_transition(s, ja)
        buckets[sigk(s)].setdefault(seed, s.deepcopy())
pairs2, md2, nstates = 0, 0, 0
for k, bys in buckets.items():
    if len(bys) < 2: continue
    ss = list(bys.values())[:4]
    base = frame(v, ss[0], g)
    nstates += 1
    for s in ss[1:]:
        md2 = max(md2, int(np.abs(base.astype(int) - frame(v, s, g).astype(int)).max()))
        pairs2 += 1
    if pairs2 > 3000: break
print("   random-rollout collisions: %d distinct candidate states, %d cross-history pairs, max abs diff = %d"
      % (nstates, pairs2, md2))

# --------------------------------------------------------------------------
hr("C. COOKING-TIMER TEXT: is it tile-local? how many distinct pot tiles?")
pot_tiles = set()
bleed = False
for ing in ing_opts:
    for tick in list(range(-1, 21)):
        s = build(Direction.NORTH, Direction.SOUTH, None, None, ing, tick, None)
        f = frame(v, s, g)
        px, py = pots[0]
        tile = f[py * TS:(py + 1) * TS, px * TS:(px + 1) * TS]
        pot_tiles.add(tile.tobytes())
        # compare all OTHER tiles against an empty-pot reference to detect bleed
        ref = frame(v, build(Direction.NORTH, Direction.SOUTH, None, None, None, -1, None), g)
        f2 = f.copy(); f2[py * TS:(py + 1) * TS, px * TS:(px + 1) * TS] = ref[py * TS:(py + 1) * TS, px * TS:(px + 1) * TS]
        if not np.array_equal(f2, ref):
            bleed = True
print("   distinct POT tiles over %d (ingredients x cooking_tick) configs : %d" % (len(ing_opts) * 22, len(pot_tiles)))
print("   timer text bleeds outside its own tile : %s" % bleed)

# distinct FLOOR/chef tiles
chef_tiles = set()
for d in Direction.ALL_DIRECTIONS:
    for h in held_opts:
        s = build(d, Direction.SOUTH, h, None, None, None, None)
        f = frame(v, s, g)
        px, py = floors[0]
        chef_tiles.add(f[py * TS:(py + 1) * TS, px * TS:(px + 1) * TS].tobytes())
print("   distinct chef tiles for player0 (4 dir x 6 held, blue hat)      : %d" % len(chef_tiles))
counter_tiles = set()
for co in [None, "onion", "tomato", "dish"] + [tuple(c) for c in
                                               [("onion",), ("onion", "onion"), ("onion",) * 3,
                                                ("tomato",), ("tomato",) * 3, ("onion", "tomato")]]:
    s = build(Direction.NORTH, Direction.SOUTH, None, None, None, None, co)
    f = frame(v, s, g)
    px, py = counters[3]
    counter_tiles.add(f[py * TS:(py + 1) * TS, px * TS:(px + 1) * TS].tobytes())
print("   distinct counter tiles (empty + 3 items + 6 done-soups)          : %d" % len(counter_tiles))

# --------------------------------------------------------------------------
hr("D. REAL JPEG/PNG DECODE BASELINE AT THE ACTUAL FRAME SIZE (one core)")
try:
    from PIL import Image
    have_pil = True
except Exception:
    have_pil = False
print("   PIL available:", have_pil)
if have_pil:
    for lay in ["cramped_room", "counter_circuit_o_1order"]:
        mm = OvercookedGridworld.from_layout_name(lay)
        gg = mm.terrain_mtx
        vv = StateVisualizer()
        f = frame(vv, mm.get_standard_start_state(), gg)
        bufs = []
        for q in range(16):
            b = io.BytesIO()
            Image.fromarray(f).save(b, format="JPEG", quality=95)
            bufs.append(b.getvalue())
        # warm
        for b in bufs[:4]:
            np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
        N = 300
        t0 = time.perf_counter()
        for i in range(N):
            np.asarray(Image.open(io.BytesIO(bufs[i % len(bufs)])).convert("RGB"))
        t1 = time.perf_counter()
        dec = N / (t1 - t0)
        print("   %-26s frame %dx%d, jpeg %d B -> decode %.0f f/s/core"
              % (lay, f.shape[1], f.shape[0], len(bufs[0]), dec))
        # renderer
        for _ in range(20): vv.render_state(mm.get_standard_start_state(), gg)
        st = mm.get_standard_start_state()
        t0 = time.perf_counter()
        for _ in range(300): vv.render_state(st, gg)
        t1 = time.perf_counter()
        ren = 300 / (t1 - t0)
        print("   %-26s pygame StateVisualizer %.0f f/s -> regen/decode = %.3fx"
              % ("", ren, ren / dec))

# --------------------------------------------------------------------------
hr("E. FAMILY-WIDE FRAME SIZES + STATE SIZES")
rows = []
for L in sorted(os.listdir(LAYOUTS_DIR)):
    if not L.endswith(".layout"): continue
    name = L[:-7]
    try:
        mm = OvercookedGridworld.from_layout_name(name)
    except Exception as e:
        rows.append((name, None, None, None, "ERR")); continue
    gg = mm.terrain_mtx
    W, H = len(gg[0]), len(gg)
    nP = sum(r.count("P") for r in gg)
    nX = sum(r.count("X") for r in gg)
    nplayers = len(mm.start_player_positions)
    rows.append((name, (W, H), (W * TS, H * TS), (nX, nP, nplayers), ""))
print("   %-38s %-9s %-12s %s" % ("layout", "grid", "frame px", "(counters,pots,players)"))
for r in rows:
    print("   %-38s %-9s %-12s %s %s" % (r[0], r[1], r[2], r[3], r[4]))
fbs = [r[2][0] * r[2][1] * 3 for r in rows if r[2]]
print("   frame bytes: min %d  max %d  median %d" % (min(fbs), max(fbs), int(np.median(fbs))))
npl = sorted(set(r[3][2] for r in rows if r[3]))
print("   player counts across family:", npl)

hr("F. HEADLESS API")
print("   the exact call chain:")
print("     os.environ['SDL_VIDEODRIVER']='dummy'")
print("     surf = StateVisualizer().render_state(state, grid)        # grid = mdp.terrain_mtx")
print("     rgb  = np.transpose(pygame.surfarray.array3d(surf),(1,0,2))   # (H,W,3) uint8")
print("   no pygame.display.set_mode() needed; render_state() calls pygame.init() itself.")
print("\nDONE")
