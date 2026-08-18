"""Decisive structural probe: is the official frame EXACTLY a gather from a
position-independent tile atlas keyed by (terrain, occupant)?

Also: cook_time range across the family (-> cooking-timer digit alphabet),
4-player layouts, and the true atlas size.
"""
from __future__ import annotations

import os, collections, itertools

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
DIRNAME = Direction.DIRECTION_TO_NAME


def hr(t):
    print("\n" + "=" * 78); print("== " + t); print("=" * 78)


def frame(vis, state, grid, **kw):
    return np.transpose(pygame.surfarray.array3d(vis.render_state(state, grid, **kw)), (1, 0, 2)).copy()


def held_code(o):
    if o is None: return ("none",)
    if o.name == "soup":
        return ("soup-onion",) if any(i.name == "onion" for i in o._ingredients) else ("soup-tomato",)
    return (o.name,)


def cell_key(state, grid, x, y):
    """Everything the renderer draws into tile (x,y), position-independent."""
    terr = grid[y][x]
    occ = []
    for i, p in enumerate(state.players):
        if p.position == (x, y):
            occ.append(("chef", i, DIRNAME[p.orientation]) + held_code(p.held_object))
    o = state.objects.get((x, y))
    if o is not None:
        if o.name == "soup":
            no = sum(1 for i in o._ingredients if i.name == "onion")
            nt = sum(1 for i in o._ingredients if i.name == "tomato")
            if terr == "P":
                status = "cooked" if o.is_ready else "idle"
            else:
                status = "done"
            tick = o._cooking_tick if terr == "P" else None
            occ.append(("soup", status, nt, no, tick))
        else:
            occ.append(("obj", o.name))
    return (terr, tuple(occ))


hr("A. COOK TIME / TIMER DIGIT ALPHABET ACROSS THE FAMILY")
cts = collections.Counter()
nplayers = collections.Counter()
for L in sorted(os.listdir(LAYOUTS_DIR)):
    if not L.endswith(".layout"): continue
    m = OvercookedGridworld.from_layout_name(L[:-7])
    nplayers[len(m.start_player_positions)] += 1
    for r in Recipe.ALL_RECIPES:
        cts[r.time] += 1
print("   distinct Recipe.time (cook_time) values seen across all 45 layouts:", sorted(cts))
print("   player-count histogram across family:", dict(nplayers))
print("   Recipe.ALL_RECIPES =", len(Recipe.ALL_RECIPES), "(multisets of size 1..3 over onion/tomato)")

hr("B. TILE-ATLAS RECONSTRUCTION (position independence + bit-exact gather)")
atlas = {}
collisions = 0
tested = 0
maxdiff = 0
per_layout = {}
LAYS = ["cramped_room", "counter_circuit_o_1order", "coordination_ring",
        "asymmetric_advantages", "forced_coordination", "scenario3", "corridor",
        "cramped_room_tomato", "long_cook_time"]
for lay in LAYS:
    m = OvercookedGridworld.from_layout_name(lay)
    g = m.terrain_mtx
    v = StateVisualizer()
    H, W = len(g), len(g[0])
    rng = np.random.default_rng(hash(lay) % 2 ** 31)
    s = m.get_standard_start_state()
    nbad = 0
    nst = 0
    for t in range(220):
        ja = tuple(Action.ALL_ACTIONS[i] for i in rng.integers(0, 6, 2))
        try:
            s, _ = m.get_state_transition(s, ja)
        except ValueError:
            continue
        f = frame(v, s, g)
        recon = np.zeros_like(f)
        for y in range(H):
            for x in range(W):
                k = cell_key(s, g, x, y)
                tile = f[y * TS:(y + 1) * TS, x * TS:(x + 1) * TS]
                if k in atlas:
                    if not np.array_equal(atlas[k], tile):
                        collisions += 1
                else:
                    atlas[k] = tile.copy()
                recon[y * TS:(y + 1) * TS, x * TS:(x + 1) * TS] = atlas[k]
        d = int(np.abs(recon.astype(int) - f.astype(int)).max())
        maxdiff = max(maxdiff, d)
        if d: nbad += 1
        nst += 1
        tested += 1
    per_layout[lay] = (nst, nbad)
print("   layouts probed: %s" % LAYS)
print("   states rendered: %d ; atlas entries learned: %d" % (tested, len(atlas)))
print("   position-independence violations (same key, different pixels): %d" % collisions)
print("   frames NOT bit-exactly reconstructed from the atlas: %d / %d ; max abs diff = %d"
      % (sum(b for _, b in per_layout.values()), tested, maxdiff))
for k, (n, b) in per_layout.items():
    print("      %-28s %3d states, %d mismatches" % (k, n, b))

hr("C. EXHAUSTIVE REACHABLE ATLAS (enumerated, not sampled)")
n_terr = 7
n_dirs = 4
held = ["none", "onion", "tomato", "dish", "soup-onion", "soup-tomato"]
n_colors = len(StateVisualizer.DEFAULT_VALUES["player_colors"])
n_chef = n_dirs * len(held) * n_colors
recipes = [(nt, no) for nt in range(4) for no in range(4) if 0 < nt + no <= 3]
ct = sorted(cts)[-1]
n_pot = len(recipes) * (ct + 2)          # tick in {-1} u [0, cook_time]
n_counter_obj = 3 + len(recipes)          # onion/tomato/dish + done-soup per recipe
total = n_terr + n_chef + n_counter_obj + n_pot
print("   terrain                      : %d" % n_terr)
print("   floor + chef(dir x held x col): %d x %d x %d = %d" % (n_dirs, len(held), n_colors, n_chef))
print("   counter + object             : %d" % n_counter_obj)
print("   pot + soup(recipe x tick)    : %d x %d = %d" % (len(recipes), ct + 2, n_pot))
print("   TOTAL reachable tiles        : %d" % total)
print("   atlas @75x75 RGB = %.2f MB | @15x15 = %.1f kB (but timer text is font-rendered POST-scale,"
      " so pot tiles are tile_size-specific)" % (total * TS * TS * 3 / 1e6, total * 15 * 15 * 3 / 1e3))
print("   with all 5 hat colours (4-player layouts): %d tiles" % (total - n_chef + n_dirs * len(held) * 5))
print("   observed atlas keys in the sampled rollouts above: %d" % len(atlas))

hr("D. HELD-SOUP AMBIGUITY CHECK (is the state->pixel map many-to-one?)")
m = OvercookedGridworld.from_layout_name("cramped_room")
g = m.terrain_mtx
v = StateVisualizer()
floors = [(x, y) for y, r in enumerate(g) for x, c in enumerate(r) if c == " "]
p0 = floors[0]


def st(h_ings):
    hp = SoupState(p0, [ObjectState(i, p0) for i in h_ings], cooking_tick=20)
    return OvercookedState([PlayerState(p0, Direction.NORTH, hp),
                            PlayerState(floors[1], Direction.SOUTH, None)], {},
                           all_orders=[{"ingredients": ["onion"] * 3}])


a = frame(v, st(("onion", "onion", "onion")), g)
b = frame(v, st(("onion", "tomato")), g)
c = frame(v, st(("tomato", "tomato", "tomato")), g)
print("   held soup onion^3 vs onion+tomato : %d px differ  (both map to 'soup-onion' sprite)"
      % int((a != b).any(axis=2).sum()))
print("   held soup onion^3 vs tomato^3     : %d px differ" % int((a != c).any(axis=2).sum()))
print("   => HELD soup renders only its dominant-ingredient class: the frame does NOT show the exact")
print("      held recipe. Render state for a held soup is 1 of 6 codes, not the full multiset.")

print("\nDONE")
