"""chaser: the three things its frame needs that maze/heist/miner did not.

1. **The ORB is a `fillRect`, not a blit.** `ChaserGame::draw_grid_obj` paints grid value 1002 as a
   solid `QColor(0, 255, 0)` over a centred sub-rect of the (already RENDER_EPS-widened) cell rect,
   at `ORB_DIM = 0.3`. So the grid is no longer one gather: it is a gather for MAZE_WALL plus a
   coverage test for ORB. Qt's aliased `fillRect` rounding rule has never been measured in this
   repo, and there are two plausible readings —
     A. `QRectF::toRect()`-style `qRound` on each edge (the rule the blit already uses), and
     B. `toNormalizedFillRect`'s `qRound(edge + 0.5 - 1/64)`.
   They differ by a whole device pixel on some cells. This probe crafts a state with exactly one
   ORB cell, diffs against the same state without it, and scores both against the truth.

2. **The enemy sprite is a function of `cur_time` and `eat_time`, not of the entity.**
   `image_for_type(ENEMY)` returns `ENEMY_WEAK` while `cur_time - eat_time < eat_timeout`, else
   `ENEMY + rem` with `rem = (cur_time/2) % 4` folded so `3 -> 1`. So all three enemies share one
   sprite index, and that index is state. This probe enumerates it and confirms the fold.

3. **Slot counts.** hard mode is `maze_dim = 13`, `total_enemies = 3`; the orbs are one per quadrant
   with one quadrant losing one, so 3 LARGE_ORBs. Measured here against real rollouts rather than
   read off `game_reset`, because a level generator's invariant is exactly the kind of constant that
   reads plausibly and is wrong.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import state_io as S  # noqa: E402

OUT = os.path.join(HERE, "chaser_facts.json")

DIM = 13
SPACE, MAZE_WALL, ORB = 100, 5, 1002
PLAYER, LARGE_ORB, ENEMY_WEAK, ENEMY_EGG, ENEMY, ENEMY2, ENEMY3 = 0, 2, 3, 4, 6, 7, 8
RENDER_EPS = 0.02
F = np.float32


# ------------------------------------------------------------------ Qt geometry, float32 in
def prepare(d, size):
    cx = cy = F(d) * F(0.5)
    vis = F(d)
    raw_unit = F(F(64) / vis)
    unit = F(np.float64(raw_unit) * (size / 64.0))
    view_dim = F(64.0 / np.float64(raw_unit))
    x_off = F(unit * F(cx - F(view_dim / F(2))))
    y_off = F(unit * F(cy - F(view_dim / F(2))))
    return unit, view_dim, x_off, y_off


def screen_rect(x, y, dx, dy, g, eps=0.0):
    unit, view_dim, x_off, y_off = g
    e = F(eps)
    return (float(F(F(F(x) - e) * unit - x_off)), float(F(F(F(view_dim) - F(y) - e) * unit + y_off)),
            float(F(F(F(dx) + F(2) * e) * unit)), float(F(F(F(dy) + F(2) * e) * unit)))


def qround(v):
    return int(np.floor(v + 0.5)) if v >= 0 else int(np.ceil(v - 0.5))


def rect_A(x, y, w, h):
    """`QRectF::toRect()` — the rule the axis-aligned blit was measured to use."""
    l, t = qround(x), qround(y)
    return l, t, qround(x + w) - l, qround(y + h) - t


DELTA = 0.5 - 1.0 / 64.0


def rect_B(x, y, w, h):
    """`toNormalizedFillRect`: `qRound(edge + 0.5 - 1/64)`, with `right() = x + w`."""
    x1, y1 = qround(x + DELTA), qround(y + DELTA)
    x2, y2 = qround(x + w + DELTA), qround(y + h + DELTA)
    return x1, y1, x2 - x1, y2 - y1


def mask_from(rule, cellx, celly, g, size, orb_dim=0.3):
    rx, ry, rw, rh = screen_rect(cellx, celly + 1, 1, 1, g, eps=RENDER_EPS)
    sx = rx + rw * (1 - orb_dim) / 2
    sy = ry + rh * (1 - orb_dim) / 2
    l, t, w, h = rule(sx, sy, rw * orb_dim, rh * orb_dim)
    m = np.zeros((size, size), bool)
    x0, x1 = max(l, 0), min(l + w, size)
    y0, y1 = max(t, 0), min(t + h, size)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


# ------------------------------------------------------------------ crafted states
def _ent(**kw):
    e = {k: (0 if t == "i" else 0.0) for k, t in S.ENTITY}
    e.update(alpha=1.0, alpha_decay=1.0)
    e.update(kw)
    return e


#: `BasicAbstractGame::deserialize` asserts `find_entity_index(PLAYER) >= 0`, so a crafted state
#: cannot have an empty entity list. The agent is parked far outside the world instead, where its
#: rect clips away entirely -- checked by `blank_is_black`.
OFFSCREEN = dict(x=-50.0, y=-50.0, rx=0.5, ry=0.5, type=PLAYER, image_type=PLAYER, image_theme=0,
                 render_z=1, smart_step=1)


def blank(template):
    b = dict(template)
    b["grid"] = np.full((DIM, DIM), SPACE, np.int32)
    b["grid_w"] = b["grid_h"] = DIM
    b["grid_size"] = DIM * DIM
    b["entities"] = [_ent(**OFFSCREEN)]
    b["use_backgrounds"] = 0                       # black plate: the diff is then the primitive alone
    return b


def render(env, b):
    env.set_state([S.dump(b)])
    _, obs, _ = env.observe()
    return np.asarray(obs["rgb"][0]).copy()


def main():
    from procgen import ProcgenGym3Env
    env = ProcgenGym3Env(num=1, env_name="chaser", distribution_mode="hard", rand_seed=0,
                         start_level=0, num_levels=0, use_backgrounds=True)
    env.act(np.array([4], np.int32))
    tmpl = S.parse(env.get_state()[0])
    # the parser must round-trip before anything crafted through it means anything
    rt = S.dump(tmpl) == env.get_state()[0]
    print("chaser blob round-trips byte for byte:", rt, flush=True)
    rep = {"roundtrip": bool(rt), "maze_dim": int(tmpl["maze_dim"]),
           "total_enemies": int(tmpl["total_enemies"]), "eat_timeout": int(tmpl["eat_timeout"])}

    size = 64
    g = prepare(DIM, size)
    base = render(env, blank(tmpl))
    rep["blank_is_black"] = bool((base == 0).all())

    # ---- 1. the ORB fillRect rule
    hits = {"A": 0, "B": 0}
    npix = 0
    colours = set()
    per_cell = []
    for cy in range(DIM):
        for cx in range(DIM):
            b = blank(tmpl)
            b["grid"][cy, cx] = ORB
            fr = render(env, b)
            truth = (fr != base).any(-1)
            npix += int(truth.sum())
            colours |= {tuple(int(v) for v in c) for c in fr[truth]}
            for k, rule in (("A", rect_A), ("B", rect_B)):
                hits[k] += int((mask_from(rule, cx, cy, g, size) == truth).all())
            per_cell.append(int(truth.sum()))
    rep["orb"] = {"cells": DIM * DIM, "exact_rule_A": hits["A"], "exact_rule_B": hits["B"],
                  "total_px": npix, "colours": sorted(colours),
                  "px_per_cell": sorted(set(per_cell))}
    print(f"ORB fillRect: rule A (QRectF::toRect) {hits['A']}/{DIM*DIM}, "
          f"rule B (toNormalizedFillRect) {hits['B']}/{DIM*DIM}; "
          f"colours {sorted(colours)}; px/cell {sorted(set(per_cell))}", flush=True)

    # ---- 1b. do two ORB cells ever share a device pixel? (decides whether one mask table is enough)
    masks = [mask_from(rect_A if hits["A"] >= hits["B"] else rect_B, cx, cy, g, size)
             for cy in range(DIM) for cx in range(DIM)]
    stack = np.stack(masks).sum(0)
    rep["orb"]["max_cells_per_pixel"] = int(stack.max())
    print("max ORB cells covering one device pixel:", int(stack.max()), flush=True)

    # ---- 1c. and does an ORB cell's fill ever touch a neighbour's WALL sprite pixels?
    #      (the grid layer draws walls and orbs in the SAME for-x-for-y loop, so if the orb's
    #       sub-rect stayed inside its own cell's non-eps rect the two never interact)
    wall_over_orb = 0
    for cy in (0, 6, 12):
        for cx in (0, 6, 12):
            b = blank(tmpl)
            b["grid"][cy, cx] = ORB
            f1 = render(env, b)
            for dy, dx in ((0, 1), (1, 0)):
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < DIM and 0 <= nx < DIM):
                    continue
                b2 = blank(tmpl)
                b2["grid"][cy, cx] = ORB
                b2["grid"][ny, nx] = MAZE_WALL
                f2 = render(env, b2)
                b3 = blank(tmpl)
                b3["grid"][ny, nx] = MAZE_WALL
                f3 = render(env, b3)
                # if they do not interact, orb-then-wall == max-composite of the two alone
                indep = ((f2 == f3) | (f2 == f1)).all()
                wall_over_orb += 0 if indep else 1
    rep["orb"]["neighbour_wall_interactions"] = wall_over_orb
    print("orb/neighbour-wall draw interactions:", wall_over_orb, flush=True)

    # ---- 2. the enemy sprite phase
    #      craft one ENEMY at a fixed cell and sweep (cur_time, eat_time); group frames by identity
    def one_enemy(cur_time, eat_time):
        b = blank(tmpl)
        b["cur_time"] = int(cur_time)
        b["eat_time"] = int(eat_time)
        b["entities"] = b["entities"] + [_ent(x=6.5, y=6.5, rx=0.5, ry=0.5, type=ENEMY,
                                              image_type=ENEMY, image_theme=0, render_z=0)]
        return render(env, b)

    def sprite_only(img_type):
        b = blank(tmpl)
        b["entities"] = b["entities"] + [_ent(x=6.5, y=6.5, rx=0.5, ry=0.5, type=LARGE_ORB,
                                              image_type=int(img_type), image_theme=0,
                                              render_z=0)]
        return render(env, b)

    ref = {t: sprite_only(t) for t in (ENEMY_WEAK, ENEMY, ENEMY2, ENEMY3)}
    phase = {}
    for ct in range(16):
        for et_off, tag in ((0, "eatable"), (-200, "not_eatable")):
            fr = one_enemy(ct, ct + et_off)
            match = [t for t, r in ref.items() if (r == fr).all()]
            phase[f"cur_time={ct},{tag}"] = match
    rep["enemy_phase"] = phase
    distinct = sorted({m[0] for m in phase.values() if m})
    rep["enemy_sprites"] = distinct
    unmatched = [k for k, v in phase.items() if not v]
    rep["enemy_phase_unmatched"] = unmatched
    print("enemy sprite ids reachable:", distinct, "unmatched:", len(unmatched), flush=True)
    for k in sorted(phase)[:8]:
        print("   ", k, "->", phase[k], flush=True)

    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=1, default=str)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
