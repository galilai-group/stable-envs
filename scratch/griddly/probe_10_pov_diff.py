"""Probe 10: what exactly are the player-observer tiles that are NOT rot90 of a global tile?

Three candidate explanations for the 62-95%% figure in probe_09:
  (a) the POV window sees out-of-bounds cells the global observer never shows;
  (b) HighlightPlayers (schema default TRUE) draws a marker on the avatar in the player view;
  (c) RotateWithAvatar resamples rather than permuting -> a rasterisation, which would be bad.

Distinguishes them: for each player tile with no exact match in the rot90-closure of the global
tiles, report the nearest global tile (min max-abs-diff over the 4 rotations), how many pixels
differ and WHERE (border ring vs interior), and whether the tile is a pure rotation of another
player tile (which (c) would break).

Also renders the same game with RotateWithAvatar forced FALSE (yaml rewrite) and repeats: if the
non-rotating POV is 100%% inside the global set, rotation is the only thing adding tiles.
"""
import re, os, json, hashlib, collections
import numpy as np
import griddly
from griddly import GymWrapper, gd

GAMES_DIR = os.path.join(os.path.dirname(griddly.__file__), "resources", "games")
T = 24


def sample_action(env, rng):
    sp = env.action_space
    if hasattr(sp, "nvec"):
        return [int(rng.integers(k)) for k in sp.nvec]
    if hasattr(sp, "n"):
        return int(rng.integers(sp.n))
    return [([int(rng.integers(k)) for k in s.nvec] if hasattr(s, "nvec") else int(rng.integers(s.n)))
            for s in sp]


def blocks(fr, T):
    H, W, _ = fr.shape
    return fr.reshape(H // T, T, W // T, T, 3).transpose(0, 2, 1, 3, 4).reshape(-1, T, T, 3)


def collect(yaml_string=None, yaml_file=None, eps=10, steps=40):
    env = GymWrapper(yaml_file=yaml_file, yaml_string=yaml_string, level=0,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     player_observer_type=gd.ObserverType.SPRITE_2D)
    G, P = {}, {}
    for ep in range(eps):
        env.seed(11 + ep); env.reset(level_id=ep % env.level_count)
        rng = np.random.default_rng(ep)
        for t in range(steps):
            g = np.array(env.game.observe(), copy=True).swapaxes(0, 2)
            for b in blocks(g, T):
                G[hashlib.md5(np.ascontiguousarray(b)).digest()] = b.copy()
            for p in range(env.player_count):
                fr = np.array(env._players[p].observe(), copy=True).swapaxes(0, 2)
                for b in blocks(fr, T):
                    P[hashlib.md5(np.ascontiguousarray(b)).digest()] = b.copy()
            try:
                _, _, done, _ = env.step(sample_action(env, rng))
            except Exception:
                break
            if np.any(done):
                break
    del env
    return G, P


def analyse(tag, G, P):
    closure = {}
    for h, b in G.items():
        for k in range(4):
            r = np.rot90(b, k)
            closure[hashlib.md5(np.ascontiguousarray(r)).digest()] = (h, k)
    miss = [h for h in P if h not in closure]
    print(f"  {tag}: global={len(G)} player={len(P)} exact-in-rot90-closure="
          f"{len(P)-len(miss)}/{len(P)} missing={len(miss)}")
    ring = np.zeros((T, T), bool); ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    ring2 = np.zeros((T, T), bool); ring2[:2, :] = ring2[-2:, :] = ring2[:, :2] = ring2[:, -2:] = True
    for h in miss[:8]:
        b = P[h]
        best = None
        for gh, gb in G.items():
            for k in range(4):
                r = np.rot90(gb, k)
                d = np.abs(r.astype(int) - b.astype(int))
                md, nd = int(d.max()), int((d.max(axis=2) > 0).sum())
                if best is None or nd < best[1]:
                    best = (md, nd, k, ((d.max(axis=2) > 0) & ~ring2).sum(),
                            ((d.max(axis=2) > 0) & ring2).sum())
        md, nd, k, interior, border = best
        print(f"     miss tile: nearest global rot{k*90}: maxabs={md} ndiff={nd}/{T*T} "
              f"(interior {interior}, 2px-border {border})  "
              f"uniform_tile={bool(b.min()==b.max())}")
    # is the player tile set closed under rot90 itself?
    selfclosed = sum(1 for h, b in P.items()
                     if all(hashlib.md5(np.ascontiguousarray(np.rot90(b, k))).digest() in P
                            for k in range(4)))
    print(f"     player tile set closed under rot90: {selfclosed}/{len(P)}")


print("### zelda_partially_observable, as shipped (RotateWithAvatar: true) ###")
yf = "Single-Player/GVGAI/zelda_partially_observable.yaml"
G, P = collect(yaml_file=yf)
analyse("rotate=true", G, P)

print("\n### same game, RotateWithAvatar forced FALSE ###")
src = open(os.path.join(GAMES_DIR, yf)).read()
ys = src.replace("RotateWithAvatar: true", "RotateWithAvatar: false")
assert "RotateWithAvatar: false" in ys
G2, P2 = collect(yaml_string=ys)
analyse("rotate=false", G2, P2)

print("\n### same game, RotateWithAvatar false AND window == whole grid (no out-of-bounds) ###")
ys2 = ys.replace("Height: 5", "Height: 3").replace("Width: 5", "Width: 3")
G3, P3 = collect(yaml_string=ys2)
analyse("rotate=false 3x3", G3, P3)
