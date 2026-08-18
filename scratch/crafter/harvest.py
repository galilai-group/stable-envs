"""Harvest the 512-state pool stable_envs/envs/crafter.py embeds, and the official frames for it.

States come from uniform-random rollouts of ``crafter.Env`` across eight seeds, keeping only the
``daylight >= 0.5`` half of the cycle (below it the reference draws noise from the world RNG and the
frame is not a function of any state). A quarter of the pool is taken after walking the player onto a
world border with the world's own ``move``, so the out-of-world code is exercised on windows whose
inside is a real neighbourhood.

Writes ``scratch/crafter/pool_gen.py`` (spliced into the env by ``splice.py``) and
``/mnt/vast/home/randall.balestriero/se_logs/crafter_pool.npz`` (states + official frames).
"""
from __future__ import annotations

import base64
import zlib

import numpy as np

import crafter
from crafter import constants

MATS = [None] + list(constants.materials)
OBJTEX = [None, "player-left", "player-right", "player-up", "player-down", "player-sleep",
          "cow", "zombie", "skeleton", "arrow-left", "arrow-right", "arrow-up", "arrow-down",
          "plant", "plant-ripe", "fence"]
ITEMS = list(constants.items)
N_OBJ = len(OBJTEX)
OUTSIDE = len(MATS) * N_OBJ
VIEW, ITEM_ROWS = 9, 2
GW, GH = VIEW, VIEW - ITEM_ROWS
OFF = (GW // 2, GH // 2)
AREA = 64
N_POOL = 512


def read_state(env):
    w, p = env._world, env._player
    cx, cy = int(p.pos[0]), int(p.pos[1])
    codes = np.full((GW, GH), OUTSIDE, np.int64)
    for x in range(GW):
        for y in range(GH):
            px, py = cx + x - OFF[0], cy + y - OFF[1]
            if not (0 <= px < AREA and 0 <= py < AREA):
                continue
            mat, obj = w[(px, py)]
            codes[x, y] = MATS.index(mat) * N_OBJ + (0 if obj is None else OBJTEX.index(obj.texture))
    inv = np.array([p.inventory[k] for k in ITEMS], np.int64)
    return np.concatenate([codes.reshape(-1), inv]).astype(np.uint8), np.float32(w.daylight)


def teleport(w, p, target):
    """Put the player on ``target`` the way walking there would, one legal cell at a time."""
    tx, ty = target
    while tuple(p.pos) != (tx, ty):
        d = (np.sign(tx - p.pos[0]), 0) if p.pos[0] != tx else (0, np.sign(ty - p.pos[1]))
        nxt = (int(p.pos[0] + d[0]), int(p.pos[1] + d[1]))
        occ = w[nxt][1]
        if occ is not None:
            w.remove(occ)
        w.move(p, nxt)


def main():
    codes, days = [], []
    edge_quota = N_POOL // 4
    n_edge = 0
    seed = 0
    while len(codes) < N_POOL:
        env = crafter.Env(seed=seed)
        env.reset()
        rng = np.random.RandomState(seed)
        for t in range(400):
            if len(codes) >= N_POOL:
                break
            if n_edge < edge_quota and t and t % 37 == 0:
                # walk to a random border cell; the window then leaves the world on one side
                side = rng.randint(4)
                k = rng.randint(1, 4)
                x = k - 1 if side == 0 else (AREA - k if side == 1 else rng.randint(4, AREA - 4))
                y = k - 1 if side == 2 else (AREA - k if side == 3 else rng.randint(3, AREA - 3))
                teleport(env._world, env._player, (int(x), int(y)))
            if env._world.daylight >= 0.5:
                c, d = read_state(env)
                if (c[:GW * GH] == OUTSIDE).any():
                    if n_edge >= edge_quota:
                        pass
                    else:
                        n_edge += 1
                codes.append(c)
                days.append(d)
            _, _, done, _ = env.step(rng.randint(0, 17))
            if done:
                env.reset()
        seed += 1
    codes = np.stack(codes[:N_POOL])
    days = np.stack(days[:N_POOL]).reshape(-1, 1)
    has_out = (codes[:, :GW * GH] == OUTSIDE).any(1)
    sleep = codes[:, OFF[0] * GH + OFF[1]] % N_OBJ == OBJTEX.index("player-sleep")
    print(f"pool {codes.shape}, edge windows {int(has_out.sum())}, sleeping {int(sleep.sum())}, "
          f"daylight {days.min():.4f}-{days.max():.4f}, distinct codes {len(np.unique(codes[:, :63]))}")

    def enc(a):
        return base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()

    def wrap(s, per=116):
        return "\n".join(f'    "{s[i:i + per]}"' for i in range(0, len(s), per))

    with open("scratch/crafter/pool_gen.py", "w") as f:
        f.write(f"_POOL_SHAPE = {codes.shape!r}\n_POOL_B64 = (\n{wrap(enc(codes))}\n)\n\n")
        f.write(f"_POOL_DAY_B64 = (\n{wrap(enc(days.astype(np.float32)))}\n)\n")
    print("wrote scratch/crafter/pool_gen.py")

    # official frames for the whole pool, through the env's own reference bridge
    import sys
    sys.path.insert(0, ".")
    from stable_envs.envs.crafter import Crafter
    st = np.concatenate([codes.astype(np.float32), days], 1)
    ref = Crafter.reference_env(64)
    frames = Crafter.reference_render(ref, st)
    np.savez_compressed("/mnt/vast/home/randall.balestriero/se_logs/crafter_pool.npz",
                        states=st, frames=frames)
    print(f"official frames {frames.shape} -> se_logs/crafter_pool.npz")


if __name__ == "__main__":
    main()
