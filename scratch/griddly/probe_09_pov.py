"""Probe 9: the partially-observable / RotateWithAvatar player observers.

12 of the 32 shipped games declare a Player.Observer window (TrackAvatar / RotateWithAvatar /
Width / Height / OffsetX / OffsetY). Those are the MiniGridPOV analogue and the ones with the
extra implementation risk: if the observer rotates the frame with the avatar, does it produce an
exact 90-degree permutation of the same tiles (a gather) or a resampled image (a rasterisation)?

Measures, per game:
  * player frame shape vs (Width*T, Height*T)
  * whether every T x T block of the player frame is an EXACT rot90^k of a block seen in the
    global observer -- if yes, the POV path is a gather with a 4x atlas
  * the G1 collision test on player frames, keyed by the render slice
  * throughput
"""
import json, hashlib, collections, time
import numpy as np
import griddly
from griddly import GymWrapper, gd

GAMES = [
    ("zelda_po", "Single-Player/GVGAI/zelda_partially_observable.yaml"),
    ("bait_po", "Single-Player/GVGAI/bait_partially_observable.yaml"),
    ("clusters_po", "Single-Player/GVGAI/clusters_partially_observable.yaml"),
    ("labyrinth_po", "Single-Player/GVGAI/labyrinth_partially_observable.yaml"),
    ("robot_tag_4", "Multi-Agent/robot_tag_4.yaml"),
]


def sample_action(env, rng):
    sp = env.action_space
    if hasattr(sp, "nvec"):
        return [int(rng.integers(k)) for k in sp.nvec]
    if hasattr(sp, "n"):
        return int(rng.integers(sp.n))
    return [([int(rng.integers(k)) for k in s.nvec] if hasattr(s, "nvec") else int(rng.integers(s.n)))
            for s in sp]


def blocks_of(fr, T):
    H, W, _ = fr.shape
    if H % T or W % T:
        return None
    b = fr.reshape(H // T, T, W // T, T, 3).transpose(0, 2, 1, 3, 4).reshape(-1, T, T, 3)
    return b


def key_render(st, extra=()):
    objs = sorted((o["Name"], tuple(o["Location"]), o["Orientation"], o["PlayerId"], o["RenderTileId"])
                  for o in st["Objects"])
    return json.dumps([st["Grid"]["Width"], st["Grid"]["Height"], objs, list(extra)])


for name, yf in GAMES:
    try:
        env = GymWrapper(yaml_file=yf, level=0,
                         global_observer_type=gd.ObserverType.SPRITE_2D,
                         player_observer_type=gd.ObserverType.SPRITE_2D)
        T = env.get_tile_size(0)[0]
        env.seed(3); env.reset(level_id=0)
        pobs = np.array(env._players[0].observe(), copy=True).swapaxes(0, 2)
        gobs = np.array(env.game.observe(), copy=True).swapaxes(0, 2)
        print(f"\n{name}: tile={T} global={gobs.shape} player={pobs.shape} "
              f"players={env.player_count} grid={env.grid_width}x{env.grid_height}")

        gset = set()
        pset = set()
        seen, coll = {}, 0
        nfr = 0
        for ep in range(8):
            env.seed(100 + ep); env.reset(level_id=ep % env.level_count)
            rng = np.random.default_rng(ep * 7 + 2)
            for t in range(40):
                st = env.get_state()
                g = np.array(env.game.observe(), copy=True).swapaxes(0, 2)
                for b in blocks_of(g, T):
                    gset.add(hashlib.md5(np.ascontiguousarray(b)).digest())
                for p in range(env.player_count):
                    fr = np.array(env._players[p].observe(), copy=True).swapaxes(0, 2)
                    nfr += 1
                    bl = blocks_of(fr, T)
                    if bl is not None:
                        for b in bl:
                            pset.add(hashlib.md5(np.ascontiguousarray(b)).digest())
                    k = key_render(st, extra=(p,))
                    m = hashlib.md5(np.ascontiguousarray(fr)).hexdigest()
                    if seen.setdefault(k, m) != m:
                        coll += 1
                try:
                    _, _, done, _ = env.step(sample_action(env, rng))
                except Exception:
                    break
                if np.any(done):
                    break

        # do player blocks live in the rotation closure of the global blocks?
        # rebuild the global block images to rotate them
        env.seed(3); env.reset(level_id=0)
        gimgs = {}
        for ep in range(6):
            env.seed(200 + ep); env.reset(level_id=ep % env.level_count)
            rng = np.random.default_rng(ep)
            for t in range(30):
                g = np.array(env.game.observe(), copy=True).swapaxes(0, 2)
                for b in blocks_of(g, T):
                    gimgs[hashlib.md5(np.ascontiguousarray(b)).digest()] = b.copy()
                try:
                    _, _, done, _ = env.step(sample_action(env, rng))
                except Exception:
                    break
                if np.any(done):
                    break
        closure = set()
        for b in gimgs.values():
            for k in range(4):
                closure.add(hashlib.md5(np.ascontiguousarray(np.rot90(b, k))).digest())
        inside = sum(1 for h in pset if h in closure)
        print(f"   player frames={nfr}  G1 collisions on player obs = {coll}")
        print(f"   distinct global blocks={len(gset)}  distinct player blocks={len(pset)}  "
              f"in rot90-closure of global blocks: {inside}/{len(pset)} "
              f"({100.0*inside/max(1,len(pset)):.1f}%)")

        sink = None

        def r():
            return np.array(env._players[0].observe(), copy=False)[0, 0, 0]
        for _ in range(20):
            r()
        t0 = time.perf_counter(); n = 300
        for _ in range(n):
            r()
        print(f"   player observe: {n/(time.perf_counter()-t0):.0f} f/s")
        del env
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"{name}: FAILED {e}")
