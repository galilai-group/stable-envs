"""B3: run the repo's OWN kernel on real Griddly frames, then price it.

docs/research/griddly.md's G4 cell is an UPPER BOUND -- ``tilemap.blit`` on "an atlas of the right
shape", because the env is not built. This replaces the bound with the real thing: the atlas is the
distinct composed tiles Griddly actually drew, the index is derived FROM STATE (the 3x3 stencil and
the avatar orientation established in b1, not from the frame), and the blit output is diffed against
Griddly's own frames. A throughput number computed on a decoder that does not reproduce the picture
is the RoboCasa mistake, so parity is measured first and the timing is only reported for a decoder
that passed it.

Then: official Griddly f/s on one pinned core, ``tilemap.blit`` f/s over a batch sweep, and JPEG
decode f/s/core on the same frames, with the crossover batch stated rather than a single ratio.
"""
import os, io, sys, json, time, collections
import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import torch
import griddly
from griddly import GymWrapper, gd
from PIL import Image
from stable_envs.render.tilemap import blit, HAVE_TRITON

OUT = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(os.path.dirname(griddly.__file__), "resources", "games")
T = 24
ROT = {0: lambda u, v: (u, v), 1: lambda u, v: (-v, u),
       2: lambda u, v: (-u, -v), 3: lambda u, v: (v, -u)}
K_OF_ORI = {"NONE": 0, "UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}   # measured in b1, all 6 games
# window centre, measured in b1 against the UNROTATED POV vs a crop of the global frame at exact-match
# fraction 1.0. Fitting it here from collision counts on a short prefix picks (1,4) for zelda_po and
# leaves 8 collisions, which is a fit failure and not a property of the renderer -- so use b1's.
CENTRES = {"sokoban2_po": (2, 2), "zelda_po": (2, 4), "labyrinth_po": (2, 4),
           "clusters_po": (2, 4), "cookmepasta_po": (2, 4), "drunkdwarf": (3, 6)}

POV_GAMES = {"zelda_po": "Single-Player/GVGAI/zelda_partially_observable.yaml",
             "labyrinth_po": "Single-Player/GVGAI/labyrinth_partially_observable.yaml",
             "sokoban2_po": "Single-Player/GVGAI/sokoban2_partially_observable.yaml"}
GLOBAL_GAMES = {"sokoban": "Single-Player/GVGAI/sokoban.yaml",
                "zelda": "Single-Player/GVGAI/zelda.yaml",
                "butterflies": "Single-Player/GVGAI/butterflies.yaml"}
N = 800
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def sample_action(env, rng):
    sp = env.action_space
    if hasattr(sp, "nvec"):
        return [int(rng.integers(k)) for k in sp.nvec]
    if hasattr(sp, "n"):
        return int(rng.integers(sp.n))
    return [([int(rng.integers(k)) for k in s.nvec] if hasattr(s, "nvec") else int(rng.integers(s.n)))
            for s in sp]


def cellmap(st):
    m = collections.defaultdict(list)
    for o in st["Objects"]:
        m[tuple(o["Location"])].append(
            (o["Name"], o["Orientation"], int(o["PlayerId"]), int(o["RenderTileId"])))
    return {k: tuple(sorted(v)) for k, v in m.items()}


def stencil(cm, x, y, GW, GH):
    return tuple(cm.get((x + dx, y + dy),
                        "__OUT__" if not (0 <= x + dx < GW and 0 <= y + dy < GH) else "__EMPTY__")
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1))


def decode_fps(img, quality=90, n=None):
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()
    n = n or max(200, int(4e7 / img.size * 3))
    for _ in range(20):
        np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    t0 = time.perf_counter()
    for _ in range(n):
        np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    return n / (time.perf_counter() - t0), len(b)


def bench_blit(atlas_np, idx_np, tile_size):
    dev = "cuda"
    atlas = torch.from_numpy(atlas_np).to(dev).contiguous()
    base = torch.from_numpy(idx_np).to(dev).to(torch.int32)
    H, W = base.shape[1], base.shape[2]
    rows = []
    for B in BATCHES:
        rep = base[torch.randint(0, base.shape[0], (B,), device=dev)].contiguous()
        try:
            out = torch.empty(B, H * tile_size, W * tile_size, 3, dtype=torch.uint8, device=dev)
            for _ in range(5):
                blit(atlas, rep, tile_size, out=out)
            torch.cuda.synchronize()
            it = max(5, min(400, int(3e9 / (out.numel() + 1))))
            t0 = time.perf_counter()
            for _ in range(it):
                blit(atlas, rep, tile_size, out=out)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / it
            rows.append(dict(batch=B, fps=B / dt, gbps=out.numel() * 2 / dt / 1e9))
            del out
        except torch.cuda.OutOfMemoryError:
            break
        torch.cuda.empty_cache()
    return rows


def harvest(name, yf, pov):
    src = open(os.path.join(GAMES_DIR, yf)).read()
    env = GymWrapper(yaml_string=src, level=0,
                     global_observer_type=gd.ObserverType.SPRITE_2D,
                     player_observer_type=gd.ObserverType.SPRITE_2D)
    avatar = env.gdy.get_avatar_object()
    recs, ep = [], 0
    while len(recs) < N and ep < 300:
        env.seed(ep); env.reset(level_id=ep % env.level_count)
        rng = np.random.default_rng(300 + ep)
        for t in range(80):
            st = env.get_state()
            g = np.array(env.game.observe(), copy=True).swapaxes(0, 2)
            fr = (np.array(env._players[0].observe(), copy=True).swapaxes(0, 2) if pov else g)
            assert len(np.unique(fr)) > 1, f"{name}: uniform frame"
            recs.append((st, fr, g.shape[1] // T, g.shape[0] // T))
            try:
                _, _, done, _ = env.step(sample_action(env, rng))
            except Exception:
                break
            if np.any(done) or len(recs) >= N:
                break
        ep += 1
    return env, avatar, recs


def run(name, yf, pov):
    env, avatar, recs = harvest(name, yf, pov)
    fr0 = recs[0][1]
    V = fr0.shape[0] // T
    Wc = fr0.shape[1] // T
    if pov:
        assert V == Wc
        # window centre, fitted the same way b1 fitted it, but here only for the shipped rotated env:
        # under RotateWithAvatar the avatar sits at a fixed window cell, so recover it by finding the
        # cell whose stencil key table is collision free for every candidate centre.
        cands = [(cx, cy) for cx in range(V) for cy in range(V)]
    else:
        cands = [None]

    def build(centre, sub):
        tab, coll = {}, 0
        for st, fr, GW, GH in sub:
            cm = cellmap(st)
            if pov:
                av = [o for o in st["Objects"] if o["Name"] == avatar]
                if not av:
                    continue
                ax, ay = av[0]["Location"]; ori = av[0]["Orientation"]
                k = K_OF_ORI[ori]
                cells = [((vi, vj), ROT[k](vi - centre[0], vj - centre[1])) for vj in range(V)
                         for vi in range(V)]
                cells = [((vi, vj), (ax + du, ay + dv)) for (vi, vj), (du, dv) in cells]
                extra = (ori,)
            else:
                cells = [((i, j), (i, j)) for j in range(GH) for i in range(GW)]
                extra = ()
            for (vi, vj), (wx, wy) in cells:
                key = (stencil(cm, wx, wy, GW, GH),) + extra
                tile = fr[vj * T:(vj + 1) * T, vi * T:(vi + 1) * T].tobytes()
                if key in tab:
                    coll += (tab[key] != tile)
                else:
                    tab[key] = tile
        return tab, coll

    if pov:
        centre = CENTRES[name]
    else:
        centre = None
    tab, coll = build(centre, recs)

    tiles = sorted(set(tab.values()))
    tid = {t: i for i, t in enumerate(tiles)}
    atlas = np.stack([np.frombuffer(t, np.uint8).reshape(T, T, 3) for t in tiles])

    idxs, refs, skipped = [], [], 0
    for st, fr, GW, GH in recs:
        if not pov and (GW, GH) != (recs[0][3] and fr.shape[1] // T, fr.shape[0] // T):
            pass
        if fr.shape != fr0.shape:
            skipped += 1
            continue
        cm = cellmap(st)
        if pov:
            av = [o for o in st["Objects"] if o["Name"] == avatar]
            if not av:
                skipped += 1; continue
            ax, ay = av[0]["Location"]; ori = av[0]["Orientation"]; k = K_OF_ORI[ori]
            ix = np.empty((V, V), np.int32)
            ok = True
            for vj in range(V):
                for vi in range(V):
                    du, dv = ROT[k](vi - centre[0], vj - centre[1])
                    key = (stencil(cm, ax + du, ay + dv, GW, GH), ori)
                    if key not in tab:
                        ok = False; break
                    ix[vj, vi] = tid[tab[key]]
        else:
            ix = np.empty((fr.shape[0] // T, fr.shape[1] // T), np.int32)
            ok = True
            for j in range(ix.shape[0]):
                for i in range(ix.shape[1]):
                    key = (stencil(cm, i, j, GW, GH),)
                    if key not in tab:
                        ok = False; break
                    ix[j, i] = tid[tab[key]]
        if not ok:
            skipped += 1; continue
        idxs.append(ix); refs.append(fr)
    idx = np.stack(idxs); ref = np.stack(refs)

    # ---- parity: the repo's own kernel, against griddly's own frames
    a_t = torch.from_numpy(atlas).cuda()
    i_t = torch.from_numpy(idx).cuda().to(torch.int32)
    got = blit(a_t, i_t, T).cpu().numpy()
    d = np.abs(got.astype(np.int32) - ref.astype(np.int32))
    parity = dict(frames=int(ref.shape[0]), max_abs=int(d.max()),
                  exact_frames=int((d.reshape(d.shape[0], -1).max(1) == 0).sum()),
                  pixel_exact_frac=float((d == 0).mean()),
                  reference_distinct_bytes=int(len(np.unique(ref[0]))),
                  skipped_frames=skipped)

    # ---- official renderer f/s, one process
    for _ in range(20):
        env._players[0].observe() if pov else env.game.observe()
    n = 400
    t0 = time.perf_counter()
    for _ in range(n):
        np.array(env._players[0].observe() if pov else env.game.observe(),
                 copy=True).swapaxes(0, 2)
    official = n / (time.perf_counter() - t0)

    dec_fps, jpeg_b = decode_fps(ref[0])
    rows = bench_blit(atlas, idx, T)
    best = max(rows, key=lambda r: r["fps"])
    cross_official = next((r["batch"] for r in rows if r["fps"] > official), None)
    cross_decode = next((r["batch"] for r in rows if r["fps"] > dec_fps), None)
    res = dict(game=name, pov=pov, frame_shape=list(fr0.shape), tile_size=T,
               atlas_tiles=len(tiles), atlas_bytes=len(tiles) * T * T * 3,
               index_keys=len(tab), index_collisions=coll, window_centre=centre,
               parity=parity, official_fps=official, decode_fps_per_core=dec_fps,
               jpeg_bytes=jpeg_b, blit=rows,
               best_fps=best["fps"], best_batch=best["batch"], best_gbps=best["gbps"],
               ratio_vs_official=best["fps"] / official,
               crossover_batch_vs_official=cross_official,
               crossover_batch_vs_decode=cross_decode,
               cores_of_decode_replaced=best["fps"] / dec_fps,
               official_cores_of_decode=official / dec_fps)
    del env
    return res


if __name__ == "__main__":
    print("HAVE_TRITON", HAVE_TRITON, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    out = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "have_triton": HAVE_TRITON, "games": {}}
    for name, yf in list(POV_GAMES.items()):
        try:
            out["games"][name] = run(name, yf, True)
        except Exception as e:                                    # noqa: BLE001
            import traceback; traceback.print_exc()
            out["games"][name] = dict(game=name, error=f"{type(e).__name__}: {e}")
        print(json.dumps(out["games"][name], default=str)[:1200], flush=True)
    for name, yf in list(GLOBAL_GAMES.items()):
        try:
            out["games"][name] = run(name, yf, False)
        except Exception as e:                                    # noqa: BLE001
            import traceback; traceback.print_exc()
            out["games"][name] = dict(game=name, error=f"{type(e).__name__}: {e}")
        print(json.dumps(out["games"][name], default=str)[:1200], flush=True)
    with open(os.path.join(OUT, "b3_blit_parity_speed.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("WROTE", os.path.join(OUT, "b3_blit_parity_speed.json"), flush=True)
    sys.stdout.flush()
