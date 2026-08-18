"""B1: is the ROTATED POV frame exactly ``tilemap.blit(atlas, idx)`` for an index that is a
function of (3x3 world stencil, avatar orientation)?

This re-measures the one thing that stopped Griddly being built (docs/research/griddly.md, "The risk
that decides the effort"): ``RotateWithAvatar`` rotates through the GPU sampler, so a rotated POV tile
is NOT an exact ``rot90`` of any global tile. The recorded conclusion was "bake orientation and view
rotation as atlas dimensions (roughly x4) -- cost, not novelty". Cost only, though, IF the rotated
frame is still a grid of tiles from a FINITE alphabet indexed by a LOCAL key. If the sampler smears
across tile boundaries, or if the tile depends on where in the window it lands, it is not a gather at
all and the obstacle is worse than recorded, not smaller.

Procedure, per game:

  0. two envs from the same GDY, one as shipped (RotateWithAvatar: true) and one with it forced false,
     stepped with identical seeds and identical actions; ``get_state()`` compared every step so a
     divergence cannot be mistaken for a rendering result. Every frame checked non-uniform.
  1. TILE ALIGNMENT. Cut both POV frames into T x T blocks and count DISTINCT blocks, with a
     saturation curve. A finite, saturating alphabet is necessary for a gather; an exploding count
     means sub-tile offsets. The distinct-block count IS the atlas; the key table is separate.
  2. WINDOW MAPPING. Fit (cx, cy) such that the UNROTATED POV cell (vi, vj) is exactly the global
     frame's cell (ax + vi - cx, ay + vj - cy). Reported as an exact-match fraction, not assumed.
  3. ROTATION MAPPING. For each avatar orientation, try all four rotations R_k of the window offset
     and score each by how many collisions the key -> tile table takes. The right one should be the
     only one with zero.
  4. RECONSTRUCTION. Build the index table on the first half, rebuild the held-out half by a pure
     gather. Reported three ways: exact frames, exact pixels, and exact pixels restricted to cells
     whose key was already in the table -- so a coverage shortfall cannot be read as a parity
     failure, and vice versa. Then the same with the table built over everything, which is the
     "is the frame a function of this key at all" number.
"""
import os, sys, json, collections
import numpy as np
import griddly
from griddly import GymWrapper, gd

OUT = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(os.path.dirname(griddly.__file__), "resources", "games")
T = 24

GAMES = {
    "sokoban2_po": "Single-Player/GVGAI/sokoban2_partially_observable.yaml",
    "zelda_po": "Single-Player/GVGAI/zelda_partially_observable.yaml",
    "labyrinth_po": "Single-Player/GVGAI/labyrinth_partially_observable.yaml",
    "clusters_po": "Single-Player/GVGAI/clusters_partially_observable.yaml",
    "cookmepasta_po": "Single-Player/GVGAI/cookmepasta_partially_observable.yaml",
    "drunkdwarf": "Single-Player/Mini-Grid/minigrid-drunkdwarf.yaml",
}
N_FRAMES = 1200          # per game; first half builds the table, second half is held out
ROT = {0: lambda u, v: (u, v), 1: lambda u, v: (-v, u),
       2: lambda u, v: (-u, -v), 3: lambda u, v: (v, -u)}


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


def blocks(fr):
    h, w = fr.shape[0] // T, fr.shape[1] // T
    keys = np.empty((h, w), dtype=object)
    for j in range(h):
        for i in range(w):
            keys[j, i] = fr[j * T:(j + 1) * T, i * T:(i + 1) * T].tobytes()
    return keys


def run_game(name, yf):
    src = open(os.path.join(GAMES_DIR, yf)).read()
    assert "RotateWithAvatar: true" in src, f"{name}: no RotateWithAvatar in yaml"
    ea = GymWrapper(yaml_string=src, level=0,
                    global_observer_type=gd.ObserverType.SPRITE_2D,
                    player_observer_type=gd.ObserverType.SPRITE_2D)
    eb = GymWrapper(yaml_string=src.replace("RotateWithAvatar: true", "RotateWithAvatar: false"),
                    level=0, global_observer_type=gd.ObserverType.SPRITE_2D,
                    player_observer_type=gd.ObserverType.SPRITE_2D)
    avatar_name = ea.gdy.get_avatar_object()

    recs, uniform, ep = [], 0, 0
    while len(recs) < N_FRAMES and ep < 400:
        lid = ep % ea.level_count
        ea.seed(ep); ea.reset(level_id=lid)
        eb.seed(ep); eb.reset(level_id=lid)
        rng = np.random.default_rng(1000 + ep)
        for t in range(80):
            sa, sb = ea.get_state(), eb.get_state()
            if cellmap(sa) != cellmap(sb):
                raise AssertionError(f"{name}: the two envs diverged at ep {ep} t {t}")
            g = np.array(ea.game.observe(), copy=True).swapaxes(0, 2)
            pa = np.array(ea._players[0].observe(), copy=True).swapaxes(0, 2)
            pb = np.array(eb._players[0].observe(), copy=True).swapaxes(0, 2)
            for fr in (g, pa, pb):
                if len(np.unique(fr)) < 2:
                    uniform += 1
            recs.append((sa, blocks(g), blocks(pa), blocks(pb), pa,
                         g.shape[1] // T, g.shape[0] // T, lid))
            a = sample_action(ea, rng)
            try:
                _, _, da, _ = ea.step(a)
                _, _, db, _ = eb.step(a)
            except Exception:
                break
            if np.any(da) or np.any(db):
                break
            if len(recs) >= N_FRAMES:
                break
        ep += 1

    V = recs[0][4].shape[0] // T

    # ---- 1. tile alphabets (this is the ATLAS; the key table below is the index)
    alpha_g, alpha_rot, alpha_norot = set(), set(), set()
    curve = []
    for i, r in enumerate(recs):
        alpha_g.update(r[1].ravel()); alpha_rot.update(r[2].ravel()); alpha_norot.update(r[3].ravel())
        if i in (24, 49, 99, 199, 399, 599, 899, len(recs) - 1):
            curve.append([i + 1, len(alpha_g), len(alpha_norot), len(alpha_rot)])

    # ---- 2. window mapping, unrotated
    score, tot = collections.Counter(), collections.Counter()
    for st, gb, pak, pbk, _pa, GW, GH, _l in recs[:250]:
        av = [o for o in st["Objects"] if o["Name"] == avatar_name]
        if not av:
            continue
        ax, ay = av[0]["Location"]
        for cx in range(V):
            for cy in range(V):
                for vj in range(V):
                    for vi in range(V):
                        gx, gy = ax + vi - cx, ay + vj - cy
                        if not (0 <= gx < GW and 0 <= gy < GH):
                            continue
                        tot[(cx, cy)] += 1
                        if pbk[vj, vi] == gb[gy, gx]:
                            score[(cx, cy)] += 1
    cx, cy = max(score, key=lambda k: score[k] / max(tot[k], 1))
    norot_frac = score[(cx, cy)] / max(tot[(cx, cy)], 1)

    def stencil(cm, x, y, GW, GH):
        return tuple(cm.get((x + dx, y + dy),
                            "__OUT__" if not (0 <= x + dx < GW and 0 <= y + dy < GH) else "__EMPTY__")
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1))

    def build(sub, k_of_ori, with_pos):
        tab, coll = {}, 0
        for st, gb, pak, pbk, _pa, GW, GH, _l in sub:
            av = [o for o in st["Objects"] if o["Name"] == avatar_name]
            if not av:
                continue
            ax, ay = av[0]["Location"]
            ori = av[0]["Orientation"]
            k = k_of_ori.get(ori, 0)
            cm = cellmap(st)
            for vj in range(V):
                for vi in range(V):
                    du, dv = ROT[k](vi - cx, vj - cy)
                    key = (stencil(cm, ax + du, ay + dv, GW, GH), ori) + ((vi, vj) if with_pos else ())
                    tile = pak[vj, vi]
                    if key in tab:
                        coll += (tab[key] != tile)
                    else:
                        tab[key] = tile
        return tab, coll

    oris = sorted({o["Orientation"] for r in recs
                   for o in r[0]["Objects"] if o["Name"] == avatar_name})
    half = recs[:len(recs) // 2]
    per_ori = {}
    for o in oris:
        sub = [r for r in half
               if any(x["Name"] == avatar_name and x["Orientation"] == o for x in r[0]["Objects"])]
        sc = {k: build(sub, {o: k}, False)[1] for k in range(4)}
        per_ori[o] = dict(collisions_by_k=sc, chosen=int(min(sc, key=lambda z: sc[z])), n=len(sub))
    k_of_ori = {o: per_ori[o]["chosen"] for o in oris}

    tab, coll = build(half, k_of_ori, False)
    tab_p, coll_p = build(half, k_of_ori, True)
    tab_all, coll_all = build(recs, k_of_ori, False)

    def reconstruct(table, sub):
        nf = ne = px_tot = px_ok = miss = cov_px = cov_ok = 0
        maxabs = 0
        for st, gb, pak, pbk, pa, GW, GH, _l in sub:
            av = [o for o in st["Objects"] if o["Name"] == avatar_name]
            if not av:
                continue
            ax, ay = av[0]["Location"]
            ori = av[0]["Orientation"]
            k = k_of_ori.get(ori, 0)
            cm = cellmap(st)
            rec = np.zeros_like(pa)
            covered = np.zeros((V, V), bool)
            for vj in range(V):
                for vi in range(V):
                    du, dv = ROT[k](vi - cx, vj - cy)
                    b = table.get((stencil(cm, ax + du, ay + dv, GW, GH), ori))
                    if b is None:
                        miss += 1
                        continue
                    covered[vj, vi] = True
                    rec[vj * T:(vj + 1) * T, vi * T:(vi + 1) * T] = np.frombuffer(
                        b, np.uint8).reshape(T, T, 3)
            d = np.abs(rec.astype(np.int32) - pa.astype(np.int32))
            nf += 1
            maxabs = max(maxabs, int(d.max()))
            px_tot += d.size; px_ok += int((d == 0).sum())
            m = np.repeat(np.repeat(covered, T, 0), T, 1)[:, :, None]
            cov_px += int(m.sum() * 3); cov_ok += int(((d == 0) & m).sum())
            ne += bool(covered.all() and d.max() == 0)
        return dict(frames=nf, exact_frames=ne, pixel_exact_frac=px_ok / max(px_tot, 1),
                    covered_pixel_exact_frac=cov_ok / max(cov_px, 1),
                    covered_cell_frac=cov_px / max(px_tot, 1), max_abs=maxabs, misses=miss)

    held = recs[len(recs) // 2:]
    res = dict(
        game=name, yaml=yf, n_records=len(recs), uniform_frames=uniform, V=V, T=T,
        levels_used=sorted({r[7] for r in recs}),
        grid_sizes=sorted({(r[5], r[6]) for r in recs}),
        pov_shape=list(recs[0][4].shape),
        orientations_seen=oris,
        atlas_tiles_global=len(alpha_g), atlas_tiles_pov_norot=len(alpha_norot),
        atlas_tiles_pov_rot=len(alpha_rot),
        atlas_tiles_union=len(alpha_g | alpha_rot | alpha_norot),
        rot_multiplier=len(alpha_g | alpha_rot | alpha_norot) / max(len(alpha_g), 1),
        atlas_bytes_union=len(alpha_g | alpha_rot | alpha_norot) * T * T * 3,
        saturation_frames_global_norot_rot=curve,
        window_center=[cx, cy], unrotated_pov_equals_global_crop_frac=norot_frac,
        rotation_fit=per_ori, k_of_orientation=k_of_ori,
        index_keys_half=len(tab), index_collisions_half=coll,
        index_keys_half_withpos=len(tab_p), index_collisions_half_withpos=coll_p,
        index_keys_all=len(tab_all), index_collisions_all=coll_all,
        recon_heldout=reconstruct(tab, held),
        recon_full_table=reconstruct(tab_all, recs),
    )
    del ea, eb
    return res


if __name__ == "__main__":
    out = {}
    for name, yf in GAMES.items():
        try:
            out[name] = run_game(name, yf)
        except Exception as e:                                   # noqa: BLE001
            import traceback; traceback.print_exc()
            out[name] = dict(game=name, error=f"{type(e).__name__}: {e}")
        print(json.dumps(out[name], default=str)[:1600], flush=True)
    with open(os.path.join(OUT, "b1_pov_gather.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("WROTE", os.path.join(OUT, "b1_pov_gather.json"), flush=True)
    sys.stdout.flush()
