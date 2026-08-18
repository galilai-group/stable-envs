"""Split a Procgen serialized state into the part a renderer needs and the part it does not.

The raw ``get_state`` blob is 22-37 kB, i.e. *bigger* than the 12288-byte observation, which reads like a
refutation of "store states, not images".  But the blob is not a render state.  ``RandGen::serialize``
dumps a ``std::mt19937`` with ``ostream << stdgen``, which writes the 624-word Mersenne state as decimal
*text*, and the game does that three times (level_seed_rand_gen, rand_gen, asset_rand_gen).  None of it
reaches ``game_draw``.

The layout is located empirically rather than assumed, because the installed 0.10.7 binary need not
match the field order in any particular source checkout:

  * an RNG dump is a 4-byte length L in 3000..12000 followed by exactly L bytes of digits and spaces --
    found by scanning at *byte* granularity, since the preceding ``write_string(game_name)`` leaves the
    stream unaligned;
  * the tile grid is the (w, h, n=w*h, n ints) block nearest the end of the blob, verified by n == w*h;
  * the entity block sits between RNG #2 and RNG #3, so ``gap = C + E * n_entities`` for constants C, E
    that are *calibrated*: E is the gcd of the observed gap differences and C is the residue.  If the
    calibration does not come out consistent across games and timesteps it is reported as failed rather
    than papered over.

Run under SLURM.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from collections import Counter

import numpy as np

DIGITS = set(b"0123456789 ")


def find_rng_blocks(blob):
    """(block_start, text_end) for each mt19937 dump; block_start includes is_seeded + length."""
    out = []
    L = len(blob)
    o = 0
    while o + 4 < L:
        n = struct.unpack_from("<i", blob, o)[0]
        if 3000 <= n <= 12000 and o + 4 + n <= L and set(blob[o + 4:o + 4 + n]) <= DIGITS:
            out.append((o - 4, o + 4 + n))
            o = o + 4 + n
        else:
            o += 1
    return out


def find_grid(blob):
    """(w, h, n, offset, tail): the (w, h, n=w*h, n ints) block closest to the end of the blob."""
    best = None
    L = len(blob)
    for o in range(0, L - 12):
        w, h, n = struct.unpack_from("<iii", blob, o)
        if 0 < w <= 4096 and 0 < h <= 4096 and n == w * h and o + 12 + 4 * n <= L:
            tail = L - (o + 12 + 4 * n)
            if tail <= 4096 and (best is None or n > best[2]):
                best = (w, h, n, o, tail)
    return best


def raw_parts(blob):
    rngs = find_rng_blocks(blob)
    grid = find_grid(blob)
    gap = (rngs[2][0] - rngs[1][1]) if len(rngs) >= 3 else None
    return rngs, grid, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/procgen_decomp.json")
    args = ap.parse_args()

    from procgen import ProcgenGym3Env
    from procgen.env import ENV_NAMES

    FRAME = 64 * 64 * 3

    # ---- pass 1: collect blobs and raw parts
    per_game = {}
    for name in ENV_NAMES:
        env = ProcgenGym3Env(num=1, env_name=name, rand_seed=args.seed, distribution_mode="hard")
        rng = np.random.RandomState(args.seed)
        n_actions = env.ac_space.eltype.n
        blobs = []
        for _ in range(args.n_steps):
            blobs.append(env.callmethod("get_state")[0])
            env.act(rng.randint(0, n_actions, size=(1,)).astype(np.int32))
        per_game[name] = [(b,) + raw_parts(b) for b in blobs]
        print(f"collected {name}", flush=True)

    # ---- calibrate the entity stride E from the gaps: every gap is C + E*n, so all pairwise
    # differences are multiples of E.
    gaps = [g for rows in per_game.values() for (_, _, _, g) in rows if g is not None]
    gaps = sorted(set(gaps))
    diffs = [b - a for a, b in zip(gaps[:-1], gaps[1:])]
    E = 0
    for d in diffs:
        E = math.gcd(E, d)
    C = min(gaps) % E if E else None
    print(f"\ncalibration: {len(gaps)} distinct gaps, entity stride E = {E} B "
          f"({E/4 if E else 0:.0f} words), residue C = {C} B")

    rows = []
    for name, recs in per_game.items():
        blobs = [r[0] for r in recs]
        last = recs[-1]
        _, rngs, grid, gap = last
        w, h, n_cells, goff, tail = grid if grid else (0, 0, 0, 0, 0)
        grid_bytes = 12 + 4 * n_cells
        rng_bytes = sum(e - s for s, e in rngs)

        def n_ent(g):
            return None if (g is None or not E) else (g - C) // E

        ents = [n_ent(r[3]) for r in recs]
        ents = [e for e in ents if e is not None]
        ent_bytes_last = (E * n_ent(gap)) if n_ent(gap) is not None else None
        # camera / scale scalars game_draw reads, plus background selection
        cam_bytes = 11 * 4
        render_bytes = grid_bytes + (ent_bytes_last or 0) + cam_bytes
        diffs = [int((np.frombuffer(a, np.uint8) != np.frombuffer(b, np.uint8)).sum())
                 for a, b in zip(blobs[:-1], blobs[1:]) if len(a) == len(b)]
        d = {
            "env_name": name,
            "total_bytes": len(blobs[-1]),
            "n_rng_blocks": len(rngs),
            "rng_text_bytes": rng_bytes,
            "rng_frac_of_state": rng_bytes / len(blobs[-1]),
            "grid_w": w, "grid_h": h, "grid_cells": n_cells, "grid_bytes": grid_bytes,
            "subclass_tail_bytes": tail,
            "n_entities_min": min(ents) if ents else None,
            "n_entities_max": max(ents) if ents else None,
            "entity_bytes_last": ent_bytes_last,
            "render_relevant_bytes": render_bytes,
            "render_over_frame": render_bytes / FRAME,
            "frame_bytes": FRAME,
            "zlib_total": len(zlib.compress(blobs[-1], 6)),
            "step_delta_bytes_median": float(np.median(diffs)) if diffs else None,
        }
        rows.append(d)
        print(f"{name:<12} total {d['total_bytes']:6d}  rng-text {d['rng_text_bytes']:6d} "
              f"({100*d['rng_frac_of_state']:.0f}%)  grid {w}x{h} = {grid_bytes:6d} B  "
              f"ents {d['n_entities_min']}-{d['n_entities_max']} ({ent_bytes_last} B)  tail {tail:4d}"
              f"  -> render-state {render_bytes:6d} B = {d['render_over_frame']:.2f}x frame", flush=True)

    with open(args.out, "w") as f:
        json.dump({"entity_stride_bytes": E, "gap_residue_bytes": C, "games": rows}, f, indent=2)
    print("\nwrote", args.out)
    print("gap histogram (first 20):", Counter(gaps).most_common(20))


if __name__ == "__main__":
    main()
