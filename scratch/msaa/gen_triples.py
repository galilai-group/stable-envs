"""``(ss=1 render, ss=2 render, OFFICIAL frame)`` triples on disjoint state sets.

Three tensors and not two, because two different questions are being asked of the same data and they
have different targets:

* ``ss1 -> official`` is the claim. These are parity cells; what they promise is the official frame,
  and a corrector that reproduces our own ss=2 render perfectly has only reproduced our approximation
  of it. So the TRAINING target is ``official``.
* ``ss1 -> ss2`` is the mechanism. ``ss2`` is stored anyway so the shipped row of every table is the
  same frames the corrector is scored on, and so the residual the corrector is asked to learn can be
  looked at directly.

**The split is a split of the GENERATOR, not of a pool.** Both families' ``sample_states`` rolls out
fresh episodes -- ``initialize_episode`` plus up to 40 random actions -- from a seed, so two seeds give
two continuous state sets whose intersection is empty. That is asserted rather than assumed: the
generator writes the minimum distance between a held-out state and its nearest training state, and
refuses to write a split where any pair is closer than ``--min-sep``. The ``decl`` split is the parity
test's own draw at ``default_rng(20260811)`` and is generated separately so that a number reported on
it is comparable to the assertion the cell has to pass; it is disjoint from training for the same
reason, and labelled as the declaration draw wherever it appears.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import common as K  # noqa: E402

import numpy as np
import torch


def states_for(cell: str, split: str, n: int, size: int) -> tuple[np.ndarray, int]:
    seed = {"train": K.SEED_TRAIN, "heldout": K.SEED_HELDOUT, "decl": K.PARITY_STATE_SEED}[split]
    n = 64 if split == "decl" else n
    env = K.make(cell, batch=1, size=size, seed=seed)
    st = np.asarray(env.sample_states(int(n), rng=np.random.default_rng(seed)), np.float32)
    del env
    torch.cuda.empty_cache()
    return st, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--n-heldout", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--min-sep", type=float, default=1e-6)
    ap.add_argument("--splits", default="train,heldout,decl")
    a = ap.parse_args()

    outroot = os.path.join(K.ROOT, a.cell, f"s{a.size}")
    os.makedirs(outroot, exist_ok=True)
    made = {}
    for split in a.splits.split(","):
        n = a.n_train if split == "train" else a.n_heldout
        t0 = time.perf_counter()
        st, seed = states_for(a.cell, split, n, a.size)
        t_states = time.perf_counter() - t0
        print(f"{a.cell} {split}: {len(st)} states from seed {seed} in {t_states:.1f}s", flush=True)
        made[split] = st

        t0 = time.perf_counter()
        official = K.official_frames(a.cell, st, a.size, chunk=a.chunk, log_every=2048)
        t_off = time.perf_counter() - t0
        env = K.make(a.cell, batch=1, size=a.size)
        t0 = time.perf_counter()
        ss1 = K.our_frames(a.cell, st, a.size, 1, chunk=a.chunk, env=env)
        ss2 = K.our_frames(a.cell, st, a.size, 2, chunk=a.chunk, env=env)
        t_ours = time.perf_counter() - t0
        del env
        torch.cuda.empty_cache()

        d = os.path.join(outroot, split)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, "states.npy"), st)
        np.save(os.path.join(d, "ss1.npy"), ss1)
        np.save(os.path.join(d, "ss2.npy"), ss2)
        np.save(os.path.join(d, "official.npy"), official)
        meta = {"cell": a.cell, "split": split, "size": a.size, "n": len(st), "seed": seed,
                "seconds": {"states": t_states, "official": t_off, "ours": t_ours},
                "official_fps": len(st) / max(t_off, 1e-9),
                "split_rule": "sample_states rolled out from default_rng(seed); train/heldout/decl "
                              f"use seeds {K.SEED_TRAIN}/{K.SEED_HELDOUT}/{K.PARITY_STATE_SEED} "
                              "and are disjoint state sets, not draws from a shared pool"}
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"  wrote {d}  official {meta['official_fps']:.0f} f/s", flush=True)

    # The disjointness ASSERTION. Distances on the GPU: 8192 x 2048 x state_dim is trivial there and
    # a python loop over it is not. A violation is a hard error, because a held-out set that shares a
    # state with training turns every number downstream into a statement about memorisation.
    if "train" in made and "heldout" in made:
        tr = torch.as_tensor(made["train"], device="cuda")
        for other in ("heldout", "decl"):
            if other not in made:
                continue
            ho = torch.as_tensor(made[other], device="cuda")
            dmin = torch.cdist(ho, tr).min().item()
            print(f"min distance {other} -> train: {dmin:.6g}", flush=True)
            if dmin < a.min_sep:
                raise SystemExit(f"{a.cell}: {other} shares a state with train (min sep {dmin:.3g} < "
                                 f"{a.min_sep}); the split is not a split")
            with open(os.path.join(outroot, other, "meta.json")) as f:
                m = json.load(f)
            m["min_dist_to_train"] = dmin
            with open(os.path.join(outroot, other, "meta.json"), "w") as f:
                json.dump(m, f, indent=1)


if __name__ == "__main__":
    main()
