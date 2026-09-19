"""Save a state corpus to HuggingFace, load it back, re-render, and assert the frames are IDENTICAL.

The repo's claim is that a state corpus can replace an image corpus. That claim is only worth anything
if the states survive a real storage round trip exactly — a corpus that comes back as float64, or with
its style dropped, or with its episode boundaries flattened, still renders *something*, and the
difference from the original is invisible in every plot anyone would draw. So the round trip is asserted
at the pixel level, with ``torch.equal``, not compared with a tolerance:

    frames(states)  ==  frames(load(save(states)))     bit for bit, all N frames

Three storage paths are exercised, because they fail differently:

  ``save_to_disk`` / ``load_from_disk``   the local Arrow directory
  ``to_parquet`` / ``from_parquet``       the wire format ``push_to_hub`` uses; this is how the hub path
                                          is verified without creating a repository under someone's
                                          account, which is not a side effect an example gets to have
  no ``datasets`` installed               the message a user without the optional dep sees

The parquet arm is not decoration. ``DatasetInfo._INCLUDED_INFO_IN_YAML`` (datasets 4.0.0) does not
include ``description``, so metadata written there survives ``save_to_disk`` and is silently dropped by
``push_to_hub``/``to_parquet`` — measured below, printed as a result. That is why ``to_hf`` also writes a
constant ``stable_envs`` column: without it, a corpus pulled off the hub no longer knows which renderer
its numbers belong to, and the failure is a wrong picture rather than an error.

    python examples/hf_roundtrip.py
    python examples/hf_roundtrip.py --corpus examples/results/pusht_rollout.npz
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
import stable_envs.data as se_data


def build_corpus(env, corpus: Path | None, n_traj: int, steps: int, seed: int):
    """States + per-trajectory style, from a saved rollout if one was given.

    The provenance is printed rather than assumed. A round-trip claim is about bytes and holds for any
    states, but a corpus of independent reset states is not a trajectory, and labelling it as one would
    misdescribe the artifact — the file written by ``rollout_with_policy.py`` contains states the
    official simulator actually visited, and that is the one to prefer.
    """
    rng = np.random.default_rng(seed)
    if corpus is not None:
        z = np.load(corpus)
        states = z["states"].astype(np.float32)
        style = z["style"].astype(np.float32) if "style" in z else None
        return states, style, f"{corpus} ({states.shape[0]} trajectories x {states.shape[1]} steps, " \
                              f"official gym_pusht simulator)"
    states = np.stack([env.sample_states(steps, rng) for _ in range(n_traj)]).astype(np.float32)
    style = env.sample_style(n_traj, rng=rng)
    return states, style, (f"{n_traj} x {steps} independent states from the env's reset distribution "
                           f"(NOT a simulated trajectory; the round trip is a statement about bytes)")


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PushT")
    ap.add_argument("--corpus", type=Path, default=None, help="npz from rollout_with_policy.py")
    ap.add_argument("--traj", type=int, default=64)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("examples/results/hf_roundtrip.json"))
    ap.add_argument("--keep", type=Path, default=None, help="write the saved dataset here and keep it")
    args = ap.parse_args()

    env = se.make(args.env, num_envs=1, size=args.size, device="cuda")
    states, style, provenance = build_corpus(env, args.corpus, args.traj, args.steps, args.seed)
    print(f"corpus: {provenance}")

    ds = se.TrajectoryDataset(env, states, style=style, window=args.window, stride=args.stride,
                              size=args.size)
    print(ds)
    print("describe   ", json.dumps(ds.describe()))
    comp = ds.compression
    print(f"compression {comp['stored_bytes']:,} B stored "
          f"({comp['state_bytes']:,} states + {comp['style_bytes']:,} style) against "
          f"{comp['frame_bytes']:,} B of {comp['frame_shape']} frames -> {comp['ratio']:.1f}x")

    # The window property, checked rather than claimed: every row a clip reads must lie inside the
    # trajectory that clip belongs to, and consecutive windows must be consecutive rows.
    rows = ds._rows(np.arange(len(ds))).numpy()
    lo = ds.offsets[ds.clip_traj][:, None]
    hi = (ds.offsets + ds.lengths - 1)[ds.clip_traj][:, None]
    inside = bool(((rows >= lo) & (rows <= hi)).all())
    consec = bool((np.diff(rows, axis=1) == ds.frameskip).all())
    print(f"windows: {len(ds)} clips, none straddles a trajectory boundary: {inside}; "
          f"rows consecutive at frameskip {ds.frameskip}: {consec}")
    assert inside and consec

    batch = ds[[0, 1, 2, 3]]
    print("batch      ", {k: tuple(v.shape) for k, v in batch.items()})

    t0 = time.perf_counter()
    before = ds.all_frames()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    n = before.shape[0]
    print(f"rendered {n:,} frames in {dt * 1e3:.1f} ms ({n / dt:,.0f} frames/s) "
          f"= {before.numel() / 1e6:.1f} MB of pixels from {comp['stored_bytes'] / 1e3:.1f} kB of state")

    tmp = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="stable_envs_hf_"))
    results = {"provenance": provenance, "describe": ds.describe(), "compression": comp,
               "n_frames": int(n), "render_s": dt, "frames_per_s": n / dt,
               "windows_inside_trajectory": inside, "windows_consecutive": consec}
    try:
        # 1. Arrow directory
        p = tmp / "corpus_arrow"
        ds.save_to_disk(p)
        back = se.TrajectoryDataset.load_from_disk(p, window=args.window, stride=args.stride)
        after = back.all_frames()
        ok = torch.equal(before, after)
        maxdiff = int((before.int() - after.int()).abs().max())
        print(f"\nsave_to_disk -> {dir_bytes(p):,} B on disk; reload: {back}")
        print(f"  states identical: {torch.equal(ds.states, back.states)}   "
              f"style identical: {torch.equal(ds.style, back.style)}")
        print(f"  FRAMES BIT-IDENTICAL: {ok}  (max abs pixel difference {maxdiff} over "
              f"{before.numel():,} bytes)")
        assert ok, "the save/load round trip changed the pixels"
        results["arrow"] = {"bytes": dir_bytes(p), "frames_identical": ok, "max_abs_diff": maxdiff,
                            "env_recovered": type(back.env).__name__}

        # 2. parquet — what push_to_hub uploads
        import datasets
        hf = ds.to_hf()
        pq = tmp / "corpus.parquet"
        hf.to_parquet(pq)
        hf2 = datasets.Dataset.from_parquet(str(pq))
        desc_survived = bool(hf2.info.description)
        col_survived = se_data.HF_META_KEY in hf2.column_names
        back2 = se.TrajectoryDataset.from_hf(hf2, window=args.window, stride=args.stride)
        after2 = back2.all_frames()
        ok2 = torch.equal(before, after2)
        print(f"\nto_parquet -> {pq.stat().st_size:,} B (the format push_to_hub uploads)")
        print(f"  info.description survived: {desc_survived}   "
              f"'{se_data.HF_META_KEY}' column survived: {col_survived} "
              f"-> env recovered as {type(back2.env).__name__} with no env= argument")
        print(f"  FRAMES BIT-IDENTICAL: {ok2}")
        assert ok2, "the parquet round trip changed the pixels"
        results["parquet"] = {"bytes": pq.stat().st_size, "frames_identical": ok2,
                              "description_survived": desc_survived, "column_survived": col_survived,
                              "env_recovered": type(back2.env).__name__}

        # 3. the message when the optional dependency is absent
        import builtins
        real = builtins.__import__

        def blocked(name, *a, **k):
            if name == "datasets":
                raise ImportError("No module named 'datasets'")
            return real(name, *a, **k)

        builtins.__import__ = blocked
        import sys
        saved = sys.modules.pop("datasets", None)
        try:
            ds.to_hf()
            msg = "NO ERROR RAISED"
        except ImportError as e:
            msg = str(e)
        finally:
            builtins.__import__ = real
            if saved is not None:
                sys.modules["datasets"] = saved
        print(f"\nwithout `datasets` installed:\n  {msg}")
        results["missing_dep_message"] = msg
    finally:
        if args.keep is None:
            shutil.rmtree(tmp, ignore_errors=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
