#!/usr/bin/env python
"""GPU evidence for the authorised OGB ``cube.color`` / ``floor.color`` follow-up.

Five claims, each measured on the card against the FROZEN Step-7 corpus rather than a fresh seed
draw, because "the registered default did not move" is a statement about the states the cell was
certified on:

    --mode default    the default frame, byte-for-byte, this tree vs $SCR/se8/baseline
    --mode sequences  custom -> default in one batch, across chunks, across renderer instances;
                      every default frame must equal the --mode default digest
    --mode isolation  cube.color moves no plate pixel; floor.color moves no mover pixel; neither
                      touches the other's style slots or the states
    --mode throughput paired default-style timing, this tree vs baseline, same card, same job

`--git-sha` is passed in rather than read from `git rev-parse`, because the trees on the cluster are
`git archive` exports and not checkouts: the SHA is a property of what was exported, and a null
provenance field is exactly what the reviewer refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

CELLS = ["OGBCube", "OGBScene"]
#: The Step-7 corpus each cell was certified on. Frozen; this script only ever reads it.
CORPUS = "examples/results/step7/ogb-manipulation/states/{cell}.npz"


def digest(a) -> str:
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=16).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def provenance(args) -> dict:
    p = {"host": platform.node(), "python": sys.version.split()[0],
         "git_sha": args.git_sha, "baseline_sha": args.baseline_sha, "tag": args.tag,
         "tree": os.path.abspath(args.tree or os.getcwd()),
         "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
         "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
         "pythonpath": os.environ.get("PYTHONPATH"), "mujoco_gl": os.environ.get("MUJOCO_GL")}
    try:
        p["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        p["gpu"] = None
    for mod in ("torch", "mujoco", "mujoco_warp", "warp", "ogbench", "numpy"):
        try:
            p[mod] = __import__(mod).__version__
        except Exception as exc:
            p[mod] = f"unavailable: {type(exc).__name__}"
    return p


def corpus(cell: str, root: str, n: int):
    path = os.path.join(root, CORPUS.format(cell=cell))
    z = np.load(path)
    st = np.asarray(z["states"], np.float32)[:n]
    return st, {"path": CORPUS.format(cell=cell), "file_sha256": sha256_file(path),
                "key": "states", "n_states": int(st.shape[0]),
                "array_sha256": hashlib.sha256(np.ascontiguousarray(st).tobytes()).hexdigest()}


def _env(cell, batch=1):
    import stable_envs as se
    return se.make(cell, batch=batch, device="cuda")


def _frames(env, states, style=None):
    f = env.render(states) if style is None else env.render(states, style=style)
    return f.detach().cpu().numpy()


def _declaration(env) -> dict:
    d = np.asarray(env.default_style(), np.float32).reshape(-1)
    names = []
    try:
        names = list(env.variation_space.names())
    except Exception:
        pass
    return {"style_dim": int(d.size), "default_style_digest": digest(d),
            "default_style": d.astype(np.float64).tolist(), "leaves": names}


# ------------------------------------------------------------------ (a) default identity

def mode_default(args):
    rows, errs = [], {}
    for cell in CELLS:
        try:
            st, prov = corpus(cell, args.corpus_root, args.n_states)
            env = _env(cell)
            f = _frames(env, st)
            rows.append({"cell": cell, "corpus": prov, "size": int(env.size),
                         "frames_shape": list(f.shape), "frames_digest": digest(f),
                         "states_digest": digest(st), **_declaration(env)})
        except Exception as exc:
            errs[cell] = f"{type(exc).__name__}: {exc}"
    return {"mode": "default", "cells": rows, "errors": errs}


# ------------------------------------------------------------------ (b) custom -> default

def mode_sequences(args):
    """Every way a custom colour could survive into a later default render."""
    from stable_envs.envs.ogb_manip import MJW_CHUNK
    RED = [1.0, 0.0, 0.0]
    DARK = [[0.9, 0.1, 0.1], [0.2, 0.8, 0.3]]
    out, errs = {}, {}
    for cell in CELLS:
        try:
            st, prov = corpus(cell, args.corpus_root, args.n_states)
            env = _env(cell)
            base = np.asarray(env.default_style(), np.float32).reshape(-1)
            layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
            B = int(st.shape[0])

            def rows(cube=None, floor=None, b=B):
                r = np.tile(base, (b, 1))
                if cube is not None:
                    o, w = layout["cube.color"]
                    r[:, o:o + w] = np.asarray(cube, np.float32)
                if floor is not None:
                    o, w = layout["floor.color"]
                    r[:, o:o + w] = np.asarray(floor, np.float32).reshape(-1)
                return r

            ref = digest(_frames(env, st))                      # the (a) default, same process
            seq = {"reference_default_digest": ref, "chunk": int(MJW_CHUNK), "steps": {}}

            # 1. custom, then default, same instance, same batch
            _frames(env, st, rows(cube=RED))
            seq["steps"]["custom_then_default"] = digest(_frames(env, st))
            # 2. both leaves custom, then default
            _frames(env, st, rows(cube=RED, floor=DARK))
            seq["steps"]["mixed_then_default"] = digest(_frames(env, st))
            # 3. per-environment custom (half red, half default), then a uniform default
            half = rows(cube=RED)
            o, w = layout["cube.color"]
            half[B // 2:, o:o + w] = base[o:o + w]
            _frames(env, st, half)
            seq["steps"]["per_env_then_default"] = digest(_frames(env, st))
            # 4. ACROSS CHUNKS: a batch larger than MJW_CHUNK, custom, then default
            big = np.tile(st, (max(2, (2 * MJW_CHUNK + 1) // B + 1), 1))[:2 * MJW_CHUNK + 1]
            _frames(env, big, rows(cube=RED, b=big.shape[0]))
            fb = _frames(env, big)
            seq["steps"]["across_chunks_big_batch_then_small_default"] = digest(_frames(env, st))
            seq["steps"]["big_default_batch_is_self_consistent"] = bool(
                np.array_equal(fb[:B], _frames(env, st)))
            # 5. ACROSS RENDERER INSTANCES: a second env, built after the custom render
            env2 = _env(cell)
            seq["steps"]["second_instance_default"] = digest(_frames(env2, st))
            _frames(env2, st, rows(cube=RED))
            seq["steps"]["first_instance_default_after_second_went_custom"] = digest(
                _frames(env, st))
            # 6. a custom render at a DIFFERENT batch size, then the default again
            _frames(env, st[:3], rows(cube=RED, b=3))
            seq["steps"]["default_after_a_smaller_custom_batch"] = digest(_frames(env, st))

            bad = {k: v for k, v in seq["steps"].items()
                   if isinstance(v, str) and v != ref}
            seq["corpus"] = prov
            seq["all_defaults_identical"] = not bad
            seq["mismatches"] = bad
            seq["declaration"] = _declaration(env)
            out[cell] = seq
        except Exception as exc:
            import traceback
            errs[cell] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
    return {"mode": "sequences", "cells": out, "errors": errs}


# ------------------------------------------------------------------ (c) isolation

def mode_isolation(args):
    """cube.color moves only movers; floor.color moves only the plate; neither moves the state.

    The plate is not inferred from the picture -- it is asked for. `render` composites
    `where(mover, mjwarp_rgb, plate)`, so "this pixel shows the plate" is exactly "this pixel equals
    the baked plate", and the plate is available from the same cache the render used.
    """
    RED = [1.0, 0.0, 0.0]
    GREEN_FLOOR = [[0.0, 0.7, 0.2], [0.6, 0.1, 0.5]]
    out, errs = {}, {}
    for cell in CELLS:
        try:
            st, prov = corpus(cell, args.corpus_root, args.n_states)
            env = _env(cell)
            base = np.asarray(env.default_style(), np.float32).reshape(-1)
            layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
            B = int(st.shape[0])
            st_before = digest(st)
            f0 = _frames(env, st)
            plate = env._plate(int(env.size) * int(env.ss)).detach().cpu().numpy()
            if plate.shape[0] != f0.shape[1]:              # supersampled bake; use the size-1 plate
                plate = env._plate(int(env.size)).detach().cpu().numpy()
            shows_plate = np.all(f0 == plate[None], axis=-1)          # (B, H, W)

            rec = {"corpus": prov, "declaration": _declaration(env),
                   "total_pixels": int(f0.shape[0] * f0.shape[1] * f0.shape[2]),
                   "plate_pixels": int(shows_plate.sum()),
                   "mover_pixels": int((~shows_plate).sum()),
                   "default_digest": digest(f0), "leaves": {}}

            for leaf, value in (("cube.color", RED), ("floor.color", GREEN_FLOOR)):
                o, w = layout[leaf]
                row = np.tile(base, (B, 1))
                row[:, o:o + w] = np.asarray(value, np.float32).reshape(-1)
                other = [i for i in range(base.size) if not (o <= i < o + w)]
                f = _frames(env, st, row)
                moved = np.any(f != f0, axis=-1)
                rec["leaves"][leaf] = {
                    "style_slice": [int(o), int(w)],
                    "npix_moved": int(moved.sum()),
                    "maxdiff": int(np.abs(f.astype(np.int32) - f0.astype(np.int32)).max()),
                    "moved_on_plate_pixels": int((moved & shows_plate).sum()),
                    "moved_on_mover_pixels": int((moved & ~shows_plate).sum()),
                    "other_style_slots_unchanged": bool(
                        np.array_equal(row[:, other], np.tile(base, (B, 1))[:, other])),
                    "states_unchanged": bool(digest(st) == st_before),
                    "digest": digest(f)}
            c, fl = rec["leaves"]["cube.color"], rec["leaves"]["floor.color"]
            rec["cube_color_moves_pixels"] = c["npix_moved"] > 0
            rec["cube_color_moves_no_plate_pixel"] = c["moved_on_plate_pixels"] == 0
            rec["floor_color_moves_pixels"] = fl["npix_moved"] > 0
            rec["floor_color_moves_no_mover_pixel"] = fl["moved_on_mover_pixels"] == 0
            rec["default_reproduced_after_both"] = digest(_frames(env, st)) == rec["default_digest"]
            out[cell] = rec
        except Exception as exc:
            import traceback
            errs[cell] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
    return {"mode": "isolation", "cells": out, "errors": errs}


# ------------------------------------------------------------------ (e) throughput

def mode_throughput(args):
    """Paired default-style render timing. No tier claim: this asks only whether the hot path moved."""
    import torch
    out, errs = {}, {}
    sizes = [int(x) for x in args.batches.split(",")]
    for cell in CELLS:
        try:
            st, prov = corpus(cell, args.corpus_root, args.n_states)
            env = _env(cell)
            rows = {}
            for B in sizes:
                s = np.tile(st, (B // st.shape[0] + 1, 1))[:B]
                for _ in range(args.warmup):
                    env.render(s)
                torch.cuda.synchronize()
                ts = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    env.render(s)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                ts = np.asarray(ts)
                rows[str(B)] = {"reps": int(args.reps), "median_s": float(np.median(ts)),
                                "min_s": float(ts.min()), "max_s": float(ts.max()),
                                "fps_median": float(B / np.median(ts))}
            out[cell] = {"corpus": prov, "declaration": _declaration(env), "batches": rows}
        except Exception as exc:
            errs[cell] = f"{type(exc).__name__}: {exc}"
    return {"mode": "throughput", "cells": out, "errors": errs}


MODES = {"default": mode_default, "sequences": mode_sequences,
         "isolation": mode_isolation, "throughput": mode_throughput}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=sorted(MODES), required=True)
    ap.add_argument("--tag", default="branch")
    ap.add_argument("--tree", default="")
    ap.add_argument("--corpus-root", required=True,
                    help="the tree the FROZEN corpus is read from -- the same one for both arms")
    ap.add_argument("--git-sha", default=None)
    ap.add_argument("--baseline-sha", default=None)
    ap.add_argument("--n-states", type=int, default=64)
    ap.add_argument("--batches", default="1,4096")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    doc = {"provenance": provenance(a)}
    doc.update(MODES[a.mode](a))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1, default=str)
    print(json.dumps(doc, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
