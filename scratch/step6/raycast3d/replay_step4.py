"""Step 6C — replay frozen Step-4 configurations and diff them against the frozen digests.

This is the regression oracle, not a new one. The digest is
``stable_envs.testing.conformance.frames_digest`` (the function Step 4 used), the states come from
``examples/results/step4/states/<config>.npz`` (read-only), and the expected values are read out of
``examples/results/step4/conformance/<shard>.json``. The render is reproduced the way
``examples/fidelity_step4.py:oracle`` produced it: a FRESH env at ``batch = len(states)`` and the
env's own default size, ``env.render(states)`` with the default appearance, plus the default-style
and corpus-style digests where Step 4 recorded them.

Two things it adds, because a digest that MOVES needs more than "it moved":

  * ``--dump-dir`` writes the rendered frames as uint8 ``.npy``. Running the same command once under
    ``PYTHONPATH=$SCR/se6/baseline`` and once under the branch, IN ONE JOB ON ONE CARD, and then
    ``--compare-dumps`` gives the pixel diff instead of a changed hash.
  * ``--sample-states SEED`` covers callers with no frozen corpus (``maze2d``'s PointMaze*,
    ``mujoco_locomotion``'s Mujoco*): they cannot be replayed against Step 4, but they CAN be held to
    before-vs-after self-consistency on the same seeded draw.

Writes only under ``examples/results/step6/raycast3d/replay/``. Never writes under
``examples/results/step4/``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# $PYTHONPATH DECIDES WHICH stable_envs IS MEASURED. A before/after run puts the baseline copy on
# PYTHONPATH for one leg and this branch for the other, so this file's own repo root must NOT be
# inserted ahead of it -- doing that made both legs import the branch and report a 1.00x speed-up as
# a real one. Appended, so the helper imports still resolve when PYTHONPATH is unset.
sys.path.append(str(ROOT))


def _root_tag(p: str) -> str:
    """An absolute Oscar path with the scratch root collapsed to ``$SCR``.

    The raw path is the evidence -- it says whether a leg loaded the baseline or the branch -- but
    written in full it reads to ``tests/unit/test_repo_contracts.py`` as a repo-relative
    ``scratch/...`` citation that cannot resolve, which is a false failure in a real check. Same
    string, written the way the sbatch files write it.
    """
    return re.sub(r"^/oscar/scratch/([^/]+)", r"$SCR", p)



STEP4 = ROOT / "examples" / "results" / "step4"
SHARDS = ("dmc", "ogb", "raster", "fetch", "swm")


def frozen_oracles() -> dict:
    """config_id -> (shard, oracle dict) from the frozen Step-4 conformance shards."""
    out = {}
    for sh in SHARDS:
        p = STEP4 / "conformance" / f"{sh}.json"
        if not p.is_file():
            continue
        for cfg, rec in json.loads(p.read_text())["configs"].items():
            if isinstance(rec.get("oracle"), dict):
                out[cfg] = (sh, rec["oracle"])
    return out


def replay_one(cfg, device, oracle, seed, dump_dir):
    import numpy as np
    import stable_envs as se
    from stable_envs.testing import conformance as C

    npz = STEP4 / "states" / f"{cfg}.npz"
    rec = {"config": cfg, "device": device, "stable_envs_from": _root_tag(se.__file__)}
    if npz.is_file():
        states = np.asarray(np.load(npz)["states"], np.float32)
        rec["states_source"] = f"examples/results/step4/states/{cfg}.npz"
    elif seed is not None:
        probe = se.make(cfg, batch=1, device=device)
        states = np.asarray(probe.sample_states(64, rng=np.random.default_rng(seed)), np.float32)
        rec["states_source"] = f"sample_states(64, rng=default_rng({seed}))  [no frozen corpus]"
        del probe
    else:
        rec["error"] = "no frozen corpus and --sample-states not given"
        return rec

    n = int(len(states))
    env = se.make(cfg, batch=n, device=device)
    rec["size"] = int(env.size)
    rec["supersample"] = int(env.ss)
    rec["batch"] = n

    frames = env.render(states).cpu().numpy()
    rec["digest_ours"] = C.frames_digest(frames)
    rec["frames_shape"] = list(frames.shape)
    rec["mean_pixel"] = float(frames.mean())
    if env.style_dim:
        import numpy as _np
        rec["digest_ours_default_style"] = C.frames_digest(env.render(
            states, style=_np.tile(_np.asarray(env.default_style(), _np.float32), (n, 1))))
    else:
        rec["digest_ours_default_style"] = rec["digest_ours"]

    if oracle:
        exp = {k: oracle.get(k) for k in ("digest_ours", "digest_ours_default_style",
                                          "size", "supersample", "batch", "device")}
        rec["frozen"] = exp
        rec["match"] = {
            "digest_ours": rec["digest_ours"] == exp["digest_ours"],
            "digest_ours_default_style":
                rec["digest_ours_default_style"] == exp["digest_ours_default_style"],
            "shape_matches": (exp["size"] == rec["size"] and exp["supersample"] == rec["supersample"]
                              and exp["batch"] == n),
        }
        rec["PASS"] = all(rec["match"].values())
    else:
        rec["frozen"] = None
        rec["PASS"] = None

    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)
        np.save(dump_dir / f"{cfg}.npy", frames)
        rec["dump"] = str((dump_dir / f"{cfg}.npy").name)
    return rec


def compare_dumps(a_dir: Path, b_dir: Path) -> dict:
    import numpy as np
    out = {"before_dir": str(a_dir), "after_dir": str(b_dir), "configs": {}}
    for f in sorted(a_dir.glob("*.npy")):
        cfg = f.stem
        g = b_dir / f.name
        if not g.is_file():
            out["configs"][cfg] = {"error": "missing in after"}
            continue
        A, B = np.load(f), np.load(g)
        if A.shape != B.shape:
            out["configs"][cfg] = {"error": f"shape {A.shape} != {B.shape}"}
            continue
        d = np.abs(A.astype(np.int32) - B.astype(np.int32))
        pm = d.max(-1)
        out["configs"][cfg] = {
            "identical": bool(d.max() == 0), "max_abs_diff": int(d.max()),
            "px_differing": int((pm > 0).sum()), "px_total": int(pm.size),
            "frames_differing": int((pm.reshape(len(A), -1).max(1) > 0).sum()),
            "mean_abs_diff": float(d.mean()),
        }
    # A COMPARISON OF NOTHING IS NOT A PASS. ``all([])`` is True, and the first maze2d run (job
    # 6347125) reported ALL_IDENTICAL=True over two empty dump directories because every config had
    # failed to construct. A verification that goes green when it verified nothing is worse than one
    # that fails.
    if not out["configs"]:
        out["all_identical"] = None
        out["error"] = ("nothing to compare: no .npy dumps in the BEFORE directory, so every "
                        "configuration failed before it rendered. This is NOT a pass.")
    else:
        out["all_identical"] = all(v.get("identical") for v in out["configs"].values())
    out["n_compared"] = len(out["configs"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=[])
    ap.add_argument("--shard", default="", help="replay every config of a frozen shard")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample-states", type=int, default=None,
                    help="seed for configs with no frozen corpus (self-consistency only)")
    ap.add_argument("--cover-compaction", choices=("on", "off"), default=None,
                    help="pin raycast3d.COVER_COMPACTION, to diff the compacted composite against "
                         "the expression it replaced on the device the change targets")
    ap.add_argument("--dump-dir", type=Path, default=None)
    ap.add_argument("--compare-dumps", nargs=2, type=Path, default=None,
                    metavar=("BEFORE", "AFTER"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.compare_dumps:
        rep = compare_dumps(*args.compare_dumps)
        args.out.write_text(json.dumps(rep, indent=2))
        for cfg, v in rep["configs"].items():
            print(f"{cfg:30s} {v}", flush=True)
        print(f"ALL_IDENTICAL={rep['all_identical']} over {rep['n_compared']} config(s)"
              + (f"  -- {rep['error']}" if rep.get("error") else "")
              + f"  wrote {args.out}", flush=True)
        return

    if args.cover_compaction is not None:
        import stable_envs.render.raycast3d as R3
        R3.COVER_COMPACTION = args.cover_compaction == "on"
        print(f"COVER_COMPACTION={R3.COVER_COMPACTION} in {R3.__file__}", flush=True)

    frozen = frozen_oracles()
    cfgs = list(args.configs)
    if args.shard:
        p = STEP4 / "conformance" / f"{args.shard}.json"
        cfgs += sorted(json.loads(p.read_text())["configs"])
    rep = {"host": socket.gethostname(), "device": args.device,
           "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "code_root": os.environ.get("PYTHONPATH", str(ROOT)), "configs": {}}
    ok = fail = err = 0
    for cfg in cfgs:
        sh_or = frozen.get(cfg)
        try:
            r = replay_one(cfg, args.device, sh_or[1] if sh_or else None,
                           args.sample_states, args.dump_dir)
        except Exception as exc:                                            # noqa: BLE001
            import traceback
            r = {"config": cfg, "error": repr(exc)[:300], "tb": traceback.format_exc()[-1500:]}
        rep["configs"][cfg] = r
        if "error" in r:
            err += 1
            print(f"{cfg:30s} ERROR {r['error'][:120]}", flush=True)
        elif r["PASS"] is True:
            ok += 1
            print(f"{cfg:30s} PASS  {r['digest_ours'][:16]}", flush=True)
        elif r["PASS"] is False:
            fail += 1
            print(f"{cfg:30s} FAIL  got {r['digest_ours'][:16]} "
                  f"want {r['frozen']['digest_ours'][:16]}  match={r['match']}", flush=True)
        else:
            print(f"{cfg:30s} (no frozen oracle) {r['digest_ours'][:16]}", flush=True)
        args.out.write_text(json.dumps(rep, indent=2))
    rep["summary"] = {"pass": ok, "fail": fail, "error": err, "total": len(cfgs)}
    args.out.write_text(json.dumps(rep, indent=2))
    print(f"\nREPLAY pass={ok} fail={fail} error={err} of {len(cfgs)}  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
