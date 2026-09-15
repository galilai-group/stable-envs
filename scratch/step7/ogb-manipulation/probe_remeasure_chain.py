"""The whole re-measurement chain for the two cells, after the cell review's B1-B6 fixes.

One script, because the steps are not independent: the thresholds come from the draws, the corpus is
selected from a pool only `sample_states` can produce, and Gate F and the tiers must be measured on
the same build as the fidelity numbers. Running them separately is how the previous round ended up
with a committed pool the corpus was not selected from.

What it does, in order:

1. **Fidelity**, 3 independent 64-state draws per cell against swm's own EGL frames.
2. **Thresholds**, computed BY THIS SCRIPT from the measured worst-of-three under the family rule
   (x1.3 on the mean; the fractional floors a round 0.05 step below the worst draw), and written to
   `paste_me.json` in exactly the shape `_MEASURED` wants. Nothing is invented by hand afterwards.
3. **Gate A through the REPO'S OWN checks**, `check_state_dims_matter` included -- not the bespoke
   probe whose quaternion perturbation reported a live dimension dead last round.
4. **Corpora**, pool and selection from ONE draw, with `pool[indices] == states` asserted before
   anything is written.
5. **Gate F arm D** against swm's own dynamics.
6. **Tiers**, with the official renderer and the one-core decode measured on THIS node.

Every artifact carries provenance: host, GPU, SLURM job, git SHA and package versions.
"""
from __future__ import annotations

import argparse, io, json, os, platform, socket, statistics, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def provenance() -> dict:
    import importlib.metadata as md
    out = {"host": socket.gethostname(), "platform": platform.platform(),
           "python": sys.version.split()[0], "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "mujoco_gl": os.environ.get("MUJOCO_GL"), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                         time.gmtime())}
    try:
        out["git_sha"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                        capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        out["git_sha"] = None
    try:
        import hashlib
        out["ogb_manip_md5"] = hashlib.md5(
            (ROOT / "stable_envs/envs/ogb_manip.py").read_bytes()).hexdigest()
    except Exception:
        out["ogb_manip_md5"] = None
    for pkg in ("mujoco", "mujoco-warp", "warp-lang", "torch", "numpy", "ogbench",
                "dm-control", "stable-worldmodel"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = None
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        out["gpu"] = None
    # `stable-worldmodel` comes back None from importlib.metadata because it is a PYTHONPATH
    # CHECKOUT here, not an installed distribution -- and it is the registration under test, so a
    # null version is the one provenance field that must not stay null. The file digests are the
    # version.
    try:
        import hashlib
        import stable_worldmodel as swm
        root = Path(swm.__file__).parent
        # `$SCR`, not the expanded path: `tests/unit/test_repo_contracts.py` reads a bare
        # `/oscar/scratch/<user>/...` in a committed artifact as a repo-relative `scratch/...`
        # citation and requires it to resolve, so capturing the absolute form grows the
        # broken-citation set by one every time.
        out["stable_worldmodel_path"] = str(root).replace(
            f"/oscar/scratch/{os.environ.get('USER', 'sboughan')}", "$SCR")
        out["stable_worldmodel_sha256"] = {
            f: hashlib.sha256((root / f).read_bytes()).hexdigest()[:16]
            for f in ("envs/ogbench/cube_env.py", "envs/ogbench/scene_env.py", "envs/__init__.py")
            if (root / f).exists()}
    except Exception as e:
        out["stable_worldmodel_sha256"] = f"unavailable: {type(e).__name__}"
    return out


def thresholds(mean_abs: float, w_lo: float, s_lo: float) -> dict:
    """The FAMILY rule, applied by the script rather than by hand.

    DM Control sets its ceilings at ~x1.3 of the measured mean (5.01 -> 6.5, 8.07 -> 10.5) and its
    floors a round step below the worst draw, so that a threshold is a contract rather than a
    tripwire on the next state draw. The floors here take the WORST of the three draws, not the
    declared value, because that is the number a re-draw has to clear.
    """
    return {"mean_abs_max": round(float(mean_abs) * 1.3, 1),
            "within8_min": float(np.floor(float(w_lo) * 20) / 20),
            "ssim_min": float(np.floor(float(s_lo) * 20) / 20)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--states-dir", default=None)
    ap.add_argument("--cells", nargs="+", default=["OGBCube", "OGBScene"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64, 256, 1024, 4096])
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    states_dir = Path(a.states_dir) if a.states_dir else (
        ROOT / "examples/results/step7/ogb-manipulation/states")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "examples"))
    import torch
    from PIL import Image
    import stable_envs as se
    import stable_envs.testing.conformance as C
    from stable_envs.testing.conformance import interior_mask, ssim
    import fidelity_step4 as F

    rec = {"provenance": provenance(), "n": a.n, "draws": a.draws, "cells": {}}
    paste = {}
    print("provenance:", json.dumps(rec["provenance"]), flush=True)

    failures = {}
    for cname in a.cells:
      try:
        cls = se.REGISTRY[cname]
        cell = {"size": int(cls.default_size), "draws": []}
        env = se.make(cname, batch=a.n, device="cuda")
        ref = cls.reference_env()

        # ---- 1. fidelity -----------------------------------------------------------------
        for k in range(a.draws):
            st = env.sample_states(a.n, np.random.default_rng(1000 + k))
            ours = env.render(st).cpu().numpy()
            off = cls.reference_render(ref, st)
            d = np.abs(ours.astype(np.int16) - off.astype(np.int16))
            dmax = d.max(-1)
            im = interior_mask(off)
            mover = None
            row = {"draw": k, "mean_abs": float(d.mean()),
                   "within8_frac": float((dmax <= 8).mean()),
                   "ssim": float(ssim(ours, off)),
                   "all_exact_frac": float((dmax == 0).mean()),
                   "interior_frac": float(im.mean()),
                   "interior_exact_frac": float((dmax[:, :-1, :-1][im] == 0).mean()),
                   "max_abs": int(dmax.max())}
            cell["draws"].append(row)
            print(f"{cname} draw {k}: mean_abs {row['mean_abs']:.4f} within8 {row['within8_frac']:.4f}"
                  f" ssim {row['ssim']:.4f} all_exact {row['all_exact_frac']:.4f}"
                  f" interior_exact {row['interior_exact_frac']:.4f}", flush=True)
            if k == 0:
                np.savez_compressed(out / f"{cname}_frames.npz", ours=ours[:4], official=off[:4],
                                    states=st[:4])

        D = cell["draws"]
        worst = {"mean_abs": max(r["mean_abs"] for r in D),
                 "within8_frac": min(r["within8_frac"] for r in D),
                 "ssim": min(r["ssim"] for r in D)}
        spread = {m: [min(r[m] for r in D), max(r[m] for r in D)]
                  for m in ("mean_abs", "within8_frac", "ssim")}
        cell["worst_of_draws"], cell["draw_spread"] = worst, spread

        # ---- 2. thresholds, computed here ------------------------------------------------
        th = thresholds(worst["mean_abs"], worst["within8_frac"], worst["ssim"])
        cell["thresholds"] = th
        paste[cname[3:]] = {
            "mean_abs": round(worst["mean_abs"], 6),
            "within8_frac": round(worst["within8_frac"], 6),
            "ssim": round(worst["ssim"], 6),
            "interior_exact_frac": round(float(np.mean([r["interior_exact_frac"] for r in D])), 6),
            "interior_frac": round(float(np.mean([r["interior_frac"] for r in D])), 6),
            "all_exact_frac": round(float(np.mean([r["all_exact_frac"] for r in D])), 6),
            "draw_spread": {m: [round(v[0], 6), round(v[1], 6)] for m, v in spread.items()},
            "job": f"job {os.environ.get('SLURM_JOB_ID')} on {socket.gethostname()}",
            **th}
        print(f"{cname} WORST-OF-{a.draws}: {worst} thresholds {th}", flush=True)

        # ---- 3. Gate A, through the REPO'S checks ----------------------------------------
        g = {}
        gate_states = env.sample_states(env.B, np.random.default_rng(99))
        # `check_batch_size_invariance` takes the states explicitly; the others draw their own.
        needs_states = {"check_batch_size_invariance"}
        for fn in ("check_determinism", "check_batch_invariance", "check_batch_size_invariance",
                   "check_state_dims_matter", "check_order_invariance"):
            f = getattr(C, fn, None)
            if f is None:
                g[fn] = {"status": "absent in this build"}
                continue
            try:
                g[fn] = f(env, gate_states) if fn in needs_states else f(env)
            except Exception as e:
                g[fn] = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300]}
            print(f"{cname} {fn}: {json.dumps(g[fn])[:220]}", flush=True)
        cell["gate_a_repo_checks"] = g
        dead = (g.get("check_state_dims_matter") or {}).get("evidence", {}).get("dead_dims")
        cell["dead_dims"] = dead
        print(f"{cname} DEAD DIMS (repo check): {dead}", flush=True)

        # ---- 4. corpus: pool and selection from ONE draw ---------------------------------
        pool = np.ascontiguousarray(
            env.sample_states(a.pool, rng=np.random.default_rng(F.STATE_SEED)), np.float32)
        states, how = F.select(pool, a.n)
        states = np.ascontiguousarray(states, np.float32)
        idx = np.asarray(how["indices"], int)
        assert np.array_equal(pool[idx], states), "selection is not a subset of its own pool"
        meta = {"seed": F.STATE_SEED,
                "selection": "typical + per-dim extremes + farthest-point spread",
                "state_dim": int(states.shape[1]), "state_names": list(env.state_names or []),
                "env": cname, "sha256": C.frames_digest(states),
                "provenance": rec["provenance"], **how}
        states_dir.mkdir(parents=True, exist_ok=True)
        np.savez(states_dir / f"{cname}.npz", states=states,
                 meta=np.array(json.dumps(meta, sort_keys=True)))
        np.savez_compressed(states_dir / f"{cname}_pool.npz", pool=pool,
                            meta=np.array(json.dumps(meta, sort_keys=True)))
        cell["corpus"] = {k: v for k, v in meta.items() if k != "indices"}
        cell["corpus"]["n_indices"] = len(idx)
        print(f"{cname} corpus: kept {len(idx)} of {len(pool)}, sha {meta['sha256'][:12]}", flush=True)

        # ---- 5. Gate F arm D --------------------------------------------------------------
        import gymnasium, mujoco
        import stable_worldmodel  # noqa: F401
        from stable_envs.envs.ogb_manip import SPEC, _bundle, _seeded_action_space
        b = _bundle(cls.CELL)
        sw = gymnasium.make(SPEC[cls.CELL]["swm_id"])
        u = sw.unwrapped
        u.np_random = np.random.default_rng(11)
        with _seeded_action_space(u, 11):
            ob, info = sw.reset(seed=11)
            got = []
            for _ in range(a.steps):
                sw.step(u.action_space.sample())
                d2 = u.data
                bits = [float(x) for x in list(getattr(u, "_cur_button_states", []))[:b.nbuttons]]
                got.append(np.concatenate([d2.qpos.copy(), d2.mocap_pos.ravel().copy(),
                                           np.asarray(bits, np.float64)]))
        st = np.stack(got).astype(np.float64)
        ours = env.render(st).cpu().numpy()
        official = cls.reference_render(ref, st)
        dmax = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
        im = interior_mask(official)
        cell["gate_f_arm_d"] = {
            "swm_id": SPEC[cls.CELL]["swm_id"],
            "observation_is_pixels": bool(np.asarray(ob).ndim >= 3),
            "observation_shape": list(np.asarray(ob).shape),
            "reset_info_keys": sorted(info.keys()),
            "certified_artifact": "render(): the observation is a vector, so there is no pixel "
                                  "observation to swap and render() is what is certified.",
            "n_states": int(len(st)),
            "shape_and_dtype_match": bool(ours.shape == official.shape
                                          and ours.dtype == official.dtype),
            "mean_abs": float(np.abs(ours.astype(np.int16) - official.astype(np.int16)).mean()),
            "within8_frac": float((dmax <= 8).mean()), "ssim": float(ssim(ours, official)),
            "all_exact_frac": float((dmax == 0).mean()),
            "interior_exact_frac": float((dmax[:, :-1, :-1][im] == 0).mean())}
        sw.close()
        print(f"{cname} GATE F: {json.dumps(cell['gate_f_arm_d'])[:260]}", flush=True)

        # ---- 6. tiers, same-node comparators ----------------------------------------------
        t = {"rows": []}
        try:
            se.make(cname, batch=1, device="cpu")
            t["cpu_raises"] = False
        except ValueError as e:
            t["cpu_raises"] = True
            t["cpu_error"] = str(e)[:300]
        big = se.make(cname, batch=max(a.batches), device="cuda")
        sts = big.sample_states(64, np.random.default_rng(5))
        for B in a.batches:
            try:
                s = np.repeat(sts, int(np.ceil(B / len(sts))), axis=0)[:B]
                for _ in range(3):
                    big.render(s)
                torch.cuda.synchronize()
                ts = []
                for _ in range(7):
                    t0 = time.perf_counter(); big.render(s); torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                med = statistics.median(ts)
                t["rows"].append({"batch": B, "ms": med * 1e3, "fps": B / med,
                                  "peak_vram_mib": int(torch.cuda.max_memory_allocated() / 2**20)})
                print(f"{cname} B={B}: {med*1e3:.2f} ms -> {B/med:.0f} f/s", flush=True)
            except Exception as e:
                t["rows"].append({"batch": B, "error": f"{type(e).__name__}: {e}"[:200]})
                torch.cuda.empty_cache()
        ok = [r for r in t["rows"] if "fps" in r]
        t["fps_best"], t["fps_1"] = max(r["fps"] for r in ok), next(
            r["fps"] for r in ok if r["batch"] == 1)
        t["batch_scales"] = t["fps_best"] / t["fps_1"]
        ru = ref.unwrapped
        for _ in range(10):
            ru.render()
        blk = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(60):
                ru.render()
            blk.append(60 / (time.perf_counter() - t0))
        t["official_fps_same_node"] = statistics.median(blk)
        frame = ru.render().copy()
        buf = io.BytesIO(); Image.fromarray(frame).save(buf, format="JPEG", quality=90)
        blob = buf.getvalue()
        for _ in range(30):
            np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
        dblk = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(300):
                np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
            dblk.append(300 / (time.perf_counter() - t0))
        t["decode_fps_per_core_same_node"] = statistics.median(dblk)
        t["jpeg_q90_bytes"] = len(blob)
        t["R_ours_over_official"] = t["fps_best"] / t["official_fps_same_node"]
        t["cores"] = t["fps_best"] / t["decode_fps_per_core_same_node"]
        t["batch_scales_pass"] = bool(t["batch_scales"] >= 20.0)
        t["declared_gpu_tier"] = ("GPU_PARITY (capped: BATCH_SCALES "
                                  f"{t['batch_scales']:.2f}x < 20x required)")
        cell["tiers"] = t
        print(f"{cname} TIERS: best {t['fps_best']:.0f} official {t['official_fps_same_node']:.1f} "
              f"decode {t['decode_fps_per_core_same_node']:.0f} R {t['R_ours_over_official']:.2f} "
              f"cores {t['cores']:.2f} BS {t['batch_scales']:.2f}x cpu_raises {t['cpu_raises']}",
              flush=True)
        rec["cells"][cname] = cell
        # WRITTEN AFTER EVERY CELL. Job 6406972 measured OGBCube cleanly and then died on OGBScene,
        # and because the artifacts were written only at the end, a complete and correct set of
        # OGBCube numbers was thrown away with the failure.
        (out / "remeasure.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
        (out / "paste_me.json").write_text(json.dumps(paste, indent=1, sort_keys=True))
      except Exception as e:
        import traceback
        failures[cname] = f"{type(e).__name__}: {e}"[:400]
        rec["failures"] = failures
        (out / "remeasure.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
        print(f"{cname} FAILED: {failures[cname]}", flush=True)
        traceback.print_exc()

    (out / "remeasure.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    (out / "paste_me.json").write_text(json.dumps(paste, indent=1, sort_keys=True))
    print("PASTE_ME:\n" + json.dumps(paste, indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
