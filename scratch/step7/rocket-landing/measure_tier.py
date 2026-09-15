"""M3/M4 on a GPU node: PFRocketLanding's visual fidelity, the corpus replay, and its throughput.

Writes examples/results/step7/rocket-landing/tier/<host>.json:
  * ``visual_fidelity``: ``check_visual_fidelity`` over ``--draws`` independent 64-state draws (the worst
    is what the class declares), plus ``check_visual_fidelity_under_variation``;
  * ``corpus_replay``: our render of the 200 frozen corpus states/styles against the frozen frames
    (mean_abs, within8, ssim, interior exact) -- the reference here is the RECORDED frame, no PyFlyt;
  * ``throughput``: f/s at batches 1/16/64/256 at 480 px, warm, best of two, CUDA-synchronised;
  * ``mandatory``: batch invariance, batch-size invariance, determinism, state dims (the suite's own).

The declared numbers in ``stable_envs/envs/rocket_landing.py`` are copied from here BY HAND -- a
threshold a script writes into the file it then tests is not a threshold.

Usage (GPU node, se7-rocket venv): python scratch/step7/rocket-landing/measure_tier.py --draws 4
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from importlib.metadata import version as pkg_version

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("MUJOCO_GL", "egl")
import stable_envs as se  # noqa: E402
from stable_envs.envs import rocket_landing as RL  # noqa: E402
from stable_envs.testing import conformance as C  # noqa: E402

REF = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "reference")
OUTDIR = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "tier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256, 1024, 4096])   # certification §6.2
    ap.add_argument("--skip-reference", action="store_true", help="corpus replay + throughput only (no PyFlyt)")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    out = os.path.join(OUTDIR, f"{platform.node().split('.')[0]}.json")
    R = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0), "python": sys.version.split()[0],
         "slurm_job": os.environ.get("SLURM_JOB_ID"), "threads": torch.get_num_threads(),
         "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
         "versions": {k: _v(k) for k in ("torch", "nvdiffrast", "PyFlyt", "pybullet", "stable-worldmodel", "numpy", "scipy")}}

    sha_file = os.path.join(ROOT, ".git_sha")
    sha = (open(sha_file).read().strip() if os.path.exists(sha_file)
           else os.popen(f"git -C {ROOT} rev-parse HEAD 2>/dev/null").read().strip() or "unknown")
    # profile_note.py's two provenance fields (its `baseline_commit` is named `harness_commit` here so it
    # cannot be misread as the $SCR/se7/baseline copy, 76dc511): the commit of the tree this script lives in, and the
    # stable_envs package ACTUALLY imported, resolved -- the only proof of which checkout was measured.
    R["provenance"] = {"sha": sha, "harness_commit": sha,   # the commit of THIS tree (never $SCR/se7/baseline = 76dc511)
                       "stable_envs_file": os.path.realpath(se.__file__),
                       "host": platform.node(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                       "threads": torch.get_num_threads(),
                       "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
                       "gpu": torch.cuda.get_device_name(0)}
    R["gates"] = {}                                                       # name -> {"pass": bool, "evidence": ...}
    gates_txt = os.path.join(OUTDIR, "gates.txt")

    def gate(name, ok, evidence):
        R["gates"][name] = {"pass": bool(ok), "evidence": evidence, **R["provenance"]}
        line = f"{'PASS' if ok else 'FAIL'}  {name}  {evidence}  [sha {R['provenance']['sha']} host {R['provenance']['host']} job {R['provenance']['slurm_job']}]"
        print("GATE", line, flush=True)
        with open(gates_txt, "a") as fh:
            fh.write(line + "\n")

    def checkpoint():
        """Written after EVERY stage, so a failure downstream never costs the numbers upstream."""
        with open(out, "w") as fh:
            json.dump(R, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    open(gates_txt, "w").close()
    if "PFRocketLanding" not in se.REGISTRY:
        gate("first_render", False, f"not registered: {RL.missing_backend()} / {se.envs.import_errors()}")
        checkpoint()
        raise SystemExit(1)
    env = se.make("PFRocketLanding", batch=16, device="cuda")
    R["describe"] = {k: v for k, v in env.describe().items() if k not in ("visual_fidelity",)}

    # ---- the job's FIRST result: one successful render, before any measurement
    from PIL import Image
    env.rng = np.random.default_rng(0)
    s1 = torch.as_tensor(env.sample_states(1), device="cuda")
    fr1 = env.render(s1).cpu().numpy()[0]
    Image.fromarray(fr1).save(os.path.join(OUTDIR, "first_render.png"))
    R["first_render"] = {"shape": list(fr1.shape), "dtype": str(fr1.dtype), "mean": float(fr1.mean()),
                         "non_white_px": int((fr1 != 255).any(-1).sum()),
                         "distinct_colours": int(len(np.unique(fr1.reshape(-1, 3), axis=0))),
                         "state": s1.cpu().numpy()[0].tolist(), "png": "tier/first_render.png"}
    gate("first_render", fr1.shape == (480, 480, 3) and 5000 < R["first_render"]["non_white_px"] < 200000,
         f"shape {fr1.shape} non-white px {R['first_render']['non_white_px']} distinct colours {R['first_render']['distinct_colours']}")
    checkpoint()

    # ---- coexistence: PyBullet DIRECT (+ the numba JIT) and an EGL context in the SAME process as the
    # CUDA raster context, then the same state rendered again must be byte-identical.
    if not a.skip_reference:
        coex = {}
        try:
            ref = RL.PFRocketLanding.reference_env(480)                   # PyBullet DIRECT, numba JIT (~57 s)
            coex["pybullet_direct"] = "ok"
            coex["reference_frame_shape"] = list(ref.unwrapped.render().shape)
        except Exception as e:                                             # pragma: no cover
            coex["pybullet_direct"] = f"FAILED {type(e).__name__}: {e}"
        try:
            import mujoco
            gl = mujoco.GLContext(64, 64)
            gl.make_current()
            coex["egl"] = "ok"
        except Exception as e:                                             # pragma: no cover
            coex["egl"] = f"FAILED {type(e).__name__}: {e}"
        fr2 = env.render(s1).cpu().numpy()[0]
        coex["render_after_identical"] = bool(np.array_equal(fr1, fr2))
        R["coexistence"] = coex
        gate("coexistence_direct_egl_cuda", coex.get("pybullet_direct") == "ok" and coex["render_after_identical"]
             and coex.get("egl", "").startswith("ok"), json.dumps(coex))
        checkpoint()

    # ---- corpus replay (no reference package needed)
    S = np.load(os.path.join(REF, "states.npz"))
    F = np.load(os.path.join(REF, "frames.npz"))["frames"]
    sty = RL.corpus_style_to_env(S["style"]).astype(np.float32)
    # ONE replay of our renderer -- render() casts states to float32, so obs30[:, :13] and state13 feed
    # bit-identical tensors and there is exactly one measurement here. It contains TWO terms: the
    # renderer's residual and the float32 quantisation of the state (the frozen frames were recorded from
    # float64 physics). The second term is measured on its own below as reference_render(state13) vs the
    # frozen frames (the same PyBullet renderer on the rounded state), and check_visual_fidelity, where
    # both sides see the same float32 state, is the renderer-only number.
    R["corpus_replay"] = {}
    st = S["state13"]
    ours = []
    for lo in range(0, len(st), 16):
        ours.append(env.render(torch.as_tensor(st[lo:lo + 16], dtype=torch.float32, device="cuda"),
                               style=sty[lo:lo + 16]).cpu().numpy())
    ours = np.concatenate(ours)
    rep = C.pixel_agreement(ours, F)
    rep["ssim"] = float(C.ssim(ours, F))
    d = np.abs(ours.astype(np.int16) - F.astype(np.int16))
    per = d.mean(axis=(1, 2, 3))
    rep["worst_frame_mean_abs"] = float(per.max())
    rep["worst_frame_index"] = int(per.argmax())
    rep["by_height"] = {}
    h = S["height"]
    for lo, hi in ((0, 12), (12, 50), (50, 260), (260, 500)):
        m = (h >= lo) & (h < hi)
        if m.any():
            rep["by_height"][f"{lo}-{hi}m"] = {"n": int(m.sum()), "mean_abs": float(d[m].mean()),
                                              "within8": float((d[m].max(-1) <= 8).mean())}
    rep["contains"] = "renderer residual + float32 state quantisation (see state_quantisation_term)"
    R["corpus_replay"]["ours_state13_vs_frozen"] = rep
    np.savez_compressed(os.path.join(OUTDIR, "corpus_replay_ours.npz"), frames=ours[:16], idx=np.arange(16))
    print("corpus replay ours(state13) vs frozen", json.dumps({k: v for k, v in rep.items() if k != "by_height"}), flush=True)
    gate("corpus_replay_ours_state13_vs_frozen", True, f"mean_abs {rep.get('mean_abs')} within8 {rep.get('within8_frac')} "
         f"ssim {rep['ssim']:.4f} interior_exact {rep.get('interior_exact_frac')} (reported; contains both terms)")
    checkpoint()

    # ---- mandatory checks
    R["mandatory"] = {}
    for name in ("check_batch_invariance", "check_batch_size_invariance", "check_determinism", "check_state_dims_matter"):
        fn = getattr(C, name, None)
        if fn is None:
            continue
        try:
            if name == "check_batch_size_invariance":       # signature (env, states, ...); windows 1/7/64/4096 (M5 ruling)
                R["mandatory"][name] = fn(env, env.sample_states(4096, rng=np.random.default_rng(11)), sizes=(1, 7, 64, 4096))
                torch.cuda.empty_cache()
            else:
                R["mandatory"][name] = fn(env)
        except Exception as e:                                                      # pragma: no cover
            R["mandatory"][name] = f"ERROR {type(e).__name__}: {e}"
        print(name, R["mandatory"][name], flush=True)
        r_ = R["mandatory"][name]
        if name == "check_state_dims_matter" and isinstance(r_, dict):
            vel = [n_ for n_ in RL.PFRocketLanding.state_names if "vel" in n_]
            dead = r_.get("dead_dims", [])
            gate("state_dims_matter_velocity_dims", not any(d in dead for d in vel),
                 f"velocity dims {vel} dead {[d for d in dead if d in vel]} all dead {dead}")
        else:
            gate(name, isinstance(r_, dict) and not str(r_).startswith("ERROR") and all(
                (v == 0 or v is True or v == "pass") for k, v in r_.items() if k in ("max_abs_diff", "differing", "ok", "status")),
                 str(r_)[:200])
    checkpoint()

    # ---- throughput, with cpu_frac = process CPU seconds / wall seconds of the timed loop (HOST_BOUND
    # at >= 1.5, certification.md section 6.7): this cell does per-row float64 host kinematics in a
    # Python chunk loop, exactly the shape that earns the qualifier, so it is measured, not assumed.
    thr, cpu_frac = {}, {}
    for B in a.batches:
        e = se.make("PFRocketLanding", batch=B, device="cuda")
        s = torch.as_tensor(e.sample_states(B), device="cuda")
        for _ in range(3):
            e.render(s)
        torch.cuda.synchronize()
        best, best_cf = 0.0, None
        for _ in range(2):
            reps = max(1, 512 // B)
            c0, t = time.process_time(), time.perf_counter()
            for _ in range(reps):
                e.render(s)
            torch.cuda.synchronize()
            wall, cpu = time.perf_counter() - t, time.process_time() - c0
            if reps * B / wall > best:
                best, best_cf = reps * B / wall, cpu / wall
        thr[str(B)] = round(best, 1)
        cpu_frac[str(B)] = round(best_cf, 3)
        print("throughput", B, thr[str(B)], "cpu_frac", cpu_frac[str(B)], flush=True)
    R["throughput_fps"] = thr
    R["cpu_frac"] = cpu_frac
    # BATCH_SCALES and NO_LARGE_BATCH_REGRESSION at the frozen definitions (certification.md §6.6)
    f1, f256, f4096 = thr.get("1"), thr.get("256"), thr.get("4096")
    R["batch_checks"] = {"BATCH_SCALES": bool(f4096 is not None and f1 and f4096 >= 20 * f1),
                         "NO_LARGE_BATCH_REGRESSION": bool(f4096 is not None and f256 and f4096 >= 0.9 * f256),
                         "fps_4096_over_fps_1": round(f4096 / f1, 2) if (f4096 and f1) else None,
                         "fps_4096_over_fps_256": round(f4096 / f256, 3) if (f4096 and f256) else None}
    print("batch checks", R["batch_checks"], flush=True)
    gate("BATCH_SCALES", R["batch_checks"]["BATCH_SCALES"], f"fps(4096)/fps(1) = {R['batch_checks']['fps_4096_over_fps_1']} (bar 20)")
    gate("NO_LARGE_BATCH_REGRESSION", R["batch_checks"]["NO_LARGE_BATCH_REGRESSION"],
         f"fps(4096)/fps(256) = {R['batch_checks']['fps_4096_over_fps_256']} (bar 0.9)")
    checkpoint()

    # ---- the state-quantisation term alone: PyBullet's own renderer at the float32 state vs the frozen
    # frames it recorded from the float64 state. CPU-only; also runnable locally (corpus_replay_state32.py).
    if not a.skip_reference:
        ref = RL.PFRocketLanding.reference_env(480)
        sq = state_quantisation_term(ref, S, F, RL.corpus_style_to_env(S["style"]))
        R["corpus_replay"]["reference_state13_vs_frozen"] = sq
        gate("corpus_replay_state_quantisation_term", True,
             f"reference_render(state13) vs frozen: exact {sq['exact']}/{sq['n']} mean_abs {sq['mean_abs']:.3f} "
             f"max diff px {sq['diff_px_max']} (the term the ours-vs-frozen row contains besides the renderer's)")
        checkpoint()

    # ---- the two comparators ON THIS NODE, same job (certification §6.3): the official renderer's
    # frames/s (TinyRenderer through the swm env, warm; the numba JIT was paid at the reference reset)
    # and one core of JPEG-q90 decode of the env's OWN frames, from memory (the Phase-0 protocol).
    if not a.skip_reference:
        import io
        from PIL import Image
        ref = RL.PFRocketLanding.reference_env(480)
        u = ref.unwrapped
        for _ in range(5):
            u.render()
        t = time.perf_counter()
        for _ in range(60):
            u.render()
        official = 60.0 / (time.perf_counter() - t)
        bufs = []
        for fr in F[:16]:
            b = io.BytesIO()
            Image.fromarray(fr).save(b, format="JPEG", quality=90)
            bufs.append(b.getvalue())
        for b in bufs[:4]:
            np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
        # ONE core: Pillow's JPEG decoder is single-threaded; torch/OMP are pinned for the duration anyway
        nt = torch.get_num_threads()
        torch.set_num_threads(1)
        c0, t = time.process_time(), time.perf_counter()
        for i in range(200):
            np.asarray(Image.open(io.BytesIO(bufs[i % 16])).convert("RGB"))
        wall = time.perf_counter() - t
        decode, decode_cpu_frac = 200.0 / wall, (time.process_time() - c0) / wall
        torch.set_num_threads(nt)
        # best CERTIFIED batch: never B = 1 (section 6.2)
        cert = [b_ for b_ in a.batches if b_ != 1]
        if not cert:
            R["tier"] = {"error": "no certifiable batch: --batches holds only B = 1 (section 6.2 never certifies on it)"}
            gate("tier_frozen_definitions_same_node", False, R["tier"]["error"])
            checkpoint()
            return
        best_b = max(cert, key=lambda b_: thr.get(str(b_), 0.0))
        best = thr[str(best_b)]
        cf = cpu_frac.get(str(best_b))
        Rr, cores = best / official, best / decode
        tier = ("GPU_REGRESSION" if Rr < 1 else "GPU_PARITY" if Rr < 2 else "GPU_FASTER" if Rr < 4
                else ("GPU_EXCEPTIONAL" if (Rr >= 40 and cores >= 10) else "GPU_FAST" if cores >= 1
                      else "GPU_FASTER"))
        quals = []
        if Rr >= 4 and cores < 1:
            quals.append("BELOW_G4")
        if not R["batch_checks"]["BATCH_SCALES"]:
            tier = "GPU_PARITY" if Rr >= 1 else tier                          # section 6.7: capped
            quals.append("BATCH_SCALES_FAIL")
        if not R["batch_checks"]["NO_LARGE_BATCH_REGRESSION"]:
            quals.append("LARGE_BATCH_REGRESSION")
        if cf is not None and cf >= 1.5:
            quals.append("HOST_BOUND")
        R["tier"] = {"class": "S (3-D scene)", "official_fps_same_node": round(official, 2),
                     "one_core_jpeg_q90_decode_fps_same_node": round(decode, 1), "decode_cpu_frac": round(decode_cpu_frac, 3),
                     "jpeg_bytes_mean": int(np.mean([len(b) for b in bufs])),
                     "best_certified_batch": best_b, "fps_at_best": best, "cpu_frac_at_best": cf,
                     "R": round(Rr, 2), "cores": round(cores, 3), "tier": tier, "qualifiers": quals,
                     "threads": R["threads"], "load_average": R["load_average"], "slurm_job": R["slurm_job"]}
        print("tier", R["tier"], flush=True)
        gate("tier_frozen_definitions_same_node", True,
             f"{tier} {quals} R={Rr:.2f} (ours {best} f/s at B={best_b} vs official {official:.2f} f/s same node) "
             f"cores={cores:.3f} (vs {decode:.1f} f/s one-core decode) cpu_frac={cf}")
        checkpoint()

    # ---- visual fidelity against the live reference (PyFlyt + TinyRenderer)
    if not a.skip_reference:
        vf, failures = [], []
        R["visual_fidelity"] = {"draws": vf, "failures": failures}
        for k in range(a.draws):
            env.rng = np.random.default_rng(1000 + k)
            try:
                r = fidelity_draw(env, a.n)
                r["draw"] = k
                vf.append(r)
                print("visual_fidelity draw", k, json.dumps(r), flush=True)
            except Exception as e:                                                  # recorded, never fatal
                failures.append({"draw": k, "error": f"{type(e).__name__}: {e}"})
                print("visual_fidelity draw", k, "FAILED", failures[-1]["error"], flush=True)
            checkpoint()
        if vf:
            # worst of the draws over the REACHABLE rows (what is declared) and over all rows (what the
            # shared check_visual_fidelity computes, flagged rows carrying the synced picture)
            worst = {"mean_abs": max(r["reachable_only"]["mean_abs"] for r in vf),
                     "within8_frac": min(r["reachable_only"]["within8_frac"] for r in vf),
                     "ssim": min(r["reachable_only"]["ssim"] for r in vf),
                     "unreachable_rows_total": int(sum(r["unreachable_rows"] for r in vf)),
                     "all_rows": {"mean_abs": max(r["mean_abs"] for r in vf), "within8_frac": min(r["within8_frac"] for r in vf),
                                  "ssim": min(r["ssim"] for r in vf)}}
            R["visual_fidelity"]["worst"] = worst
            R["visual_fidelity"]["spread"] = {k_: [min(r["reachable_only"][k_] for r in vf), max(r["reachable_only"][k_] for r in vf)]
                                              for k_ in ("mean_abs", "within8_frac", "ssim")}
        gate("visual_fidelity_4x64", len(vf) == a.draws and not failures,
             f"draws {len(vf)}/{a.draws} rejections {len(failures)} worst {R['visual_fidelity'].get('worst')} "
             f"spread {R['visual_fidelity'].get('spread')}")
        try:
            R["variation_fidelity"] = C.check_visual_fidelity_under_variation(env, n=a.n, seed=7)
            R["variation_fidelity"]["unreachable_rows"] = int(np.asarray(RL.PFRocketLanding.last_reference_unreachable, bool).sum())
        except Exception as e:                                                      # pragma: no cover
            R["variation_fidelity"] = f"ERROR {type(e).__name__}: {e}"
        print("under variation:", R["variation_fidelity"], flush=True)
        gate("visual_fidelity_under_variation", isinstance(R["variation_fidelity"], dict) and "skipped" not in R["variation_fidelity"],
             str(R["variation_fidelity"])[:300])
        checkpoint()
    R["seconds"] = round(time.time() - t0, 1)
    checkpoint()
    print("wrote", out)


def fidelity_draw(env, n: int) -> dict:
    """``stable_envs.testing.conformance.check_visual_fidelity`` line for line (same rng consumption, same
    reference, same metrics), plus the cell's per-row ``last_reference_unreachable`` flag: the numbers over
    ALL rows are exactly what the shared check computes (flagged rows carry swm's synced picture), the
    numbers over the REACHABLE rows are what the declaration carries, with the flagged share recorded."""
    cls = type(env)
    ref = C.reference_env(cls, env.size)
    states = env.sample_states(int(n))
    # return_flags: the supported path. The class attribute is last-call-wins mutable state and the
    # cell's own docstring says so; the only caller that still needs it is the shared
    # check_visual_fidelity_under_variation, which cannot ask for the flags.
    official, flags = cls.reference_render(ref, states, return_flags=True)
    ours = env.render(states).cpu().numpy()
    r = C.pixel_agreement(ours, official)
    r["ssim"] = C.ssim(ours, official)
    r["n_states"] = int(n)
    r["unreachable_rows"] = int(flags.sum())
    keep = ~flags
    if flags.any() and keep.any():
        rr = C.pixel_agreement(ours[keep], official[keep])
        rr["ssim"] = C.ssim(ours[keep], official[keep])
        rr["n_states"] = int(keep.sum())
        r["reachable_only"] = rr
    else:
        r["reachable_only"] = {k: r[k] for k in ("mean_abs", "exact_frac", "within8_frac", "interior_exact_frac", "ssim", "n_states")}
    return r


def state_quantisation_term(ref, S, F, sty_env) -> dict:
    """``reference_render(state13)`` vs the frozen frames: what float32 rounding of the state costs, alone."""
    st = S["state13"].astype(np.float64)
    diffs, exact, rej = [], 0, 0
    for i in range(len(st)):
        try:
            fr = RL._reference_frame(ref, st[i], sty_env[i])
        except AssertionError:
            rej += 1
            continue
        d = np.abs(fr.astype(np.int16) - F[i].astype(np.int16))
        px = int((d.max(-1) > 0).sum())
        diffs.append((px, float(d.mean())))
        exact += px == 0
    return {"n": len(st), "rendered": len(diffs), "rejections": rej, "exact": exact,
            "diff_px_max": max(p for p, _ in diffs) if diffs else None,
            "diff_px_mean": float(np.mean([p for p, _ in diffs])) if diffs else None,
            "mean_abs": float(np.mean([m for _, m in diffs])) if diffs else None}


def _v(k):
    try:
        return pkg_version(k)
    except Exception:
        return None


if __name__ == "__main__":
    main()
