"""Produce the sealed render corrector for one cell: fit it, measure it, and refuse to seal it.

    python examples/rendernet/train_corrector.py RoboCasaCounterToSink

This is the documented way to get the weights ``stable_envs/render/rendernet.py`` looks for, and the
ONLY supported one. The repository ships code, not weights — the same rule that makes it ship no
compiled RoboCasa kitchen and no MiniGrid atlas — so a corrector is built on the machine that wants it
and cached under ``~/.cache/stable-envs/rendernet/``. On one H200 the whole run is about 40 minutes,
most of it the official renderer producing targets.

FIVE STAGES, AND THE LAST ONE CAN SAY NO
----------------------------------------

1. **Split.** Disjoint at the level of the RECORDED STATE, not of the draw. RoboCasa's pool is 200
   recorded states doubled into 400 rows by yaw randomisation, so splitting the rows would put the
   same arm pose on both sides and the held-out number would measure a lookup table. Recorded states
   0-149 train, 150-199 held out.
2. **Pairs.** ``(cheap render at 112 px with texture LOD 2, official 256 px frame)``, both at a
   per-sample appearance DRAWN FROM THE ENV'S OWN DECLARED VARIATION SPACE, with a quarter of the rows
   pinned at the default. That is not an augmentation preference, it is the difference between a
   corrector that works and one that is harmful: fitted at one camera pose, a corrector scores 3.25
   at that pose and 9.06 under the env's own ±8° of pan — worse than not correcting at all, and worse
   than the render it replaces. Both sides of the pair get the same drawn appearance, which is why
   ``reference_render`` takes a style.
3. **Fit.** L1 against the official frame, which is the quantity the tier declares, on the un-rounded
   output. The head is zero-initialised, so training starts exactly at the bilinear baseline and every
   number is a departure from a measured point.
4. **Measure.** The corrected path and the plain class, on the SAME draws, in the exact protocol a
   ``visual_fidelity`` entry is asserted in: 64 states x 4 independent draws at 256 px, worst of the
   four — and then again under the declared variation space, which is where a corrector's training
   distribution shows up and where a default-appearance measurement would have said nothing.
5. **Gate, then seal.** The corrector must beat the plain class on mean |diff|, within-8 AND SSIM, in
   BOTH regimes. Anything else does not get written. This is the mechanical form of the measured fact
   that a corrector applied where it does not belong is worse than doing nothing — and the reason it
   is a gate rather than a warning is that the failure is invisible in the picture and invisible in
   the pixel mean; only SSIM and within-8 catch it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch
import torch.nn.functional as F

import stable_envs as se
from stable_envs.render import rendernet as RN
from stable_envs.testing import conformance as C

#: The recommended configuration per family, measured in ``scratch/altrender/rendernet/README.md``.
#: A RoboCasa room camera gives away resolution almost for free once a corrector exists (256 -> 112 px
#: costs 0.08/255 held out) and texture LOD 2 buys the throughput that makes the whole thing worth
#: doing; HumanoidBench's floor is already a single dominant texel, so its LOD lever is inert.
DEFAULTS = {
    "robocasa": {"net": "unet_112t128_w48_d2_u2", "tex_lod": 2},
    "humanoidbench": {"net": "espcn_112t128_w96_d6_u2", "tex_lod": 1},
}

#: Fraction of training rows pinned at the DEFAULT appearance. A declaration is measured there, so a
#: corrector that never saw it would be scored on a picture it was never trained for.
FRAC_DEFAULT = 0.25


# ------------------------------------------------------------------------------------- the split

def split_states(cls, split: str, n: int, seed: int, p_interp: float = 0.3) -> np.ndarray:
    """``(n, state_dim)`` from the declared half of this cell's state pool.

    Family-specific because the pool is: there is no generic "half of the states" that is disjoint in
    the way that matters. A family whose split is not written here is refused rather than given
    ``sample_states`` twice, which would draw from one distribution and call it two.
    """
    fam = cls.corrector_cell_key()["family"]
    if fam != "robocasa":
        raise SystemExit(
            f"no disjoint state split is defined for the {fam!r} family. Two draws from "
            f"sample_states are not a split -- they are the same distribution twice, and a corrector "
            f"measured that way reports how well it memorised the pool. Add the split here, at the "
            f"level of the underlying recorded state, before training this family.")
    import mujoco

    import stable_envs.envs.robocasa as RC
    s = RC._scene(cls.KITCHEN)
    live = RC._live_dims(cls.KITCHEN, cls.CAMERA, 256)
    pool = RC._yawed_pool(cls.KITCHEN)                    # (400, nq); rows 2i, 2i+1 <- recorded i
    n_rec = len(pool) // 2
    n_train_rec = int(round(0.75 * n_rec))
    rng = np.random.default_rng(seed)
    free = [int(s.model.jnt_qposadr[j]) + 3 for j in range(s.model.njnt)
            if int(s.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
    rec = (np.arange(n_train_rec) if split == "train" else np.arange(n_train_rec, n_rec))
    base = pool[np.concatenate([2 * rec, 2 * rec + 1])]
    q = base[rng.integers(0, len(base), n)].copy()
    if split == "train" and p_interp > 0:
        # Interpolating two TRAINING rows reaches arm poses between the recorded ones without leaving
        # their convex hull. Quaternions do not lerp, so a free joint keeps an endpoint's rotation and
        # the yaw redraw below replaces it anyway.
        k = int(p_interp * n)
        other = base[rng.integers(0, len(base), k)]
        t = rng.uniform(0.0, 1.0, (k, 1))
        mix = q[:k] * (1 - t) + other * t
        for qa in free:
            mix[:, qa:qa + 4] = q[:k, qa:qa + 4]
        q[:k] = mix
    for qa in free:                                        # the same generative process the env uses
        yaw = rng.uniform(-np.pi, np.pi, len(q))
        q[:, qa:qa + 4] = np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw),
                                    np.sin(yaw / 2)], 1)
    return q[:, live].astype(np.float32)


def draw_style(env, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = env.sample_style(n, rng=rng)
    v[rng.random(n) < FRAC_DEFAULT] = env.default_style()
    return v.astype(np.float32)


# -------------------------------------------------------------------------------------- the pairs

def make_pairs(cls, cheap, states, style, render_size, chunk=256, label=""):
    """``(cheap uint8 (n,r,r,3), official uint8 (n,256,256,3))`` for these states and appearances."""
    ref = C.reference_env(cls, 256)
    hr = np.empty((len(states), 256, 256, 3), np.uint8)
    t0 = time.perf_counter()
    for lo in range(0, len(states), chunk):
        hi = min(lo + chunk, len(states))
        hr[lo:hi] = cls.reference_render(ref, states[lo:hi], style=style[lo:hi])
        print(f"  [{label}] official {hi}/{len(states)}  "
              f"{hi / (time.perf_counter() - t0):.1f} f/s", flush=True)
    lr = np.empty((len(states), render_size, render_size, 3), np.uint8)
    t0 = time.perf_counter()
    for lo in range(0, len(states), chunk):
        hi = min(lo + chunk, len(states))
        lr[lo:hi] = cheap.render(states[lo:hi], size=render_size,
                                 style=style[lo:hi]).cpu().numpy()
    print(f"  [{label}] cheap {len(lr)} in {time.perf_counter() - t0:.1f}s", flush=True)
    return lr, hr


# ---------------------------------------------------------------------------------------- the fit

def quick_metrics(pred_u8: torch.Tensor, hr_u8: torch.Tensor) -> dict:
    """Same reduction as ``conformance.pixel_agreement``: max over channels, then mean over pixels."""
    d = (pred_u8.to(torch.int16) - hr_u8.to(torch.int16)).abs().amax(-1).float()
    return {"mean_abs": float(d.mean()), "within8_frac": float((d <= 8).float().mean()),
            "exact_frac": float((d == 0).float().mean())}


@torch.no_grad()
def predict(model, lr: torch.Tensor, batch: int = 128) -> torch.Tensor:
    return torch.cat([model(lr[i:i + batch]) for i in range(0, len(lr), batch)])


def fit(model, tr_lr, tr_hr, ho_lr, ho_hr, epochs: int, batch: int, lr_max: float, clip: float,
        seed: int) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max)
    steps = max(epochs * (len(tr_lr) // batch), 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr_max, total_steps=steps, pct_start=0.15)
    g = torch.Generator(device="cuda").manual_seed(seed)
    model.eval()
    base = quick_metrics(predict(model, ho_lr), ho_hr)
    print("epoch  -1 (bilinear baseline, the zero-initialised head) "
          + json.dumps({k: round(v, 4) for k, v in base.items()}), flush=True)
    hist, best, best_sd = [], 1e9, None
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr_lr), device="cuda", generator=g)
        tot, nb = 0.0, 0
        for i in range(0, len(perm) - batch + 1, batch):
            idx = perm[i:i + batch]
            x = tr_lr[idx].permute(0, 3, 1, 2).float().mul_(1 / 255.0)
            x = x.contiguous(memory_format=torch.channels_last)
            y = tr_hr[idx].permute(0, 3, 1, 2).float().mul_(1 / 255.0)
            y = y.contiguous(memory_format=torch.channels_last)
            # bf16 autocast with NO loss scaler: bf16 carries fp32's exponent range, so there is
            # nothing to scale, and a scaler that silently skips steps would hide the divergence the
            # gradient clip is here to catch -- one spike is what killed the first sweep.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.l1_loss(model.forward_float(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        model.eval()
        m = quick_metrics(predict(model, ho_lr), ho_hr)
        m.update(epoch=ep, train_l1_255=255.0 * tot / max(nb, 1),
                 seconds=time.perf_counter() - t0)
        hist.append(m)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d} " + json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
        if m["mean_abs"] < best:
            best, best_sd = m["mean_abs"], {k: v.detach().clone()
                                            for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd)
    print(f"best held-out mean |diff| {best:.4f}/255 against the bilinear baseline's "
          f"{base['mean_abs']:.4f}", flush=True)
    return hist


# ------------------------------------------------------------------------------- the measurement

def worst(draws: list[dict]) -> dict:
    """Worst of the draws: the largest error and the smallest agreement, per column.

    NOT the first draw and not the mean. A declared threshold set from one draw has a margin that is a
    guess, and the wrist camera's spread over four draws is the biggest term in its declaration.
    """
    out = {}
    for k in ("mean_abs", "interior_mean_abs"):
        vals = [d[k] for d in draws if d.get(k) is not None]
        if vals:
            out[k] = max(vals)
    for k in ("within8_frac", "ssim", "interior_exact_frac", "exact_frac"):
        vals = [d[k] for d in draws if d.get(k) is not None]
        if vals:
            out[k] = min(vals)
    return out


def spread(draws: list[dict], keys) -> dict:
    return {k: [min(d[k] for d in draws), max(d[k] for d in draws)]
            for k in keys if all(d.get(k) is not None for d in draws)}


def measure(make_env, seeds: int, n: int, label: str) -> tuple[dict, list[dict], list[dict]]:
    """Both regimes, over ``seeds`` independent draws: default appearance, then declared variation."""
    at_default, under_variation = [], []
    for k in range(seeds):
        e = make_env(k)
        r = C.check_visual_fidelity(e, n=n)
        at_default.append(r)
        print(f"  [{label}] draw {k} default    " + json.dumps(
            {j: round(v, 4) for j, v in r.items() if isinstance(v, float)}), flush=True)
        v = C.check_visual_fidelity_under_variation(e, n=n, seed=k)
        if "skipped" not in v:
            under_variation.append(v)
            print(f"  [{label}] draw {k} variation  " + json.dumps(
                {j: round(x, 4) for j, x in v.items() if isinstance(x, float)}), flush=True)
    return worst(at_default), at_default, under_variation


def gate(corrected: dict, plain: dict, regime: str) -> list[str]:
    """Every reason this corrector must not ship. Empty means it may."""
    bad = []
    if not corrected or not plain:
        return [f"{regime}: not measured on both paths, so there is nothing to compare"]
    if corrected["mean_abs"] >= plain["mean_abs"]:
        bad.append(f"{regime}: mean |diff| {corrected['mean_abs']:.3f} is not better than the plain "
                   f"render's {plain['mean_abs']:.3f}")
    for k, name in (("within8_frac", "within-8"), ("ssim", "SSIM")):
        if corrected[k] <= plain[k]:
            bad.append(f"{regime}: {name} {corrected[k]:.4f} is not better than the plain render's "
                       f"{plain[k]:.4f}")
    return bad


def thresholds(w: dict) -> dict:
    """Assertion floors set from the WORST draw with the margin every other cell in the repo uses.

    A threshold equal to a measurement flakes on the next state draw; a measurement inferred from a
    threshold is not a measurement. These are ~1.3x on the error and 0.05 absolute on the two
    fractions, which is what ``robocasa.FIDELITY`` and ``humanoidbench.FIDELITY`` already carry.
    """
    return {"mean_abs_max": float(np.ceil(w["mean_abs"] * 1.3 * 10) / 10),
            "within8_min": float(np.floor((w["within8_frac"] - 0.05) * 100) / 100),
            "ssim_min": float(np.floor((w["ssim"] - 0.05) * 100) / 100)}


# ------------------------------------------------------------------------------------------ main

def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                                # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("env", help="the PLAIN visual-tier class to fit a corrector for")
    ap.add_argument("--net", default="", help=f"architecture; default per family, see {list(DEFAULTS)}")
    ap.add_argument("--tex-lod", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=16384)
    ap.add_argument("--n-heldout", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decl-seeds", type=int, default=4, help="independent draws behind a declaration")
    ap.add_argument("--decl-states", type=int, default=64, help="states per draw; 64 is what the "
                                                                "conformance suite asserts at")
    ap.add_argument("--from-prototype", default="", help="adopt an already-fitted state_dict instead "
                                                         "of fitting; still measured and still gated")
    ap.add_argument("--no-seal", action="store_true", help="measure and report, write nothing")
    ap.add_argument("--out", default="", help="checkpoint path; default is the cache directory")
    ap.add_argument("--report", default="", help="also write the full measurement here as json")
    a = ap.parse_args()

    if a.env not in se.registry.REGISTRY:
        raise SystemExit(f"unknown env {a.env!r}; import errors: {se.import_errors()}")
    cls = se.registry.REGISTRY[a.env]
    if getattr(cls, "render_decoder", "renderer") != "renderer":
        raise SystemExit(f"{a.env} already draws through a learned decoder. Fit a corrector for the "
                         f"PLAIN class it was derived from, not for a corrected one.")
    if not hasattr(cls, "corrector_cell_key"):
        raise SystemExit(f"{a.env} does not declare a corrector_cell_key(), so a checkpoint could "
                         f"not be locked to it. That lock is the only thing standing between a "
                         f"corrector and the cell it silently ruins; see render/rendernet.py.")
    cell = dict(cls.corrector_cell_key())
    cell["env_class"] = f"{a.env}Corrected"
    fam = cell["family"]
    net = a.net or DEFAULTS[fam]["net"]
    tex_lod = a.tex_lod or DEFAULTS[fam]["tex_lod"]
    r = RN.render_size(net)
    render = {"render_size": r, "out_size": 256, "tex_lod": int(tex_lod), "backface_cull": False}
    print(f"{a.env}: {net} ({RN.n_params(RN.build(net))/1e3:.0f}k params), render {r} px "
          f"tex_lod {tex_lod} -> 256 px\ncell {cell}", flush=True)

    torch.manual_seed(a.seed)
    plain = se.make(a.env, num_envs=1, size=256, device="cuda", seed=a.seed)
    # The cheap render, through the SAME class with the corrector's profile. Not registered: it is a
    # render configuration, not an env, and the thing that makes it one is the checkpoint this run is
    # trying to produce.
    Cheap = type(f"{a.env}CheapProfile", (cls,), dict(TEX_LOD=int(tex_lod), BACKFACE_CULL=False))
    cheap = Cheap(num_envs=1, size=256, device="cuda", seed=a.seed)

    model = RN.build(net).cuda().to(memory_format=torch.channels_last)
    hist: list[dict] = []
    if a.from_prototype:
        ck = torch.load(a.from_prototype, map_location="cpu", weights_only=False)
        if ck.get("net") != net:
            raise SystemExit(f"{a.from_prototype} holds {ck.get('net')!r}, not {net!r}")
        if ck.get("env") not in (None, a.env):
            raise SystemExit(f"{a.from_prototype} was fitted on {ck.get('env')!r}, not {a.env!r}. A "
                             f"corrector is per cell; this is the mistake the whole module exists to "
                             f"prevent.")
        model.load_state_dict(ck["state_dict"])
        print(f"adopted weights from {a.from_prototype} (tag {ck.get('tag')!r})", flush=True)
    else:
        pairs = {}
        for split, n in (("train", a.n_train), ("heldout", a.n_heldout)):
            st = split_states(cls, split, n, a.seed + 1000 * (split == "heldout"))
            sty = draw_style(plain, n, a.seed + 7 + 1000 * (split == "heldout"))
            pairs[split] = make_pairs(cls, cheap, st, sty, r, label=split)
        tr_lr = torch.as_tensor(pairs["train"][0], device="cuda")
        tr_hr = torch.as_tensor(pairs["train"][1], device="cuda")
        ho_lr = torch.as_tensor(pairs["heldout"][0], device="cuda")
        ho_hr = torch.as_tensor(pairs["heldout"][1], device="cuda")
        hist = fit(model, tr_lr, tr_hr, ho_lr, ho_hr, a.epochs, a.batch, a.lr, a.clip, a.seed)
        del tr_lr, tr_hr, ho_lr, ho_hr
        torch.cuda.empty_cache()

    # ---- measure both paths on the same draws, in the protocol a declaration is asserted in ----
    corr = RN.unsealed(model, cell=cell, render=render, net=net)
    Cand = type(cell["env_class"], (_corrected_base(fam),),
                dict(**_cell_attrs(cls), TEX_LOD=int(tex_lod), BACKFACE_CULL=False,
                     default_size=256))

    def make_corrected(k):
        e = Cand(num_envs=16, size=256, device="cuda", seed=k)
        e._corrector_override = corr
        return e

    c_def, c_draws, c_var = measure(make_corrected, a.decl_seeds, a.decl_states, "corrected")
    p_def, p_draws, p_var = measure(lambda k: se.make(a.env, num_envs=16, size=256, device="cuda",
                                                      seed=k),
                                    a.decl_seeds, a.decl_states, "plain")
    c_varw, p_varw = worst(c_var), worst(p_var)

    reasons = gate(c_def, p_def, "at the default appearance")
    if c_var and p_var:
        reasons += gate(c_varw, p_varw, "under the declared variation space")
    else:
        reasons.append(
            "under the declared variation space: not measured, because this family's "
            "reference_render does not take a style. A corrector fitted at one appearance is "
            "measured at 9.06/255 under a declared variation it never saw against 8.91 for doing "
            "nothing, so an unmeasured variation regime is not an absent risk, it is the known one.")

    vf = {**{k: round(c_def[k], 4) for k in ("mean_abs", "within8_frac", "ssim",
                                             "interior_mean_abs", "interior_exact_frac")},
          **thresholds(c_def),
          "draw_spread": spread(c_draws, ("mean_abs", "within8_frac", "ssim", "interior_exact_frac")),
          "plain_mean_abs": round(p_def["mean_abs"], 4),
          "plain_within8_frac": round(p_def["within8_frac"], 4),
          "plain_ssim": round(p_def["ssim"], 4),
          "decoder": (f"{net} ({RN.n_params(model)/1e3:.0f}k parameters) applied to MJWarp at "
                      f"{r} px with texture LOD {tex_lod}, fp16, compiled with "
                      f"torch.compile(mode={RN.COMPILE_MODE!r}) at a fixed batch of {RN.NET_BATCH} "
                      f"and replayed from a CUDA graph"),
          "measured_at": (f"256 px, {a.decl_states} states x {a.decl_seeds} independent draws from "
                          f"sample_states (fresh env per draw, seeds 0-{a.decl_seeds - 1}), "
                          f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}; the values "
                          f"are the WORST of the draws and draw_spread is [min, max]. The plain_* "
                          f"columns are {a.env} measured on the same draws."),
          }
    varf = ({**{k: round(c_varw[k], 4) for k in c_varw},
             "plain_mean_abs": round(p_varw["mean_abs"], 4),
             "plain_within8_frac": round(p_varw["within8_frac"], 4),
             "plain_ssim": round(p_varw["ssim"], 4),
             "mean_abs_max": thresholds(c_varw)["mean_abs_max"],
             "within8_min": thresholds(c_varw)["within8_min"],
             "ssim_min": thresholds(c_varw)["ssim_min"],
             "measured_at": ("as above, with a per-state appearance drawn from the env's own "
                             "variation_space and written into BOTH renderers")}
            if c_var else {})

    report = {"env": a.env, "cell": cell, "render": render, "net": net,
              "visual_fidelity": vf, "variation_fidelity": varf,
              "baseline": {k: round(v, 4) for k, v in p_def.items()},
              "baseline_variation": {k: round(v, 4) for k, v in p_varw.items()},
              "corrected_draws": c_draws, "plain_draws": p_draws,
              "corrected_variation_draws": c_var, "plain_variation_draws": p_var,
              "gate_failures": reasons, "history": hist[-5:]}
    print("\n== declaration ==\n" + json.dumps(vf, indent=1), flush=True)
    print("\n== under the declared variation ==\n" + json.dumps(varf, indent=1), flush=True)
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1)
        print("wrote", a.report, flush=True)

    if reasons:
        raise SystemExit(
            "REFUSING TO SEAL. A corrector that is not better than the render it replaces is not a "
            "trade anyone asked for, and the way this fails is invisible: it lowers a frame's average "
            "error while making more of its pixels wrong.\n  " + "\n  ".join(reasons))
    if a.no_seal:
        print("--no-seal: measured and gated, nothing written", flush=True)
        return
    path = a.out or RN.path_for(cell)
    RN.seal(path, cell=cell, render=render, net=net, state_dict=model.state_dict(),
            visual_fidelity=vf, variation_fidelity=varf,
            baseline={k: round(v, 4) for k, v in p_def.items()},
            provenance={"git": git_sha(), "host": platform.node(),
                        "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
                        "date": time.strftime("%Y-%m-%d"), "args": vars(a),
                        "compile_mode": RN.COMPILE_MODE, "net_batch": RN.NET_BATCH,
                        "n_params": RN.n_params(model),
                        "split": "recorded states 0-149 train / 150-199 held out, yaw redrawn per "
                                 "draw, 30% of training draws interpolating two training rows",
                        "from_prototype": a.from_prototype or None})
    print(f"\nsealed {path}\n  {a.env}Corrected will appear in the registry on the next import",
          flush=True)


def _corrected_base(family: str):
    if family == "robocasa":
        from stable_envs.envs.robocasa import RoboCasaCorrectedBase
        return RoboCasaCorrectedBase
    raise SystemExit(f"no corrected base class for the {family!r} family")


def _cell_attrs(cls) -> dict:
    return {k: getattr(cls, k) for k in ("KITCHEN", "CAMERA", "ROBOT", "TASK") if hasattr(cls, k)}


if __name__ == "__main__":
    main()
