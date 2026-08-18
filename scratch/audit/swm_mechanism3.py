"""Name the mechanism behind the swm-nav interior-parity flake, by decomposition rather than by guess.

The symptom: `test_render_parity_with_official_renderer` draws states unseeded, and on a few percent of
draws one or two INTERIOR pixels disagree with the official frame by exactly 1 on exactly one channel,
at the 254/255 boundary.

Three candidate mechanisms, and this separates them by construction. Our renderer and the reference
differ in exactly two places: the DEVICE the arithmetic runs on (CUDA vs CPU) and the PRECISION of the
transcendental (`exp` widened to float64 and rounded once, vs the reference's float32 `expf`). So build
four frame stacks that differ one factor at a time:

    G64   ours, CUDA, exp in float64      -- what the env ships
    C64   ours, CPU,  exp in float64
    C32   ours, CPU,  exp in float32      -- the reference's own expression, batched
    REF   the official renderer

    REF vs C32   must be 0.  If it is, C32 is a faithful model of the reference and the rest of the
                 decomposition is meaningful rather than a comparison against a black box.
    C64 vs C32   the exp-precision term.
    G64 vs C64   the device term.

Then, for every disagreeing INTERIOR pixel, the alpha that produced it is recovered under both exp
variants and the composite is re-evaluated in float32 by hand, so the claim "a 1-ulp alpha flips this
pixel" is arithmetic on the record rather than an inference from a pixel count.

Phase 2 measures the rate over many seeded draws, which is what a floor (if one is declared) has to be
set from.

    python scratch/audit/swm_mechanism3.py [--draws 200]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import stable_envs as se
from stable_envs.envs import swm_nav as SN
from stable_envs.testing import conformance as C

ENVS = ["TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones"]

_F64_DOT = SN._gaussian_dot


def _dot_f32(gy, gx, pos, radius):
    """`_gaussian_dot` with the transcendental left in float32 -- the reference's own expression."""
    dx = gx.unsqueeze(0) - pos[:, 0].view(-1, 1, 1)
    dy = gy.unsqueeze(0) - pos[:, 1].view(-1, 1, 1)
    dist2 = dx * dx + dy * dy
    std = radius.double().clamp_min(1e-6)
    denom = (2.0 * std * std).to(torch.float32).view(-1, 1, 1)
    dot = torch.exp(-dist2 / denom)
    m = dot.amax(dim=(1, 2), keepdim=True)
    return torch.where(m > 0, dot / m, dot)


def render_variant(name, states, device, exp64):
    SN._gaussian_dot = _F64_DOT if exp64 else _dot_f32
    try:
        env = se.make(name, batch=1, device=device)
        return env.render(states).cpu().numpy()
    finally:
        SN._gaussian_dot = _F64_DOT


def alpha_at(name, states, frame, y, x, device, exp64):
    """The normalised alpha of the LAST dot composited (the agent), at one pixel, as a float32 bit
    pattern. That is the dot the disagreement can come from: it is blended last, over everything."""
    dot = _F64_DOT if exp64 else _dot_f32
    env = se.make(name, batch=1, device=device)
    s = torch.as_tensor(states[frame:frame + 1], device=env.device, dtype=torch.float32)
    a = dot(env._gy, env._gx, s[:, 0:2], env._style["agent.radius"][:1])
    return np.float32(a[0, y, x].item())


def blend_u8(bg: float, col: float, a: np.float32) -> int:
    """The reference's composite for one channel, in float32, exactly as both renderers evaluate it."""
    out = np.float32(np.float32(bg) * np.float32(np.float32(1.0) - a) + np.float32(col) * a)
    return int(np.uint8(out))            # C-style truncation, like torch's .to(uint8)


def phase1(name, seeds):
    cls = se.REGISTRY[name]
    env = se.make(name, batch=64, device="cuda")
    ref = C.reference_env(cls, env.size)
    print(f"\n================ {name}")
    for seed in seeds:
        states = env.sample_states(64, rng=np.random.default_rng(seed))
        REF = cls.reference_render(ref, states)
        G64 = render_variant(name, states, "cuda", True)
        C64 = render_variant(name, states, "cpu", True)
        C32 = render_variant(name, states, "cpu", False)

        def nd(a, b):
            return int((np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > 0).sum())

        m = C.interior_mask(REF)
        r = C.pixel_agreement(G64, REF)
        print(f"  seed {seed:3d}  interior_exact {r['interior_exact_frac']!r}  max_abs {r['max_abs']}  "
              f"| disagreeing px: REFvsC32 {nd(REF, C32)}  C64vsC32 {nd(C64, C32)}  "
              f"G64vsC64 {nd(G64, C64)}  G64vsREF {nd(G64, REF)}")

        d = np.abs(G64.astype(np.int16) - REF.astype(np.int16)).max(-1)
        bad_int = np.argwhere(d[:, :-1, :-1] * m > 0)
        for f, y, x in bad_int:
            ch = int(np.argmax(np.abs(G64[f, y, x].astype(np.int16) - REF[f, y, x].astype(np.int16))))
            a64 = alpha_at(name, states, f, y, x, "cuda", True)
            a32c = alpha_at(name, states, f, y, x, "cpu", False)
            a64c = alpha_at(name, states, f, y, x, "cpu", True)
            bg, col = 255.0, float(SN.AGENT_C[ch])
            u64, u32 = blend_u8(bg, col, a64), blend_u8(bg, col, a32c)
            # how far is this pixel from a place where the REFERENCE image changes value at all?
            g = REF[f].astype(np.int16)
            rad = 0
            for k in range(1, 40):
                p = g[max(y - k, 0):y + k + 1, max(x - k, 0):x + k + 1]
                if not (p == p.reshape(-1, 3)[0]).all():
                    rad = k
                    break
            print(f"    frame {f:2d} ({y:3d},{x:3d}) ch{ch}  ours={G64[f, y, x]} ref={REF[f, y, x]}  "
                  f"flat radius {rad}px\n"
                  f"      alpha  gpu-f64 {a64.view(np.uint32):#010x}  cpu-f64 {a64c.view(np.uint32):#010x}"
                  f"  cpu-f32 {a32c.view(np.uint32):#010x}   ulp(f64,f32)="
                  f"{int(a64.view(np.int32)) - int(a32c.view(np.int32))}\n"
                  f"      blend(bg={bg:.0f}, col={col:.0f}) -> f64-alpha {u64}, f32-alpha {u32}  "
                  f"(1-a)={np.float32(1.0) - a64:.9g}")
    del env
    torch.cuda.empty_cache()


def phase2(name, draws):
    cls = se.REGISTRY[name]
    env = se.make(name, batch=64, device="cuda")
    ref = C.reference_env(cls, env.size)
    fr, bad, worst = [], [], None
    n_int = None
    for d in range(draws):
        states = env.sample_states(64, rng=np.random.default_rng(d))
        REF = cls.reference_render(ref, states)
        ours = env.render(states).cpu().numpy()
        r = C.pixel_agreement(ours, REF)
        f = r["interior_exact_frac"]
        if f is None:
            continue
        m = C.interior_mask(REF)
        n_int = int(m.sum())
        npx = int(((np.abs(ours.astype(np.int16) - REF.astype(np.int16)).max(-1)[:, :-1, :-1]) * m > 0).sum())
        fr.append(f)
        bad.append(npx)
        if worst is None or f < worst[1]:
            worst = (d, f, npx)
    fr = np.asarray(fr)
    print(f"\n{name:15s} draws {len(fr)}  below-1.0 {int((fr < 1.0).sum())}  min {fr.min():.10f}  "
          f"interior px/draw {n_int}  worst bad-px {max(bad)} (seed {worst[0]})  "
          f"mean bad-px {np.mean(bad):.3f}", flush=True)
    del env
    torch.cuda.empty_cache()
    return {"env": name, "draws": len(fr), "below": int((fr < 1.0).sum()), "min": float(fr.min()),
            "interior_px": n_int, "worst_bad_px": int(max(bad)), "mean_bad_px": float(np.mean(bad))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--phase1-seeds", default="25,34,20,2")
    a = ap.parse_args()
    seeds = [int(s) for s in a.phase1_seeds.split(",")]
    print("PHASE 1 -- decomposition on known-bad seeds", flush=True)
    for n in ENVS:
        try:
            phase1(n, seeds)
        except Exception as e:                                           # noqa: BLE001
            print(f"{n}: phase1 failed {type(e).__name__}: {e}", flush=True)
    print("\n\nPHASE 2 -- rate over seeded draws", flush=True)
    rows = []
    for n in ENVS:
        try:
            rows.append(phase2(n, a.draws))
        except Exception as e:                                           # noqa: BLE001
            print(f"{n}: phase2 failed {type(e).__name__}: {e}", flush=True)
    print("\nSUMMARY")
    for r in rows:
        print(r, flush=True)
