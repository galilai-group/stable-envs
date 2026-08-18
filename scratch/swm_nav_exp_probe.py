"""Where do the last 0.08% of pixels come from — geometry, or ``expf``?

The swm navigation frames are an exact arithmetic formula, so a residual disagreement with the
official CPU renderer can only be (a) a different formula, or (b) the same formula evaluated with a
different transcendental. This separates the two: the argument to ``exp`` is a float32 value computed
identically on both sides, so evaluating ``exp`` in float64 and rounding once gives the correctly
rounded float32 answer. If the residual vanishes under that, it was libm; if it does not, it is
geometry and no amount of precision will fix it.
"""
from __future__ import annotations

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing import conformance as C
from stable_envs.envs import swm_nav as M


def variant(mode: str):
    gy_gx_cache = {}

    def dot(gy, gx, pos, radius):
        dx = gx.unsqueeze(0) - pos[:, 0].view(-1, 1, 1)
        dy = gy.unsqueeze(0) - pos[:, 1].view(-1, 1, 1)
        dist2 = dx * dx + dy * dy
        std = radius.double().clamp_min(1e-6)
        denom = (2.0 * std * std).to(torch.float32).view(-1, 1, 1)
        arg = -dist2 / denom
        if mode == "f32":
            d = torch.exp(arg)
        elif mode == "f64exp":
            d = torch.exp(arg.double()).to(torch.float32)
        elif mode == "f64all":
            d = torch.exp((-dist2.double()) / denom.double()).to(torch.float32)
        else:
            raise ValueError(mode)
        m = d.amax(dim=(1, 2), keepdim=True)
        return torch.where(m > 0, d / m, d)

    return dot


def main() -> None:
    for name in ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones"):
        cls = se.REGISTRY[name]
        env = se.make(name, batch=16, device="cuda")
        ref = C.reference_env(cls, env.size)
        states = env.sample_states(64, rng=np.random.default_rng(3))
        official = cls.reference_render(ref, states)
        print(f"\n== {name}")
        original = M._gaussian_dot
        for mode in ("f32", "f64exp", "f64all"):
            M._gaussian_dot = variant(mode)
            ours = env.render(states).cpu().numpy()
            r = C.pixel_agreement(ours, official)
            d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
            print(f"  {mode:8s} interior_exact={r['interior_exact_frac']}  "
                  f"all_exact={r['exact_frac']:.6f}  bad_px={int((d>0).sum())}  max={r['max_abs']}")
        M._gaussian_dot = original


if __name__ == "__main__":
    main()
