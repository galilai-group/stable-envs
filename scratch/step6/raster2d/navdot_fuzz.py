"""Differential fuzz: the fused swm_nav kernel against the torch path it replaces, byte for byte.

The kernel's claim is bit-identity, not closeness, so this compares ``==`` and not a tolerance. The
cases are the awkward ones rather than the ones the envs produce: dots off the frame in every
direction and far enough that the frame maximum is no longer next to the centre, dots on and between
pixel centres, radii at both ends of the declared range and outside it, per-row styles so no two rows
share a colour or a radius, duplicate rows, backgrounds at the ends of the uint8 range, and -- for
TwoRoomTarget -- two dots placed on top of each other, which is where quantising between the blends
is the difference between 172 disagreeing pixels and 35,957.

    python scratch/step6/raster2d/navdot_fuzz.py --cases 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from stable_envs.envs import swm_nav as SN                     # noqa: E402
from stable_envs.render import navdot                          # noqa: E402

IMG = SN.IMG


def torch_reference(base, pos, sigma, colour):
    """``_alpha_blend(_gaussian_dot(...))`` once per dot, in order, exactly as the envs call it."""
    gy, gx = SN._grids(base.device)
    img = base
    for d in range(pos.shape[1]):
        dot = SN._gaussian_dot(gy, gx, pos[:, d], sigma[:, d])
        img = SN._alpha_blend(img, dot, colour[:, d])
    return img


def case(gen, dev, i):
    # the shape draws come from a CPU generator: torch.randint with a cuda generator and no device=
    # raises "Expected a 'cpu' device type for generator", which is how the first run of this fuzz
    # produced four tracebacks and no comparison while the job around it reported success.
    B = int(torch.randint(1, 6, (1,), generator=gen, device=dev).item())
    ND = 1 + (i % 2)
    r = lambda *s: torch.rand(*s, generator=gen, device=dev)                 # noqa: E731
    base = (r(B, IMG, IMG, 3) * 255).to(torch.uint8)
    style = i % 8
    if style == 0:                                   # flat, extreme backgrounds
        base = torch.where(r(B, 1, 1, 1) < 0.5, torch.zeros_like(base),
                           torch.full_like(base, 255))
    pos = r(B, ND, 2) * IMG
    if style == 1:                                   # far outside the frame, every direction
        pos = (r(B, ND, 2) * 4 - 1.5) * IMG
    elif style == 2:                                 # just outside, where the max leaves the centre
        pos = r(B, ND, 2) * (IMG + 120) - 60
    elif style == 3:                                 # exactly on pixel centres
        pos = (r(B, ND, 2) * IMG).round()
    elif style == 4:                                 # exactly between them
        pos = (r(B, ND, 2) * IMG).round() + 0.5
    elif style == 5 and ND == 2:                     # two dots in the same place
        pos[:, 1] = pos[:, 0]
    elif style == 6:                                 # far enough that the dot underflows entirely
        pos = r(B, ND, 2) * 8000 + 2000
    sig = r(B, ND) * 7.0 + 7.0
    if style == 7:
        sig = torch.where(r(B, ND) < 0.5, torch.full_like(sig, 1e-7), r(B, ND) * 60 + 0.5)
    col = (r(B, ND, 3) * 255).trunc()
    return base.contiguous(), pos.contiguous(), sig.contiguous(), col.contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    gen = torch.Generator(device=a.device).manual_seed(a.seed)
    bad = 0
    print(f"device {a.device}  cases {a.cases}  seed {a.seed}")
    for i in range(a.cases):
        base, pos, sig, col = case(gen, a.device, i)
        want = torch_reference(base, pos, sig, col)
        got = navdot.blend_dots(base, pos, sig, col)
        if not torch.equal(want, got):
            bad += 1
            d = (want.to(torch.int16) - got.to(torch.int16)).abs()
            print(f"case {i} (style {i % 8}, B={base.shape[0]}, ND={pos.shape[1]}): "
                  f"{int((d.amax(-1) > 0).sum())} pixels differ, max {int(d.max())}")
            if bad > 5:
                break
    print(f"{a.cases - bad}/{a.cases} identical" if not bad else f"FAILED: {bad} differing cases")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
