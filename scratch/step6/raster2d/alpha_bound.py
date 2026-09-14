"""The smallest alpha that can still move a pixel, measured against swm_nav's own ``_alpha_blend``.

Step 6's raster2d findings asserted that "an alpha of 2e-9 still moves a pixel", and used that to
conclude the Gaussian dot has unbounded support and cannot be windowed. The assertion was wrong and
so was the conclusion. ``_alpha_blend`` is

    out = img.to(float32) * (1 - a) + colour * a   ->  .to(uint8)

and ``img`` is an exact small integer in float32, so the question is not whether ``a`` is nonzero but
whether ``a * (img - colour)`` is large enough to change the float32 result at all -- which is a
question about float32 spacing at ``img``, not about the truncation of a real number. This measures
it rather than reasoning about it, and then converts the answer into the radius at which a normalised
Gaussian falls below it.

    python scratch/step6/raster2d/alpha_bound.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from stable_envs.envs import swm_nav as SN                              # noqa: E402


def main() -> None:
    print(f"tree {SN.__file__}")
    # Every (background, colour) pair the four configs can produce is some pair of uint8 values, so
    # the search is over the widest possible contrast, which is where a small alpha bites hardest.
    smallest, where = 1.0, None
    for img_v in (0, 1, 127, 128, 254, 255):
        for col_v in (0, 1, 127, 128, 254, 255):
            if img_v == col_v:
                continue
            img = torch.full((1, 1, 1, 3), img_v, dtype=torch.uint8)
            rgb = torch.full((1, 3), float(col_v))
            lo, hi = 0.0, 1.0
            for _ in range(200):                       # bisect on "does this alpha change a pixel"
                mid = (lo + hi) / 2
                a = torch.full((1, 1, 1), mid)
                lo, hi = ((lo, mid) if not torch.equal(SN._alpha_blend(img, a, rgb), img)
                          else (mid, hi))
            if hi < smallest:
                smallest, where = hi, (img_v, col_v)
    # The MINIMUM over contrasts, not the maximum: a bound has to hold for the pixel that is easiest
    # to move. The two directions are not symmetric and that is the whole point. Painting a bright
    # colour onto a dark pixel needs a*|img-colour| >= 1 before truncation sees it, i.e. a >= 1/255.
    # Darkening a bright one goes through ``img * (1 - a)``, and 1 - a rounds to exactly 1.0 in
    # float32 for every a below 2^-24 -- so the threshold there is a property of float32 spacing at
    # 1.0 and is four orders of magnitude smaller.
    print(f"smallest alpha that changes any pixel, over every uint8 contrast: {smallest:.6e}"
          f"   (at img={where[0]}, colour={where[1]})")
    print(f"   = 2^{math.log2(smallest):.2f}   (float32 has 24 bits of significand; 2^-24 = "
          f"{2.0**-24:.6e})")
    worst = smallest

    # A normalised Gaussian alpha is exp(-d^2/(2 s^2)) / max, and the max sits at the sample nearest
    # the centre, so the normalisation cannot raise the tail relative to the peak by more than the
    # centre offset -- bounded below by exp(-0.5/(2 s^2)) for a half-pixel offset. The radius where
    # the alpha falls under the threshold above is therefore a real bound on the dot's support.
    print("\nradius at which the normalised dot can no longer move a pixel:")
    for s in (7.0, 10.0, 14.0):                        # the declared agent.radius range, 7..14
        # worst case: peak underestimated by the half-pixel offset, so require
        #   exp(-d^2/(2 s^2)) / exp(-0.5/(2 s^2)) < worst
        d = s * math.sqrt(2.0 * (-math.log(worst)) + 0.5 / (s * s))
        print(f"   sigma {s:5.1f} px -> {d:7.2f} px = {d/s:5.2f} sigma   "
              f"(frame is {SN.IMG} px, so the bound is "
              f"{'INSIDE' if 2*d < SN.IMG else 'the whole frame'})")


if __name__ == "__main__":
    main()
