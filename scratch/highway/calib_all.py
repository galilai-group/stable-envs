"""Parity for every registered highway env, plus the two calibration questions the new pair raises.

    python scratch/highway/calib_all.py [n_states]

Reports interior and all-pixel agreement against the official pygame renderer, and then answers, by
measurement rather than by reading ``road/graphics.py``:

  * does ``RoadObjectGraphics`` cull off-screen objects the way ``VehicleGraphics`` does? Rendered with
    the culling test on and off, against the reference, on states where a parking wall's CENTRE is
    outside the frame;
  * what do the tyres cost? Rendered with and without them, against the reference.
"""
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "offscreen"
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import stable_envs as se                                             # noqa: E402
from stable_envs.testing import conformance as C                     # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
NAMES = ["Highway", "Merge", "Roundabout", "Intersection", "Racetrack", "Parking"]

print(f"{'env':13s} {'dim':>4s} {'prims':>6s} {'interior':>10s} {'all':>10s} {'mean':>8s} {'max':>5s}")
for name in NAMES:
    e = se.make(name, batch=8, device="cuda")
    st = e.sample_states(N, rng=np.random.default_rng(0))
    f = e.render(st).cpu().numpy()
    ref = C.reference_env(type(e), e.size)
    off = type(e).reference_render(ref, st)
    r = C.pixel_agreement(f, off)
    prims = e.scene(torch.as_tensor(st[:2], device="cuda")).n_prims
    ie = r["interior_exact_frac"]
    print(f"{name:13s} {e.state_dim:4d} {prims:6d} {ie if ie is None else round(ie, 6)!s:>10s} "
          f"{r['exact_frac']:10.6f} {r['mean_abs']:8.4f} {r['max_abs']:5d}")

# ---------------------------------------------------------------- object culling
print("\nobject culling: is RoadObjectGraphics.display gated on is_visible, like VehicleGraphics?")
e = se.make("Parking", batch=8, device="cuda")
rng = np.random.default_rng(3)
st = e.sample_states(64, rng=rng)
# drive the ego far enough that at least one wall's CENTRE leaves the frame by more than the 50 px
# margin: half a 600 px frame at scaling 7 is 42.9 m, so 20 m of offset does it
st[:, 0] = rng.uniform(-34, 34, len(st))
st[:, 1] = rng.uniform(-20, 20, len(st))
ref = C.reference_env(type(e), e.size)
off = type(e).reference_render(ref, st)
ours = e.render(st).cpu().numpy()
d = np.abs(ours.astype(np.int16) - off.astype(np.int16)).max(-1)
print(f"  cull=False (shipped): all-pixel exact {float((d == 0).mean()):.6f}, mean {d.mean():.4f}")

import stable_envs.envs.highway as H                                  # noqa: E402
_orig = H.HighwayEnvBase._add_box


def _culled(self, *a, **kw):
    kw["cull"] = True
    return _orig(self, *a, **kw)


H.HighwayEnvBase._add_box = _culled
ours2 = e.render(st).cpu().numpy()
H.HighwayEnvBase._add_box = _orig
d2 = np.abs(ours2.astype(np.int16) - off.astype(np.int16)).max(-1)
print(f"  cull=True  (wrong):   all-pixel exact {float((d2 == 0).mean()):.6f}, mean {d2.mean():.4f}, "
      f"differing px/frame {int((d2 > 0).sum()) / len(st):.0f}")

# ---------------------------------------------------------------- what the tyres cost
print("\ntyres: rendered with them, and with them suppressed")
for name in ("Racetrack", "Parking"):
    e = se.make(name, batch=8, device="cuda")
    st = e.sample_states(N, rng=np.random.default_rng(0))
    ref = C.reference_env(type(e), e.size)
    off = type(e).reference_render(ref, st)
    with_t = e.render(st).cpu().numpy()
    orig = H.HighwayEnvBase._add_tires
    H.HighwayEnvBase._add_tires = lambda *a, **k: None
    without = e.render(st).cpu().numpy()
    H.HighwayEnvBase._add_tires = orig
    a = np.abs(with_t.astype(np.int16) - off.astype(np.int16)).max(-1)
    b = np.abs(without.astype(np.int16) - off.astype(np.int16)).max(-1)
    print(f"  {name:10s} with tyres {float((a==0).mean()):.6f} exact ({int((a>0).sum())/len(st):6.1f} px/frame), "
          f"without {float((b==0).mean()):.6f} ({int((b>0).sum())/len(st):6.1f} px/frame)")

# ---------------------------------------------------------------- steering is load-bearing
print("\nsteering: does the reference frame change when only ego_steering changes?")
e = se.make("Racetrack", batch=8, device="cuda")
st = e.sample_states(16, rng=np.random.default_rng(1))
ref = C.reference_env(type(e), e.size)
a = type(e).reference_render(ref, st)
st2 = st.copy()
st2[:, e.steer_off] = 0.0
b = type(e).reference_render(ref, st2)
dd = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
print(f"  official frames at steering=s vs steering=0: {int((dd>0).sum())/len(st):.1f} px/frame, "
      f"max {int(dd.max())}/255")
