# M1 — where OGBCube's MJWarp-vs-MuJoCo interior gap actually is

Phase 0 measured raw MJWarp on `swm/OGBCube-v0` at **0.0101 interior-exact** against MuJoCo's own
EGL frames with only `gl_rounding` applied, against Fetch's corrected 0.9988, and none of the four
known Fetch corrections applied. This is the term-by-term attribution the coordinator authorised
before any cell is written.

**Method.** Each term is neutralised **in the `MjModel`**, and then BOTH renderers are driven from
that same modified model. A term MJWarp implements correctly leaves agreement unchanged when it is
removed; a term MJWarp gets wrong makes agreement jump. That is the attribution — it isolates the
term rather than changing one side's picture. Job **6396646**, `gpu2109`, RTX 3090, 8 real states,
224 px, `front_pixels`, `gl_rounding` on throughout. Raw data: `m1_attribution.json`.

## The table

| term neutralised on both sides | interior exact | Δ vs baseline | interior mean abs | affine slope | reading |
|---|---:|---:|---:|---:|---|
| *baseline* | 0.0082 | — | 8.78 | 0.858 | MJWarp systematically darker |
| **spot light** | **0.4568** | **+0.4486** | **1.54** | 0.965 | **dominant term** |
| headlight | 0.0454 | +0.0372 | 8.25 | 0.780 | second term |
| MSAA (4x → 1x) | 0.0111 | +0.0029 | 8.89 | 0.865 | edge-only, as phase 0 predicted |
| specular | 0.0078 | −0.0004 | 9.56 | 0.863 | not a term |
| shadows | 0.0040 | −0.0042 | 13.06 | 1.179 | **not** the problem; removing it hurts |
| skybox | 0.0039 | −0.0043 | 8.90 | 0.852 | sky only (sky is already 1.000 exact) |
| textures | 0.0073 | −0.0009 | 213.25 | 0.127 | control: the floor texture is really being sampled |

**The signed histogram is the confirmation.** On interior pixels at baseline the difference is spread
over **−3 to −8** (no value above 16 %), i.e. MJWarp is uniformly *darker* by several levels — not a
quantisation signature. With the spot neutralised it collapses to **{0: 61.8 %, +1: 21.3 %,
+2: 7.3 %}**, which is what "the same shading, quantised twice" looks like. Mean |Δ| falls **5.7x**,
from 8.78 to 1.54.

## The mechanism, read from both sources rather than inferred

`mujoco_warp/_src/render.py` attenuates a spot light as

```python
if lighttype == 0:                                  # spot
    cos_theta  = wp.dot(-L, lightdir)
    cos_cutoff = wp.cos(lightcutoff_rad)            # lightcutoff * pi/180
    if cos_theta < cos_cutoff:
        return diff_rgb, spec_rgb                   # hard cone
    attenuation *= wp.pow(wp.max(cos_theta, 0.0), lightexp)
```

This scene's spot is `cutoff = 100°`, `exponent = 10`, diffuse 0.3, at (0.25, 0, 0.5) pointing down.
`cos(100°) = −0.174`, so MJWarp's cone rejects almost nothing — but `pow(cos_theta, 10)` falls off
hard off-axis (0.24 at 30°, 0.028 at 45°, 0.001 at 60°), so MJWarp extinguishes the spot away from
directly beneath it.

**MuJoCo does not apply that attenuation at all, because it discards the cutoff.** Measured on the
laptop through MuJoCo alone — no MJWarp involved, so this half needs no GPU
(`m1_mujoco_cutoff_probe` below):

| spot cutoff | pixels differing from cutoff = 100° | max abs | frame mean |
|---:|---:|---:|---:|
| 100° (the model's own) | — | — | 57.241 |
| **180°** | **0.00 %** | **0** | 57.241 |
| 90° | 95.21 % | 77 | 49.830 |
| 89° | 95.21 % | 77 | 49.830 |
| 45° | 95.21 % | 77 | 49.811 |
| 10° | 95.25 % | 88 | 47.964 |

OpenGL's `GL_SPOT_CUTOFF` accepts `[0, 90]` or the sentinel `180`; anything else is
`GL_INVALID_VALUE` and the parameter is left at its default **180 = not a spotlight**. The table is
that rule, sharply: at 100° MuJoCo renders **bit-identically to 180°** — max abs difference **0** —
while a *valid* 90° changes 95 % of pixels and darkens the frame by 7.4 levels of mean. The boundary
is exactly where the GL spec puts it. Probe: `scratch/step7/ogb-manipulation/probe_m1_spot_cutoff.py`,
data `m1_mujoco_spot_cutoff.json`; it needs no GPU because it is a statement about MuJoCo alone.

**What "discarded" does and does not mean — measured, because the obvious reading is wrong.**
A follow-up sweep (`m1_mujoco_light_forms.json`, same probe file) pins the behaviour down:

| MuJoCo-side change | frame mean | vs shipped |
|---|---:|---|
| *shipped: SPOT, cutoff 100, exponent 10* | **58.26** | — |
| **exponent 10 → 0**, cutoff still 100 | **58.26** | **bit-identical, max 0** |
| cutoff 100 → 90 | 50.66 | 94.6 % differ, max 63 |
| cutoff 100 → 45 | 50.64 | 94.6 % differ, max 63 |
| light disabled | 48.55 | 94.6 % differ, max 66 |
| re-typed to POINT | 48.55 | **identical to disabled** |

Two things follow, and the second killed a plausible cell design before it was written:

1. **At an out-of-range cutoff the angular term is gone entirely.** The exponent has *no effect at
   all* — changing 10 to 0 is bit-identical — and the frame is at its brightest. So MuJoCo is
   lighting the scene with this light unattenuated by angle, which is exactly "not a spotlight",
   while `mujoco_warp` applies both the cone and `cos^10`. That is the disagreement.
2. **Re-typing the light to POINT does NOT express that.** In MuJoCo a POINT light here renders
   identically to the light being *switched off* — 9.7 levels of mean darker than shipped. So the
   tempting correction "make MJWarp see a POINT light, since that is what GL is doing" is measurably
   false on the reference side, and a cell built on it would have been wrong. The correction that
   matches the measurement is instead to neutralise the angular term MJWarp applies — `exponent = 0`
   (its `pow(cos_theta, exp)` becomes 1) with a cutoff wide enough that its cone rejects nothing —
   which is what the M3 design gate tests on the GPU.

## What this changes## What this changes

* **The gap is one term, and it is upstream.** 0.0082 → 0.4568 from a single light. The remaining
  0.543 is led by the headlight (independently worth +0.0372), and the combination has not yet been
  measured — queued.
* **The interior metric here is 99.2 % floor.** Measured class shares of the interior mask:
  static 0.9917, sky 0.0043, mover 0.0040. So the whole-frame interior number is almost entirely a
  statement about the plate — which the shipped cell bakes through MuJoCo exactly and never asks
  MJWarp to draw. `interior_exact_mover` is 0.667 at baseline and 0.852 with the headlight
  neutralised, but over 0.4 % of interior pixels, so it is a weak estimate and must be reported with
  its sample size rather than quoted as a mover-pass parity number.
* **Shadows and specular are not terms.** Neutralising either makes agreement *worse*. MJWarp's
  shadow implementation is not the thing to spend phase 1 on.
* **MSAA behaves exactly as phase 0 predicted**: +0.0029 interior, i.e. invisible to the interior
  metric, while it moves 5.4 % of all pixels. The edge-only reading holds.
