# M3 design gate — the spot correction works, and is not sufficient

Job **6397454**, `gpu2101`, RTX 3090, 8 real states, 224 px, `front_pixels`, `gl_rounding`
throughout. The reference is always the **unmutated** MuJoCo model; only what MJWarp reads is
changed, which is model surgery of the same shape as `fetch.py`'s four existing corrections.

| MJWarp-side correction | interior exact | interior mean abs | all exact | within 8 | signed interior |
|---|---:|---:|---:|---:|---|
| *baseline* | 0.0091 | 8.900 | 0.0355 | 0.5567 | −8: 14.8 %, −4: 12.6 %, −6: 11.8 % |
| **`exponent = 0` + `cutoff = 180`** | **0.3507** | **2.786** | 0.2743 | 0.8003 | 0: 54.6 %, +1: 15.2 %, −1: 6.5 % |
| `exponent = 0` only | 0.3506 | 2.741 | 0.2746 | 0.8013 | 0: 54.6 %, +1: 15.2 %, −1: 6.4 % |
| re-typed to POINT (contrast) | 0.3497 | 3.051 | 0.2729 | 0.7957 | 0: 54.0 %, +1: 14.9 %, −1: 6.5 % |

## What it settles

* **The correction is real and large.** Interior exactness rises **38x**, 0.0091 → 0.3507; mean |Δ|
  falls 3.2x; whole-frame "within 8" rises from 0.557 to 0.800. The signed distribution moves from a
  systematic −3..−8 darkening to **centred on zero with ±1 tails**, which is the shape of two
  implementations of one shading law rather than two different laws.
* **All three neutralisations land in the same place** (0.3497–0.3507). That is the expected result
  and it is worth stating: on the MJWarp side, zeroing the exponent, widening the cone, and re-typing
  to POINT all remove the same angular term, so the gate does not depend on picking the "right" one.
  `exponent = 0` alone is therefore the minimal form, and the cutoff change is cosmetic. Note this
  does **not** contradict M1's finding that POINT is wrong on the *MuJoCo* side — there, POINT
  renders identically to the light switched off. The two sides are asymmetric, which is exactly why
  the correction is applied to the MJWarp model only.
* **It is not sufficient.** 0.3507 is far from Fetch's corrected 0.9988, and it is also **below**
  M1's `no_spot` row (0.4568), where the light was removed from *both* sides. That gap — 0.35 against
  0.46 — is the part of the spot's contribution the two renderers still compute differently once the
  angular term is gone: MJWarp takes `attenuation = 1 / (1, d, d²)·(1, 0, 0) = 1`, i.e. no distance
  falloff, and MuJoCo evidently does something else with the same `light_attenuation`. M1's second
  term, the headlight (independently worth +0.0372 of interior), is untouched here and is next.

## Consequence for the cell

A cell written today would carry the correction and declare **`visual`**, not `parity`: 0.35
interior-exact is not a parity claim under any reading, and the remaining terms are identified but
not yet closed. The honest sequence is to finish the attribution — spot distance term, then
headlight — before writing `stable_envs/envs/` code, because each correction is a line in the cell's
model surgery and the set of them is what the cell *is*. **No cell has been written.**

Two further facts the cell will need, already measured: the correction applies to **1** light in this
model (`neutralised_lights: 1`), and the predicate is "a spot light whose cutoff is outside
OpenGL's legal `[0, 90]`" — a model property, checkable at construction, not a per-env constant.
