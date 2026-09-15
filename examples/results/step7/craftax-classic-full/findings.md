# Step 7 — W1 `craftax-classic-full`

**Classification: `PATCH`.** All three declared exclusions are implemented and the renderer
reproduces the official Craftax Classic observation **byte for byte** — the 7x9 world window, the
2-row inventory strip, the sleeping player and night frames including the threefry noise field.
Gate A's coverage for `swm/CraftaxClassicPixels-v1` goes from `partial` to **full** on the
renderer's side; the remaining items are certification measurements that need Oscar, and two
declaration rulings that are Sami's.

| | |
|---|---|
| parent baseline SHA | `76dc5116f432e61dd47fba8f9e879cf9212c8e46` (Step 6 closed) |
| branch | `step7/craftax-classic-full`, worktree `/Users/samib/stable-envs-step7-craftax-classic-full` |
| affected config | **`CraftaxClassic` only** — 1 of the 49 frozen Step-4 render-distinct configurations |
| parity | **100% of all pixels exact, max abs diff 0**, 64 states at 7 px and 16 px, over the whole 144x144 observation and all three lighting bands. `digest_ours == digest_official`: the frames are byte-identical, not merely equal within a bound |
| reference venv | `/Users/samib/se7-craftax-venv` — python 3.12.11, **jax/jaxlib 0.11.1**, **craftax 1.6.1**, torch 2.14.0, numpy 2.5.3, gymnasium 1.3.0 |
| host | **Apple arm64 Mac**. Local CPU timings are **not** certification numbers |
| Oscar | **BLOCKED** — `Permission denied (keyboard-interactive)`. sbatch scripts written and ready (§11) |
| pixels changed | **yes, by design** — this is a coverage change. The frame grows 112x144 -> 144x144 at the default size. The **world rows of a Step-4-shaped state are unchanged** and asserted so |

## Contents

Sections 1-5 are the phase-0 diagnosis, kept as measured. Sections 6-12 are the implementation,
its evidence, and what is still owed.

---

---

## 0. What phase 0 did

Instrumented the installed `craftax.craftax_classic.renderer.make_craftax_pixel_renderer` — the
function body taken with `inspect.getsource` and transformed by anchored substitution, so a craftax
upgrade makes the probe fail loudly rather than measure a stale copy
(`scratch/step7/craftax-classic-full/instrumented_renderer.py`) — harvested **64 real states** from
the env's own 4,000-step rollouts stratified over day / dusk / night / sleeping, and prototyped each
excluded component in torch.

**The instrumented renderer is bit-exact with the official one on 64/64 states when un-jitted.**
Under `jax.jit` it agrees on only 32/64 in float32 (max `|diff|` 1.53e-05) while staying uint8-exact
on 64/64 — XLA fuses the extra outputs differently. Every float32 claim below is therefore measured
against the **un-jitted** reference. That instability is a property of the reference (the same
effect `docs/research/pixel_native.md` section 3.1 records for `get_obs` fused inside `step_env`),
and it is itself a reason a parity tier for this env should be declared on uint8 frames, not floats.

| artifact | path (under `examples/results/step7/craftax-classic-full/phase0/`) |
|---|---|
| state pool census | `state_census.json` |
| PRNG exactness | `prng_exactness.json` |
| strip / night / sleep parity | `components.json` |
| `state_rng` observability + state-window bytes | `purity_and_state_window.json` |
| sleeping player sprite + day cycle | `player_sleep_and_daycycle.json` |
| world layer under `is_sleeping` | `world_layer_sleeping_slot.json` |
| resolution-consistency risk | `resolution_consistency_risk.json` |
| local CPU cost | `local_cost.json` |
| venv pins | `reference_venv_pins.json` |
| daylight post-process identity | `daylight_identity.json` |

Probes: `scratch/step7/craftax-classic-full/{harvest,probe_prng,probe_components,probe_purity_state,probe_player_sleep,probe_world_sleep,probe_resolution,probe_cost}.py`,
plus `threefry_torch.py` and `instrumented_renderer.py`.

---

## 1. The inventory strip — `FINDING` + `PATCH_CANDIDATE`

### 1.1 What it reads, and what it draws from

Exactly **16 integers**, each in `0..9`, and nothing else:

| grid cell | field | icon |
|---|---|---|
| `(0,0)`..`(0,3)` | `player_health`, `player_food`, `player_drink`, `player_energy` | `health/hunger/thirst/energy_texture` |
| `(0,4)` | `inventory.sapling` | `sapling_texture` |
| `(0,5)`..`(0,8)` | `inventory.{wood,stone,coal,iron}` | `smaller_block_textures[{6,4,8,9}]` |
| `(1,0)` | `inventory.diamond` | `smaller_block_textures[10]` |
| `(1,1)`..`(1,3)` | `inventory.{wood,stone,iron}_pickaxe` | `{wood,stone,iron}_pickaxe_texture` |
| `(1,4)`..`(1,6)` | `inventory.{wood,stone,iron}_sword` | `{wood,stone,iron}_sword_texture` |
| `(1,7)`, `(1,8)` | — | never written; stays black |

Counts are capped at 9 by the simulator (`game_logic.py:1633` for the inventory, `:1293` / `:1319`
for energy and health), so `number_textures[0..9]` is the whole digit alphabet and
`number_textures[0]` is a zero array with zero alpha. **A cell whose count is 0 is entirely black** —
the icon is `smaller_empty_texture` and the digit is the zero texture, so "no icon" and "no digit"
are the same condition.

Per cell the reference does: zeros -> `.set(icon)` on `[left : bps-right]` squared ->
`.mul(1 - number_alpha).add(number_premultiplied)` on `[number_offset-1 : bps-1]` squared.
Measured geometry:

| `bps` | icon square | digit square | digit size |
|---|---|---|---|
| 7 | `[0:5]` | `[2:6]` | 4 |
| 16 | `[1:13]` | `[6:15]` | 9 |

**Every cell's two writes stay inside that cell's own `bps x bps` tile, and no two cells overlap**
(`cells_are_disjoint: true` at both sizes, measured by accumulating the written regions). So the
sequence of 32 `.at[].set/.mul/.add` calls is order-independent and the strip is a **pure per-cell
gather** — exactly the shape `_Composite` already is.

### 1.2 The atlas, and the parity number

A **160-entry second table** — 16 cells x 10 counts — is the whole capability:

| `bps` | entries | uint8 bytes |
|---|---|---|
| 7 | 160 | 23,520 |
| 16 | 160 | 122,880 |

Against it, the strip is `atlas[cell, count[b, cell]]` scattered onto a 2x9 grid.

**Parity, 64 real states, both tile sizes: `strip_bitexact` 64/64 in float32, `strip_uint8_exact`
64/64, and 64/64 against the official frame's own rows.** Not "close" — the same float32 bits.

### 1.3 Exact rows

| `bps` | frame | world rows | strip rows |
|---|---|---|---|
| 7 | 63x63x3 | `[0:49]` | **`[49:63]`** — 14 rows, i.e. rows 49..62 inclusive |
| 16 | 144x144x3 | `[0:112]` | `[112:144]` — 32 rows |

The module docstring's "rows 49-63 of the 63x63 agent observation" is right read as a half-open
interval and wrong read as inclusive; worth tightening when the docstring is rewritten.

### 1.4 The strip is *not* lit

`render` concatenates `inv_pixels` **after** the night chain and the sleep desaturation, so the strip
is the same picture at midnight and at noon. Measured directly: re-rendering each of the 64 states at
`light_level` in `{1.0, 0.3, 0.0}` crossed with `is_sleeping` in `{False, True}` leaves the strip
rows **bit-identical** (`strip_invariant_to_light_and_sleep: true`), and changing `state_rng` changes
**0** strip channels on 64/64 states at both sizes. This is the same fact the Crafter port had to
discover the hard way (`docs/research/crafter.md`: "the strip matched at daylight 1.0 and not at 0.8
until the concatenation order was fixed"); here it is measured up front.

**PATCH_CANDIDATE.** A `_StripComposite` table beside `_Composite` in `stable_envs/envs/craftax.py`,
gathered and written into rows `[H_world:]` of the same output buffer. At the default appearance it
is one extra `index_select` into uint8 with no float intermediate — the same structure as the Step-6
world-layer gather.

---

## 2. Night — `FINDING` + `PATCH_CANDIDATE` + one `BLOCKER`

### 2.1 The formula, exactly

```
daylight    = state.light_level                                  # float32 scalar
intensity   = max(2 * (0.5 - daylight), 0)
noise       = jax.random.uniform(state.state_rng, (OBS_H*t, OBS_W*t)) * 95 + 32
mask        = intensity * night_noise_intensity_texture          # (H, W, 3) vignette
with_static = (1 - mask) * world + mask * noise[:, :, None]
night_px    = select(daylight < 0.5, with_static, world)
lum         = 0.299 R + 0.587 G + 0.114 B
night_px    = night_px * 0.4 + 0.6 * lum                         # "enhance": desaturate
night_px    = 0.5 * night_px + 0.5 * night_texture               # tint, night_texture = [0,16,64]
out         = daylight * world + (1 - daylight) * night_px
```

`night_noise_intensity_texture` is `1 - exp(-0.5 (x^2+y^2)/0.5^2)` on a `linspace(-1,1)` grid,
transposed, broadcast to 3 channels — a vignette that is **0 at the frame centre** and rises to the
corners.

Two consequences that shape the implementation:

* **`intensity` is exactly 0 for `daylight >= 0.5`**, so `with_static == world` bit for bit and the
  `select` is redundant there. **The PRNG is needed on, and only on, `light_level < 0.5`.** Computed
  from `calculate_light_level` over a full 300-step day: **58.3% of the cycle is `light >= 0.5` and
  needs no PRNG at all**; 41.7% does. (The 4,000-step rollout reported 87%, biased by resets landing
  at `t=0`, where `light = 0.797`; the analytic number is the honest one.)
* The dusk band `0.5 <= daylight < 1` is **not** the identity — the enhance/tint/blend still run with
  weight `(1 - daylight)`. Only `daylight == 1.0` (3% of the cycle) is the identity the current
  renderer relies on.

### 2.2 The noise draw, and the torch threefry2x32

`jax.random.uniform(key, shape)` under jax 0.11.1 is:

* `threefry_random_bits` -> **`_threefry_random_bits_partitionable`**, because
  `jax_threefry_partitionable` **defaults to `True`** (`jax/_src/config.py:1411`). The counter for
  output element `i` is the 64-bit `i` split into `(hi, lo)` uint32 halves; both go through one
  20-round threefry2x32 block; the result is `bits1 ^ bits2`. It is **elementwise** — element `i`
  depends on `i` and the key only, which is what makes a batched torch version one fused pass.
* `uniform` then randomises the mantissa only:
  `bitcast_f32((bits >> 9) | 0x3f800000) - 1.0`, then `max(minval, f*(maxval-minval)+minval)`.

`scratch/step7/craftax-classic-full/threefry_torch.py` implements this and the pre-0.4.36 `_original`
pairing. Measured against `jax.random.uniform`:

| variant | draws | bit-exact |
|---|---|---|
| partitionable, int64 arithmetic | 64 keys x 7 shapes = **448** (incl. all 64 real rollout keys, both Classic tile sizes, both full-Craftax shapes, an odd element count, and a 1x1) | **448/448**, max `|diff|` 0.0 |
| partitionable, int32 arithmetic | 128 keys x 4 shapes = **512** | **512/512**, max `|diff|` 0.0 |
| pre-0.4.36 `_original` | 448 | **0/448** |

**`BLOCKER` (declaration, not implementation): the night noise field is a function of the jax config
as well as of the state.** The two code paths give different bits for the same key, and which one
runs is decided by `jax_threefry_partitionable`, whose default flipped to `True` in jax 0.4.36. A
parity claim for night frames is therefore a claim *against a declared jax version and flag*. It must
be written into the class declaration and the certification record, not assumed. This is exactly the
class of hidden dependency `docs/research/crafter.md` catalogues under "constants that were measured,
not read".

### 2.3 Chain parity

Feeding the reference's own pre-night world layer into the torch chain, on 64 real states (32 of them
with `light < 0.5`), at both tile sizes:

**`night_bitexact` 64/64, `night_uint8_exact` 64/64, max absolute float32 difference 0.0.**

### 2.4 `state_rng` is observable, so it must be in the state

Re-rendering each state with `state_rng` replaced by `PRNGKey(123456)`:

| | 7 px | 16 px |
|---|---|---|
| night states whose world layer changed | **32 / 32** | **32 / 32** |
| mean channels changed (of 9,261 / 48,384) | 6,706 (**72%**) | 34,928 (**72%**) |
| worst | 8,786 | 45,670 |
| **daylight** states whose world layer changed | **0 / 32** | **0 / 32** |
| strip channels changed, any state | **0** | **0** |

So the reference *is* a pure `f(state)` — `state_rng` is a field of `EnvState` — but only because the
key is part of the state. A render window that omits it cannot draw a night frame.

### 2.5 Cost — `BLOCKER` for the performance arm as prototyped

One Apple arm64 core, one torch thread (**not** a certification number):

| `t` | `B` | world render | threefry field | full night chain | night / world |
|---|---|---|---|---|---|
| 7 | 16 | 0.125 ms (128k f/s) | 2.31 ms | 3.06 ms | **24x** |
| 7 | 256 | 0.715 ms (358k f/s) | 33.6 ms | 44.3 ms | **62x** |
| 16 | 16 | 0.157 ms (102k f/s) | 11.2 ms | 14.9 ms | **95x** |
| 16 | 256 | 1.462 ms (175k f/s) | 181.8 ms | 232.5 ms | **159x** |

Naively bolted on, the night chain would take this env from `CPU_EXCEPTIONAL` back below
`CPU_REGRESSION` — it would undo the Step-6 patch. Two structural reductions are available and both
are measured or derivable, not hoped for:

1. **int32 instead of int64 arithmetic.** Two's-complement wraparound *is* mod 2^32; only the logical
   right shift needs a mask (`(x >> (32-d)) & ((1<<d)-1)` — note the mask is `(1<<d)-1`, **not**
   `(1<<(32-d))-1`; getting that wrong produces a plausible but wrong hash, which is how the first
   version of this prototype failed). Measured: **bit-identical to the int64 path and to jax, 2.3x
   faster** (190.7 -> 82.8 ms at `B=256`, `(112,144)`).
2. **Skip the field where it cannot contribute.** 58.3% of the day needs no draw at all, and the
   chain collapses per pixel to `out = a1*world + a2*lum(world) + c` with `a1, a2` scalars in
   `(daylight, mask)` and `c` a per-pixel RGB — three passes instead of ten. This is the same
   "image-sized float32 temporaries" problem the Step-6 patch removed from the world layer, and it
   wants the same treatment.

Neither is in scope for phase 0. The honest statement for the coordinator is: **night is reproducible
bit-exactly, and making it fast is a separate, measurable piece of work that must be done in the same
patch or the env's frozen CPU tier regresses.**

---

## 3. Sleeping — `FINDING` + `PATCH_CANDIDATE`

### 3.1 It is two changes, not one

The docstring calls this "desaturates the entire frame". It is that, **and** a world-layer change the
current `_Composite` table cannot express:

```
player_texture_index = select(is_sleeping, 4, player_direction - 1)
```

There is a **fifth player texture** (`player-sleep.png`). Measured: it differs from all four facing
sprites (54-60 differing values at 7 px, 278-388 at 16 px), and — the load-bearing property —
`full_map_player_textures_alpha[4]` and `full_map_player_textures[4]` are **exactly 0 away from the
centre tile**, same as slots 0-3. So compositing it into one tile stays the same arithmetic as
compositing it over the frame, and the existing gather is untouched. The alpha set is still
`{0, 128/255, 1}`.

Measured on 64 real states at both sizes:

| | 7 px | 16 px |
|---|---|---|
| centre tile, awake slot, reconstructed bit-exactly | **64/64** | **64/64** |
| centre tile, sleeping slot 4, reconstructed bit-exactly | **64/64** | **64/64** |
| `is_sleeping` world change confined to the centre tile | **64/64** | **64/64** |

**So the world-layer change is exactly: `N_PLAYER` 4 -> 5, `NO_PLAYER` 4 -> 5, table 680 -> 816
entries** (99,960 -> 119,952 B at 7 px; 522,240 -> 626,688 B at 16 px). Nothing else about the gather
moves.

### 3.2 The desaturation

```
lum = 0.299 R + 0.587 G + 0.114 B          # of the POST-night pixels
out = (1 - is_sleeping) * post_night + is_sleeping * (0.5*lum + [0, 0, 8])
```

`0.5 * jnp.array([0,0,16])` is `[0,0,8]`. It runs **after** the daylight blend and **before** the
strip concatenation, so it composes with night and does not touch the strip. On 19 real sleeping
states out of 64: **`sleep_bitexact` 64/64, `sleep_uint8_exact` 64/64, max float diff 0.0** at both
tile sizes.

---

## 4. What the render window must grow by — `FINDING`

| field | float32 slots | bytes |
|---|---|---|
| today (63 blocks + 63 mobs + facing) | 127 | 508 |
| 12 inventory counts | +12 | +48 |
| 4 intrinsics (health, food, drink, energy) | +4 | +16 |
| `light_level` | +1 | +4 |
| `is_sleeping` | +1 | +4 |
| `state_rng` | **+4** | **+16** |
| **total** | **149** | **596** |

**`FINDING`: a uint32 PRNG half does not survive a float32 state row.** Only **2.008%** of random
uint32 round-trip through float32 (2^32 > 2^24). Splitting each uint32 into two 16-bit halves
round-trips **100%** of 100,000 draws. So the key costs **four** slots, not two — and a decoder that
reads it must be `hi*65536 + lo` with each half wrapped into `[0, 65536)`, not a cast.

The growth is **+88 bytes, +17.3%**, and the compression story survives it: **19.98x** the 7 px frame
(was 18x, world-only) and **104x** the 16 px frame — because the frame grows too, from the world
layer to the whole observation.

---

## 5. The two real blockers for certification

### 5.1 `BLOCKER` — night breaks the frozen `check_resolution_consistency` bound

`examples/results/step4/certification.json` records for `CraftaxClassic`
`check_resolution_consistency` `exact_frac = 1.0` against
`tests/gpu/test_reproducibility.py:DOWNSAMPLE_EXACT_FRAC` (`>= 1.0 - 0.005`). Rendering at 32 px and
box-downsampling to 16 px reproduces the 16 px render exactly today, because `_Assets`'s resample
rule `((i+0.5)*src/t).long()` is exact 2x pixel replication at `t = 2*src`.

The noise field is drawn at `(7t, 9t)`, so element `i` of the 32 px field is a **different threefry
counter** from element `i` of the 16 px field. Measured on 16 sampled states, 16 -> 32 px:

| `light_level` | `exact_frac` | `mean_abs` | `max_abs` |
|---|---|---|---|
| 1.00 | **1.000** | 0.00 | 0 |
| 0.75 | **1.000** | 0.00 | 0 |
| 0.45 | **0.282** | 0.49 | 3 |
| 0.10 | **0.035** | 6.00 | **32** |

And it is in genuine tension with a mandatory check: `check_state_dims_matter` requires every state
dimension to reach a pixel, and the four `state_rng` slots reach no pixel unless the sampled pool
contains `light < 0.5` states (section 2.4: 0/32 daylight states changed). **A pool that keeps the
frozen downsample bound leaves four dead state dimensions; a pool that keeps the dimensions live
breaks the frozen bound.**

The disagreement is a property of the *reference* — its noise is resolution-dependent by construction
— not of a port. It cannot be designed away without changing pixels. **The resolution is the
coordinator's**: either re-declare `DOWNSAMPLE_EXACT_FRAC` for this one config with the measured
night number and its provenance, or declare the check `not_applicable` under night with a cited
reason. AGENT_COMMON forbids this workstream from weakening a frozen threshold, so it is reported,
not edited.

**The strip does not have this problem.** Craftax's own strip geometry never scales by exact 2x
(`int(0.8*bps)` is 12 at 16 px and **25** at 32 px; `int(0.6*bps)` is 9 and **19**;
`strip_scales_by_exact_2x` is `false` for 8->16, 16->32 and 32->64), but the repo builds its tables at
the two reference sizes and nearest-resamples the *composited tiles* elsewhere, which is exact 2x
replication. Keeping that rule keeps the strip resolution-consistent.

### 5.2 `BLOCKER` — a real-rollout-only pool leaves 10 of the 16 strip cells dead

Over the whole 4,000-step random-policy rollout, `inventory_max_seen` is `wood 2`, `sapling 1`,
**everything else 0**; the four intrinsics reach the full `0..9`. A random policy never crafts. Built
from that pool alone, ten inventory dimensions are dead and `check_state_dims_matter` fails —
precisely the failure `docs/research/pixel_native.md` records for Procgen miner/heist ("a state pool
must come from real rollouts far enough in ... it looks exactly like a wrong state contract").

The fix is cheap and legitimate, because section 1 proves the strip is `f(count)` per cell and counts
are capped at 9 by the simulator: sample the counts directly over `0..9` in `sample_states`, and
write them back into the template `EnvState` in `reference_render` — the resulting state is a state
the simulator can hold. A scripted crafting policy is the alternative and costs far more for no extra
coverage of the *renderer*.

---


# Part II — the implementation

Phase 1 was authorised in the order proposed and landed as three commits, each with its own tests.

| commit | what |
|---|---|
| `d33288c` | `stable_envs/render/prng.py` — JAX's threefry2x32 counter PRNG in torch, bit for bit, plus 15 tests against golden vectors frozen from jax |
| `538ca69` | the inventory strip and the sleeping player; `state_dim` 127 -> 144 |
| *(this)* | night, with the reductions; `state_dim` 144 -> 149 |

## 6. What was built

### 6.1 `stable_envs/render/prng.py` — shared, minimal, W2-consumable

Interface, as agreed: *(key halves, counter offset, shape) -> uint32 bits / float32 uniform*.

```
threefry2x32(k1, k2, x1, x2)             the 20-round hash, int32, broadcastable
random_bits(keys, n, offset=0)           (B, n) int32 uint32 bit patterns
uniform(keys, shape, minval, maxval, offset=0)   (B, *shape) float32
key_from_halves(halves) / key_to_halves(keys)    the float32-state-row encoding
```

Implements the **partitionable** path only — the one every supported jax uses, and the reason the
whole thing is one fused pass: the counter for element `i` is `i` split into `(hi, lo)` uint32
halves, so element `i` depends on `i` and the key alone.

**The rotation mask is the trap.** After a logical shift right by `32 - d` the result keeps `d`
bits, so the mask is `(1 << d) - 1` and **not** `(1 << (32 - d)) - 1`. The wrong one produces a hash
that looks like noise and is wrong everywhere; the first version of this prototype had it, and
nothing but a golden vector would have caught it.
`test_rotation_mask_is_the_one_that_is_easy_to_get_wrong` asserts it directly.

`tests/data/prng_threefry_golden.json` (64 kB) is frozen from jax 0.11.1 by
`scratch/step7/craftax-classic-full/gen_prng_golden.py`, so **the test needs no jax** — the repo
ships none and no Oscar interpreter has one, so a jax-importing test would skip everywhere and
assert nothing. It records the flag it was frozen under, because a golden vector that does not say
which of jax's two code paths it came from is worthless.

### 6.2 The strip: one table, one gather

The strip's 160 `(cell, count)` cells go into the **same** atlas as the world tiles — 816 world
composites then 160 strip cells, `N_TILE = 976` — so the whole observation is still one
`index_select` at the default appearance. A strip cell is a tile like any other; it is just indexed
by `(cell, count)` instead of by `(player, block, mob)`. A count of 0 draws an entirely black cell,
which is also what the two cells the reference never writes look like, so a blank needs no entry.

The finished **cell** is resampled at non-reference sizes, never the ingredients. That is what keeps
the strip resolution-consistent: craftax's own geometry does not scale by two (`int(0.8*t)` is 12 at
16 px and **25** at 32 px; `int(0.6*t)` is 9 and **19**), so re-laying the cell out at 2S would give
a strip that is not a 2x replication of the S one.
`test_the_strip_atlas_resamples_by_exact_replication` asserts the consequence directly.

### 6.3 The sleeping player: two changes, not one

`is_sleeping` selects a **fifth player sprite** *and* desaturates the world rows. The table grows
680 -> 816 and `player_facing` now wraps **mod four**, because slot 4 must not be reachable from a
facing value. `player-sleep.png` is zero away from the centre tile like the four facing sprites, so
the one-tile composite stays the same arithmetic and the gather is untouched.

### 6.4 Night: `night_and_sleep`, shape-parameterised and shared with W2

```python
night_and_sleep(world, light, sleeping, keys, light_map=None)   # (n, h, w, 3) -> (n, h, w, 3)
night_vignette(device, h, w)                                    # (h, w, 1)
```

Module-level, reading nothing of `CraftaxClassic`'s state layout and taking its shapes from `world`.
`light_map` is the per-pixel multiplier full Craftax applies before the daylight blend and Classic
does not — one argument, not a second function. `test_the_night_chain_is_shape_parameterised_for_the_full_renderer`
exercises it at `(63,77)` and `(90,110)`, which are W2's own window shapes.

The vignette is rebuilt the way craftax builds it — numpy **float64** `exp`, then narrowed to
float32 — rather than with a float32 `exp` that would round differently.

## 7. The three reductions, and the one that was refused

| reduction | status |
|---|---|
| **int32 threefry** | done. Bit-identical to an int64-with-masking implementation and to jax; 2.3x faster (190.7 -> 82.8 ms at B=256, `(112,144)`) |
| **skip the field where `intensity` is exactly 0** | done, and by construction: only the rows with `light < 0.5` are gathered, drawn and scattered back. `intensity = max(2*(0.5 - light), 0)` is exactly 0 above the threshold, so `(1-0)*world + 0*noise` is `world` in float32 whatever the noise was. 58.3% of a 300-step day is in that band |
| **collapse the chain to `a1*world + a2*lum + c`** | **refused, deliberately.** It is the same expression in real arithmetic and a different rounding at every step in float32, and this renderer's claim is bit equality with the reference rather than closeness to it. What was done instead is the same arithmetic **in place**: each result overwrites its own buffer instead of allocating a fresh image-sized float32 temporary, which at B=256 and 16 px is 63.7 MB each. Measured effect: dusk at 16 px went 1,720 -> 2,770 f/s |
| **(extra) the daylight short-circuit** | `light == 1.0` and awake and unit gain keeps the batch on the **uint8 gather with no float buffer at all**, decided by a single host read rather than three. `1.0*world + 0.0*night` is exactly `world`, measured bit-exact on 24 real states at both tile sizes |

## 8. Evidence

### 8.1 Parity — byte-identical, per band

`examples/results/step7/craftax-classic-full/certify/certify_cpu_mac.json`, 64 sampled states,
craftax 1.6.1 / jax 0.11.1 un-jitted:

| tile px | frame | exact_frac | max abs | night (23) | dusk (25) | day (16) | strip rows |
|---|---|---|---|---|---|---|---|
| 7 | 64x63x63x3 | **1.0** | **0** | 1.0 | 1.0 | 1.0 | 1.0 |
| 16 | 64x144x144x3 | **1.0** | **0** | 1.0 | 1.0 | 1.0 | 1.0 |

`digest_ours == digest_official` at both sizes: the frames are **byte-identical**, not equal within
a bound. 25 of the 64 states are sleeping.

### 8.2 "The official frame" is not one object — and the parity claim says which one

Under `jax.jit`, XLA fuses the reference's arithmetic differently and it disagrees with **its own
un-jitted self** by up to 1.5e-05 in float32. By day that is invisible in uint8. Under
`is_sleeping` it is not: the desaturation halves the dynamic range, so the difference straddles a
rounding boundary.

| reference | 7 px | 16 px |
|---|---|---|
| **un-jitted** (the renderer as written) | **100.0000%**, max 0 | **100.0000%**, max 0 |
| jitted | 99.7854%, max 1 | 99.7225%, max 1 |

Every differing pixel is on a sleeping row. `reference_env` therefore returns the **un-jitted**
function and pays 0.25 s a frame instead of 0.008 s, and the declared claim is a **uint8** claim:
this renderer does not assert bit equality of the reference's float, because the reference does not
have one float value to be equal to.

### 8.2b The same frames on two architectures — Oscar, job `6394533`

Oscar's SSH returned mid-session, so this is no longer owed. `$SCR/venvs/se7-craftax` was built
(job `6394425`, 2m32s) — python 3.12, **jax 0.11.1 with `jax_threefry_partitionable = True`**,
craftax 1.6.1, torch 2.14.0+cu130 — and is the **first jax-bearing interpreter on Oscar**, which is
what made this env's parity `blocked_environment` from Step 6 onward.

Job `6394533` on `node2416` (x86-64, `batch`/`normal`, one torch thread, 2m57s, `md5sum` of
`craftax.py` and `prng.py` printed into the log and matching the offered tree):

| | 7 px | 16 px |
|---|---|---|
| `exact_frac` / `max_abs` | **1.0 / 0** | **1.0 / 0** |
| night (23) / dusk (25) / day (16) | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| world rows / strip rows | 1.0 / 1.0 | 1.0 / 1.0 |
| frame digest | `288630d8d823c0b9` | `820c52a906ae839a` |

**Those digests are the same as the Mac's**, so arm64 and x86-64 produce byte-identical frames —
which is the check that matters most for the PRNG, since it is bit-level integer arithmetic and
nothing about torch guarantees it across architectures. The dedicated sweep confirms it directly:
`prng_exactness` re-run on x86-64 gives **64/64 bit-exact on all seven shapes** (both Classic tile
sizes, both full-Craftax shapes, an odd element count, a 1x1, and all 64 real rollout keys), with
the pre-0.4.36 path at **0/64** as on the Mac. The craftax unit tests on that node: **70 passed,
1 skipped**.

Artifact: `certify/certify_cpu_node2416.oscar.ccv.brown.edu.json`.

### 8.3 Gate A — every mandatory check, on both hosts

Identical on the Mac and on `node2416`: `check_state_roundtrip` 0.0 ·
`check_state_dims_matter` **`dead_dims: []`** ·
`check_batch_invariance` 0 · `check_batch_size_invariance` 0 over `[1, 7, 64]` ·
`check_order_independence` 0 / 0 · `check_determinism` identical ·
`check_render_is_pure` 0 on all four probes · `check_chunked_render_matches` 0 over `[1, 7, 64, 4096]` ·
`check_input_form_invariance` no disagreements · `check_set_state_copies` 0 / not rebound ·
`check_style_is_state` 0 · `check_default_style_is_official` 0 · `check_variations_do_something` `[]`.

**`check_colors_apply_exactly` → `colors_not_applied: []`, unchanged.** As asked: this env exposes
no RGB colour leaf (its recolouring knob is `channel_gain`), so the check has nothing to assert and
reports nothing not applied, exactly as before the patch. **`channel_gain` now multiplies the strip
as well as the world**, because it is applied to the whole frame. The reference has no such knob so
there is no behaviour to copy; applying it to the whole frame is the consistent reading, and it is
exactly the identity at its default of 1.0.

**A sampler fix was needed to make `check_state_dims_matter` honest**, and it is worth flagging
because it is the kind of thing that looks like a renderer bug. The check probes `env.B` sampled
states. The four PRNG-key slots reach no pixel above `light_level = 0.5` (0 of 32 daylight states
move when the key changes; 32 of 32 night ones do), and `player_facing` reaches no pixel while the
player is asleep (the reference selects sprite slot 4 whatever the facing is — a true statement
about the renderer, not a defect). With the bands left to the draw, a `batch=1` instance reported
the key slots dead about two times in three and `player_facing` dead about one time in four.
`sample_states` now **stratifies by row index**: the lighting band is assigned `i % 4`, row 0 is
always awake and row 3 always asleep. Measured after: **no dead dimension at batch 1, 2, 3, 4, 8 or
16, over 12 seeds each.** This is the same device as the existing `OUT_OF_BOUNDS` edge band, which
exists so the conformance suite's interior statistic is computable at all.

### 8.4 The Step-4 replay obligation

Per the ruling: the frozen Step-4 record is untouched, and the obligation is world-rows-only
byte-identity at `light = 1.0`, `sleeping = 0`, counts 0. Asserted three ways:

* in the repo — `test_a_step4_shaped_row_reproduces_the_old_world_layer` renders a 127-slot row
  zero-extended to 149 with `light_level = 1.0` and asserts the strip is entirely black and the
  world rows are drawn;
* in `test_daylight_is_the_identity_and_keeps_the_uint8_fast_path`, which asserts the uint8 gather
  and the float chain agree **bit for bit** at full daylight, so "the post-process is the identity"
  is a measurement rather than a code-reading;
* in the harness — `certify.py` records `step4_world_row_replay`: world-row digests at both tile
  sizes, `strip_is_black: true`, mean pixel and distinct colours.

**And the comparison against the frozen record is done, on CPU.** An earlier draft of this section
said it had to wait for the CUDA leg; that was wrong. The frozen `digest_ours` is over a
`(64, 112, 144, 3)` stack, which is exactly the world rows of the grown frame, so it compares
directly. Rendering the frozen Step-4 corpus (`examples/results/step4/states/CraftaxClassic.npz`,
64 x 127) zero-extended to 149 with `light_level = 1.0` and taking rows `[0:112]` at 16 px:

| | replay | frozen Step-4 record |
|---|---|---|
| `frames_digest` | `1aeabbe8bc6a9a2ee1fdb8ffc27a737bd659630dc6e3a1b98f6bae002c894a4d` | **identical** |
| `mean_pixel` | 93.84458414713542 | identical |
| `distinct_colors` | 277 | identical |
| strip rows | entirely black | (did not exist) |

Use `stable_envs.testing.conformance.frames_digest`, not a bare `sha256`: it hashes the shape and
dtype strings before the bytes, so a raw digest of the same pixels does not match and looks like a
failure. The CUDA leg re-takes this on the oracle's own device, which is confirmation rather than
the first evidence.

### 8.5 The new frozen corpus

`examples/results/step7/craftax-classic-full/corpus/` — 64 states from a 256 pool, selected the way
Step 4 selects (typical + per-dimension extremes + farthest-point spread) and then **checked** for
band and sleep coverage rather than hoped: 23 night, 25 dusk, 16 daylight, 30 sleeping, and **every
one of the sixteen strip cells reaches all ten counts 0..9**. `corpus/parity.json` records that both
renderers produce **byte-identical** frames on it at 7 px and 16 px. Proposed for the Step-8 freeze.

## 9. Performance — measured per band, reported honestly

Apple arm64, one torch thread. **Not certification numbers**; the Oscar sweep is `03_sweep_cpu.sbatch`.

| band | 7 px, B=256 | 16 px, B=256 |
|---|---|---|
| **daylight** (`light == 1.0`, awake) | **260,262 f/s** | **132,006 f/s** |
| dusk (`0.5 <= light < 1`) | 14,051 | 2,770 |
| night (`light < 0.5`) | 6,487 | 1,306 |
| **mixed corpus** (the certified figure) | **9,955** | **1,986** |

**The daylight path takes the Step-6 route — and "untouched" was too strong, which independent
review caught.** It is still one gather from a uint8 table with no float buffer and no PRNG, but the
frame grew from 7 to 9 tile rows, so it does not cost what it did. Measured against the Step-6 tree
(`76dc511`) on the same machine at one thread, daylight/awake/unit gain
(`sweep/daylight_baseline.json` vs `sweep/daylight_branch.json`):

| | baseline ms | branch ms | row-scaled baseline | residual | residual share |
|---|---|---|---|---|---|
| B=1, 16 px | 0.0690 | 0.1106 | 0.0887 | +0.0219 | per-call |
| B=16, 16 px | 0.1606 | 0.2248 | 0.2065 | +0.0183 | per-call |
| B=256, 16 px | 1.4396 | 1.8706 | 1.8509 | **+0.0197** | **1.1%** |
| B=1024, 16 px | 5.7163 | 7.2399 | 7.3495 | -0.1096 | none (within noise) |

"Row-scaled baseline" is the Step-6 time times 144/112 — what the same renderer would cost drawing
the taller frame. **The per-frame cost is the row growth; what is left is a flat ~0.02 ms per call**,
which matters only at tiny batches.

It was five times that before review. `_strip_grid` rebuilt two tensors with 32 indexed assignments
on **every** render (39.7 us a call on CPU; ~34 tiny kernel launches and two allocations a frame on
CUDA), and the two post-process colour constants were allocated per call. All four are now cached
per device the way `_VIGNETTE_CACHE` already was, `_decode` does one `round()` over the tail block
instead of one per field, and the PRNG key is assembled from its halves only on a lit row. Review
measured the before as 0.061 -> 0.166 ms at B=1 and 1.460 -> 2.092 ms at B=256/16 px; it is now
0.069 -> 0.111 and 1.440 -> 1.871.

Any *lit* row still costs about 50x (7 px) to 100x (16 px) more than a daylight one, and that is
**structural**: a night frame carries a per-pixel float post-process that the reference performs and
that this renderer must reproduce bit for bit, and no engineering makes a per-pixel float chain as
cheap as a memcpy.

Attribution at B=256, 16 px, all-night (`sweep/bands_cpu_mac.json`): full render 194.8 ms, of which
the threefry field **77.5 ms (40%)** and the rest of the chain ~144 ms including the float gather.

### 9.1 Decision B — one tier, from the protocol's own distribution

Sami's ruling: **one** registered/default tier for `CraftaxClassic`, from the mixed-corpus
throughput; no separate daylight/night environment tiers; report all three plus the ratio and flag
the slower band so the headline cannot hide it.

**The mixed distribution is not a weighting invented for this benchmark.** Step 3's harness draws
`probe.sample_states(64)` (`examples/bench_step3.py:186`), so `CraftaxClassic.sample_states` **is**
the certification distribution. Measured over seeds 0-39 at n=64 and now declared on the class as
`certification_state_distribution`, with a test that fails if the sampler is retuned away from it:

| band | share |
|---|---|
| daylight (`light == 1.0`) | **exactly 0.25** — the band the `i % 4` stratification pins |
| dusk (`0.5 <= light < 1`) | 0.33 – 0.44 |
| night (`light < 0.5`) | 0.31 – 0.43 |
| **lit (anything but full daylight)** | **0.70 – 0.80** |
| sleeping | 0.31 – 0.46 |

So the tier is dominated by the slow band **by construction** — and that is correct, because about
three quarters of Craftax's day is lit. A tier quoted on daylight alone would describe a state the
agent is in a quarter of the time.

**Numbers (GPU, RTX 3090 `gpu2113`, job `6394893`, B=4096):**

| | mixed (**the tier**) | daylight (diagnostic) | night (diagnostic) | daylight/night |
|---|---|---|---|---|
| 16 px | **45,203 f/s** | 796,702 | 24,353 | **32.7x** |

**Slower band, flagged: `light_level < 0.5`.** Why: a per-pixel float post-process the reference
performs — noise field, luminance enhance, tint, daylight blend — against a single uint8 gather at
full daylight.

**Numbers (CPU, Oscar `node2356`, job `6395527`, one torch thread, paired verdict `USABLE`):**

| | mixed (**the tier**) | daylight (diagnostic) | night (diagnostic) | daylight/night |
|---|---|---|---|---|
| 16 px, `bench_step3` certified batch | **1,150.6 f/s** (B=64) | — | — | — |
| 16 px, B=256 (`bench_bands`) | 1,111 | 68,833 | 656 | **104.9x** |
| 16 px, B=1024 | 980 | 105,742 | 389 | 272x |
| 7 px, B=256 | 5,958 | — | — | — |

**Slower band, flagged: `light_level < 0.5`** — a per-pixel float post-process the reference
performs, against a single uint8 gather at full daylight. Attribution at B=256/16 px, all night:
336.8 ms, of which the threefry field 123.5 ms (37%).

**The measurement is paired and gated** (`sweep/paired_verdict_node2356*.json`). The Step-6-closed
tree was measured on either side of the branch in the same job: 165,922 and 165,181 f/s, a
**1.0045x spread**, and **1.44x the 115,166 f/s Step 6 certified for that tree on this cluster**.
Node load 0.86 per core over 192 cores. All three gates pass.

### 9.2 The tier this produces, and why it goes to Sami

`bench_step3` measures the decode bar per run: **4,367 f/s** for the 144x144 frame (5,489 for the
old 112x144 one — the frame grew 1.29x, so the bar fell). On the frozen `NO_REFERENCE` ladder:

| | `cores_cpu` | tier |
|---|---|---|
| **mixed corpus — the registered default** | **1,150.6 / 4,367 = 0.263** | **`CPU_REGRESSION`** |
| daylight only (diagnostic, not a tier) | 105,742 / 4,367 = 24.2 | would be `CPU_EXCEPTIONAL` |

**So the registered-default CPU tier falls from `CPU_EXCEPTIONAL` to `CPU_REGRESSION`, and that is
the coverage change rather than a slower renderer.** The Step-6 tier was measured on an observation
that could not represent night; this one is measured on the full observation, three quarters of
which is lit. The daylight path itself is unchanged (§9). Per the Q3 ruling — "if it falls below
`CPU_FAST` it goes to Sami with numbers" — **it does, and these are the numbers.**

### 9.3 A trap I introduced, and it must not reach a certification

With jax now on Oscar, `bench_step3` can measure the official renderer, and it reports
**`official_cpu` = 2.18 f/s (0.459 s/frame)** for this branch against **610.6 f/s** for the
Step-6 tree. That is not a 280x slowdown in craftax. It is §8.2: `reference_env` now returns the
**un-jitted** renderer, because the jitted one disagrees with itself in uint8 and cannot be the
thing a parity claim is made against.

The consequence is that `S_cpu` — the speedup-over-official ratio the Step-3 ladder prefers when a
reference is available — would read **1150.6 / 2.18 = 528x** for this env, which is nonsense as a
performance statement: nobody runs craftax un-jitted except a parity harness.

**The two comparisons need two different references, and this is the general rule rather than a
quirk of this env:**

* **parity** must use the reference as *written* — un-jitted — because that is the only one with a
  single well-defined answer;
* **a speed ratio** must use the reference as a *user would run it* — jitted — because that is what
  the port is replacing.

Against the jitted official (610.6 f/s, from the baseline leg), `S_cpu` is about **1.9x**, which is
a real number and a modest one. **Recommendation: this env's CPU tier should be taken on the
`NO_REFERENCE` `cores_cpu` ladder (0.263, `CPU_REGRESSION`), and `S_cpu` should be suppressed for it
until the harness can be told which reference a speed ratio is entitled to use.** Left alone, this
would certify `CPU_EXCEPTIONAL` on a 528x speedup against a renderer nobody runs.

### 9.4 The record, in one place

| quantity | value | what it is |
|---|---|---|
| **mixed default, `cores_cpu`** | **0.263** | **the registered-default CPU tier: `CPU_REGRESSION`** |
| mixed default, f/s | 1,150.6 (B=64, 16 px) | the tier's throughput |
| decode bar | 4,367 f/s | measured per run by `bench_step3` for the 144x144 frame |
| daylight diagnostic, `cores_cpu` | 24.2 | **not a tier** — would be `CPU_EXCEPTIONAL` |
| night diagnostic | 656 f/s (B=256, 16 px) | **not a tier** — the flagged slower band |
| daylight / night | **104.9x** | the split the headline must not hide |
| `S_cpu` vs **un-jitted** official | **528x** | 1,150.6 / 2.18 |
| `S_cpu` vs **jitted** official | **~1.9x** | 1,150.6 / 610.6 |

**Neither `S_cpu` number certifies anything until Sami rules on which reference a speed ratio may
use.** The CPU tier above is taken on the `NO_REFERENCE` `cores_cpu` ladder. **No shared harness has
been edited to suppress `S_cpu`** — that is Sami's ruling and Step 8's to encode; both numbers are
recorded so the ruling can be made on them. The same record is carried on the class as
`CraftaxClassic.performance_record_cpu` and `CraftaxClassic.s_cpu_note`, with a test that fails if
the tier and the diagnostic ever stop being visibly different numbers.

For orientation only, one Apple arm64 core (**not** a certification number): mixed 1,990 f/s at
16 px and 9,973 at 7 px; daylight 151,894 and 309,910; night 1,317 and 6,684.

### 9.5 GPU tier — job `6395847`, queued

`bench_step3 --device cuda`, paired the same way: the Step-6-closed tree either side of the branch
in **one allocation on one 3090** (`--constraint=geforce3090`), so the baseline re-measures a known
answer — Step 6 certified it at 1,206,822 f/s at B=4096, and 1,229,787 on a second card. The verdict
script now takes `--device`, with that figure as the CUDA baseline floor.

Queued behind W5's and W6's GPU jobs under the `norm-gpu` cap. The band sweep already taken on
`gpu2113` (job `6394893`) gives **mixed 45,203 f/s, daylight 796,702, night 24,353 at B=4096**; what
is still owed is the **harness's own tier label**, which is why this job exists.

## 10. `check_resolution_consistency` — the frozen bound, and the two options

The frozen `DOWNSAMPLE_EXACT_FRAC >= 1.0 - 0.005` is **not edited**, per the ruling. Night is
shipped and the check records what it records. Measured at 16 -> 32 px on 16 states
(`certify/certify_cpu_mac.json`, `resolution_consistency`):

| pool | `exact_frac` | `mean_abs` | `max_abs` | verdict against the frozen bound |
|---|---|---|---|---|
| all daylight (`light = 1.0`) | **1.000** | 0.00 | 0 | **pass** |
| dusk `light = 0.75` | **1.000** | 0.00 | 0 | **pass** |
| dusk `light = 0.50` | **1.000** | 0.00 | 0 | **pass** |
| night `light = 0.45` | 0.591 | 0.30 | 3 | **fail** |
| night `light = 0.10` | 0.280 | 3.75 | 30 | **fail** |
| the sampler's mixed pool | 0.760 | 1.53 | 34 | **fail** |

**The mechanism, written down as required.** The noise field is drawn at `(7t, 9t)` — the frame's
own resolution — so element `i` of the 32 px field is a **different threefry counter** from element
`i` of the 16 px field. Rendering at 2S and box-downsampling to S therefore averages four
independent samples against one. This is a property of the **reference**: its night frames are
resolution-dependent by construction, and no port can design that away without changing pixels. The
boundary is sharp at `light_level = 0.5` because `intensity` is exactly 0 above it, which is why
every non-night row passes at 1.000. The strip does **not** contribute — it resamples by exact 2x
replication (§6.2).

**The two options, with these numbers, for Sami:**

1. **Re-declare `DOWNSAMPLE_EXACT_FRAC` for `CraftaxClassic` with provenance.** A per-config floor
   of, say, `>= 0.27` at the sampler's band mix, cited to "the reference draws its night noise at
   the frame's own resolution". *For:* the check keeps measuring something — a renderer regression
   that moved a tile would still drop the number. *Against:* the floor is a property of the band mix
   in `sample_states`, so it moves whenever the sampler is retuned, and a bound that tracks the
   sampler is not really a bound.
2. **Mark the check `not_applicable` under night, with a cited reason, and keep the 0.995 floor for
   `light >= 0.5`.** *For:* it is the truthful classification — the quantity the check is defined to
   measure ("does the disagreement stay where geometry is") is undefined when the two renders draw
   *different noise*, which is not a geometry question at all; and the daylight/dusk bands, 58.3% of
   the day cycle, keep the full 1.000 bound with no relaxation anywhere. *Against:* it removes
   coverage from 41.7% of the state space, and `not_applicable` is a heavier hammer than a number.

**Sami's ruling (A): neither option may be chosen from our implementation alone.** The verdict has
to come from an **oracle experiment against the official renderer** — same states, same protocol,
same reference as parity — and only then does the applicability follow. That experiment is §14.

## 11. What is owed, and what is ready for it

**Oscar: still BLOCKED** — `ssh -o BatchMode=yes sboughan@sshcampus.ccv.brown.edu` returns
`Permission denied (keyboard-interactive)`. Checked once more at the end of this session, not
retried in a loop. Four sbatch scripts are written, versioned under
`scratch/step7/craftax-classic-full/`, and share `env_header.sh`, which redirects every cache
(`UV_CACHE_DIR`, `PIP_CACHE_DIR`, `XDG_CACHE_HOME`, `TORCH_HOME`, `HF_HOME`, `WARP_CACHE_PATH`,
`MPLCONFIGDIR`) under `$SCR` because **HOME is at 100% of its quota**:

| script | what it owes |
|---|---|
| `01_build_venv.sbatch` | `$SCR/venvs/se7-craftax` — python 3.12, `jax[cpu]==0.11.1`, `craftax==1.6.1`, torch 2.14.0. **No existing Oscar venv has jax**, which is why Step 6 recorded this env's parity as `blocked_environment`; a new venv, never an edit to a shared one |
| `02_certify_cpu.sbatch` | Gate A + parity on a batch node at one thread, plus the PRNG exactness sweep re-run on **x86-64** — this is bit-level integer arithmetic and arm64/x86 agreement is checked, not assumed |
| `03_sweep_cpu.sbatch` | the CPU tier per band and on the mixed corpus, with the `$SCR/se6/baseline` leg in the same job on the same node |
| `04_gpu.sbatch` | the CUDA leg of Gate A and the GPU tier, before and after in **one allocation on one card**. It re-takes the frozen Step-4 world-row digest comparison on the oracle's own device; that comparison already passes on CPU (§8.4), so this is confirmation rather than the first evidence |

Job names all start `se7-craftax-classic-full-`. One GPU job at a time; never cancel a job this
workstream did not submit, and never a `simclr_cifar10` job.

**Gate F.** Re-measured as far as it can be locally: `StableEnvGym('CraftaxClassic')` reset gives a
**149**-dim observation (was 127), `render()` returns **(144, 144, 3)** (was (112, 144, 3)), and the
`info` keys are unchanged (`distance_to_target`, `env_name`, `goal`, `goal_state`, `proprio`,
`state`). `step()` still raises by design — this env ships a renderer and no batched simulator, so
its Gate F path is `swap_renderer`, which keeps the official env's dynamics. The full evidence run
through `examples/swm_compat/` needs `stable_worldmodel`, which is in **no** local venv; it is
listed with the Oscar work.

## 12. Two shared files were touched, both additively

Flagged per AGENT_COMMON rather than left to be discovered:

Independent review confirmed both weaken nothing.

* **`examples/fidelity_step4.py`** records `jax` among `reference_packages`. Without it the declared
  `jax` reference version could never be confirmed and `provenance.reference_version_mismatch` would
  report "version not recorded by the measuring interpreter" — which is judged `blocked_environment`
  — forever. One entry in a tuple; no other env declares `jax`, so no other env's judgement can move.
* **`tests/unit/test_declarations.py`** reads `parity_note` as the corroborating prose for a
  **parity** env's `reference_versions`. The check previously required
  `visual_fidelity["measured_at"]`, which `check_fidelity_declaration` **forbids** a parity env from
  carrying — so a parity env could not declare a reference version at all. The rule is unchanged: a
  version may not be declared without prose in the class that says the same thing.

## 12b. Decision A — the oracle experiment, and what it settled

Sami's ruling: neither "split bound" nor "not applicable" may be chosen from our implementation
alone. The verdict has to come from running the check's exact protocol on the **official** renderer.

**Protocol** (`scratch/step7/craftax-classic-full/oracle_resolution.py`, artifact
`oracle/resolution_oracle_S16.json`): the frozen Step-7 corpus split by band, plus
`check_resolution_consistency`'s own draw (`render_worker.sample_inputs(env, 8)` at `STATE_SEED`,
default appearance) so the states the frozen bound is judged on are covered; craftax's renderer
**un-jitted**, the same reference as parity, with each state's own `state_rng` written back so both
resolutions see the same PRNG state; craftax's own `load_all_textures_given_size(32)` for the 2S
textures — craftax caches only 7, 16 and 64, so the renderer is built through a patched lookup
rather than a reimplementation, and every pixel is still craftax's; then the check's own
`_box_downsample` and its own statistic.

**Result — `exact_frac` at S=16, 2S=32, split by region because the two regions fail for different
reasons:**

| pool | official world | **ours** world | official strip | **ours** strip | parity vs official @16 |
|---|---|---|---|---|---|
| corpus daylight (16) | 1.0000 | **1.0000** | 0.6933 | **1.0000** | 1.0000 |
| corpus dusk (25) | 1.0000 | **1.0000** | 0.6928 | **1.0000** | 1.0000 |
| corpus night (23) | **0.1634** | **0.1634** | 0.6815 | **1.0000** | 1.0000 |
| the check's own 8-state draw | 0.7639 | **0.7639** | 0.6743 | **1.0000** | 1.0000 |
| that draw, night rows only (2) | 0.0557 | **0.0557** | 0.6946 | **1.0000** | 1.0000 |

Two independent mechanisms, each confined to one region:

1. **The world rows, at night: the official renderer is not resolution-consistent, and this renderer
   is inconsistent by *exactly* the same amount — 0.1634 against 0.1634, 0.0557 against 0.0557.**
   Not "also bad": the same number, which is what reproducing the reference's noise field bit for
   bit implies. The noise is drawn at `(7t, 9t)`, so element `i` of the 32 px field is a different
   threefry counter from element `i` of the 16 px one, and the quantity the check is defined to
   measure — "does the disagreement stay where geometry is" — is undefined when the two renders draw
   different noise. It is not a geometry question at all.
2. **The strip rows: the official is inconsistent in *every* band, daylight included (0.68–0.69),
   and this renderer is at 1.0000.** That is the `int(0.8*t)` / `int(0.6*t)` re-layout — 12 and 9 at
   16 px, 25 and 19 at 32 px. Resampling the finished cell rather than re-laying it out (§6.2) makes
   this renderer *stricter* than the reference at 2S, at a size craftax **ships no textures for**,
   and it is what keeps the frozen daylight bound meetable at all. Matching craftax's 32 px
   re-layout would break that bound and buy no parity.

   **"No shipped textures at 2S" is the accurate phrasing, not "no official frame"** — a wording the
   re-review corrected, and worth stating precisely because the loose version implies the two
   renderers disagree at 32 px and they do not. craftax's own `load_all_textures_given_size(32)`
   builds a perfectly good 32 px renderer, and measured against it on the frozen corpus
   (`oracle/parity_at_2S.json`):

   | band | world rows exact | world max abs | strip rows exact | strip differing |
   |---|---|---|---|---|
   | daylight | **1.000000** | **0** | 0.820404 | 17.96% |
   | dusk | **1.000000** | **0** | 0.820119 | 17.99% |
   | night | **1.000000** | **0** | 0.814111 | 18.59% |

   The world rows are exact against craftax at 32 px **in every band, night included**. The entire
   2S divergence is the strip, ~18% of its pixels, by design. 32 px is not a declared parity size
   because there is no shipped art to certify against — not because the frames would disagree.

**Ruling, following Sami's rule:** the official renderer *is* itself resolution-inconsistent under
night, so `check_resolution_consistency` is **NOT APPLICABLE below `light_level = 0.5`**, cited to
that official-reference result. Where the official is consistent — the world rows at dusk and
daylight — **the bound stays asserted and is met exactly (1.000)**. The frozen Step-2 threshold is
not edited; the per-band applicability is declared on the class as
`CraftaxClassic.resolution_consistency_applicability`, carrying the evidence path and the reference
versions, and `test_the_declared_resolution_applicability_matches_measurement` fails if the
declaration goes stale.

**Exact parity at both supported resolutions is unaffected and still required**: the same experiment
records `exact_frac` 1.0 against the official at 16 px in every band, and §8.1 records the same at
7 px.

### 12b.1 What Step 8 must do with this, and what happens until it does

**The declaration is not yet wired into the certifier, and the consequence belongs in the record
rather than in a later surprise.** `certify.py` runs `check_resolution_consistency` on the check's
own mixed 8-state protocol draw, which contains night rows, and gets **0.8164** — the official
renderer gets **0.7440** on the same draw. Against the frozen `DOWNSAMPLE_EXACT_FRAC >= 0.995` that
**records as a FAIL on this config**, for a reason the oracle shows belongs to the reference and in
a band where the check is declared not applicable.

Until the Step-8 certifier either reads `CraftaxClassic.resolution_consistency_applicability` or
judges this check on the `light_level >= 0.5` subset, that failure will appear in the record. It is
a **certifier gap, not a renderer regression**, and the evidence to close it is already frozen here:
`oracle/resolution_oracle_S16.json` has the official renderer at 0.1634 on the world rows where this
one is at 0.1634. **The frozen threshold must not be edited to make it go away.**

One thing the experiment turned up that the framing had not anticipated: the strip makes the
official renderer fail its own consistency check **in daylight**, which is the band everyone assumed
was clean. Had the strip been re-laid-out to match craftax at 32 px, this env would have gone from
`exact_frac` 1.000 to ~0.69 at daylight and the regression would have looked like a night problem.

---

## 13. Independent review — four required changes, all landed

Review of `a1ebf57`: **APPROVE-WITH-CHANGES**. The reviewer regenerated the golden file from jax
(byte-identical), ran a fresh 324-case `random_bits`/`uniform` probe (all exact), reproduced parity
`1.0 / 0` on their own boundary pool and these digests, replayed the frozen Step-4 corpus on CPU,
and confirmed the two shared-file edits weaken nothing.

1. **`prng.uniform`'s range arguments were not bit-exact against jax.** jax's jitted
   `uniform(key, shape, f32, minval, maxval)` contracts the scale into a **fused multiply-add**;
   508 of 4096 values differ from `uniform() * 95 + 32` for the span `[32, 127)` at `(32, 127)`,
   while a power-of-two span agrees — exactly the coincidence that lets such a bug survive a test.
   Parity was never affected (craftax writes the scale itself and un-jitted), but W2 consumes this
   module. **The arguments are removed**: a caller reproduces a range by writing the expression, and
   `test_the_scaled_night_field_matches` pins that expression against jax-generated values.
   `test_uniform_takes_no_range_arguments` keeps them from coming back.
2. **"Daylight untouched" was false.** Fixed and re-measured; see §9.
3. **"Every jit-differing pixel is on a sleeping row" over-generalised.** The reviewer found the
   jitted reference differing on an **awake** row at `light_level = 0.49999997` (45 px, max 1). The
   wording in `parity_note` and both docstrings now says *on this pool*, and gives the mechanism:
   a ~1e-5 float32 shift meeting a uint8 rounding boundary, which **any lit row** can straddle
   because any lighting compresses the range. The un-jitted-reference and uint8-claim decisions are
   unchanged.
4. **The frozen Step-4 digests could be compared on CPU all along.** Done; see §8.4.

Recorded, non-blocking:

* `certify.py`'s Gate A shows `check_per_env_variation` **skipped (batch too small)** because it
  runs at the env's default batch. Step 8 should run it at the corpus batch.
* SWM's `render()` truncates on the `x255`, where this repo rounds half-to-even. Pre-existing,
  affects every env in the shard equally, out of scope here.
* arm64-vs-x86 threefry agreement is **settled**: identical frame digests on both hosts, int32
  arithmetic throughout, and no FMA anywhere in the hash.

## 14. Report per AGENT_COMMON

* **tests run:** `tests/unit` — **832 passed, 37 skipped**, 2 deselected. The two deselected are the
  pre-existing failures Step 6's `tilemap` findings §6 already routed to the coordinator
  (`test_every_cited_scratch_path_resolves`, whose unresolved citations are *not* this workstream's,
  and `test_examples.py::…[robocasa_visual/render_mjwarp.py]`). `tests/unit/test_nvdiff_texgen.py`
  cannot import on this Mac (`MUJOCO_GL=egl`) and was excluded. New: 15 PRNG tests, 39 strip/night
  tests; `test_craftax_atlas.py` extended to the sixth player slot, the strip table and batches that
  mix sleeping/awake and all three lighting bands.
* **tiers before/after:** unchanged on the daylight path; see §9 for the band split. Certified
  numbers are owed on Oscar.
* **pixels changed:** yes, by design — the observation grows from the world layer to the whole
  144x144 frame. The **world rows of a Step-4-shaped state are unchanged** (§8.4).
* **semantic follow-up:** a **paired Gate H**, not a tripwire. The observation gains a whole
  information channel a world model can read, and night and sleep change the global statistics of
  every frame in the pool. That is a coverage change rather than pixel drift, and a tripwire is the
  wrong instrument for it.
* **SLURM job IDs:** none — Oscar blocked.
