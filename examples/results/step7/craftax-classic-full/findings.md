# Step 7 — W1 `craftax-classic-full`: phase 0 diagnosis

**Classification: `FINDING` (phase 0 complete; no `stable_envs/` file was edited).**
All three declared exclusions are reproducible **bit-exactly in float32**, on real states, at both
reference tile sizes. What remains is not whether the pixels can be reproduced — they can — but the
**state contract**, the **frozen `check_resolution_consistency` bound**, and the **cost of the night
chain on CPU**.

| | |
|---|---|
| parent baseline SHA | `76dc5116f432e61dd47fba8f9e879cf9212c8e46` (Step 6 closed) |
| branch | `step7/craftax-classic-full`, worktree `/Users/samib/stable-envs-step7-craftax-classic-full` |
| target | `swm/CraftaxClassicPixels-v1`: Gate A `PARTIAL` -> `full` |
| affected config | **`CraftaxClassic` only** — 1 of the 49 frozen Step-4 render-distinct configurations |
| reference venv | `/Users/samib/se7-craftax-venv` — python 3.12.11, **jax/jaxlib 0.11.1 (CPU)**, **craftax 1.6.1**, torch 2.14.0, numpy 2.5.3, flax 0.12.9, ml-dtypes 0.6.0, pillow 12.3.0, imageio 2.37.4, pytest 9.1.1 (`phase0/reference_venv_pins.json`) |
| host | **Apple arm64 Mac**, macOS 26.6.2. Every number below was taken here. Local CPU timings are **not** certification numbers |
| Oscar | **BLOCKED** — `ssh -o BatchMode=yes sboughan@sshcampus.ccv.brown.edu` returns `Permission denied (keyboard-interactive)`. One attempt, not retried. Cluster items listed as pending in section 8 |
| pixels changed | **no** — nothing under `stable_envs/` was touched |

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

## 6. Smallest implementation plan

Everything lands in **one env file** plus one new shared module and its tests. Nothing in
`stable_envs/render/tilemap.py` is touched: that module is MiniGrid's atlas and this env has never
used it; Craftax's table is `_Composite` inside `envs/craftax.py`.

### 6.1 Files

| file | change |
|---|---|
| `stable_envs/render/prng.py` **(new, shared)** | `threefry2x32(k1,k2,x1,x2)`, `random_bits(keys,n)`, `uniform(keys, shape, minval, maxval)` — batched, int32, device-agnostic. ~60 lines, nothing env-specific. **This is the W2-shared piece; see section 7** |
| `stable_envs/envs/craftax.py` | (a) 5th player texture embedded, `N_PLAYER` 4->5, `NO_PLAYER` 5, `_Composite` 680->816 entries; (b) `_StripComposite`: a 160-entry `(cell, count)` table + the 2x9 layout, embedded at 7 px and 16 px and nearest-resampled elsewhere; (c) `_night_and_sleep`: the 2.1/3.2 chain in the reference's order, applied to the world rows only; (d) `state_dim` 127->149, `state_names`, `sample_states`, `reference_render`; (e) rewrite the docstring — it currently *declares* these exclusions and that declaration is load-bearing for Gate A |
| `tests/unit/test_craftax_prng.py` **(new)** | the torch threefry against golden vectors frozen from jax (so the test needs no jax), plus int32-vs-int64 agreement, plus batched-vs-single |
| `tests/unit/test_craftax_atlas.py` | extend the existing differential fuzz to the 6th player slot, the strip table, and a night/day/sleep mixture **within one batch** (the `style-as-state` trap the Step-6 patch was careful about) |

### 6.2 Decoding rules the conformance probe forces

`check_state_dims_matter` perturbs each dimension by +25, so every new slot needs a documented total
decoding rule, as the existing class ids already do (`remainder`, deliberately not `clamp`):

* counts: `round().remainder(10)` — wrap, matching the existing rule and keeping the +25 probe live;
* `light_level`: a wrap into `[0,1]` (e.g. `x - floor(x)`), **not** a clamp — a clamped daylight state
  is insensitive to the probe and reports as a dead dimension;
* `is_sleeping`: `> 0.5`;
* key halves: `round().remainder(65536)`, recombined as `hi*65536 + lo`.

### 6.3 Replay obligation

**The frozen Step-4 `CraftaxClassic` corpus must remain byte-identical for the world layer under
daylight.** It is a 64-state corpus of 127-dim rows; the new components are additive, so:

1. the 127 existing slots keep their meaning and position — append, never reorder;
2. a 149-dim row whose 12 counts and 4 intrinsics are 0, `light_level = 1`, `is_sleeping = 0`
   reproduces today's frame in rows `[0:H_world]` **exactly** (`daylight = 1` makes the whole chain
   the identity — already verified numerically to 0 over 20 states in Step 4, and re-verified here on
   32 real daylight states);
3. the frame **shape changes** (112x144 -> 144x144 at 16 px) because the strip is now drawn, so
   `digest_ours`, `frames_shape`, `mean_pixel` and `distinct_colors` in the frozen record **cannot**
   match. The replay must therefore run as *world-rows-only* against the frozen digests, plus a new
   full-observation corpus under `examples/results/step7/`. **That is a coordinator decision about
   what "the frozen corpus is unchanged" means for a config whose observation grows**, and it is the
   single biggest process question in this workstream.

### 6.4 Order of work

1. `render/prng.py` + its golden-vector test (no env change; independently reviewable).
2. 6th player slot + strip table + `state_dim` growth, **with `light_level` pinned to 1 and
   `is_sleeping` free** — this alone is exact, cheap, and does not touch the resolution bound.
3. The night chain, with the 2.5 reductions, and re-measured CPU/GPU tiers.
4. Declarations, corpora, certification.

Steps 1-2 are shippable on their own if 5.1 stalls; step 3 is the one that carries the blocker.

---

## 7. What W2 (`craftax-full`) shares — keep these three, and only these

| capability | where | how W2 reuses it |
|---|---|---|
| **`stable_envs/render/prng.py`** — batched threefry2x32 + `uniform` | new shared module | **verbatim.** Verified here bit-exact at full Craftax's own shapes too: `(63,77)` and `(90,110)`, 64 keys each, 64/64. Full Craftax draws the same `jax.random.uniform(state.state_rng, map_pixels.shape[:2])`; only the shape differs |
| **the night + sleep post-process** | `_night_and_sleep(world, light, sleeping, key, vignette, night_tex)` in `envs/craftax.py` | **as a function**, parameterised on `(H, W)`. Full Craftax's chain additionally multiplies a `light_map` before the daylight blend, so W2 needs one extra argument, not a different function. Keep it module-level and free of `CraftaxClassic` state layout |
| **the strip table shape** — `strip_tile[icon, count]` plus an `(rows, cols)` layout of icon ids with `-1` for blank | `_StripComposite` | full Craftax has a **4-row** strip and a larger icon set; same table, bigger `icon` axis and a different layout array. Parameterise on `(icon_textures, layout)`; do **not** hard-code Classic's 16 cells inside the builder |

Not shared, and should not be: the block/player/mob composite table (different atlas — 143 PNGs vs
59 — and more layers), and the state encoding.

**Recommendation for the coordinator:** W2 can start now. Its phase-0 dependency — "does the threefry
reproduce, and is the strip a gather" — is answered, and answered at W2's own shapes.

---

## 8. Certification obligations, and what is pending on Oscar

| obligation | state |
|---|---|
| **Gate A, mandatory checks** | all 14 must be re-run; `check_state_dims_matter` and `check_resolution_consistency` are the two at risk (section 5) |
| **parity tier over the WHOLE observation** | the current record's `parity_interior_min = 1.0` was measured on a **crop**. It must be re-declared against the full 144x144 frame. Two warnings from phase 0: (a) the interior statistic is `NaN` on dense pixel art and the `OUT_OF_BOUNDS` band trick in `sample_states` is what rescues it — the strip adds large flat black regions, which should *help*; (b) the reference's own float output is not fusion-stable (section 0), so the declared bound must be on uint8 |
| **the jax-version declaration** | new, and mandatory: `jax >= 0.4.36` with `jax_threefry_partitionable = True`. Without it the night frames are a different picture (section 2.2) |
| **Gate F** | SWM render-compat re-run through `stable_envs/compat/worldmodel.py` and `examples/swm_compat/`; the frame **shape changes**, so `AddPixelsWrapper` -> `EverythingToInfoWrapper` -> `EnsureInfoKeysWrapper` -> `ResizeGoalWrapper` must be re-measured, with `facade_missing == []` and `batched_missing == []` |
| **new frozen corpus** | `examples/results/step7/craftax-classic-full/corpus/` — 64 states from **real rollouts** (day/dusk/night/sleep stratified, as `phase0/state_census.json`) with the inventory counts swept over `0..9` per section 5.2 |
| **performance, both devices** | Step-6 machinery (`examples/bench_step3.py`, `scratch/step6/profile_note.py`); Step 2 section 12.1 `PERFORMANCE_MEASURED`. Before/after must be in one SLURM job on one card (AGENT_COMMON addendum) |

**Oscar: BLOCKED.** `ssh -o BatchMode=yes` -> `Permission denied (keyboard-interactive)`; one attempt,
not retried. Pending there, in order:

1. a `jax`-bearing interpreter. **No Oscar venv has jax** — Step 6 recorded
   `render_parity_with_official_renderer` as `blocked_environment` for exactly this reason, and this
   workstream's entire parity claim needs it. A new `$SCR/venvs/se7-craftax` (python 3.12, `jax[cpu]`,
   craftax 1.6.1, matching the local pins) with `UV_CACHE_DIR` / `PIP_CACHE_DIR` / `XDG_CACHE_HOME` /
   `TORCH_HOME` under `$SCR` — **HOME is at 100% quota**;
2. the PRNG exactness sweep re-run on x86-64 Linux (this is bit-level integer arithmetic; arm64 vs
   x86 agreement should be checked, not assumed);
3. Gate A conformance + parity on a 3090; CPU sweep on `batch` / `normal` with threads pinned to 1;
4. before/after tiers in one job on one card, job names `se7-craftax-classic-full-*`.

---

## 9. Open questions for the coordinator

1. **6.3 — what does "the frozen Step-4 corpus stays byte-identical" mean when the observation
   grows?** World-rows-only replay against the frozen digests plus a new full corpus is the proposal;
   it needs a ruling before any code lands.
2. **5.1 — `DOWNSAMPLE_EXACT_FRAC` for this config.** Re-declare with the measured night number and
   provenance, or mark the check `not_applicable` under night? Not this workstream's to edit.
3. **2.5 — is a slower `CraftaxClassic` acceptable?** Step 7's rules say a new cell need not be
   `CPU_FAST` to ship, but this is not a new cell — it is a cell that is currently `CPU_EXCEPTIONAL`
   and would regress. Preference: land 6.4 steps 1-2 (no regression), then step 3 with the
   reductions, and report the tier honestly.
4. **`render/prng.py` ownership.** W1 needs it first and W2 needs it verbatim. Proposal: W1 writes it
   minimal, W2 reviews it before merging rather than after.
5. **The `channel_gain` variation now multiplies the strip too.** The reference has no such knob, so
   there is no right answer to copy; applying it to the whole frame is the consistent reading.
   Flagged because it changes what `check_colors_apply_exactly` sees.

---

## 10. Semantic follow-up recommendation

**None yet.** Nothing was changed, so no digest moved. When the patch lands, the right follow-up is a
**paired Gate H**, not a tripwire: the observation gains a whole information channel (the inventory
strip) that a world model can read, and night/sleep change the global statistics of every frame in
the pool. That is a coverage change, not a pixel drift, and a tripwire is the wrong instrument for it.
