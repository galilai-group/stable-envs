# Step 7 — W2 `craftax-full`, phase 0

**Classification: `PATCH_CANDIDATE`**, with **two `BLOCKER`s that are rulings, not code** (§8) and one
finding that belongs to the coordinator rather than to this workstream (§8.3).

Target: `swm/CraftaxPixels-v1` — full Craftax, nine levels, a 130x110 observation, a 143-PNG atlas,
mobs / projectiles / player / `light_map` layers.

Phase 0 did not stop at a census. **Every layer of the official renderer was reconstructed in torch
and measured bit-for-bit against the reference's own intermediates on 49 real states**, so the phase-1
design rests on measurements rather than on a reading of the source:

| layer | reconstruction | result |
|---|---|---|
| world window (blocks, items, player, mobs, projectiles) | per-tile **5-integer stencil composite** | **49/49 bit-exact, max abs 0.0** |
| lighting (`light_map` -> night -> sleep) | craftax's chain op for op, W1's `prng` unchanged | **49/49 bit-exact, max abs 0.0** |
| 4-row inventory strip | per-cell **layered stencil** (icon, ones digit, tens digit, enchantment) | **49/49 bit-exact, max abs 0.0** |

| | |
|---|---|
| parent baseline | `76dc5116f432e61dd47fba8f9e879cf9212c8e46`; branch cut from `step7/integration` at `ada4075` (W1 + W4 merged) |
| branch / worktree | `step7/craftax-full` · `/Users/samib/stable-envs-step7-craftax-full` |
| reference venv | `/Users/samib/se7-craftax-venv` — jax/jaxlib **0.11.1** (`jax_threefry_partitionable = True`), craftax **1.6.1**, torch 2.14.0, gymnasium 1.3.0 |
| host | Apple arm64 Mac, one thread. **Local CPU timings are not certification numbers** |
| Oscar | **reachable**; `$SCR/venvs/se7-craftax` verified (jax 0.11.1, partitionable `True`, craftax 1.6.1, torch 2.14.0+cu126). No jobs submitted in phase 0; three sbatch scripts written (§10) |
| shared files edited | **none** |

Artifacts, all under `examples/results/step7/craftax-full/phase0/`:
`census.json` · `state_census.json` · `purity_and_state_window.json` · `layers.json` ·
`composite.json` · `chain.json` · `strip.json` · `resolution_and_dead_dims.json` · `bars.json` ·
`window.json`.
Probes: `scratch/step7/craftax-full/{harvest,probe_census,probe_layers,probe_composite,probe_chain,probe_strip,probe_purity_state,probe_resolution,probe_bars,probe_window}.py`
plus `instrumented_renderer.py`. The state pool (`_pool.pkl`, 27 MB) is **not committed**; `harvest.py`
regenerates it deterministically from seed 20260915.

---

## 1. Registration and the observation contract

`stable_worldmodel/envs/__init__.py` registers `swm/CraftaxPixels-v1` at
`stable_worldmodel.envs.craftax.craftax:CraftaxPixelsWrapper`, which is
`make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=False)` with the observation space
**hard-coded to `(130, 110, 3)`** — the wrapper's own comment says craftax's declared
`observation_space` "has H and W swapped, so we cannot trust it" (it does not, at this size, but the
hard-coding is what the id actually contracts).

| fact | value | where measured |
|---|---|---|
| craftax env | `Craftax-Pixels-v1`, `auto_reset=False` | `stable_worldmodel/envs/craftax/craftax.py:123` |
| renderer | `make_craftax_pixel_renderer(BLOCK_PIXEL_SIZE_AGENT)` | `craftax/craftax/envs/craftax_pixels_env.py:27` |
| observation | `render_craftax_pixels(state) / 255.0`, float32 in `[0, 1]` | `craftax_pixels_env.py:62` |
| world window | **9 x 11 tiles** (`OBS_DIM`) | `craftax/craftax/constants.py:15` |
| strip | **4 rows** x 11 columns (`INVENTORY_OBS_HEIGHT`) | `constants.py:21` |
| tile sizes craftax ships | **10** (agent, the registered one), 16 (img), 64 (human) | `constants.py:18-20` |
| frame at 10 px | `(9 + 4) * 10 = 130` rows x `11 * 10 = 110` cols x 3 = **42,900 B** | `census.json` |
| frame at 16 px | 208 x 176 x 3 = **109,824 B** | `census.json` |
| action repeat | **none**; `step` and `reset` are `jax.jit`-wrapped by the SWM wrapper | `craftax.py:56` |
| levels | **9** (`StaticEnvParams.num_levels`), maps 48 x 48 | `craftax_state.py` |

**The observation is the whole frame** — world window *and* strip *and* night — not a crop, and it is
the `/255` of the renderer's float32 output. SWM's `render()` then does `clip(obs,0,1)*255` and
**truncates**; this repo rounds half-to-even. That is the same pre-existing, shard-wide discrepancy W1
recorded and is out of scope here.

**How the level shows in the frame.** Three separate ways, and all three must be in the state window:
the per-tile ids come from `map/item_map/light_map[player_level]`; the **level digit** is drawn at
strip cell `(x=6, y=0)`; and `daylight = select(player_level == 0, light_level, 1.0)` — see §3.3.

---

## 2. Full-renderer census

### 2.1 Draw order and layers

`render_craftax_pixels` (craftax 1.6.1, `craftax/craftax/renderer.py:201-912`), in order:

| # | stage | what it is | pure gather? |
|---|---|---|---|
| 1 | block layer | `scan` over 37 `BlockType`s, `pixels + full_map_block_textures[i] * (idx == i)` | **gather** (one-hot accumulate; `x + 0.0 == x`) |
| 2 | item layer | 4 `ItemType`s (`TORCH`, `LADDER_DOWN`, `LADDER_UP`, `LADDER_DOWN_BLOCKED`), RGBA, alpha over | **stencil select** (alpha binary) |
| 3 | player | `player_texture_index = select(is_sleeping, 4, player_direction - 1)`, one of **5** sprites | **stencil select**, one tile |
| 4 | melee mobs | 3 slots, `type_id` in `0..7` (8 sprites), `dynamic_slice`/`dynamic_update_slice` per mob | **stencil select**, one tile each |
| 5 | passive mobs | 3 slots, 3 sprites | as above |
| 6 | ranged mobs | 2 slots, 8 sprites | as above |
| 7 | mob projectiles | 3 slots, 8 sprites x 4 orientations (flip / transpose) | as above |
| 8 | player projectiles | 3 slots, same 8 x 4 | as above |
| 9 | `light_map` | per-tile float in `[0,1]`, `pixels = light_map_pixels[:,:,None] * pixels` | **per-pixel arithmetic** (tile-constant) |
| 10 | night | noise field, vignette mask, tint, daylight blend | **per-pixel arithmetic** |
| 11 | sleep | `sleep_level = 1 - is_sleeping * 0.5`; `pixels * sleep_level` | **per-pixel arithmetic** |
| 12 | strip | 36 written cells over a 4 x 11 grid, then `concatenate([map_pixels, inv_pixels])` | **stencil select** per cell |

Two **global substitutions** happen before the gather and are resolved into the stored tile id:

* `NECROMANCER -> NECROMANCER_VULNERABLE` when `melee_mobs.mask[level].sum() == 0` **and**
  `ranged_mobs.mask[level].sum() == 0` **and** `boss_timesteps_to_spawn_this_round <= 0` — note this
  reads **off-screen** mobs. Measured: 274 pixels move, **confined to the necromancer tiles**.
* `LADDER_DOWN -> LADDER_DOWN_BLOCKED` when `monsters_killed[player_level] < 8`. Measured: 168 pixels
  move, **confined to the ladder tile**.

Both are therefore absorbed by storing the *resolved* id; neither needs its own state dimension
(`layers.json`).

### 2.2 The atlas

143 PNGs; 47 texture-dict keys; **6.82 MB as float32 at 10 px** (`census.json`). What the renderer
actually indexes:

| family | shape at 10 px | entries |
|---|---|---|
| `block_textures` | `(37, 10, 10, 3)` | 37 `BlockType` |
| `full_map_item_textures` | `(5, 90, 110, 4)` — item `i` tiled over the frame (**asserted tiled**) | 5 `ItemType` |
| `player_textures` | `(5, 10, 10, 4)` (and a zero-padded full-frame copy) | 4 facings + `player-sleep` |
| `melee_mob_textures` / alphas | `(8, 10, 10, 3)` | 8 |
| `passive_mob_textures` / alphas | `(3, 10, 10, 3)` | 3 |
| `ranged_mob_textures` / alphas | `(8, 10, 10, 3)` | 8 |
| `projectile_textures` / alphas | `(8, 10, 10, 3)` | 8 x 4 orientations |
| `smaller_block_textures` | `(37, 8, 8, 3)` — the strip's `int(0.8*t)` icons | 37 |
| `number_textures{,_with_zero}` + alphas | `(10, 4, 4, 3)` | 10 each |
| ~25 single inventory icons (`health`, `mana`, `xp`, `dex`, …) | `(8, 8, 3)` | 1 each |
| `armour_textures` / `armour_enchantment_textures` | `(3, 4, 8, 8, 3)` / `(3, 4, 8, 8, 4)` | level x slot |
| `sword_textures`, `pickaxe_textures`, `bow_textures`, `potion_textures` | `(5/5/2/6, 8, 8, 3)` | |
| `night_texture`, `night_noise_intensity_texture` | `(90, 110, 3)` | 1 each |

**The decisive measurement: every overlay alpha in the full game is binary `{0, 1}` — at 10 px, 16 px
and 64 px** (`layers.json`, `all_world_overlay_alphas_binary` true at all three). Item alpha, player
alpha, all three mob alphas, projectile alphas, the enchantment overlays and the digit alphas: all
`{0, 1}`.

**That is the structural difference from Classic.** Classic's alphas take `{0, 128/255, 1}`, so its
composite is a real blend and W1 could only make it a gather by baking the whole
`(player, block, mob)` product — 816 entries. Full Craftax's composite is a **select**, which is what
makes the per-tile layered formulation of §3 possible *and* what makes the product bake impossible:

| candidate bake | entries | uint8 @10px | uint8 @16px |
|---|---|---|---|
| block | 37 | 11 kB | 28 kB |
| block x item | 185 | 56 kB | 142 kB |
| block x item x player x mob | 22,200 | 6.7 MB | 17.0 MB |
| **block x item x player x mob x projectile** | **732,600** | **220 MB** | **563 MB** |
| *(Classic's own table, for scale)* | 816 | 245 kB | 627 kB |

So "generalise `_Composite` to the full atlas" in the Step-1 gap table is **the wrong shape**: the
product is 898x Classic's and does not fit. §3 is what replaces it.

### 2.3 The render window: what determines the observation

| group | slots | range |
|---|---|---|
| **per tile (99 tiles = 9 x 11)** | | |
| block id, post-substitution | 99 | `0..36` |
| item id, post-substitution | 99 | `0..4` |
| mob slot (`none` \| 8 melee \| 3 passive \| 8 ranged) | 99 | `0..19` |
| projectile slot (`none` \| 8 types x 4 orientations) | 99 | `0..32` |
| `light_map`, per tile | 99 | float32 `[0,1]` |
| **per frame, world** | | |
| player slot (4 facings \| sleeping \| none) | 1 | `0..5` |
| `player_level` | 1 | `0..8` |
| `light_level` | 1 | float32 |
| `is_sleeping` | 1 | `0/1` |
| `state_rng`, as **four 16-bit halves** (W1's encoding) | 4 | `0..65535` |
| **per frame, strip** | | |
| health / food / drink / energy / mana | 5 | `0..99` |
| xp / dexterity / strength / intelligence | 4 | `0..9` (the gather clamps) |
| wood, stone, coal, iron, diamond, sapphire, ruby, sapling, torches, books, arrows | 11 | `0..99` |
| `potions[6]` | 6 | `0..99` |
| pickaxe / sword / bow | 3 | `0..4`, `0..4`, `0..1` |
| `armour[4]` | 4 | `0..2` |
| `sword_enchantment`, `bow_enchantment` | 2 | `0..2` |
| `armour_enchantments[4]` | 4 | `0..2` |
| `learned_spells[2]` | 2 | `0/1` |
| **total** | **544** | **2,176 bytes as float32** |

### 2.4 Bytes per state against bytes per frame

| | bytes | vs the 10 px frame | vs the 16 px frame |
|---|---|---|---|
| **render window** | **2,176** | **19.7x smaller** | **50.5x smaller** |
| frame at 10 px (the registered observation) | 42,900 | — | |
| frame at 16 px | 109,824 | | — |
| craftax's own `EnvState` | **273,276** | **6.37x LARGER** | 2.49x larger |
| *(Classic's window, for scale)* | 596 | 20.0x / 104x | |

Storing `EnvState` loses to storing the image by 6.37x, for the reason the module docstring already
records: it carries all nine 48x48 levels of `map`, `item_map` and `light_map` while the frame shows
one 9x11 window of one of them. The render window wins by 19.7x. **The Step-1 estimate of "~5.3 kB"
was 2.4x pessimistic.**

---

## 3. The smallest missing capability

### 3.1 What it is

**A per-tile, N-layer binary-stencil composite over a small background table.** In one line:

```
tile = where(proj_a, proj, where(mob_a, mob, where(player_a, player, bg[block, item])))
```

Five small integers per tile, four gathers, three selects. `probe_composite.py` builds exactly this in
torch from `EnvState` and compares against the instrumented renderer's `world` intermediate — the
last state before any lighting — on 49 real states (40 from a 4,000-step rollout, 9 level-shifted):

> **`per_tile_composite_bitexact` 49/49, `max_abs` 0.0.**

Bit-identical to craftax's own scan of up to fourteen `dynamic_update_slice` blits. The tables it
needs total **46,900 bytes as uint8 at 10 px** (block 11.1 kB, item RGBA 2.0 kB, player RGBA 2.0 kB,
mob rgb+alpha 12.0 kB, projectile rgb+alpha 19.8 kB) — three orders of magnitude under the product
bake, and small enough to stay resident at 64 px.

**One trap, and it is silent.** `ItemType.NONE` is slot 0 and the reference's scan starts at 1, so it
*never draws slot 0*. Its texture is `debug.png` and is **not** transparent. An implementation that
gathers slot 0 like any other slot paints a debug tile over every empty square — which is what the
first run of this probe did (0/49, max abs 255). The fix is to zero slot 0's alpha in the table, not
to special-case it in the gather.

### 3.2 What this cannot represent, measured

The formulation carries **one mob slot and one projectile slot per tile**. Over 800 sampled frames of
a 4,000-step rollout: `mob_on_mob` **0**, `proj_on_proj` **0**, `proj_on_mob` **0**, and the maximum
entities on any one tile was **1** (`state_census.json`). Craftax's `mob_map` keeps the three mob
classes off each other's tiles; a projectile *can* in principle share a tile with a mob, and that
state is not representable here. This is the same documented limitation Classic carries ("it carries
one mob per tile"), stated up front rather than discovered.

### 3.3 The lighting chain — and why W1's is not it

The full game's chain is **not** Classic's. Side by side:

| | Classic | **full Craftax** |
|---|---|---|
| `light_map` | absent | `pixels = light_map[:, :, None] * pixels`, **before everything** |
| daylight | `state.light_level` | `select(player_level == 0, light_level, 1.0)` |
| noise | `uniform(rng, (7t, 9t)) * 95 + 32` | `uniform(rng, (9t, 11t)) * 95 + 32` — **the same expression** |
| vignette | `1 - exp(-0.5 (x^2+y^2)/0.5^2)` on `linspace(-1,1)`, numpy float64 | **normalised radial distance** `sqrt((x-H//2)^2 + (y-W//2)^2) / max`, in jnp float32 |
| `select(daylight < 0.5, …)` guard | present (redundant) | absent (`night_mask` is 0 there anyway) |
| "enhance" (pull towards luminance, factor 0.4) | **present** | **absent** |
| tint | `0.5*px + 0.5*[0,16,64]` | `[0,16,64]*0.5 + 0.5*px` (same in IEEE754) |
| blend | `daylight*world + (1-daylight)*night` | same |
| sleep | luminance at half brightness, lifted `+[0,0,8]` | **`pixels * 0.5`**, towards black |

Offered the reference's own inputs both ways its signature allows,
`stable_envs.envs.craftax.night_and_sleep` reproduces the full game's chain on
**4 of 49** states with `light_map=` and **8 of 49** with the light map pre-multiplied, max abs
**230** and **121** (`chain.json`). It is not a near miss; it is a different function.

The `light_map` argument W1 added for this workstream is in the wrong place as well as insufficient:
it multiplies only the night branch, where craftax multiplies `map_pixels` **before** the chain, so it
also scales the `daylight * map_pixels` term.

**The full chain transcribed in torch is 49/49 bit-exact, max abs 0.0** (`chain.json`), consuming
`stable_envs/render/prng.py` **unchanged**.

**One ULP, and it is the whole difference.** craftax normalises the vignette as `ntx / ntx.max()`
under jax, and **XLA lowers a float32 divide by a scalar as a reciprocal multiply**. Against a true
float32 divide that disagrees on **2,265 of the 9,900** vignette elements, by one ULP each — and one
ULP in the vignette is one ULP in the night mask, which is enough to move a uint8 pixel. With
`v / v.max()` the chain was 35/49 bit-exact (max abs 3.05e-05); with `v * (1/v.max())` it is 49/49 at
0.0. This is the same class of trap as W1's rotation mask and must be pinned by a test.

Two consequences for the fast paths, both exact rather than approximate:

* `daylight` is forced to `1.0` on levels 1-8, so the night chain is **the identity on eight of the
  nine levels whatever `light_level` holds** — measured: changing `light_level` from 0.0 to 1.0 moves
  **29,700** pixels on level 0 and **0** on levels 1, 4 and 8 (`chain.json`).
* the `light_map` alphabet is small (**23 distinct values over four worlds**, min 0.0, max 1.0), and
  on the overworld the visible window is **all 1.0**, so a fully-lit awake batch stays on the uint8
  gather with no float buffer at all.

### 3.4 The strip

36 written cells and 8 blanks over a 4 x 11 grid. Every write's span is `[T*y + a : T*(y+1) - b]` with
`0 <= a, b < T` (icon `[1, 9)`, ones digit `[6, 10)`, tens digit `[2, 6)` at 10 px), so **every write
lies inside its own cell and the cells are disjoint** — the strip is a per-cell table, as in Classic.

Three things are new:

1. **Counts are capped at 99, not 9.** `game_logic.clip_inventory_and_intrinsics` does
   `jnp.minimum(x, 99)` over the whole `Inventory`; `max_health = 8 + strength`,
   `max_food/drink/energy = 7 + 2*dexterity`, `max_mana = 6 + 3*intelligence`, `max_attribute = 5`. So
   a cell renders a **two-digit** number: ones digit from `number_textures_with_zero` (or
   `number_textures` when the value is exactly 0), tens digit from `number_textures`, whose slot 0 is
   empty and therefore suppresses a leading zero.
2. **Five cells carry a second, alpha-blended icon**: the sword `(8,1)` and the four armour slots
   `(7,0..3)` carry their enchantment, and the arrows cell `(6,2)` carries
   `bow_enchantment * (arrows > 0)`.
3. Four cells are bare digits with no count semantics (`player_level` at `(6,0)`; xp / dex / str / int
   at `(9,0..3)`), and two are boolean spell icons.

A **flat `(cell, value)` table** would be **2,511 entries** = 753 kB at 10 px, 1.93 MB at 16 px — which
is affordable. But the **layered** form needs only the icon sprites plus 21 digit sprites, because a
two-digit cell is `icon -> ones -> tens -> enchantment` with binary alphas — *the same stencil the
world layer uses*. `probe_strip.py` builds the layered form and compares against the reference's own
`inv` intermediate: **49/49 bit-exact, max abs 0.0**.

### 3.5 Reuse map — W1 unchanged vs new

| piece | verdict |
|---|---|
| `stable_envs/render/prng.py` — `uniform(keys, shape, offset)`, `key_from_halves`/`key_to_halves` | **consumed unchanged.** The noise expression is character-for-character craftax's own `uniform(rng, shape) * 95 + 32`, which is exactly what the review ruling (no range arguments) provides for. The four-halves float32 encoding applies verbatim |
| W1's **un-jitted reference** ruling | **adopted unchanged** — the same 1.5e-05 jit disagreement is measured here (§4.2) |
| W1's `certification_state_distribution` convention | **adopted**; the sampler is the certification distribution and its band shares are declared on the class |
| W1's **applicability-declaration** pattern (oracle experiment on the official renderer, then declare) | **adopted**; the oracle is already run (§7.2) |
| W1's `_Assets` resample rule `((i+0.5)*src/t).long()` and "resample the finished tile, never the ingredients" | **adopted as a rule** |
| `night_and_sleep(..., light_map=…)` | **NOT reusable.** 4/49 bit-exact, max abs 230 (§3.3). The `light_map` argument added for this workstream does not do what this workstream needs |
| `night_vignette` | **NOT reusable.** Different function (Gaussian vs normalised radial distance) |
| `_Composite` product bake | **NOT scalable.** 732,600 entries, 563 MB at 16 px |
| `_strip_cells` / `STRIP_LAYOUT` / `strip_geometry` | **NOT reusable.** Different grid, different cells, 0..99 counts, enchantment overlays |
| `_render_blend`, `_tile_ids`, `_decode`, `sample_states` | Classic-specific; the *patterns* transfer, the code does not |

### 3.6 Shared vs env-local — recommendation

**No new file under `stable_envs/render/`, and `stable_envs/envs/craftax.py` should not be edited.**

The capability of §3.1 has exactly one consumer. Classic cannot use it (its alphas are not binary, and
its product bake is already smaller than the layered form would be); `render/tilemap.py` is MiniGrid's
atlas and shares no concept with it; Step 7's own rule is that no workstream redesigns tilemap for one
env. So the smallest interface is:

* **new file `stable_envs/envs/craftax_full.py`**, importing `stable_envs.render.prng` and nothing
  else from W1's module, carrying the layered composite, the full chain, the vignette and the strip;
* `stable_envs/envs/craftax.py` **untouched**, which also keeps W1's `CraftaxClassic` parity claim and
  its frozen Step-4 replay out of this workstream's blast radius.

If the coordinator would rather see the composite lifted to `stable_envs/render/`, the honest interface
is one function —
`composite_layers(bg: (N0,t,t,3), layers: [((Nk,t,t,3), (Nk,t,t,3))], ids: (n,H,W,K+1)) -> (n,H*t,W*t,3)` —
but it should be lifted **when a second env wants it**, not before.

---

## 4. Purity and determinism, on real frames

Un-jitted reference, 49 states (40 rollout + 9 level-shifted), 10 px
(`purity_and_state_window.json`).

### 4.1 Purity

| probe | result |
|---|---|
| same state rendered twice | **49/49 identical** |
| the same 12-step history replayed from the same seed | **12/12 identical frames** |

### 4.2 Jitted vs un-jitted — the same disagreement as Classic, and it persists

| | value |
|---|---|
| max float32 `\|diff\|` over the pool | **1.52587890625e-05** |
| states whose uint8 frames are identical | **48 / 49** |
| worst uint8 `exact_frac` | **0.99997669** |
| worst uint8 max abs | **1** |

Same mechanism as W1's: XLA fuses the reference's arithmetic differently and a ~1e-5 float32 shift
straddles a uint8 rounding boundary wherever the picture is compressed towards one value. **W1's
ruling carries over: `reference_env` must return the un-jitted renderer and the parity claim must be a
uint8 claim.** The full game compresses the range in *more* places than Classic (the `light_map`
multiply on eight of nine levels, plus the sleep halving), so the effect will not be rarer here.

### 4.3 Window sufficiency — the strongest single result

Twenty-one `EnvState` fields the proposed window does **not** carry were each perturbed to a
legal-looking value, and the frame was diffed pixel by pixel:

`timestep` · `achievements` · `player_recover` · `player_hunger` · `player_thirst` · `player_fatigue`
· `player_recover_mana` · `is_resting` · `potion_mapping` · `chests_opened` · `growing_plants_age` ·
`growing_plants_mask` · `down_ladders` · `up_ladders` · `boss_timesteps_to_spawn_this_round` ·
`map` on another level · `map` off-screen on the same level · `mob_map` · `melee_mobs.health` ·
`melee_mobs.attack_cooldown` · `monsters_killed` on another level

> **Every one moved exactly 0 pixels.** `window_sufficiency_all_zero: true`.

And the fields the window *does* carry each move pixels: `light_level` 29,700 · `is_sleeping` 29,205 ·
`player_direction` 70 · `state_rng` **0 by day and 29,697 at night** · `light_map` 27,457 ·
`player_level` 29,664 · `player_xp` 87.

(`mob_map` moving 0 pixels is worth noting on its own: the renderer's mob-map overlay is commented out
in craftax 1.6.1, so it is a field that looks load-bearing and is not.)

---

## 5. What a random policy does *not* reach

4,014 states from a 4,000-step random-policy rollout (`state_census.json`):

* **levels visited: `[0]` only.** A random policy never descends. All nine levels are generated at
  `reset` (`state.map` is `(9, 48, 48)`), so a level-k frame is reached by writing `player_level = k`
  and standing the player on level k's own up-ladder — a state the simulator itself produces on a
  `DESCEND`. That is how the nine level-shifted states in this pool were made, and it is how the
  frozen corpus must be built.
* **light bands: 19 daylight, 2,667 dusk, 1,328 night.** Full daylight is 0.5% of the rollout.
* **25 of the ~42 strip fields never leave 0**: every one of `stone`, `coal`, `iron`, `diamond`,
  `ruby`, `sapphire`, `torches`, `books`, `arrows`, `bow`, `pickaxe`, `potions[6]`, `armour[4]`,
  `armour_enchantments[4]`, `sword_enchantment`, `bow_enchantment`, `learned_spells[2]`, `player_xp`.
  `dexterity`/`strength`/`intelligence` never leave 1. This is W1's Procgen failure mode, worse: a
  pool built from rollouts alone would present **~25 dead state dimensions**.
* on-screen entity counts: melee max 1 (mean 0.05), passive max 3 (mean 1.05), ranged max 2, mob
  projectiles max 2, player projectiles **0** — a random policy never fires.

---

## 6. Throughput bars

Apple arm64, one core, one torch thread, `JAX_PLATFORMS=cpu`. **Not certification numbers**; the
Oscar batch-node re-take is `02_bars_cpu.sbatch` (`bars.json`).

| | **10 px (registered)** | 16 px |
|---|---|---|
| frame | 130x110x3 = 42,900 B | 208x176x3 = 109,824 B |
| official renderer, **un-jitted** | **2.98 f/s** | 2.92 f/s |
| official renderer, jitted | **1,516 f/s** | 935 f/s |
| jit speedup | **509x** | 320x |
| JPEG-q90 blob | 7,212 B | 12,458 B |
| **one core of JPEG-q90 decode** | **11,270 f/s** | 6,464 f/s |
| **absolute f/s for `S_cpu >= 5`** vs the **jitted** official | **7,582** | 4,673 |
| absolute f/s for `S_cpu >= 5` vs the un-jitted official | 14.9 | 14.6 |
| **absolute f/s for `cores_cpu >= 1`** | **11,270** | 6,464 |

**The binding bar at the registered size is `cores_cpu >= 1` at ~11,270 f/s**, not `S_cpu`. For scale,
W1's `CraftaxClassic` reaches 132,006 f/s on the daylight path and 1,986 f/s on its mixed corpus at
16 px on this machine; full Craftax's frame is 0.39x the 16 px Classic frame at the registered 10 px
size, and its daylight path is a gather of the same shape, so the fully-lit band should clear both
bars comfortably. **The lit band is the risk**, exactly as in Classic.

### 6.1 The tier will be one mixed-corpus number

Sami's ruling for Classic applies unchanged: **one** registered/default tier, taken on the
distribution `sample_states(64)` produces (Step 3's own protocol, `examples/bench_step3.py:186`), with
per-band diagnostics reported beside it and the slower band named. For this env the bands are
**richer than Classic's**, and the declaration must say so:

| band | cost shape |
|---|---|
| levels 1-8, fully lit (`light_map == 1`), awake | one uint8 gather, **no float buffer, no PRNG** |
| levels 1-8, dark (`light_map < 1`) or asleep | per-pixel float multiply, no PRNG |
| level 0, `light_level == 1`, `light_map == 1`, awake | as the first row |
| level 0, `0.5 <= light_level < 1` | float chain, no PRNG |
| **level 0, `light_level < 0.5`** | float chain **+ the threefry field** — the slow band |

The PRNG is needed on a **narrower** set than in Classic: only `player_level == 0` **and**
`light_level < 0.5`. Eight of the nine levels never draw it at all.

---

## 7. Correctness plan

### 7.1 Tier: parity, byte-identical, as for Classic

Phase 0 has already reproduced all three layers bit-exactly, so `parity_all_exact_min = 1.0` is the
target and `reference_versions = {"craftax": "1.6.1", "jax": "0.11.1"}` with the
`jax_threefry_partitionable` assertion is the guard. `reference_env` returns the **un-jitted**
renderer and refuses a size craftax does not ship (10, 16, 64).

### 7.2 Gate A — the two checks at risk

**(a) `check_state_dims_matter`.** Three families of dimension are unobservable outside a specific
band, and every one of them is a property of the *reference*
(`resolution_and_dead_dims.json`):

| dimension | reaches a pixel only when | measured |
|---|---|---|
| the four `state_rng` halves | `player_level == 0` **and** `light_level < 0.5` | level 0 day **0**, level 0 night **29,697**, levels 1/4/8 at `light = 0.1` **0 / 0 / 0** |
| `light_level` | `player_level == 0` | level 0 **29,700**, levels 1/4/8 **0 / 0 / 0** |
| the 99 `light_map` slots | the level is not fully lit — on the **overworld the window is all 1.0** | `light_map_window_min_by_level["0"] = {min: 1.0, max: 1.0}` |
| ~25 strip fields | they are sampled directly | §5 |
| the 99 projectile slots | a projectile is placed | a random policy fires **0** |

**Plan**, following W1's device exactly: `sample_states` **stratifies by row index** over
`(player_level, lighting band, sleeping, light_map)` and samples the strip values directly over their
legal ranges rather than harvesting them — sound because §3.4 proves each cell is a function of its own
value alone and the simulator caps every count at 99. The declared
`certification_state_distribution` then records the shares, with a test that fails if the sampler is
retuned away from them.

**(b) `check_resolution_consistency` — and the oracle is already run.** W1's ruling (Sami's decision A)
is that applicability may only be declared from an experiment on the **official** renderer. Done, at
S=10 / 2S=20, craftax's own `load_all_textures_given_size(20)` patched into the module cache so every
pixel is still craftax's, un-jitted, each state's own `state_rng`, the check's own box downsample and
statistic:

| pool | official `exact_frac` | world rows | strip rows | max abs |
|---|---|---|---|---|
| all 49 | **0.4895** | **0.2885** | 0.9418 | 191 |
| daylight (= the 8 level>0 states) | 0.6227 | **0.4817** | 0.9400 | 191 |
| dusk (27) | 0.5181 | 0.3295 | 0.9424 | 191 |
| night (14) | 0.3584 | **0.0991** | 0.9418 | 191 |

**The official full renderer fails the frozen `DOWNSAMPLE_EXACT_FRAC >= 0.995` in every band,
including full daylight, and in the world rows as well as the strip.** Two independent mechanisms:

1. **Texture resampling.** `load_texture(fname, size)` resamples the source PNG at each size, so the
   20 px block textures are **not** a 2x replication of the 10 px ones. This is the daylight number
   (world 0.4817 with no noise and no lighting at all) and it is new — Classic's equivalent affected
   only the strip.
2. **The noise field**, drawn at `(9t, 11t)`: element `i` of the 20 px field is a different threefry
   counter from element `i` of the 10 px one. This takes the world rows from 0.48 to 0.099 at night.

Unlike Classic, the strip geometry *does* scale by two here (`int(0.8*10)=8 -> 16`,
`int(0.4*10)=4 -> 8`, left space `1 -> 2`); its 0.94 is the icon resampling, not the layout.

**Recommendation for the coordinator, in W1's own form:** build the atlas at craftax's own reference
sizes and **resample the finished tile**, which makes this renderer exactly 2x-consistent where the
reference is not, and declare `resolution_consistency_applicability` **not_applicable** citing the
official numbers above — with the distinction from Classic stated plainly: there, only night was
inapplicable; here the reference is resolution-inconsistent **in every band**, so the check has no
band left in which it measures what it is defined to measure. The frozen threshold is not edited.
*This ruling is Sami's, not this workstream's.*

### 7.3 The frozen corpus

`examples/results/step7/craftax-full/corpus/`, built the way Step 4 builds (typical + per-dimension
extremes + farthest-point spread) and then **checked**, not hoped, for:

* all **9 levels** present;
* all five cost bands of §6.1 present, with level 0 x night guaranteed so the PRNG halves are live;
* `light_map` windows that are not all-1.0 (underground);
* sleeping and awake;
* every strip cell reaching both ends of its range, and every cell with an enchantment overlay
  reaching all three enchantment levels;
* mobs of all three classes and all `type_id`s, and projectiles in all four orientations — none of
  which a random policy produces.

---

## 8. Blockers and open questions

### 8.1 `BLOCKER` (ruling) — `check_resolution_consistency` has no applicable band

§7.2(b). The oracle is run and the numbers are in hand; the decision is Sami's.

### 8.2 `BLOCKER` (ruling) — what "the official renderer" costs, for the tier

The parity claim requires the **un-jitted** reference (§4.2). The Step-3 performance denominator is
taken through **the same hook** (`examples/bench_gpu_vs_cpu.py:cpu_throughput` calls
`cls.reference_render`). At 10 px those two numbers differ by **509x**. A tier quoted against an
un-jitted denominator would certify `CPU_EXCEPTIONAL` for anything above 15 f/s, which is not a claim
worth making. The workstream's recommendation is to quote `S_cpu` against the **jitted** official
renderer, declared as such, and to keep `reference_env` un-jitted for parity — but that needs either a
second hook or a ruling.

### 8.3 `FINDING` for the coordinator — W1 has already moved Classic's denominator

This is not this workstream's to fix and must not be found later. At the Step-6 baseline `76dc5116`,
`CraftaxClassic.reference_env` returned `jax.jit(make_craftax_pixel_renderer(size))`. On
`step7/integration` it returns the un-jitted renderer — correctly, for parity. Measured on this Mac
(`window.json`):

| | f/s |
|---|---|
| `CraftaxClassic.reference_render` on the branch, 16 px | **4.62** |
| the same renderer jitted, 16 px | **2,073** |
| what Step 3 recorded as `CraftaxClassic` official | **1,046** |

Re-running Step 3 against the integration branch divides Classic's official denominator by ~450 and
multiplies its `S_cpu` by the same factor. `docs/research/step3_benchmarks.md:44` currently records
`S_cpu = 5.01`, `CPU_FASTER`.

### 8.4 Smaller open questions

1. **Is the strip lit in the full game?** Inferred here from the concatenation order — the reference
   concatenates `inv_pixels` after the sleep multiply, as in Classic — but not yet measured directly.
   Phase 1 should measure it the way W1 did rather than infer it.
2. **`player_xp` above 9.** `_render_digit(state.player_xp, 9, 0)` indexes a 10-entry table; jax
   gathers clamp, so xp 12 draws a "9". Phase 1 must reproduce the **clamp**, not a wrap — which is
   the opposite of the decode rule Classic uses for its conformance probe, and the two must not be
   confused.
3. **`do_night_noise`.** `make_craftax_pixel_renderer(bps, do_night_noise=True)` has a second,
   noiseless branch. The registered env uses the default `True`; a declaration should say so, since
   the flag is part of the reference the way `jax_threefry_partitionable` is.
4. **Which sizes to certify.** The registered observation is **10 px**; craftax also ships 16 and 64.
   Phase 1 proposes certifying at 10 px and 16 px and refusing anything else, as Classic does.
5. **`state_rng` under the SWM wrapper.** The wrapper jits `step`, so a deployed rollout sees the
   *jitted* reference and its 1.5e-05 float drift. The parity claim is against the un-jitted renderer;
   phase 1 should state that explicitly in `parity_note`, as W1 does.

---

## 9. Phase-1 milestones

| # | milestone | evidence it must produce |
|---|---|---|
| 1 | `stable_envs/envs/craftax_full.py`: tables + the per-tile layered stencil; **no edit to `craftax.py`** | world layer bit-exact against the un-jitted reference on a fresh pool at 10 and 16 px |
| 2 | the lighting chain (`light_map` -> night -> sleep) with the reciprocal-multiply vignette, consuming `prng` unchanged | chain bit-exact; a test pinning `v * (1/v.max())` against a jax-generated vignette |
| 3 | the layered strip | strip bit-exact; a test that the strip is invariant to `light_level`, `light_map` and `is_sleeping` |
| 4 | the state window, `_decode`, `sample_states` stratified over (level, band, sleep, light_map) + directly-sampled strip values | Gate A with `dead_dims: []` at batch 1..16 over 12 seeds |
| 5 | fast paths: the uint8 gather for "fully lit, awake, unit gain" (which now includes **all of levels 1-8 at `light_map == 1`**), and drawing the noise field only on level-0 night rows | measured per band, plus a bit-exactness test that each skip is the identity |
| 6 | reference bridge: `reference_env` (un-jitted, flag-asserted), `reference_render` writing the window back into a full `EnvState` | parity 1.0 / max abs 0 on the frozen corpus at both sizes |
| 7 | frozen corpus under `examples/results/step7/craftax-full/corpus/` | coverage checked per §7.3 |
| 8 | Oscar: x86-64 re-take of every bit-exactness claim; the two bars on an `--exclusive` batch node; the GPU leg | `01`/`02`/`03` sbatch outputs, job IDs recorded |

Risk order: the state window and the sampler (milestone 4) are the largest unknown, because Gate A's
dead-dimension check is where this env's nine levels and five bands can make a correct renderer look
wrong. The renderer itself is de-risked: all three layers are already reproduced bit-exactly.

---

## 10. Oscar

Reachable; `$SCR/venvs/se7-craftax` verified from the login node (jax 0.11.1 with
`jax_threefry_partitionable = True`, craftax 1.6.1, torch 2.14.0+cu126). `$SCR/se7/` holds the other
workstreams' copies; **`$SCR/se7/craftax-full/` does not exist yet** and will be seeded by rsync at the
start of phase 1. No jobs were submitted in phase 0 and nothing was cancelled.

| script | what it owes |
|---|---|
| `scratch/step7/craftax-full/01_phase0_x86.sbatch` | every phase-0 bit-exactness claim re-taken on x86-64 (`batch`/`normal`, one thread) — float32 reconstruction and a bit-level integer PRNG, so cross-architecture agreement is measured, not assumed |
| `scratch/step7/craftax-full/02_bars_cpu.sbatch` | the two bars on an **`--exclusive`** batch node with `/proc/loadavg` printed either side, because W1 measured a contended node at 6,219 f/s where the same code does 126,845 on a laptop |
| `scratch/step7/craftax-full/03_gpu.sbatch` | **phase 1 only.** The CUDA leg of Gate A and the GPU tier, before and after in one allocation on one card. `--partition=gpu --qos=norm-gpu --constraint=geforce3090`, excluding `gpu2108,gpu2113,gpu2110,gpu2116,gpu2104`, `--cpus-per-task=4 --mem=48G --time=03:00:00` |

Job names all start `se7-craftax-full-`. One GPU job in flight at a time; never cancel a job this
workstream did not submit, and never a `simclr_cifar10` job.
