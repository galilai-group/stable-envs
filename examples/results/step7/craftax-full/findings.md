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

---
---

# Part II — the implementation

Phase 1 was authorised with five rulings (structure, resolution applicability, the performance
denominator, milestone order, Oscar caps) and landed in the order the coordinator set. Sections 1-10
above are the phase-0 diagnosis, kept as measured; what follows is what was built and what it
measures.

| commit | milestone |
|---|---|
| `6f92a3b` | **M1 + M2** — `stable_envs/envs/craftax_full.py`, the baked tables, the 544-slot window, the stratified sampler, 24 unit tests |
| `99f2b48` | **M3** — the frozen Step-7 corpus and parity on it |
| `231dac2` | **M4 + M5 (local)** — Gate F shape, the uint8-composite reduction, the tier per band |

**Classification: `PATCH`.** `swm/CraftaxPixels-v1` goes from **gap** to **full** on the renderer's
side. Gate A is clean, parity is byte-identical at both certified sizes, and the two open items are
the rulings already escalated (§8.1, §8.2/§8.3), not code.

## 11. What was built

Per the coordinator's ruling: a **new** `stable_envs/envs/craftax_full.py`;
`stable_envs/envs/craftax.py` **untouched**; nothing new under `stable_envs/render/`;
`render/prng.py` consumed unchanged. `git diff --name-only 76dc5116..HEAD -- stable_envs` is
`envs/_craftax_full_tex.py` and `envs/craftax_full.py`, both new.

### 11.1 The cell

`stable_envs/envs/_craftax_full_tex.py` (41 kB) holds craftax 1.6.1's textures at 10 px and 16 px as
**uint8, losslessly**: `load_texture` reads each PNG as int, forces `alpha // 255` so every alpha is
exactly 0 or 1, and resizes with `Image.NEAREST`; over all 47 texture keys at both sizes the only
non-integral entry is the vignette, which is a function of the frame size rather than an asset.

Four tables, all small enough to stay resident at any size craftax draws at:

| table | entries | what |
|---|---|---|
| `bg` | **596** | 185 `block x item` composites, one black tile, **410 strip icon cells** |
| `ov` | **660** | `mob x projectile` pairs — projectile over mob, the reference's order |
| `pl` | 6 | the four facings, `player-sleep`, and "no player" |
| `dg` | 200 | `ones x tens` digit pairs; entry 0 is (empty, empty), the no-op a digit-less cell takes |

and the frame is **one gather plus three selects**: gather `bg` for all 143 tiles at once — a strip
cell's *icon is its background*, so the strip costs a gather like a world tile does — then the player
on the centre tile (1/99 of the frame), then `ov` on the world rows and `dg` on the strip rows.

### 11.2 The reductions, including the one that was refused

| reduction | status |
|---|---|
| **the composite never leaves uint8** | **done.** Every table entry is integral and a stencil composite of integers is an integer, so a float table would hold the same numbers. The gathers and selects move one byte a channel instead of four, the strip's 4 of 13 rows never become float at unit gain, and only the world rows make the float round trip. Measured: mixed corpus at 10 px, B=256, **3,475 -> 4,636 f/s** |
| **draw the noise field only where it can contribute** | **done, and by construction.** `night_intensity` is exactly 0 for `daylight >= 0.5`, so only those rows are gathered, drawn and scattered back. Eight of the nine levels are in that band unconditionally |
| **one host read, not four** | **done.** "is the appearance default" and "can this batch skip the float chain" come off one stacked tensor, so a CUDA render syncs once |
| **a one-channel stencil mask** | **tried, measured, refused.** It moves a third of the bytes and lets `torch.where` broadcast, and is **1.7x slower** on a fully lit frame (18,928 -> 10,852 f/s at 10 px, B=256). The number is in the code beside the three-channel mask so it is not tried again |
| **collapse the lighting chain algebraically** | **refused, as in Classic.** It is the same expression in real arithmetic and a different rounding at every step in float32, and the claim here is bit equality rather than closeness |

### 11.3 The two traps, and how each is pinned

* **The vignette is a reciprocal multiply.** craftax normalises `ntx / ntx.max()` under jax and XLA
  lowers a float32 divide by a scalar as `x * (1/s)`; against a true divide that is one ULP out on
  **2,265 of 9,900** elements at 10 px, and one ULP there is one ULP in the night mask. The chain
  went 35/49 -> **49/49** bit-exact when the line changed.
  `test_the_vignette_is_a_reciprocal_multiply_and_not_a_divide` asserts the sha256 of the whole
  texture against a value frozen from craftax, six exact probe values, **and** that the naive divide
  is still wrong — because a test whose wrong branch stopped being wrong has stopped testing
  anything.
* **`ItemType.NONE` is never drawn and `debug.png` is opaque.** Its alpha is zeroed at bake time, so
  the gather needs no special case. `test_item_slot_zero_is_never_drawn` asserts it at both sizes.

### 11.4 The resampling rule, which is what makes the cell stricter than the reference

`_source_size` prefers a baked size that **divides** the requested one over a nearer one that does
not: a 20 px render comes from the **10 px** art, not from the nearer 16 px art, and is therefore an
exact 2x replication of the 10 px frame. Measured: with the nearest rule
`check_resolution_consistency` reported `exact_frac` **0.323**; with the divisor rule it reports
**0.7946**, and `test_a_render_at_twice_the_size_resamples_from_the_size_that_divides_it` asserts
`big[:, ::2, ::2] == small` exactly on a fully lit batch. The residual is the night noise, which is
drawn at the frame's own resolution and which no port can make consistent without changing pixels.

## 12. Parity — byte-identical at both certified sizes

`examples/results/step7/craftax-full/certify/certify_cpu_mac_corpus.json`, the **frozen Step-7
corpus**, craftax 1.6.1 / jax 0.11.1 (`jax_threefry_partitionable` True), **un-jitted**:

| tile px | frame | `exact_frac` | max abs | world rows | strip rows | fast (21) | lit (21) | night (22) |
|---|---|---|---|---|---|---|---|---|
| **10** (registered) | 64x130x110x3 | **1.0** | **0** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **16** | 64x208x176x3 | **1.0** | **0** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

`digest_ours == digest_official` at both sizes (`b3dcba4dea708d15`, `43414708ce31a3d7`): the frames
are **byte-identical**, not equal within a bound. A second, independently drawn 64-state pool
(`certify_cpu_mac.json`, seed 20260915) gives the same 1.0 / 0.

The claim is declared as a **uint8** claim with the jit-gap note, as W1 declared Classic's:
`parity_all_exact_min = 1.0`, `reference_versions = {craftax 1.6.1, jax 0.11.1}`, and a `parity_note`
that names `jax_threefry_partitionable`, `do_night_noise`, the un-jitted choice and the 1.53e-05
measurement.

## 13. The frozen corpus

`examples/results/step7/craftax-full/corpus/` — 64 states from a 512 pool, selected the way Step 4
selects (typical + per-dimension extremes + farthest-point spread) in Step-4's schema, with the pool
written beside the selection so a later run can show which 64 of which 512 were kept. Proposed for
the Step-8 freeze; no frozen Step-4/5/6 artifact is touched.

| | |
|---|---|
| levels | **all 9** |
| level 0 | night 22 · dusk 10 · day 10 |
| underground | lit 11 · dark 11 |
| uint8 fast path | 14 · PRNG drawn 22 · sleeping 17 |
| entities | 255 mob tiles across all three classes · 218 projectile tiles in **all four** orientations |

**Spread is not coverage**, and the corpus says so rather than hiding it: level 4 did not survive the
farthest-point selection on this seed, so a repair pass swaps a missing level in against the most
over-represented one and records the swap in `corpus.json`. Widening the pool until the draw happened
to work would have been the same corpus with the reason removed.

## 14. Gate A — every check, and the two that needed design

At the env's default batch, on the Mac (the x86-64 re-take is §16):

`check_state_roundtrip` 0.0 · **`check_state_dims_matter` `dead_dims: []`** · `check_batch_invariance`
0 · `check_batch_size_invariance` 0 over `[1, 7, 64]` · `check_order_independence` 0 / 0 ·
`check_determinism` identical · `check_render_is_pure` 0 on all four probes ·
`check_chunked_render_matches` 0 over `[1, 7, 64, 4096]` · `check_input_form_invariance` no
disagreements · `check_set_state_copies` 0 / not rebound · `check_style_is_state` 0 ·
`check_default_style_is_official` 0 · **`check_variations_do_something` `[]` at batch 1, 2, 4 and
16** · `check_colors_apply_exactly` `[]` (this env exposes no RGB colour leaf; its recolouring knob is
`channel_gain`) · `check_backend_agreement` skipped (no `Scene2D`; this is a tile path).

### 14.1 `check_state_dims_matter` — `dead_dims: []` at batch 1 through 16

**Measured: no dead dimension at batch 1, 2, 3, 4, 8 or 16, over twelve seeds each.** That took four
devices in the sampler, and each one is documented in the code with the measurement that forced it,
because each looks arbitrary and none is:

1. **Row 0 is always the richest band** — level 0, night, dimmed, awake. It is the only band in which
   the four PRNG halves (`player_level == 0` **and** `light_level < 0.5`), `light_level`
   (`player_level == 0`) and the 99 `light_map` slots (a tile that is not fully lit) are *all*
   observable at once, and a `batch=1` instance gets exactly one row.
2. **Band 0 dims to 0.6 and no further, and its night stops at 0.3.** The frame is uint8: at
   `light_map` 0.11 a block swap worth four levels lands inside one uint8 step and the dimension
   reads as dead. The deep values live in band 4, where other rows can see them.
3. **Band 0 carries no `DARKNESS` block.** `DARKNESS` renders as an all-zero tile and zero times any
   light is still zero, so a `DARKNESS` tile has a dead `light_r*c*` dimension.
4. **Band 0 carries a ladder on every tile.** The item alphabet is NONE, TORCH and three ladders, and
   the check's fallback nudge on a NONE slot is `+1` — which lands on TORCH, a small icon that
   disappears entirely under the opaque part of a mob or projectile sprite. Measured: with a plain
   draw a `batch=1` instance reported one or two `item_r*c*` dimensions dead on **half of twelve
   seeds**, every one of them a tile under an overlay. Exactly one mob sprite of the twenty is fully
   opaque (ranged type 5) and band 0 keeps it out, which
   `test_exactly_one_mob_sprite_is_fully_opaque_and_the_sampler_knows` pins.

The strip's 41 numbers are **drawn directly rather than harvested**, which is sound for the reason
§3.4 measures — each cell is a function of its own value alone and the simulator caps every count at
99 — and necessary for the reason §5 measures: a 4,000-step random-policy rollout leaves 25 of the 41
at zero for every step and never leaves level 0.

### 14.2 `check_resolution_consistency` — declared `not_applicable` in all bands

Per the coordinator's ruling, W1's applicability pattern with the official renderer's own evidence.
The declaration on the class carries the numbers, the two mechanisms, the evidence path, the
reference versions, and — as instructed — **the note that this scope is broader than Classic's**:
there only `light_level < 0.5` is inapplicable and the frozen bound is kept above it; here the
reference is resolution-inconsistent **in every band**, including full daylight, because
`load_texture` re-resamples every PNG at every size with `Image.NEAREST`. The frozen Step-2 threshold
is **not edited**. This renderer measures `exact_frac` **0.7946**, which is *better* than the
official's 0.4895 because it resamples the finished tile; exact parity at 10 px and 16 px is
unaffected and still required, and is met.

`test_the_declared_resolution_applicability_matches_measurement`-style staleness cover is provided by
`test_the_resolution_applicability_declares_its_scope_and_its_evidence`, which fails if the scope
line, the evidence path or the "still required" clause goes away.

## 15. Gate F

`StableEnvGym("CraftaxFull")` reset gives a **544**-dim observation, `render()` returns
**(130, 110, 3)** — exactly the `obs_shape` `stable_worldmodel/envs/craftax/craftax.py` hard-codes
for `swm/CraftaxPixels-v1` — and (208, 176, 3) at 16 px. `info` carries the shard's usual six keys.
`step()` raises by design: this cell ships a renderer and no batched simulator, so its Gate F path is
`swap_renderer`, which keeps the official env's dynamics and replaces only its pixels, exactly as for
`CraftaxClassic`. Artifact: `gate_f/gate_f_local.json`. The run through
`examples/swm_compat/run_swm_workflow.py` needs `stable_worldmodel`, which is in no local venv, and
is owed on Oscar.

## 16. Performance — one tier, five band diagnostics, and a denominator that is not yet a ruling

Apple arm64, one core, one torch thread. **Not certification numbers**; the Oscar batch-node leg is
§17. `sweep/bench_cpu_mac.json`.

| band | 10 px, B=256 | 16 px, B=256 | what it costs |
|---|---|---|---|
| level 0, full daylight | **18,970 f/s** | 9,604 | uint8 gather, no float buffer, no PRNG |
| underground, fully lit | 18,869 | 9,598 | the same path — the daylight override makes 8 of 9 levels noise-free |
| underground, dark | 6,946 | 2,983 | per-tile `light_map` multiply |
| level 0, dusk | 6,894 | 3,023 | the float chain, no PRNG |
| **level 0, night** | **2,808** | 1,129 | the float chain **+ the threefry field** |
| sleeping | 4,682 | 1,940 | |
| **mixed corpus — THE TIER** | **4,636 f/s** | **1,946 f/s** | |

* **fastest/slowest ratio 6.75x**, and the **flagged slower band is `player_level == 0` with
  `light_level < 0.5`**: the reference performs a per-pixel float post-process there — light map,
  noise field, tint, daylight blend — against a single uint8 gather when the frame is fully lit and
  awake. No engineering makes a per-pixel float chain as cheap as a memcpy.
* The ratio is **five times gentler than Classic's 32.7x**, for a structural reason: full Craftax's
  night chain has no luminance "enhance" step, and eight of its nine levels never draw the noise
  field at all.

### 16.1 The bars, and the denominator question

One core of JPEG-q90 decode of this env's own 10 px frame: **9,795 f/s** (9,992-byte blob against a
42,900-byte frame).

| | value | reading |
|---|---|---|
| `cores_cpu` at 10 px | **0.47** | **BELOW_G4** — where Step 3 also put `CraftaxClassic` (0.44-0.49) |
| `S_cpu` vs the **jitted** official | **3.00** (10 px), 2.05 (16 px) | `CPU_FASTER` band |
| `S_cpu` vs the **un-jitted** official | 3,456 (10 px), 1,092 (16 px) | not a number worth quoting |

**Neither `S_cpu` certifies anything until Sami rules**, and this report does not choose. The parity
claim requires the un-jitted reference (§4.2) and the Step-3 denominator is read through the **same**
`cls.reference_render` hook, and the two differ by **1,151x** at 10 px and 533x at 16 px. Measured
here: the un-jitted bridge is 1.34 f/s at 10 px against 1,544 f/s for the jitted renderer. The tier
is taken on the **NO_REFERENCE ladder** as instructed, and both denominators are recorded beside it.

This workstream's §8.3 finding — that W1's (correct) un-jitted `reference_env` has already moved
`CraftaxClassic`'s own Step-3 denominator from a recorded 1,046 f/s to a measured 4.62 — is added to
that escalation and is **not** fixed here.

## 17. Oscar

Reachable. `$SCR/venvs/se7-craftax` verified (jax 0.11.1 with `jax_threefry_partitionable = True`,
craftax 1.6.1, torch 2.14.0+cu126). The worktree is rsynced to `$SCR/se7/craftax-full/`. Caps
respected: one GPU job in flight, `--cpus-per-task=4`, `--mem<=48G`, `--time<=03:00:00`,
`--exclude=gpu2108,gpu2113,gpu2110,gpu2116,gpu2104`. Nothing was cancelled and no `simclr` job was
touched.

| job | node | result |
|---|---|---|
| **6397697** `02_certify_cpu.sbatch` | `node2352`, x86-64, 8m01s | Gate A + parity on the frozen corpus, one thread |
| **6397916** `06_tier_cpu_paired.sbatch` | `node2352`, 5m46s | the CPU tier by the paired protocol, verdict **USABLE** |
| **6397699** `03_gpu.sbatch` | `gpu2106`, RTX 3090, 7m36s | CUDA Gate A + parity + the GPU tier per band |
| ~~6397698~~ `03_tier_cpu.sbatch` | — | cancelled by this workstream: `--exclusive` on `batch`/`normal` was scheduled for **05:46**, ~4 h out. The `normal` QOS's `cpu=64` cap blocks whole-node allocations here, exactly as it did for W1's `6395167` and in Step 6. Replaced by the paired protocol |

Caps respected throughout: one GPU job in flight, `--cpus-per-task=4`, `--mem<=48G`,
`--time<=03:00:00`, `--exclude=gpu2108,gpu2113,gpu2110,gpu2116,gpu2104`. Nothing was cancelled but
this workstream's own `6397698`, and no `simclr` job was touched.

### 17.1 The same frames on three backends

| | 10 px digest | 16 px digest | `exact_frac` | max abs |
|---|---|---|---|---|
| Apple arm64, CPU | `b3dcba4dea708d15` | `43414708ce31a3d7` | 1.0 | 0 |
| **x86-64, CPU** (`node2352`) | **`b3dcba4dea708d15`** | **`43414708ce31a3d7`** | **1.0** | **0** |
| **CUDA, RTX 3090** (`gpu2106`) | **`b3dcba4dea708d15`** | **`43414708ce31a3d7`** | **1.0** | **0** |

**The same digests on all three.** That is the check that matters most for this cell: the composite
is float32 arithmetic over a bit-level integer PRNG, and nothing about torch guarantees either across
architectures or between CPU and CUDA. It is measured, not assumed.

Gate A is identical on all three hosts — `dead_dims: []` at batch 1, 2, 3, 4, 8 and 16 over twelve
seeds each; no inert variation at batch 1, 2, 4 or 16; `check_per_env_variation`
`collapsed_variations: []`; `check_resolution_consistency` 0.7945367 to the seventh digit. On the GPU
node **`check_reference_renderer_is_stable` reports `max_abs_diff: 0`** — the node's own reference
renderer is deterministic, so nothing here is being blamed on the wrong thing. The 24 unit tests pass
on `node2352`.

### 17.2 The CPU tier — paired, gated, USABLE

`--exclusive` could not start, so the tier was taken by the paired rule instead. **One adaptation,
and it changes what the pair means:** `CraftaxFull` is a **new cell** and does not exist on
`$SCR/se7/baseline`, so there is no before and **no ratio to report**. The baseline legs measure
`CraftaxClassic` on the baseline tree purely to **qualify the node** against an answer that is
already known; the branch leg's absolute number is then reported or quarantined. W1's three gates,
their constants and their reasoning are reused rather than re-derived; this workstream's copy is
`scratch/step7/craftax-full/paired_verdict.py`, and the gates come from W1's, which lives on W1's
branch at commit `84ecb03` and not on this one — citing it from here is what broke
`test_every_cited_scratch_path_resolves`.

`$SCR/se7/baseline` was verified to **be** 76dc511 *before* submitting rather than after: its
`stable_envs/envs/craftax.py` md5s to `ec41decf3fb8ad2b5fb7e0054884ef26`, byte-identical to
`git show 76dc5116`. That is precisely the `$SCR/se6/baseline` trap W1's floor gate exists to catch.

| gate | measured | verdict |
|---|---|---|
| baseline spread (legs either side of the branch) | 97,275 and 99,257 f/s → **1.020x** (≤ 1.25) | pass |
| baseline floor vs the known 115,166 f/s | **0.853x** (≥ 0.25) | pass |
| load per **node** core | 38.86 over 48 cores → **0.81** (≤ 1.5) | pass |
| | | **USABLE** |

**The tier: `CraftaxFull` at 1,337 f/s**, at `bench_step3`'s own certified batch of 64, 10 px, one
torch thread, `node2352` — and it is a **lower bound**, not a clean number. The three gates say the
node was usable, not that it was quiet: loadavg was 38.9 over 48 cores, and contention is visible in
the band sweep taken in the same job, where 16 px at B=256 (265 f/s) came out *below* B=16 (589 f/s),
which is not a shape a renderer produces. The 10 px tier is quoted because it is the registered size
and its own sweep is monotonic; a quieter node would give the same number or a better one.

| | value | reading |
|---|---|---|
| `cores_cpu` (1,337 / 3,751 f/s of decode on that core) | **0.356** | **BELOW_G4** — where Step 3 also put `CraftaxClassic` (0.44-0.49) |
| `S_cpu` vs the **jitted** official renderer (350.9 f/s at 10 px) | **3.81** | `CPU_FASTER` band |
| `S_cpu` vs the **un-jitted** official (0.307 f/s) | 1,870 | not a number worth quoting |

**Neither `S_cpu` certifies anything until Sami rules**, and this report does not choose: on that
node the un-jitted bridge is 0.307 f/s against 350.9 f/s for the jitted renderer — a **1,144x**
difference read through the same `cls.reference_render` hook the parity claim requires. The tier is
taken on the NO_REFERENCE ladder as instructed and both denominators are recorded.

Per-band on the same node at B=256, 10 px (`sweep/bench_cpu_node2352….json`), which is the
diagnostic rather than the tier:

| band | f/s |
|---|---|
| level 0, full daylight | **8,652** |
| underground, fully lit | 8,655 |
| underground, dark | 1,661 |
| level 0, dusk | 1,660 |
| **level 0, night** | **418** ← the flagged slower band |
| sleeping | 1,113 |

### 17.3 The GPU tier

RTX 3090, `gpu2106`, `sweep/bench_cuda_gpu2106….json`:

| band | 10 px, B=4096 | 10 px, B=256 | 16 px, B=4096 |
|---|---|---|---|
| level 0, full daylight | **928,434 f/s** | 113,328 | 184,220 |
| underground, fully lit | 926,549 | 113,138 | 184,193 |
| underground, dark | 216,763 | 90,183 | 68,486 |
| level 0, dusk | 217,047 | 90,048 | 68,662 |
| **level 0, night** | **44,607** | 36,787 | 16,634 |
| sleeping | 94,183 | 53,923 | 33,501 |
| **mixed corpus — THE TIER** | **94,187 f/s** | 54,132 | 33,559 |

Fastest/slowest **20.8x** at B=4096, and against one core of decode on that node (4,882 f/s) the
mixed tier is **19.3 cores**. The ratio is wider on the GPU than on the CPU (6.75x on the Mac,
20.7x on `node2352`) for the expected reason: the uint8 gather scales with the card and the
per-pixel float chain does not scale as far.

## 18. What is owed, and what is not this workstream's

1. **The two escalations stand, unchanged and unfixed here.** The resolution-consistency scope is
   declared per the coordinator's ruling and flagged onward to Sami; the
   `reference_render`-as-denominator question carries §8.3 with it — W1's (correct) un-jitted hook has
   already moved `CraftaxClassic`'s own Step-3 denominator from a recorded 1,046 f/s to a measured
   4.62 f/s on the Mac and 0.31 f/s on `node2352`.
2. **`examples/swm_compat/run_swm_workflow.py`** has no Craftax arm — it drives TwoRoom and PushT and
   is CUDA shaped. The Gate F statement for this cell is the `swap_renderer` one instead, and it is
   **measured against the real wrapper** in §21.

### 18.1 Step-8 carry-overs (shared files; reported, not edited)

* **`conformance.check_reference_renderer_is_stable` hard-codes `device="cuda"`.** It builds
  `cls(batch=1, size=size, device="cuda")` whatever the caller is doing, so on a CPU host it raises
  `Torch not compiled with CUDA enabled` rather than running. It is a CUDA-host check by
  construction; this workstream records it `not_applicable` on CPU and runs it on the GPU node, where
  it returns `max_abs_diff: 0`. Making it device-aware is a one-line change to a shared testing
  module and belongs to whoever owns that file in Step 8.
* **Neither Step-7 corpus carries the `meta` key `examples/fidelity_step4.py:corpus_for` reads.**
  `corpus_for` loads a frozen corpus as `z["states"]` **and** `json.loads(str(z["meta"]))`; both
  `examples/results/step7/craftax-classic-full/corpus/` and
  `examples/results/step7/craftax-full/corpus/` write `states=` only, with the provenance in a
  sibling `corpus.json`. Adopting either into the Step-4 pipeline as-is would `KeyError`. **Both** are
  affected, so the fix belongs to the Step-8 freeze rather than to one workstream changing its own
  file and leaving the two inconsistent.
* **The paired CPU tier is a lower bound**, not a clean number — see §17.2.

## 19. Report per AGENT_COMMON

* **classification:** `PATCH`. `swm/CraftaxPixels-v1` goes from **gap** to **full**.
* **affected configs:** `CraftaxFull` only — a new cell. No existing config's pixels move;
  `git diff --name-only 76dc5116..HEAD -- stable_envs` is `envs/craftax_full.py` and
  `envs/_craftax_full_tex.py`, both new. `examples/results/step4|5|6`: **0 files changed**.
* **tests:** `tests/unit/test_craftax_full.py` **25 passed** (24 at the time of the Oscar run);
  `tests/unit` **772 passed, 39 skipped**, one pre-existing failure
  (`test_every_cited_scratch_path_resolves`, none of whose unresolved citations are this branch's).
* **Step-4 replay:** not applicable — this is a new cell with no frozen Step-4 corpus. The new
  Step-7 corpus is proposed for the Step-8 freeze.
* **tiers:** CPU `cores_cpu` 0.356 (BELOW_G4), `S_cpu` 3.81 against the jitted official, paired
  verdict USABLE; GPU mixed 94,187 f/s, 19.3 cores of decode. Reported, not certified, pending the
  denominator ruling.
* **pixels changed:** no existing pixel. The cell is new and is byte-identical to the reference.
* **semantic follow-up:** a **paired Gate H**, not a tripwire. This is new coverage rather than pixel
  drift: a world model gains a whole environment, and a tripwire is the wrong instrument for that.
* **SLURM job IDs:** 6397697, 6397916, 6397699 (and 6397698, cancelled by this workstream).

---

## 20. Independent review of `59ab465` — the six required changes

**APPROVE-WITH-CHANGES.** The reviewer reproduced parity 1.0 / 0 at 10 px and 16 px on their own
2,009-state rollout with their own `EnvState` → window extractor, on 90 level-shifted states across
all nine levels and on 49 hand-built edge states; purity and window sufficiency; a sampler power
test; Gate A and F including the **real** SWM wrapper; regression against the Classic frozen rows and
W1's corpus. All six changes landed in one follow-up commit and **none moves a pixel, a corpus or a
digest** — the frozen corpus still renders to `b3dcba4dea708d15` (10 px) and `43414708ce31a3d7`
(16 px), the same digests the Mac, `node2352` and `gpu2106` recorded.

| # | finding | what changed |
|---|---|---|
| 1 | the module docstring claimed **a mob and a projectile cannot share a tile** — false: the baked overlay table *is* the `mob x projectile` product and draws the pair | the real limits are stated instead: a **second entity of the same kind** on one tile (two mobs, or two projectiles), and the six-projectile bridge budget, now also in the class docstring. Re-measured here on an `EnvState` built with the pair on one tile rather than quoting review's figure: the disagreement is **confined to that tile**, and its magnitude depends on which pair collides — two melee mobs 156 at both sizes, two mob projectiles 237 at 10 px and 253 at 16 px, against review's 113–187 on its own pairs. The docstring therefore names the mechanism and gives both ranges rather than one spuriously precise number |
| 2 | "4 gathers and 3 selects" undercounted the mask lookups | one background gather plus three stencil layers = **seven `index_select`s and three selects**, in both docstrings |
| 3 | a citation to `tests/unit/test_craftax_full_night.py`, **which does not exist** | the pin is `test_craftax_full.py::test_the_vignette_is_a_reciprocal_multiply_and_not_a_divide` |
| 4 | `certification_state_distribution["uint8_fast_path_share"]` declared **2/6; it is 0.25** | fixed, with the mechanism written down: the lighting bands give 2/6, but `is_sleeping` is stratified independently on `i % 4` and a sleeping row takes the float chain, so three of every twelve rows stay on the gather. **The test tolerance that hid it — `< 0.1` against an error of 0.083 — is now one row (`<= 1/64`).** The corpus's own 14 of 64 agreed with 0.25 all along and was read as agreeing with 2/6 |
| 5 | findings cited **W1's branch** for `paired_verdict.py` — the one `tests/unit` regression this branch caused (broken citations 11 → 12) | cites this workstream's copy, and names W1's as the source of the gates |
| 6 | `check_state_dims_matter` at batch 1 reported **`inv_ench_bow` dead on 3 of 200 seeds** (100, 202, 248): the reference draws `arrow_enchantment_textures[bow_enchantment * (arrows > 0)]`, so a lone row with no arrows makes the slot unobservable | a **fifth sampler guard** — band 0 always carries at least one arrow — documented like the other four. Re-measured: **0 dead dimensions over 200 seeds at batch 1** |

**The fifth guard is a clamp and not a redraw, and that is the whole of why it is corpus-neutral.** A
redraw would consume another value from `rng` and shift every subsequent draw, changing the frozen
pool and every digest with it. `test_the_arrow_guard_leaves_the_frozen_corpus_untouched` regenerates
the 512-state pool from the recorded seed and asserts its sha256, and the selection's, against the
values `corpus.json` recorded **before the guard existed** — so corpus-neutrality is measured rather
than argued.

Nits also taken: the declared 1.53e-05 jit gap is now stated as a property of the 49-state phase-0
pool (review measured 3.05e-05 over 402 states); the last assertion of
`test_a_window_carrying_more_projectiles…` compared against an all-zero state and was vacuous — it now
compares six projectiles against eight and checks that **only tiles 6 and 7 move**; §17.2 labels the
CPU tier a lower bound; §18.1 records the two Step-8 carry-overs.

One thing turned up while fixing another: the tile scan in that same test sliced `moved[r0:r0+t, …]`
on an array whose first axis is the batch, so it was reading the wrong region entirely. It was caught
because the corrected assertion failed, which is the test doing its job.

---

## 21. Gate F against the real `CraftaxPixelsWrapper`

`examples/swm_compat/run_swm_workflow.py` drives TwoRoom and PushT and is CUDA shaped — it has no
Craftax arm — and this cell has no batched simulator, so its Gate F path is `swap_renderer`. The
statement worth making is therefore end-to-end rather than a method-signature one:

> step the **official** `swm/CraftaxPixels-v1`, read the render window out of the `EnvState` it
> produces, render that window with `CraftaxFull`, and compare against the observation the wrapper
> itself hands the workflow.

`stable_worldmodel` was installed into `$SCR/venvs/se7-craftax` from the **checkout**
(`176d54532b3c0a251a4f509b74f2fdacc143a63d`, rsynced to `$SCR/se7/stable-worldmodel`) rather than from
PyPI, because W5 found 0.0.6 on PyPI differs from it. Oscar job **6399614**, `node2350`, script
`scratch/step7/craftax-full/07_gate_f_swm.sbatch`, probe `scratch/step7/craftax-full/gate_f_swm.py`,
artifacts under `gate_f/`.

### 21.1 The install moved no pin, and the digests survived it

That venv is what **every** parity number on this branch was measured in, and it is shared with W1, so
the install is constrained rather than plain: the pins are frozen into a constraints file first, the
install runs under it, and the job **exits 1 if any of them moved**.

| | before | after |
|---|---|---|
| `craftax` | 1.6.1 | 1.6.1 |
| `gymnasium` | 1.3.0 | 1.3.0 |
| `jax` / `jaxlib` | 0.11.1 | 0.11.1 |
| `numpy` | 2.5.3 | 2.5.3 |
| `torch` | 2.14.0+cu126 | 2.14.0+cu126 |
| `torchvision` | (absent) | 0.29.0**+cu126** — new, not moved |

`gate_f/pins_before.txt`, `gate_f/pins_after.txt`. And the parity certify re-run **inside the modified
venv** reproduces the frozen corpus exactly — `exact_frac` 1.0, max abs 0, digests
`b3dcba4dea708d15…` (10 px) and `43414708ce31a3d7…` (16 px) — on three different nodes across the
attempts (`certify/certify_cpu_swmvenv_node2302|2349|2350….json`). Installing `stable_worldmodel` did
not disturb what the branch was certified against.

### 21.2 The contract the wrapper actually offers

| | measured |
|---|---|
| construction | `gymnasium.make('swm/CraftaxPixels-v1')` |
| `observation_space` | `Box(0.0, 1.0, (130, 110, 3), float32)` — **matches the registration and our `render()`** |
| `action_space` | `Discrete(43)` |
| `reset` info keys | `['env_name']` |
| `step` info keys | **69** — 67 `Achievements/*` plus `discount` and `env_name` |
| render window | 544 slots, **2,176 B** against a 42,900 B frame (**19.7x**), **0 tiles dropped** over the rollout |

### 21.3 The pixel comparison — and the one honest disagreement

121 frames from a 120-step random-policy rollout of the official env
(`gate_f/gate_f_swm.json`):

| compared against | `exact_frac` | max abs | frames identical |
|---|---|---|---|
| the **un-jitted** reference, on the same extracted windows | **1.0** | **0** | 8 / 8 |
| **the wrapper's own observation** (`round(obs * 255)`) | 0.99977 | **1** | **118 / 121** |
| — fully lit, awake band (1 frame) | **1.0** | **0** | 1 / 1 |
| — lit band (120 frames) | 0.99977 | 1 | 117 / 120 |

**The residual is the jit gap, and it is a property of the reference rather than a defect.**
`CraftaxWrapper` wraps `env.step` in `jax.jit`, so what a live workflow consumes is the **jitted**
frame — and under jit XLA fuses craftax's arithmetic differently from the un-jitted renderer this
cell is certified against, by ~1.5e-05 in float32, which straddles a uint8 rounding boundary wherever
the picture is compressed towards one value (§4.2). The signature fits that and nothing else: every
disagreeing frame is in a **lit** band, the one fully-lit awake frame in the rollout is
byte-identical, the error never exceeds **1**, and the same windows through the un-jitted reference
are byte-identical.

Two consequences, stated plainly rather than buried:

* **a workflow consuming the wrapper's observation and one consuming our `render()` will differ on
  about 0.02% of the pixels of a lit frame, by one level.** This renderer cannot drive that to zero
  without certifying against the jitted reference instead, which §4.2 explains is the wrong object —
  it disagrees with itself.
* it is now measured on 121 real frames rather than inferred. Before this run it had only been
  checked against craftax's own env (`gate_f/gate_f_swm_nowrapper.json`), which is what the wrapper
  wraps. The real wrapper gives the same answer.

### 21.4 Five job ids, four of them failures

Recorded because three of the four were mine and a reader should not have to guess:

| job | outcome |
|---|---|
| `6398628` | FAILED in 1 s — `$VENV/bin/pip freeze` under `set -o pipefail`; the venv was built by `uv venv` (W1's `01_build_venv.sbatch`) and **has no pip at all**. Rewritten for `uv` |
| `6398700` | FAILED — `--no-build-isolation` cannot build `stable_worldmodel`, whose backend is `uv_build` and is not in the venv. Build isolation turned on: it fetches the backend into a throwaway environment and leaves the target venv's runtime packages alone |
| `6398740` | COMPLETED, wrapper **not** exercised — `RuntimeError: operator torchvision::nms does not exist`. PyPI's `torchvision==0.29.0` is built against `torch 2.14.0+cu130`; this venv carries **+cu126**. The constraint correctly held torch still and left torchvision mismatched against it. Fixed the way W1 fixed torch: from the cu126 index with `--reinstall-package`, because `+cu130` and `+cu126` both satisfy `==0.29.0` and uv would otherwise audit the wrong wheel as a match |
| `6398918` | COMPLETED, wrapper **not** exercised — `ModuleNotFoundError: stable_worldmodel.data`. **My error**: the rsync used `--exclude data`, which rsync matches against *any* path component, so it dropped the `stable_worldmodel/data/` **package** along with the top-level `data/` directory it was meant to skip. Re-synced with anchored excludes |
| **`6399614`** | **COMPLETED, wrapper exercised.** The last blocker was `cv2`: `stable_worldmodel/wrapper/visual.py` imports it and it lives in the `[env]` extra, so the package cannot be *imported* on a base install. The extra was **not** installed wholesale — it pulls `gymnasium[all]` and this venv's `gymnasium==1.3.0` is a pin — only the modules the import chain reaches (`opencv-python pygame pymunk shapely`), under the same constraints file |

None of the five is a renderer or a wrapper defect; all five are packaging, and the Gate F substance
was identical in every run that produced numbers.

**One operational lesson, since it cost a restore.** `rsync`-ing results *back* from Oscar over a
directory that has been edited locally overwrites the local edits: pulling
`examples/results/step7/craftax-full/` clobbered this file with the copy that had been pushed before
the review follow-up, losing 190 lines. Recovered from git. Pull into a staging directory, or pull
only the specific new artifacts.
