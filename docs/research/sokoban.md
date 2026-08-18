# Sokoban / Boxoban

Verdict: **landed, exactly, and it is the second family in the repo whose SIMULATOR is as exact as its
renderer.** `gym-sokoban` composes its frame by blitting one of seven 16x16 PNG assets per grid cell,
so it belongs on the tilemap path (`stable_envs/render/tilemap.py`, the one MiniGrid uses) rather than
on the polygon IR: pushing it through `Scene2D` would mean re-deriving, approximately, images the
package already ships exactly. Its transition is a nine-entry decision table over a seven-value
alphabet, which is a thing a port either reproduces on every `(state, action)` pair or gets wrong.

Package: `gym-sokoban` 0.0.6 (installs against the legacy `gym`, not `gymnasium`).
Module: `stable_envs/envs/sokoban.py`. Probes: `scratch/sokoban/`.

## 1. Results

| quantity | value |
|---|---|
| tile alphabet | **7** images, 16x16x3 uint8 |
| whole atlas | **5,376 bytes**, **10** distinct colours |
| render parity vs `render(mode="rgb_array")` | **interior 100%, all pixels 100%, max abs diff 0** |
| step parity vs `SokobanEnv._push`/`_move` | **9,216 transitions, 0 mismatched cells** |
| GPU throughput, B = 4,096, one H200 | **8,405,843 frames/s** (645.6 GB/s) |
| official CPU renderer | **197 frames/s** (5.08 ms a frame) |
| speedup | **42,700x** |
| state | 100 numbers against a 76,800-byte frame, **768x** |
| render at 2S, box-downsampled to S | `exact_frac` **1.000000** |

The reconstruction check is the point, and it is run over many rooms rather than one: rebuilding the
frame by indexing a 7-entry atlas with `room_state` reproduces the official render bit for bit across
120 frames spanning 12 generated rooms. That is the definition of a tilemap env.

Why the official renderer is 197 f/s for a 100-cell gather: `room_to_rgb` calls
`pkg_resources.resource_filename` and `imageio.imread` for **all seven PNGs on every single call**. The
frame is not what costs 5 ms; re-decoding the atlas is.

## 2. The state is one index per cell, and that is not the env's own layout

`room_to_rgb` picks its tile from `room_state` *after* rewriting id 5 to 6 wherever
`room_structure == 2` — a player standing on a target — so the drawn tile is a function of **two**
grids, `room_state` and `room_fixed`. Storing both would double the state and, worse, would carry
dimensions that cannot change the picture: `room_fixed` is invisible under a box.

The 7-value tile id is the join of the two, and it is invertible. Measured over 1,200 cells x 120 real
states, the only `(room_state, room_fixed)` pairs that occur are:

| tile | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| | wall | floor | empty target | box on target | box | player | player on target |
| `room_state` | 0 | 1 | 2 | 3 | 4 | 5 | 5 |
| `room_fixed` | 0 | 1 | 2 | 2 | 1 | 1 | 2 |

so both grids are recovered from the tile id exactly. `reference_render` asserts that round trip on
every driven state rather than trusting it, because a reference that did not reach the requested scene
would turn every parity number into a comparison of two different pictures.

## 3. Ten colours, and why they are ten separate knobs

`room_to_rgb` **composites nothing**: it assigns whole 16x16 tiles into a uint8 buffer. Every pixel has
exactly one writer, there is no supersampling and no box filter, so `palette[label]` is not an
approximation of a tile — it *is* the tile. That makes a recolouring variation exact by construction
and makes `check_colors_apply_exactly` a meaningful check here, which it is not for MiniGrid (whose
every tile is downsampled and, in the POV view, blended with the field-of-view overlay).

Measured colour census, with the tiles each colour appears in:

| colour | painted in | name |
|---|---|---|
| `(0,0,0)` | floor (256 px), box_target, player, player_on_target | `background` |
| `(41,202,26)` | player (148 px), player_on_target | `player` |
| `(176,61,0)` | wall (168 px) | `wall_brick` |
| `(176,154,142)` | wall (88 px) | `wall_mortar` |
| `(215,103,0)` / `(215,114,0)` / `(215,196,0)` | box, box_on_target | `box_dark` / `box_mid` / `box_light` |
| `(238,0,0)` | box_on_target (116 px), box_target, player_on_target | `target_dark` |
| `(239,22,22)` | box_target (52 px) | `target_mid` |
| `(242,26,26)` | box_target (5 px) | `target_light` |

`_Atlas` asserts `palette[labels] == the shipped PNGs` at the default palette before it will build, so
a wrong constant cannot survive construction — and it would otherwise be invisible, because parity is
measured at the default palette, where the label map is unused.

## 4. `sample_states` is stratified, and the reason is measured

Under random actions a box sits on a target in only **7.0%** of states and the player stands on one in
**9.4%**, so **221 of 500** 16-state batches drawn uniformly from a rollout pool were missing at least
one tile id. A missing tile means the knob that colours it cannot move a pixel, and
`check_colors_apply_exactly` would then pass or fail on the draw rather than on the renderer.

Every state in the pool is one the env actually reached; only *which* of them are selected is arranged,
and the repair loop replaces a row at a time until all seven ids are present.

## 5. The transition, by enumeration

A random rollout is a weak test of a decision table — it walks into walls far more often than it pushes
a box onto a target, and `box_type = 3` is exactly the branch a port gets wrong. So the check is the
cross product (`python scratch/sokoban/enumerate_step.py`):

| | |
|---|---|
| states | 1,024, from `sample_states` |
| actions | all 9 (no-op, 4 pushes, 4 moves) |
| transitions | **9,216**, i.e. 921,600 cells |
| mismatched cells | **0** |
| outcomes exercised | pushed onto target 131, pushed off target 9, plain push 354, walked 3,947, blocked 4,775 |

That is the same bar MiniGrid set (10,752 transitions, 0 mismatches) and for the same reason: the
alphabet is small enough that the cross product IS the proof, so there is nothing to sample.

`exact_frac` is the only statistic worth quoting for a decision table; a "median error" for a grid would
be a category mistake.

One boundary case is decided in the port rather than in the reference, and it is unreachable in a
generated room. `_move` indexes `room_state` with **no bounds check at all**, so a player one cell
outside the grid would wrap under numpy's negative indexing or raise; `_push` bounds-checks only the
FAR edge (`new_box_position >= shape`) and, when it trips, returns *without* falling back to `_move`.
The port treats any cell outside the grid as a wall, so a push whose box would land off the edge falls
through to the move, which then finds a wall and does nothing. The two agree on every reachable state,
because every room `generate_room` produces is surrounded by a wall ring.

`reference_step` drives `_push`/`_move` directly rather than `SokobanEnv.step`, which would re-decode
seven PNGs per step to produce an observation nothing reads.

## 6. Not ported

**Room generation.** `room_utils.generate_room` runs a reverse-play search and is where Sokoban's CPU
time actually goes — but it is a level generator, not a transition, and this repo ports renderers and
transitions. Boxoban shares both the renderer and the transition and changes only how rooms are
generated, so it is covered by the same atlas and the same `step`.

**`room_to_tiny_world_rgb`**, which renders at 1 px per cell with flat colours. It is a second atlas of
seven entries and would be a second registered class; nothing needs it yet.

## 7. The dependency is hard, and the module says so at import

There is no degraded mode here. The seven PNGs *are* the renderer, and `sample_states` rolls the
official env out because a Sokoban room is generated by a reverse-play search with no closed form. So
`stable_envs/envs/sokoban.py` probes the whole runtime at **import** — imageio, the seven files
themselves, and `SokobanEnv` constructing — and raises `ImportError` if any of it is missing.
`stable_envs/envs/__init__.py`'s auto-discovery then records it, `import_errors()` names it, and the
env is cleanly **absent**.

That is not a style preference, it is a measured one: registering anyway and letting the dependency
fail at render time turned one missing package into **15 test failures**, every one of them raised
inside a fixture where an absent env is indistinguishable from a broken one. `import X` succeeding is
not the same as X working, which is why the probe constructs a `SokobanEnv` (with `reset=False`, since
what is in question is the class rather than the level generator) rather than importing the module and
hoping.

## 8. Sizes

The art is 16x16 and the reference has no tile-size parameter, so there is no reference frame at any
other resolution. `render(size=)` accepts whole multiples — `160 * k` for a 10x10 room — and
implements them by replicating each art pixel `k x k`, which is still exactly what the reference draws.
Anything else raises. That is also why the 2S-render-and-box-downsample check is exactly 1.000000 here:
the 2S frame *is* the S frame at twice the size, so the box filter inverts it.
