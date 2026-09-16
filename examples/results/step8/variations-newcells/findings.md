# Step 8 — Workstream C findings: VariationMaps for the Step-7 new cells + OGB / Rocket / mazes

**Verdict: `PATCH_CANDIDATE` + `FINDING`** (no `BLOCKER`; nothing re-frozen; no renderer changed).

- baseline SHA: `43ac95238f7472137619f3e69e7636e11dd088b1`
- branch: `step8/variations-newcells`
- hardware/venvs: local Apple M5 (`se6-venv`, `se7-ogb-venv`, `se7-rocket-venv`, `MUJOCO_GL=glfw`)
  for every model-side and reference-pixel probe; Oscar `gpu2264`, **NVIDIA GeForce RTX 3090**,
  driver `575.57.08`, SLURM job **6443480** (overlapped on the held `se7-interactive-3090-b`
  allocation) for the GPU evidence, with `$SCR/venvs/se7-ogb` + `$SCR/se7/stable-worldmodel` on
  `PYTHONPATH` and `$SCR/venvs/se7-rocket`.

## What was asked and what was found

Five swm ids, **49 offered variation leaves**, every one now classified:

| bucket | count |
|---|---|
| `mapped` | 14 |
| `unsupported` | 13 |
| `dead_upstream` | 4 |
| `not_applicable` | 18 |

Plus `swm/CraftaxClassicPixels-v1` and `swm/CraftaxPixels-v1`, confirmed live to expose **0** leaves
(`CraftaxWrapper.__init__` sets `self.variation_space = None`), recorded as `not_applicable`.

Per-id counts, generated table and full evidence:
`examples/results/step8/variations-newcells/leaf_classification.json` and `leaf_tables.md`;
narrative in `docs/research/step8_variations_newcells.md`.

## The methodological finding, first, because it invalidated a whole sweep

**A probe that silently fails to apply a variation returns all zeros, which is indistinguishable
from a dead leaf.** The first Rocket sweep reported all nine leaves dead. They are not — the probe
passed `options={'variation_values': ...}` and `PFRocketLanding` reads only
`options={'variation': [names to resample]}`. Each id accepts a different mechanism
(`OGBCube`/`OGBScene`/`OGBMaze` take `variation_values`; `PFRocketLanding` and `SimplePointMaze`
take only `variation` and ignore `variation_values` silently). Every probe here is validated by
showing a known-live leaf actually moves before any zero is believed.

Two noise floors also had to be pinned: OGBench draws `cur_task_id` from the **global** `np.random`,
giving a 3,000–5,500 px reset noise floor larger than several real leaf effects; and `SceneEnv`'s
reset is not seed-reproducible at all.

## Evidence highlights

- **`environment.sky_color` is dead upstream, re-measured.** 5 seeds spanning the RGB cube,
  **0 of 691,200 px**, identical digests. Static picture agrees: declared once in swm, absent from
  PyFlyt, TinyRenderer's clear colour fixed.
- **Upstream defect — OGB `agent.color`.** `CubeEnv.modify_mjcf_model` computes
  `agent_color_changed = np.allclose(current, desired)` — the negation is missing — so the dirty flag
  is never set and the staged MJCF edit is never compiled. Measured: RED alone leaves `mat_rgba` at
  the default; RED co-varied with `light.intensity` applies. Dead in isolation, live when co-varied.
- **Upstream finding — OGBMaze in states mode.** The registered id defaults to `ob_type='states'`,
  and there **all five leaves move 0 px**: `compile_model()` rebuilds ogbench's `custom_renderer`
  only for `'pixels'`, so a states-mode render draws the pre-recompile model.
- **Upstream finding — two declared defaults that a plain reset does not produce.** Rocket's
  `pad.color` declares `(200,200,200)` but a plain reset loads the unmodified URDF (white);
  OGBMaze's `floor.pixel_encoding` declares 0 while OGBench constructs the env with the gradient on.
- **Apparent default conflict resolved by measurement.** OGBMaze `wall.color` reads back `[1,1,1]`,
  our default is `[0.6,0.6,0.6]`; ogbench rewrites wall rgba to `.6` only in pixels mode, and
  setting the leaf to 0.6 reproduces the default frame **bit-exactly (0 of 50,176 px)**.

## GPU evidence — job 6443480, RTX 3090, gpu2264

**Default reproduction, baseline tree vs branch tree, same card, same job:**

| cell | size | style_dim | frames digest |
|---|---|---|---|
| `OGBCube` | 224 | 0 → 0 | **IDENTICAL** `13f725346c21e3eb…` |
| `OGBScene` | 200 | 0 → 0 | **IDENTICAL** `dd22ec9fce3e9a69…` |
| `PFRocketLanding` | 480 | 16 → 16 | **IDENTICAL** |

**Isolation, `PFRocketLanding`, all five mapped leaves:**

| leaf | style slice | px moved | maxdiff | other dims unchanged |
|---|---|---|---|---|
| `pad.color` | [1,3] | 339 | 225 | yes |
| `rocket.body_color` | [4,3] | 105,504 | 253 | yes |
| `rocket.booster_color` | [7,3] | 2,019 | 152 | yes |
| `rocket.fin_color` | [10,3] | 51,625 | 147 | yes |
| `rocket.leg_color` | [13,3] | 663 | 242 | yes |

Worth noting: `pad.color` moves 339 px in **our** renderer while moving 0 in the reference probe —
our sampled states reach low altitude where the pad is framed. It is mapped, and live on our side.

## Pixel-movement classification

**No pixels changed, in any cell.** The patch adds a declaration module, one import/update line, tests
and artifacts; it touches no renderer, no env class and no `make_variation_space`, so no `style_dim`
or `style_digest` declaration moved. Measured rather than merely argued: the three CUDA cells are
byte-identical baseline vs branch above.

## Tests

- `tests/unit/test_variation_maps_newcells.py`: **58 passed, 5 skipped** (`se6-venv`, no swm).
- Live-enumeration rows pass where swm is importable: 3 in `se7-ogb-venv` (Cube, Scene, Maze), 2 in
  `se7-rocket-venv` (Rocket, SimplePointMaze) — all five ids confirmed against the live wrapper.
- Full `tests/unit`: **1069 passed, 2 failed, 12 skipped**. Both failures
  (`test_every_cited_scratch_path_resolves`, the `robocasa_visual/render_mjwarp.py` argparse row)
  reproduce **unchanged at the baseline commit** with the branch stashed. Pre-existing, not caused here.
- `scratch/step8/variations-newcells/generate_report.py --check` passes: the committed artifacts are
  a faithful render of the module, so the docs table and the JSON cannot drift from the code.

## Shared files touched

- `stable_envs/compat/worldmodel.py` — **one import + one `VARIATION_MAPS.update` line**, placed
  after `VariationMap`/`VARIATION_MAPS` exist. The `VariationMap` class itself is **not** modified
  (Workstream A owns it). A test asserts the five pre-existing maps still register.
- Everything else is new and owned by this workstream.

## Open items for the coordinator

1. **ESCALATION (plan §4) — leaves that look feasible and semantically required but need a renderer
   change.** Not declared unsupported on this workstream's authority:
   - OGB Cube/Scene `light.intensity`, `camera.angle_delta`, `cube.size` — each needs a **per-world
     MODEL array** inside one mujoco_warp context (measured: `light_diffuse`, `cam_mat0`/`cam_quat`,
     `geom_size` on mover geoms).
   - OGBMaze `light.intensity` — needs the fitted lighting law re-derived as a function of intensity
     and re-certified.
2. **Cheap follow-ups, if authorised.** OGB `cube.color` needs **no new machinery at all** — it
   writes mover `geom_rgba` with no recompile, exactly the route `_upload_button_colours` already
   uses. OGB `floor.color` needs only the plate cache key extended (it writes `tex_data` only, and
   the floor is static). Both move a certified cell's style declaration, so both want a deliberate
   re-baseline rather than a quiet edit.
3. **Ownership request to Workstream A.** `VariationMap` can express `ours = theirs * factor` and
   nothing else. OGBMaze `floor.pixel_encoding` is a switch we **already render**, with the opposite
   sense (`ours = 1 - theirs`), and it is recorded `unsupported` purely because the class cannot say
   so. One `inverted` flag, or an affine `(factor, offset)`, would make it `mapped`.
4. **Upstream findings bundle** — the `agent.color` inverted dirty flag; OGBMaze states-mode renderer
   staleness; `SceneEnv` reset ignoring its seed entirely (seed 0 and seed 7 give byte-identical
   models while two seed-0 resets differ); `pixel_encoding` and `pad.color` declared defaults not
   matching a plain reset; `PFRocketLanding` and `SimplePointMaze` silently ignoring
   `variation_values`; `sky_color` confirmed dead.
5. **Provenance caveat.** The swm trees on Oscar are not git checkouts, so the brief's
   `@176d545` could not be confirmed there by SHA. It was confirmed functionally instead: the live
   leaf inventory matches the recorded one for all three ids exercised on the card.
6. **Environment note.** No venv was rebuilt. `stable_worldmodel` was layered onto
   `$SCR/venvs/se7-ogb` via `PYTHONPATH=$SCR/se7/stable-worldmodel` because no single Oscar venv has
   swm + ogbench + mujoco_warp together; matplotlib was layered onto the local rocket venv the same
   way for `SimplePointMaze`. Nothing was installed into an existing venv.

## Semantic follow-up recommendation

**None.** No pixel moved in any cell, so every Step-5/7 semantic claim carries forward unchanged.

## SLURM jobs

`6443480` (overlapped `srun` on the held `se7-interactive-3090-b` allocation, gpu2264, RTX 3090).
No new GPU job was queued; no job was cancelled.

---

# Follow-up — the authorised OGB `cube.color` / `floor.color` implementation

**Verdict: `PATCH_CANDIDATE`.** Nothing re-frozen, no frozen corpus or threshold touched, no oracle
replaced. One renderer file changed: `stable_envs/envs/ogb_manip.py`.

- branch: `step8/variations-newcells-followup`, on top of C's `a09b407`
- merged: `step8/integration` at `302098f` (Workstream A), additively — A's
  `variation_maps` installer now registers this module through `MAPS`
- baseline compared against: `43ac95238f7472137619f3e69e7636e11dd088b1` (`$SCR/se8/baseline`,
  whose `.git_sha` file records exactly that)
- GPU evidence tree SHA: `a9e68cedfba32bae58a25a41dbc9182547f38a6a`, exported with `git archive`
  and carried in every JSON's `provenance.git_sha` — the null-SHA defect in the first evidence round
  is fixed by passing the SHA in rather than running `git rev-parse` in a non-checkout

## The declaration delta, stated before anything else

This is the part that is NOT a no-op, and it is the reason the follow-up needed measuring rather
than arguing. Both cells moved from an empty style space to a nine-dimensional one:

| cell | `style_dim` | `default_style_digest` | leaves |
|---|---|---|---|
| `OGBCube` | **0 → 9** | `cae66941d9efbd40…` → `8384a1141f52cbb7…` | `cube.color` (3), `floor.color` (2×3) |
| `OGBScene` | **0 → 9** | `cae66941d9efbd40…` → `8384a1141f52cbb7…` | `cube.color` (3), `floor.color` (2×3) |

The registered default row is
`[0.96, 0.26, 0.33, 0.08, 0.11, 0.16, 0.15, 0.18, 0.25]` on both cells, read from swm's own
`variation_space` at bundle time rather than copied into our source, so "our default is the
registered default" is true by construction. `cae66941…` is the digest of the *empty* float32 row,
which is what a zero-dimensional style space hashes to; it is quoted so the before-state is a value
and not a description.

Two Gate-A checks change applicability as a direct consequence:
`check_default_style_is_official` and `check_style_is_state` were recorded `not_applicable` ("env
declares no style") at Step 7 and are now applicable and must pass. See gate (d).

## Shape contract

swm declares `cube.color` as `(num_cubes, 3)` and both registered cells are one-cube configurations
(`CubeEnv(env_type='single')`, `SceneEnv` pinning `_num_cubes = 1`). We carry the leaf **flat, as
`(3,)`**; `floor.color` stays `(2, 3)`. The one-cube assumption is asserted where the default is read,
not assumed at the `Box`, so a multi-cube configuration fails loudly instead of silently dropping a
colour. `style_from_variation` flattens before slicing, so swm's `(1, 3)` lands in our width-3 slot
unchanged; `variation_from_style` emits a flat `(3,)` (and `(6,)` for the floor), which needs a
reshape before being fed back to a live swm env. That asymmetry is a property of
`compat.worldmodel`'s flat layout and is shared by every multi-row leaf in the repository — recorded
here as a boundary reshape rather than papered over, because this workstream does not own that
module.

## GPU evidence — job `6443480`, `gpu2264.oscar.ccv.brown.edu`, NVIDIA GeForce RTX 3090, driver `575.57.08`

Overlapped `srun` on the held `se7-interactive-3090-b` allocation. No job was queued and none was
cancelled. `MUJOCO_GL=egl`, `$SCR/venvs/se7-ogb` with `$SCR/se7/stable-worldmodel` on `PYTHONPATH`;
torch `2.14.0+cu126`, mujoco `3.12.0`, mujoco_warp `3.12.0`, warp `1.17.0`, numpy `2.5.2`.
Artifacts under `examples/results/step8/variations-newcells/gpu/colors_*.json`.

### (a) Default identity on the FROZEN Step-7 corpus — `colors_default_{baseline,branch}.json`

Not a fresh seed draw. The states are `examples/results/step7/ogb-manipulation/states/{cell}.npz`
key `states`, and each JSON records the array SHA-256, which matches
`examples/results/step7/closure2/gate_a/cuda_ogbmanip.json` **exactly** — the same 64 states Gate A
certified these cells on:

| cell | corpus `array_sha256` | style_dim base → branch | default frames digest |
|---|---|---|---|
| `OGBCube` | `560c1398a8a49a3f…` | 0 → 9 | **IDENTICAL** `e7e6804f9aa88fbccb45fda7932aab11` |
| `OGBScene` | `7a0cdff1849ee221…` | 0 → 9 | **IDENTICAL** `c60370c7d9ba80154d71f84c18cfb186` |

Both arms ran in the same job on the same card, minutes apart, reading the corpus from one tree.

### (b) Custom → default, every route a colour could survive — `colors_sequences.json`

Eight sequences per cell; every default frame must equal the digest in (a). **All 16 identical, no
mismatches.**

| sequence | OGBCube | OGBScene |
|---|---|---|
| custom (red cube) → default | ✔ | ✔ |
| both leaves custom → default | ✔ | ✔ |
| per-environment custom (half the batch) → uniform default | ✔ | ✔ |
| batch of 513 (> 2 × `MJW_CHUNK` = 256) custom → small default | ✔ | ✔ |
| big default batch self-consistent with the small one | ✔ | ✔ |
| second renderer instance's first default, after instance 1 went custom | ✔ | ✔ |
| instance 1's default, after instance 2 went custom | ✔ | ✔ |
| default after a custom render at a *smaller* batch size | ✔ | ✔ |

This is the defect the reviewer reproduced on CPU, measured on the card in every arrangement that
could have hidden it: chunk boundaries, batch-size changes, shared module caches and two renderer
instances in one process.

### (c) Isolation — `colors_isolation.json`

"Shows the plate" is not inferred from the picture: `render` composites
`where(mover, mjwarp_rgb, plate)`, so a plate pixel is exactly a pixel equal to the baked plate, and
the plate is taken from the same cache the render used.

| cell | leaf | slice | px moved | maxdiff | on plate px | on mover px | other style slots | states |
|---|---|---|---|---|---|---|---|---|
| `OGBCube` | `cube.color` | [0,3] | 31,812 | 124 | **0** | 31,812 | unchanged | unchanged |
| `OGBCube` | `floor.color` | [3,6] | 2,814,715 | 215 | 2,814,715 | **0** | unchanged | unchanged |
| `OGBScene` | `cube.color` | [0,3] | 13,795 | 123 | **0** | 13,795 | unchanged | unchanged |
| `OGBScene` | `floor.color` | [3,6] | 1,597,112 | 216 | 1,597,112 | **0** | unchanged | unchanged |

`cube.color` moves **no plate pixel** and `floor.color` moves **no mover pixel**, on both cells.
Both leaves move pixels, so neither is inert. After both, the default reproduces the (a) digest.
(`OGBCube` is 64 × 224², `OGBScene` 64 × 200².)
