# Step 8 — final certification harness and provenance

!!! info "Workstream E"

    Branch `step8/certification-provenance`, from
    `43ac95238f7472137619f3e69e7636e11dd088b1`. Nothing here was pushed.

The question this workstream answers is narrower than "does the library work":

> does the certification harness measure **exactly what the frozen contracts say**, for the complete
> Step-8 universe?

Eight tasks, each with before/after evidence. Two came out `NO_OP` and one is an escalation, and
those three are as much of the answer as the five patches: a harness that adds a check nobody froze,
or re-freezes a pixel oracle to silence a test, is failing this question in a way that looks like
progress.

**The rule this workstream held to throughout: no tuning until pass.** Every bound value in the tree
is the value it was before. What changed is *which states a bound is asserted on*, *which backends a
class is computed from*, and *whether a corpus can be read at all*.

---

## Summary

| # | task | verdict | what moved |
|---|---|---|---|
| 1 | census + classifier | `PATCH_CANDIDATE` | census 88 → 100 envs; `unknown` 1 → 0; **0 cells changed class** |
| 2 | Step-7 corpora freeze | `PATCH_CANDIDATE` | all seven load; 3 metadata sidecars; **49 frozen configs untouched** |
| 3 | resolution applicability | `PATCH_CANDIDATE` | CraftaxClassic 0.816382 → **1.000000** on its declared band, against an unmoved 1.000000 bound |
| 4 | downsample provenance | `PATCH_CANDIDATE` | 2 entries added, 2 deliberately not added, 1 unchanged, 2 verified |
| 5 | `check_colors_apply_exactly` | **`FINDING` + escalation** | nothing — decision packet only |
| 6 | `check_order_invariance` | **`NO_OP`** | nothing — it was never missing |
| 7 | nvdiff x86 ULP | **`FINDING`** (`DOCUMENTED_LIMITATION`) | nothing — no re-freeze |
| 8 | support matrix | `PATCH_CANDIDATE` | generator + 42-row matrix |

**Replay on the 49 frozen Step-4 configurations: 0 of 49 changed.** Same verdicts (43 PASS / 5 FAIL /
1 PARTIAL), same failures, same blocked, same `no_declared_bound`, same `not_applicable`, same
per-check records, and `docs/research/step4_fidelity_baseline.md` regenerates byte-identical.
`examples/results/step8/certification/replay49.json`.

---

## 1. Census and classifier

Two separate defects produced one symptom.

**The census was stale.** `scratch/census/` predated six registered cells, so the universe and the
classifier could not see CraftaxFull, DMCQuadrupedWalk, DMCManipulatorBringBall, OGBCube, OGBScene or
PFRocketLanding at all. Regenerated across eight interpreters on `gpu2264` (alloc `6443480`, RTX
3090, driver 575.57.08): `se-step6`, `se-step5`, `se-s3b-fetch311`, `se-vis311`, `se7-craftax`,
`se7-craftax-gpu`, `se7-ogb`, `se7-rocket`. Union **88 → 100 envs, nothing lost.**

The new files are named so they sort *after* the existing ones, and `read_censuses` uses
`setdefault`, so **an env the old census already declared keeps its old declaration by
construction** — the preservation property is structural, not something that has to be re-checked
after every future census.

**The classifier keyed on module names.** `backend_class` consulted the frozen `backend_classes`
lists and then, if no backend matched, fell through to `NO_BACKEND_MODULE_CLASS`, keyed on the
module's last path segment. That table has `"craftax"` and not `"craftax_full"`, which is the whole
reason CraftaxFull came out `unknown` — a fact about where a file lives, not about what the renderer
does.

The fall-through itself had also quietly stopped working. `NO_BACKEND_MODULE_CLASS` only ever fires
for an env with an **empty** backend list, and once `craftax_full` imported `stable_envs.render.prng`
its list stopped being empty. `prng` is a counter-based random number generator: it draws the night
noise *field*, not the picture. It is not a renderer and must never decide a class.

Three additions, and **no frozen threshold is edited** — `scratch/certification.json` is read
unchanged and the ladder each class is judged on is exactly Step 2's:

- `POST_STEP2_BACKEND_CLASS` — the four render modules written after the frozen lists. `cpu3d` and
  `mjwarp_lightspec` are class S (3-D scene); `navdot` and the two `gfxdraw` variants are class R
  (2-D raster/tile). This restores the frozen *definition* to modules that postdate the frozen
  *lists*.
- `NOT_A_BACKEND` — `prng` and `rendernet`, listed explicitly rather than left to fall off the end.
- `NO_BACKEND_MODULE_CLASS` gains `craftax_full` and `minihack`, both atlas gathers on their own
  modules' account (`minihack.py:1-12`: *"the frame is a gather from a finite atlas indexed by the
  21×79 glyph grid… no sub-pixel geometry, no blending and no float arithmetic anywhere in the
  path"*). Step 9 moved the CraftaxFull implementation to `stable_envs/envs/_craftax_full.py`;
  `stable_envs.envs.craftax_full` remains a `sys.modules` alias, and the classifier table keeps both
  spellings so a census regenerated on either tree still classifies the cell.

### Before / after

`examples/results/step8/certification/census_class_diff.json`:

```text
class changed for a previously classified cell   0
unknown -> classified                            CraftaxFull (backends ['prng'] -> R)
                                                 MiniHack   (backends []       -> R)
still unknown                                    0
counts                                           S 61 -> 72,  R 26 -> 27,  unknown 1 -> 0
```

The backend declarations that *did* move are all additive and all class-inert: seventeen DM Control
cells gained `cpu3d` beside `mjwarp`/`raycast3d` (S either way), and the four `swm_nav` cells gained
`navdot` where they previously declared nothing (R either way — but now from the backend rather than
from the module name).

And the check that matters most: the configuration list `scratch/step4/universe.py` builds is
**byte-identical** on
the old and the new census — 49 configurations, 0 added, 0 removed, 0 entries changed.

---

## 2. Freezing the seven Step-7 corpora

`corpus_for` read `z["meta"]` unconditionally. Three of the seven corpora have no such key: both
Craftax cells wrote `states=` only with the provenance in a sibling `corpus.json`, and the Rocket
corpus is a multi-array reference file whose states live under `state13`. Adopting any of them raised
`KeyError`.

**The arrays are read-only and none was rewritten.** Their sha256 values are already quoted in
committed Step-7 Gate-A records — `PFRocketLanding`'s `f4707e54f3007b2e…` appears verbatim in
`closure2/gate_a/cuda_rocket.json` — so rewriting the bytes would invalidate a record that is
currently true. `corpus_for` now reads a metadata **sidecar** when the `.npz` carries no `meta` key,
trying `<stem>.meta.json` then `corpus.json` in the same directory, and raises a message that says
what a corpus without provenance cannot claim rather than a bare `KeyError`. `meta` inside the npz
still wins.

That fallback is provably unreachable for the frozen 49: **all 49 carry an inner `meta` key.**

`scratch/step8/certify/universe8.py` composes Step 4's universe with the seven. It joins through
`examples/results/step7/closure2/coverage_matrix.json` — 42 rows, `integration_head_verified` — and
**not** through `scratch/swm_matrix.json`, which is a Step-1 artifact whose six new rows still carry
an empty `stable_envs.envs`. That file is the SWM-integration workstream's join, and regenerating it
is a separate change with a separate review.

```text
corpus_for returns the frozen 64 for all seven, every sha256 matching the index
  CraftaxClassic 64x149   CraftaxFull 64x544   DMCManipulatorBringBall 64x15
  DMCQuadrupedWalk 64x23  OGBCube 64x24        OGBScene 64x30   PFRocketLanding 64x13
Step-4 universe:  49 in, 49 out, 0 changed, 0 removed, 6 added
```

### Escalation — CraftaxClassic has two frozen corpora

`examples/results/step4/states/CraftaxClassic.npz` is **64 × 127** and frozen in Step 4.
`examples/results/step7/craftax-classic-full/corpus/states.npz` is **64 × 149** and frozen in Step 7,
because the cell's state grew when the full nine-row world window was implemented. At most one of
them can be set on the cell as it stands, which is why the Step-7 closure replay already records
CraftaxClassic as `NOT REPLAYED` — that is the state of the world at the Step-8 baseline, not
something Step 8 caused.

**Retiring a frozen Step-4 corpus is a re-freeze of an authoritative oracle** (plan §19). Both are
recorded, neither is edited, and the choice is escalated.

---

## 3. Resolution applicability

Two Step-7 cells already declared, on the class and with the official renderer's own oracle result
cited, that the 2×-render-and-box-downsample **relation** does not hold in some or all of their state
space. Nothing read the declaration.

- **CraftaxClassic** — `light_level >= 0.5` asserted at the frozen bound; `light_level < 0.5` not
  applicable, because the noise field is drawn at the *frame's* resolution, so 2S draws a different
  field rather than a finer sampling of the same one. The official craftax renderer measures 0.1634
  on world rows there; so do we — *the same number*.
- **CraftaxFull** — all bands not applicable: `craftax.load_texture` resamples every PNG with
  `Image.NEAREST` at each size, so the official 2S atlas is not a 2× replication of its S one
  (official world rows 0.4817 in daylight, 0.0991 at night, max abs 191).

Judging a renderer against a relation its own reference does not satisfy measures the reference. Both
rulings are Sami's, 2026-09-15.

`conformance.resolution_consistency_applicability(env, states)` reads the declaration and returns
which states the bound may be asserted on. A band is `"<state name> <op> <value>"` resolved against
`env.state_names`, or `"all bands"` for the whole cell.

Three safety properties, each with a test, and all three point the same way — **a mistake must not be
able to silently narrow the judged set**, because a narrowed set looks like a pass:

1. an env with no declaration is untouched and returns "every state";
2. a declaration naming only the *inapplicable* band still leaves its complement judged;
3. a band naming a dimension the env does not have widens nothing and narrows nothing — it is
   reported, and every state stays judged.

Wired into all three consumers: `examples/fidelity_step4._resolution`,
`tests/gpu/test_reproducibility`, and the `_downsample_judge` in `scratch/step4/certify.py`.

### Before / after — and why this is the strongest evidence in the workstream

CraftaxClassic, the bound's own protocol draw of 8 states:

| judged on | `exact_frac` | against the frozen floor of 1.000000 |
|---|---|---|
| all 8 states (**before**) | 0.816382 | fails by 0.18 |
| the 6 at `light_level >= 0.5` (**after**) | **1.000000** | holds **exactly** |

The frozen bound was recorded before the night band existed. On the band it was recorded for, it
holds to the last decimal. **No bound value was edited to achieve that** — `DOWNSAMPLE_EXACT_FRAC`
and the 0.005 margin come out of `bounds.py` exactly as before. The applicability declaration
narrows the state *set*, never the bar.

---

## 4. Downsample provenance

Every value confirmed against the committed artifact it came from, measured on the bound's own
protocol (`conformance.downsample_protocol_states`: 8 states at `render_worker.STATE_SEED`, default
appearance). Full record in `examples/results/step8/certification/downsample_provenance.json`.

**Added**

| cell | value | provenance |
|---|---|---|
| `PFRocketLanding` | `0.956137` | cuda, RTX 3090, torch 2.14.0+cu126 / nvdiffrast 0.4.0; identical in the closure and closure2 sweeps; states sha `a5f128ad…`. That is `0.95614` to five decimals — the carry-over's value. No cpu row: the cell raises on `device="cpu"` by design. |
| `DMCQuadrupedWalk` | `0.811877` | the **minimum** over four committed runs (cpu 0.8118772/0.8118797, cuda 0.8120117/0.8120217, mujoco 3.11.0 and 3.12.0), so the floor holds on every device and reference pair measured; spread 2.5e-05, well inside the 0.005 margin. |

**Deliberately not added** — each recorded in the table itself, so a future agent does not "fix" it:

| cell | why | measured for the record |
|---|---|---|
| `CraftaxFull` | declares `resolution_consistency_applicability` "all bands: not_applicable" on the official renderer's own oracle. A floor here would be a number for a relation the reference does not satisfy in **any** band. | 0.794537 |
| `DMCManipulatorBringBall` | carries `PLATE_PATHS` and is not a `PLATE_REPRESENTATIVE`, so the test skips it exactly as it skips every other PARITY-tier DM Control plate cell. Adding a floor would create a gate the frozen protocol does not have. | 0.850842 |

**Unchanged**: `CraftaxClassic` stays `1.000000`; task 3's rule judges it where that holds exactly.
**Verified**: `OGBCube` 0.885782 and `OGBScene` 0.805306 both match `closure2` exactly. One provenance
nit recorded and not edited — their shared comment cites states sha `055d4057…`, which is OGBScene's
draw; OGBCube's is `ae1cc5e1…`.

---

## 5. `check_colors_apply_exactly` — escalation, no code change

Full packet: **`examples/results/step8/certification/colour_check_decision.md`**. Summary here; the
decision is Sami's.

The check asserts that a requested RGB triple appears **verbatim**, bit-exact, in at least one pixel.
On `PFRocketLanding` all five colour knobs fail, and the mechanism is arithmetic. bullet3's
TinyRenderer applies a **multiplicative** law:

```text
out_i = int( 255 · c_i · ( A + shadow·( D·max(0, n·l) + S·max(0, r_z)^spec ) ) · L_i )
```

I re-derived the ceiling rather than quoting it — maximising
`f(n) = 0.6 + 0.35·max(0, n·l) + 0.05·max(0, r_z)²` over a 2001 × 4001 grid of unit normals:

```text
max f = 0.993567   at n ~ (-0.3257, 0.1954, 0.9251),  n·l = 0.9904,  r_z = 0.9686
```

matching W6's figure to six decimals — and notably **not** at `n = l`, where `f = 0.987313`; the
maximum trades a little diffuse for a lot of specular and sits slightly off the light axis. So
200 → **198**, 33 → **32**, 17 → **16**. Both probe targets are out of reach in every channel,
**for any correct implementation of this reference** — and the official PyFlyt frames do not contain
the exact colour either. MuJoCo's specular *adds* white instead of multiplying, so its bracket reaches
and clamps at 1.0, which is why every MuJoCo-backed cell passes.

The knobs are independently known to be wired correctly: the cell's colour table matches
`getVisualShapeData` to 1.19e-08 on every geom. That is the entire property this check exists to
establish.

The corpus has been routing around this for a long time. **Nine env modules** explain in their own
docstrings that they deliberately declare no RGB leaf because their output path cannot carry a
literal colour — CarRacing (a 10.4× downscale turns `v` into `v−2`), Crafter, Procgen, PGX, Jumanji,
SimplePointMaze, and by contrast Sokoban and Overcooked, which *do* declare RGB leaves precisely
because their paths composite nothing. Read together they say the check's real subject is not "is the
knob wired correctly" but "is this renderer's output path colour-preserving".

**Recommendation: (c) declared, measurement-backed inapplicability for lit RGB leaves now** — which
applies Sami's own CraftaxClassic ruling to the same shape of problem — **with (b), the
law-aware/reference-compared form, commissioned as the successor** on its own review. Three
conditions so (c) is not a loophole: the declaration must carry the *measured* ceiling and its
artifact; the check must start recording `best_abs_diff` per leaf so a wiring regression still moves a
number; and the Gate-A reporting inconsistency below must be fixed in the same change.

### A second, independent finding

Step 7's Gate-A harness records `family_specific` checks without judging them, so the Rocket row reads
`result: PASS`, `failing_checks: []` while carrying ten `colors_not_applied` entries and a note calling
the failure *"neither a Gate-A check"*. That contradicts `docs/research/certification.md:143,149`,
where a family-specific check **is** part of Gate A once applicable. The cell is currently recorded as
both passing Gate A and failing a Gate-A check. That is a reporting defect independent of the
invariant question.

---

## 6. `check_order_invariance` — `NO_OP`

**Step 2 did freeze it as mandatory, and the harness does implement it.** The check is called
`check_order_independence`. Nothing is missing; nothing was restored; no new mandatory check was
created.

`docs/research/certification.md:121-137` is Step 2's frozen mandatory table (there is no
`docs/research/step2*.md` — `certification.md` *is* Step 2's deliverable, marked *"frozen policy, not
a measurement"*). Row 5, verbatim:

```text
| `check_order_independence` | permuting the batch changes per-state output |
```

It lives at `conformance.py:1252`, runs as item 5 of `fidelity_step4.mandatory()`'s fourteen, is in
`certify.py`'s `MANDATORY` tuple with a judge requiring `permutation_max_abs_diff == 0` and
`duplicate_max_abs_diff == 0`, is asserted in both `tests/gpu` and `tests/unit`, and stands at
**49 pass / 0 fail**.

No other check subsumes it. `check_batch_invariance` never shuffles — it compares index *i* of a batch
to *i* rendered alone. `check_batch_size_invariance` varies the window *width* with row order
preserved. `check_chunked_render_matches` varies the chunk boundary. Only this one permutes, and only
this one checks that a duplicated row renders identically twice.

The Step-7 "absent in this build" claim root-causes to
`scratch/step7/ogb-manipulation/probe_remeasure_chain.py:191`, which asks
`getattr(C, "check_order_invariance", None)` — a lookup that could never succeed against a function
named `check_order_independ**ence**`. The probe honestly reported a check that was never missing, only
misspelled in the driver's own tuple. It was filed under `recommended`, not `required`, and the same
Step-7 corpus contradicts it twice using the correct name.

Adding a `check_order_invariance` function would duplicate an existing mandatory check — which is
precisely what the plan forbids without escalation.

**Residual**: the Step-8 plan carries the same misspelling at `00_COORDINATOR_PLAN.md:606` and `:923`.
Normalising it would stop this recurring.

---

## 7. nvdiff x86 sparse-stage ULP — `DOCUMENTED_LIMITATION`, no re-freeze

Full record: `examples/results/step8/certification/nvdiff_ulp_classification.json`.

The stage is `torch.sparse.mm(S4, …)` in `Scene.clip` (`nvdiff.py:860`). `S4` has exactly four
nonzeros per row, so every output element is a sum of four products whose accumulation order the
kernel chooses by the dense operand's column count, `4·B`. `mvp_rows` is *not* the cause — it has been
matmul-free float64 since `d1e61a1` and is bit-identical at every width on every machine measured.

### The finding this workstream added

**The effect is host-dependent, not code-dependent, and it does not reproduce on the Step-8
baseline's own sweep host.** No existing prose remarks on this.

| host | arch | threads | widths tested | differing |
|---|---|---|---|---|
| `login010` | x86_64 | 1 | 1-10, 12, 14, 16, 20, 24, 32, 48, 63 | positions at 1-6; normals at 4 and 8 |
| `node1939` | x86_64 | 1, 2, 4 | 1, 2, 7, 8, 31, 32, 63 | positions at 1 and 2 (3-6 not sampled) |
| **`gpu2264`** | **x86_64** | **1 and 4** | **1-16, 31, 32, 63** | **nothing — 0 of 19** |
| Apple M5 | arm64 | 6 | 1, 2, 7, 8, 31, 32, 63 | nothing |

The `gpu2264` sweep ran at `fec4f6e`, and `stable_envs/render/nvdiff.py` is byte-identical between
`fec4f6e` and `43ac952` (sha256 `03e6d0cf9a971c3d…`), so that null result is evidence about the
baseline. A difference that appears on one x86 host and not another, at the same torch build and the
same source, is a property of that machine's CPU BLAS dispatch — which is exactly why no portable
bit-exactness assertion can own it.

**The right invariant at the intermediate stage is a numeric tolerance on cpu and bit-exactness on
cuda** — which is what `tests/unit/test_nvdiff_triangles.py:188-270` already asserts: one float32 ulp
at the quantity's own scale, written as an assertion rather than an exemption.

**Rendered pixels: none change.** The six locomotion cells re-render 6/6 bit-identical to the
re-freeze, and no shipped renderer draws through nvdiff on cpu — `PFRocketLanding` raises on cpu by
design and the locomotion cells render through nvdiffrast on the card.

**The Step-7 nvdiff re-freeze remains authoritative**: authorised by Sami under nine reviewer-verified
conditions, measured in one job on one card (`6410105`, `gpu2106`, RTX 3090), reviewer
APPROVE-WITH-CHANGES with RD-1 fixed in `4eb1c35`, merged at `fd2b8ef` — and it touched **zero** frozen
artifacts, because the need was measured rather than assumed.

The available fix (explicit contraction, `4dc3d4b`) is batch-invariant everywhere but moves four of
the six frozen cells on CUDA (Ant `mean_abs` +2.7e-07 / SSIM −3.8e-10; Walker2d, Humanoid and Swimmer
SSIM 1e-10…1e-09). That trades a non-observable cpu residual for four observable digest changes on
cells that are currently bit-identical to their oracle. **Plan §19's bar is not met**: no old oracle is
proven wrong, and the "corrected" implementation does not satisfy the intended invariant better on
the device that is certified.

**Residual**: the carry-over and the dashboard ruling still say "widths 1-3" and "cpu bitwise for
widths ≥ 4". The measurements say neither.

---

## 8. Support matrix

`scratch/step8/report.py` → `docs/research/step8_support_matrix.md` and
`examples/results/step8/support_matrix.json`, with the plan §22 columns. It never constructs an env,
never renders and never re-measures: it is a **join**, and every column names the file it came from.

The five words of §3 are kept apart, with the reason for each spelled out on the page. `FAIL` is
deliberately not one of them — a real failure is reported as a failure, not dressed as a limitation.
The two tier columns use a sixth word, `MEASURED`, and only those columns do, because **a tier is a
grade, not a gate**: calling `CPU_REGRESSION` "PASS" would read as *fast enough* and calling it "FAIL"
would read as a defect, and neither is what a tier says.

Two distinctions the generator makes rather than collapsing:

- `CPU_MISSING` splits on the `RAISES_BY_DESIGN` qualifier — an env that raises on `device="cpu"` by
  design is `UNSUPPORTED_BY_DESIGN` and *closed*; one that simply has no CPU backend written yet is
  `UNMEASURED` and is *work*.
- The backends column reports the **union** over every census while the class comes from the
  precedence-ordered reader. Precedence is the right rule for a verdict — it is what keeps a
  previously classified cell's class — and the wrong one for an inventory.

At this HEAD:

```text
rows 42 · pixel-producing 40 · symbolic 2 · render covered 40
Gate A        PASS 36 · FAIL 3 · UNMEASURED 1 · UNSUPPORTED_BY_DESIGN 2
GPU tier      MEASURED 40 · UNMEASURED 2
CPU tier      MEASURED 24 · UNMEASURED 15 · UNSUPPORTED_BY_DESIGN 3
VariationMap  complete 4 · missing 34 · n/a 2 · unknown 2
backend class S 28 · R 12 · n/a 2
```

The page lists what is `UNMEASURED` and which workstream closes each — Gate F is Workstream D,
semantic is H, the VariationMaps are A/B/C, the quarantined CPU tiers are G.

---

## Verification on hardware

Paired, both legs in the **same SLURM allocation on the same card**: baseline `$SCR/se8/baseline`
(a pristine `git archive` of `43ac952`) against `$SCR/se8/cert`, on `gpu2264.oscar.ccv.brown.edu`,
NVIDIA GeForce RTX 3090, driver 575.57.08, alloc `6443480`, `MUJOCO_GL=egl`. Selection
`-k "downsample or resolution_applicability or order"`.
Record: `examples/results/step8/certification/oscar_paired_tests.json`.

| venv | baseline | branch | fixed | **new failures** |
|---|---|---|---|---|
| `se-vis311` | 7 failed, 50 passed, 27 skipped | 4 failed, 53 passed, 28 skipped | 3 | **0** |
| `se-step6` | 2 failed, 70 passed, 52 skipped | **0 failed**, 72 passed, 53 skipped | 2 | **0** |

The five fixed are exactly the three cells this workstream's tasks 3 and 4 concern —
`test_resolution_downsample_agrees_within_the_measured_bound` for `CraftaxClassic` (now judged on its
declared band), `CraftaxFull` (now skipped on its all-bands declaration) and `DMCQuadrupedWalk` (now
has a measured bound instead of failing the missing-bound assertion). Error counts are identical in
both legs; the four remaining `se-vis311` failures are present in both and are unrelated.

The applicability reader was also exercised directly on GPU hardware, on both devices, under
`se7-craftax-gpu` (`examples/results/step8/certification/oscar/applicability_oscar.txt`) — and it
reproduces the local numbers exactly:

| cell / device | applicable | judged | `exact_frac` judged | `exact_frac` all bands |
|---|---|---|---|---|
| `CraftaxClassic` / cpu | yes | 6 of 8 | **1.000000** (mean 0.0, max 0) | 0.816382 |
| `CraftaxClassic` / cuda | yes | 6 of 8 | **1.000000** (mean 0.0, max 0) | 0.816382 |
| `CraftaxFull` / cpu | no | 0 | — | — |
| `CraftaxFull` / cuda | no | 0 | — | — |

### Local CPU suite

`tests/unit`, `se6-venv`, Apple M5: **2 failed, 1035 passed, 9 skipped** against the baseline's
**2 failed, 1034 passed, 10 skipped** — the same two pre-existing failures
(`test_every_argparse_script_can_print_its_own_help[robocasa_visual/render_mjwarp.py]` and
`test_every_cited_scratch_path_resolves`, the latter at the baseline's own 9 broken citations), and
the same 1044 total.

`mkdocs.yml` is the one file outside this workstream's ownership that had to change: adding two docs
pages makes `test_every_docs_page_is_reachable_from_the_nav` fail until they are in the nav. Two nav
entries, nothing else.

---

## What this workstream did not do

Recorded here because each was considered and declined, and a later reader should be able to see the
reason rather than rediscover the question:

- **no threshold, floor or margin was edited.** `scratch/certification.json` is read and never
  written.
- **no frozen Step-4/5/6 corpus was rewritten**, including the one (CraftaxClassic) where a
  successor exists and the current pairing does not work.
- **no pixel re-freeze**, and no second nvdiff re-freeze.
- **no new mandatory check**, including the one the plan asked about — because it already exists.
- **no change to `check_colors_apply_exactly`**, because the honest fix edits frozen Step-2 text.
- **`stable_envs/envs/*` and `stable_envs/render/*` were not touched.** Where a certification fix
  would need an env declaration change — the colour-check applicability declaration is the live
  example — it is reported for the coordinator to route rather than made here.

!!! note "Superseded, in one place only, by the follow-up below"

    After Sami ruled, the follow-up adds **one class attribute** to `stable_envs/envs/rocket_landing.py`
    (`color_exactness_applicability`), authorised by the coordinator, declaration only — and adds a
    reported-never-judged `best_abs_diff` to `check_colors_apply_exactly`, whose verdict is unchanged.
    No renderer code, and everything else on this list still holds.

---

## Escalations

**Escalations 1, 2 and 4 were ruled on by Sami on 2026-09-16 and are closed in the follow-up section
at the bottom of this page. They are left here as written so the question is readable beside the
answer.**

1. **`check_colors_apply_exactly`** — five numbered questions in
   `examples/results/step8/certification/colour_check_decision.md` §7. Adopting the recommendation
   edits frozen Step-2 text (`VARIATIONS_COMPLETE.complete_requires`), scoped to applicability only:
   no tolerance, no threshold.
2. **CraftaxClassic's two frozen corpora** — 64 × 127 (Step 4) and 64 × 149 (Step 7). Retiring the
   Step-4 one is a re-freeze under §19. Both recorded, neither edited.
3. **`scratch/swm_matrix.json` is stale** — six of 42 rows carry an empty `stable_envs.envs`. Not
   this workstream's file; Workstream D's join. Worked around by reading the Step-7 closure matrix.
4. **Gate-A reporting inconsistency** — a `family_specific` check on an applicable cell either is
   part of Gate A or is not; the Rocket row currently says both.

---

# Follow-up — the truthful Gate-A record (2026-09-16)

!!! info "Branch `step8/certification-followup`"

    After Sami ruled on the two escalations above. This section is **additive**: nothing earlier on
    this page was rewritten, because nothing earlier on it became untrue.

## What this is, and what it is not

It is a **rejudgment of historical measurements**. The measurements are the Step-7 closure2 Gate-A
shards, unchanged and read by sha256; the judge is `scratch/step4/certify.py:verdict`, unchanged; and
`scratch/step8/rejudge_gate_a.py` runs the second over the first. **Nothing was rendered.** No
threshold, no `DOWNSAMPLE_EXACT_FRAC` entry, no frozen corpus and no frozen mandatory-table text in
`docs/research/certification.md` was touched.

It is **not** a fresh Step-8 certification. A cell whose measuring interpreter recorded something
incompletely still records that gap here; rejudging cannot fill it.

What changed is *which group of checks is judged*. The Step-7 harness judged only the `mandatory`
group, so a cell could be recorded as **passing Gate A while carrying a failing Gate-A check** — and
`docs/research/certification.md:143,149` says a `family_specific` check **is** part of Gate A once
the env declares the capability. Both groups are now judged. That is escalation 4 above, closed.

## The two applicability declarations

Both are **read off the class** — `type(env)`, with no env constructed, so neither PyFlyt nor craftax
needs to be installed to see what a cell declares. Neither is invented by the harness, and both are
refused unless their evidence is attached.

### `check_resolution_consistency` on `CraftaxClassic`

Sami's Step-7 ruling: night states (`light_level < 0.5`) are **not applicable**, because the official
craftax renderer is itself resolution-inconsistent there and measures the same 0.1634 we do. The
class has declared this since Step 7 (`craftax.py:865`); §3 above made the harness read it.

The closure2 shard records only the **aggregate** over its 8-state protocol draw — there is no
per-state array in it, so the applicable subset cannot be recomputed from it. The subset judgment was
made on GPU hardware, on both devices, and is committed in
`examples/results/step8/certification/oscar_paired_tests.json`. The rejudgment **joins** it by
`(cell, device)`, and the join is recorded in `gate_a_rejudged.json.applicability_joins` with the
sha256 of every file read. It is **refused** unless:

- the committed subset judgment's recorded bands are exactly the class's bands, and
- its `all_bands` numbers are exactly this shard's numbers — i.e. the two are the same measurement.

| judged on | `exact_frac` | against the frozen floor of 1.000000 |
|---|---|---|
| all 8 states | 0.816382 | recorded as `all_bands`, **never judged** |
| the 6 at `light_level >= 0.5` | **1.000000** | holds exactly |

**Both frozen corpora are kept.** Step-4 `64 × 127` stays the historical world-row oracle under the
zero-extension interpretation; Step-7 `64 × 149` certifies the complete observation. No retirement
and no re-freeze; the roles are recorded in
`examples/results/step8/certification/corpus_index.json` under `cells.CraftaxClassic.two_frozen_corpora`.

### `check_colors_apply_exactly` on `PFRocketLanding`

Sami's ruling of 2026-09-16 on `colour_check_decision.md`: literal RGB equality is **not applicable**
for a style leaf whose official renderer applies a known lighting/material law that makes literal
equality mathematically unreachable — provided that is demonstrated **against the official renderer**
and scoped **narrowly** to those leaves.

The evidence, all committed:

| what | value | where |
|---|---|---|
| the law | `out = int(255·c·(A + D·max(0,n·l) + S·max(0,r_z)^spec)·L)`, truncated | bullet3 `TinyRenderer.cpp` `Shader::fragment` |
| measured ceiling | **0.993567**, recomputed over a 2001 × 4001 normal grid | `colour_law_ceiling.json` |
| consequence | 200 → 198, 33 → 32, 17 → 16 — unreachable at every pixel, for any correct implementation | same |
| official renderer | the **official PyFlyt frame fails this check too**: brightest body pixel `0.991·int(255·c)` | `step7/rocket-landing/findings.md:251-257` |
| wiring, independently | colour table matches `getVisualShapeData` to **1.19e-08** on every geom | `review_phase1.md:885-894` |

`PFRocketLanding.color_exactness_applicability` names the **five** lit leaves one by one and nothing
else. `conformance.color_exactness_applicability()` reads it and **REFUSES** it unless it carries all
five of `leaves`, `mechanism`, a measured `max_shading_factor` **strictly below 1**,
`official_renderer_fails_the_same_check`, and `evidence` — there is a unit test per key, because a
declaration that costs nothing to write is a way to silence a check. A probe naming a leaf the class
does not declare leaves the check **measured and failing**.

The check now also records **`best_abs_diff` per leaf** — reported, never judged, and the verdict is
byte-identical — so a knob that comes off its primitive still moves a number where no bound is
asserted. Sami's condition. The Step-7 measurement predates it, so the record says so explicitly
rather than showing a blank.

## The successor law-aware check — a PROPOSAL only

Option (b) of the decision packet: *the requested triple appears verbatim, **or** this renderer's
response to the knob matches the **official** renderer's response to the same knob*. It is strictly
stronger than literal equality for a lit cell, because it compares against a measured reference
rather than a constant, and a wrong primitive, a scale error and a channel swap all still fail it.

**It is not implemented, it is not mandatory, and it is not part of the frozen Step-2 contract.** It
needs the official renderer available at check time (so a `blocked_environment` path, like the parity
checks have), and it is a genuine extension of a Gate-A check in policy-frozen `tests/gpu/`. It
deserves its own review, and presenting it retroactively as part of the Step-2 mandatory contract
would be exactly the dishonesty this workstream exists to avoid.

## The reference-provenance gap the rejudgment exposed — measured, and closed

With the colour check ruled not applicable, two `family_specific` checks the mandatory-only harness
never judged came into view on `PFRocketLanding`:

```text
check_visual_fidelity                 blocked_environment
check_visual_fidelity_under_variation blocked_environment
```

Not because the numbers were bad — they were well inside the declaration — but because the cell
declares `reference_versions = {PyFlyt 0.29.0, pybullet 3.2.7}` and **the measuring interpreter never
recorded either**. `fidelity_step4.provenance` tracked eleven packages and neither was one of them,
so `block_on_reference_mismatch` reported *"version not recorded by the measuring interpreter"* — and
"could not confirm" is the blocked case, not the passing one. A **provenance gap, not a renderer
failure**, and no amount of rejudging could close it: only a new measurement could.

### The Step-8 re-sweep

`examples/fidelity_step4.provenance` now records `PyFlyt` and `pybullet`, and the cell was
**re-measured on that harness**, on the same card, through the same driver:

| | |
|---|---|
| artifact | `examples/results/step8/certification/gate_a_resweep/cuda_rocket.json` |
| driver | `scratch/step7/closure/gate_a_new_cells.py --cells PFRocketLanding --device cuda --n-states 64 --pool 256` — the **same** script that produced the closure2 shard |
| corpus | the **same** 64 states, sha256 `f4707e54f3007b2e9afb1c236e932626373ee767eaa705161d53faa937d6eedf` |
| host / GPU | `gpu2264.oscar.ccv.brown.edu`, NVIDIA GeForce RTX 3090, driver 575.57.08 |
| SLURM job | **6443480** (the held allocation; `srun --overlap`, no new job submitted, nothing cancelled) |
| interpreter | `$SCR/venvs/se7-rocket` — **PyFlyt 0.29.0, pybullet 3.2.7**, torch 2.14.0+cu126, nvdiffrast 0.4.0 |
| tree | `$SCR/se8/cert-followup`, git `1ca3866` |
| runtime | 114.0 s, `status: measured` |

The versions recorded are **exactly** the versions the class declares, so the two checks are now
judged rather than blocked. The closure2 shard is **kept, unchanged**, as the history of what was
measured when; the rejudgment reads it, sees a Step-8 re-sweep for that `(cell, device)`, judges the
re-sweep, and records `supersedes` with the closure2 path, sha256 and previous verdict.

### `PFRocketLanding` / cuda — every check

**Gate A: `PASS`.** 14 of 14 mandatory pass; 9 `family_specific` checks resolve as:

| check | status |
|---|---|
| `check_visual_fidelity` | **pass** — mean_abs 7.814 ≤ 10.0, within8 0.9607 ≥ 0.91, SSIM 0.8630 ≥ 0.80, against the class's own declaration, reference version now **confirmed** |
| `check_visual_fidelity_under_variation` | `no_declared_bound` — the class declares no separate `variation_fidelity`, so the numbers (mean_abs 7.369, within8 0.9606, SSIM 0.8943) are recorded and nothing holds them |
| `check_resolution_consistency` | **pass** — `exact_frac` 0.956137 ≥ 0.956137 − 0.005; identical to closure2 to every digit |
| `check_variations_do_something` | **pass** |
| `check_colors_apply_exactly` | **`not_applicable` (declared, evidence attached)** |
| `check_backend_agreement`, `check_task_object_visibility`, `render_parity_with_official_renderer` | `not_applicable` — the cell has one backend, declares no task object, and is VISUAL tier |

`cpu` stays `REFUSED_BY_DESIGN`: the cell raises on `device="cpu"` on purpose.

**And the colour check's distances are now measured**, which is what the ruling asked for — the
Step-7 shard recorded only the knob names:

| leaf | `best_abs_diff` (summed absolute channel error, closest pixel) |
|---|---|
| `rocket.body_color`, `rocket.booster_color` | 4 |
| `pad.color` | 5 |
| `rocket.leg_color` | 6 |
| `rocket.fin_color` | 10 |

All ten probes still miss — that is the point, and it is what the 0.993567 ceiling predicts. Every
distance is **strictly positive**, which is the property a test now asserts: a leaf that could reach
its colour would not need the declaration, and a zero would mean the declaration had gone stale.

## Gate A over the 40 pixel-producing rows

| | Workstream E, `9d286ce` | after the rejudgment and the re-sweep |
|---|---|---|
| `PASS` | 36 | **36** |
| `FAIL` | 3 | 3 |
| `UNMEASURED` | 1 | 1 |

**No row's Gate-A verdict differs from E's matrix** — but every verdict now includes the
`family_specific` group, the two applicability rulings are honoured with their evidence recorded in
the row, and the Rocket's reference provenance is *recorded* rather than assumed. The three
`trackcom` `FAIL` rows and the one `UNMEASURED` row (`HumanoidDMControl`) are untouched. The only
other column that moved is `VariationMap`, where the A and B merges take 29 ids out of `missing`:
17 to `complete` and the 12 Fetch ids to **`partial`**, because their maps classify
`background.color` as a leaf whose upstream effect is reproduced only in part. That is
`DOCUMENTED_LIMITATION` — known, measured, bounded — and the column reports it as its own word with
the leaf named in the row, rather than rolling it into `complete` and claiming a mapping is done
while a leaf is knowingly approximate.

```text
variation map   complete 21 · partial 12 · missing 5 · n/a 2 · unknown 2
```

### The GPU assertion, before and after, on hardware

`tests/gpu/test_conformance.py::test_color_knobs_apply_the_requested_color`, whole selection, same
interpreter (`$SCR/venvs/se7-rocket`), same card, in one `srun` on job `6443480`:

| tree | result |
|---|---|
| `43ac952` (baseline) | 3 failed, 28 passed — `MountainCar`, `MountainCarContinuous`, **`PFRocketLanding`** |
| this branch | 2 failed, 29 passed — `MountainCar`, `MountainCarContinuous` |

**Exactly one test flipped, and it is the declared cell.** The two `MountainCar` failures are
pre-existing, unrelated, and declare nothing — which is the point: the assertion was narrowed by a
declaration, not weakened for everyone.

**Replay of the 49 frozen Step-4 configurations through the frozen judge: 0 of 49 changed any
field** — verdict, failures, blocked, `no_declared_bound`, `not_applicable` or any per-check record.

## Merge gate (plan §18)

| # | item | state |
|---|---|---|
| 1 | explicit workstream scope | E follow-up: the Gate-A record, two applicability declarations, the support-matrix join. One additive class attribute on `rocket_landing.py` (C's file), authorised by the coordinator, declaration only. |
| 2 | before/after diagnosis | above, per row and per check; every join recorded with sha256 in `gate_a_rejudged.json.applicability_joins` |
| 3 | targeted unit tests | `tests/unit/test_step8_gate_record.py`, 30 tests: one per required declaration key, two refused-join cases, five on the partition helper the GPU assertion uses, the Rocket `not_applicable`-with-evidence and re-sweep assertions, the CraftaxClassic applicable-subset assertion, three on the variation column's `partial` taxonomy |
| 4 | affected Gate-A / Gate-F checks | Gate A: `check_colors_apply_exactly` (GPU assertion now reads the declaration through `partition_colour_exactness`), `check_resolution_consistency`, and the two visual-fidelity checks, re-measured and now judged. Gate F: untouched (Workstream D) |
| 5 | frozen-corpus replay | 49 of 49 identical; both CraftaxClassic corpora kept and neither rewritten |
| 6 | pixel-movement classification | **none.** No renderer, shader, atlas or geometry code was touched; `check_colors_apply_exactly`'s verdict is byte-identical and only gains a reported diagnostic. The re-sweep confirms it on hardware: `check_resolution_consistency` reproduces 0.956137 to every digit against the closure2 shard |
| 7 | performance regression check | **not applicable:** no hot-path code changed. The one added computation is `d.min()` inside a check that already computes `d` |
| 8 | independent reviewer verdict | **OPEN** — none exists for this follow-up |
| 9 | semantic follow-up decision | **none warranted.** No pixel moved, so no downstream semantic result can have. The re-sweep is a new measurement of an unchanged renderer, and it reproduces the closure2 numbers |
| 10 | dashboard update | **OPEN** — `examples/results/step8/dashboard.yaml` is the coordinator's file |
