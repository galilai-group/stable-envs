APPROVE

# Independent review — Step 8, Workstream E (final certification harness and provenance) + follow-up

- **Reviewed SHA:** `2af1e058394f7e008d53c9822759c4975603abfb` (branch `step8/certification-followup`,
  worktree `/private/tmp/stable-envs-step8-e-followup`)
- **Delta reviewed:** `git diff fce9036 2af1e05` — 47 files, +16829/-25. (`fce9036` and `43ac952` have
  identical trees for every file E touches, so the delta against the integration tip equals the delta
  against the baseline `43ac95238f7472137619f3e69e7636e11dd088b1`.)
- **Reviewer environment:** local CPU only, `/Users/samib/se6-venv/bin/python`, `MUJOCO_GL=glfw`,
  `-p no:cacheprovider`. Nothing in the reviewed worktree was edited, committed, stashed or reset; no
  Oscar/SLURM contact. All regeneration ran into a scratch directory.

## Verdict

**APPROVE.** The frozen contracts are byte-unchanged where they must be; the two applicability
mechanisms are narrow, declaration-driven, refuse incomplete declarations and keep the measurement in
the record; and every artifact that has a generator regenerates byte-identically apart from a HEAD
stamp (and one tempdir name). Residual defects below are cosmetic or are hand-offs for the
coordinator, none of them contract-adjacent.

## Merge gate (plan §18)

| # | item | status |
|---|---|---|
| 1 | explicit workstream scope | **PASS** — census/classifier, corpora freeze, applicability, downsample provenance, colour packet, order-invariance NO-OP, nvdiff classification, support matrix, follow-up rejudgment + Rocket re-sweep. One additive class attribute on C's `rocket_landing.py` (declaration only). |
| 2 | before/after diagnosis | **PASS** — every join recorded with sha256 in `gate_a_rejudged.json.applicability_joins`; paired baseline-vs-branch GPU legs in `oscar_paired_tests.json` (5 fixed, 0 new failures, errors unchanged). |
| 3 | targeted unit tests | **PASS** — `tests/unit/test_step8_gate_record.py` (370 lines, one case per required declaration key, refused joins, partition helper, Rocket/Craftax records, `partial` taxonomy) plus `tests/gpu/test_reproducibility.py:703` band-reader cases. |
| 4 | affected Gate-A/Gate-F checks | **PASS** — Gate A: `check_colors_apply_exactly`, `check_resolution_consistency`, the two visual-fidelity checks (re-measured, now judged). Gate F untouched. |
| 5 | frozen-corpus replay | **PASS** — `replay49.json`: 0 of 49 configurations changed verdict/failures/blocked/n-a/per-check record; verified by regenerating `universe8 --check` (step4 configs 49, `changed: []`, `removed: []`). |
| 6 | pixel-movement classification | **PASS** — none. No renderer/shader/atlas/geometry code touched; the colour check's verdict is unchanged and only gains `best_abs_diff`; the Rocket re-sweep reproduces `exact_frac` 0.9561371527777778 and `mean_abs` 0.9922828052662037 to every digit against closure2. |
| 7 | performance regression | **N/A, accepted** — only added compute is `d.min()` inside a check that already materialises `d`. |
| 8 | independent reviewer verdict | **CLOSED by this document: APPROVE.** |
| 9 | semantic follow-up | **PASS** — none warranted; no pixel moved. |
| 10 | dashboard update | **OPEN (coordinator)** — `examples/results/step8/dashboard.yaml` is not E's file. |

## 1. Contract integrity (verified by diff, not prose)

| file | 43ac952 → 2af1e05 | judgment |
|---|---|---|
| `docs/research/certification.md` | **no diff** | frozen Step-2 policy text byte-unchanged, incl. the mandatory table |
| `scratch/certification.json` | **no diff** | thresholds/ladders untouched |
| `tests/gpu/test_reproducibility.py` `DOWNSAMPLE_MARGIN` | `0.005` → `0.005` | unchanged |
| `DOWNSAMPLE_EXACT_FRAC` | 106 → 108 keys; **0 changed, 0 removed**; added `DMCQuadrupedWalk 0.811877`, `PFRocketLanding 0.956137` | permitted — plan §13 "Downsample provenance" mandates the Rocket entry; both floors ≤ the minimum measured value in the committed shards (Rocket 0.9561371527777778; Quadruped min 0.8118771922831632 over four closure/closure2 runs) |
| `scratch/step4/certify.py` | +18 lines, recording only (`applicability`, `judged_on_subset`, `all_bands_recorded_not_judged`) | permitted — no threshold, no comparator, no floor touched; the colour judge at `:164` is untouched |
| `examples/results/step4/**` | **no diff** (no M/D/R entries) | frozen Step-4 oracle intact |
| `examples/results/step7/**` | 3 **additions only** (`craftax-classic-full/corpus/states.meta.json`, `craftax-full/corpus/states.meta.json`, `rocket-landing/reference/states.meta.json`); no modify/delete anywhere under step7 | permitted — sidecar metadata for `fidelity_step4.corpus_for`, written **beside** the read-only arrays precisely so no committed `.npz` byte changes |
| `examples/results/step7/closure2/gate_a/cuda_rocket.json` | **no diff**, sha256 `244f375b…` | the superseded shard is kept byte-unchanged |
| `scratch/step3/certify.py` | classifier tables extended (`POST_STEP2_BACKEND_CLASS`, `NOT_A_BACKEND`, `NO_BACKEND_MODULE_CLASS`) | permitted — `scratch/certification.json`'s frozen S/R lists are read and never edited; class is assigned by what the backend does, and regeneration proves **0 class changes** |
| `examples/fidelity_step4.py` | corpus-meta sidecar reader; `PyFlyt`/`pybullet` added to recorded provenance; resolution applicability read | permitted, conservative (a corpus with neither `meta` nor sidecar now raises with an explicit message rather than `KeyError`) |
| `stable_envs/testing/conformance.py` | +231 lines: two applicability readers, `partition_colour_exactness`, `best_abs_diff` | see §2/§3 |
| `stable_envs/envs/rocket_landing.py` | +47 lines, one class attribute | declaration only, no render path touched |
| `tests/gpu/test_conformance.py` | assertion reads `partition_colour_exactness` | see §2 |

**Ruling (c) honoured:** no threshold, no `DOWNSAMPLE_EXACT_FRAC` value, no frozen corpus and no frozen
mandatory-table text changed.

## 2. Colour-exactness applicability — narrow, declaration-driven, evidenced

- `conformance.color_exactness_applicability` (`conformance.py:709`) reads `type(env)` only and
  **REFUSES** a declaration missing any of `leaves`, `mechanism`, `max_shading_factor`,
  `official_renderer_fails_the_same_check`, `evidence`; refuses `max_shading_factor` not strictly in
  (0,1); refuses `official_renderer_fails_the_same_check is not True`. Every refusal returns the
  fully-judged result, i.e. failure is in the conservative direction.
- `PFRocketLanding.color_exactness_applicability` (`rocket_landing.py:673`) names five leaves one by
  one — `pad.color`, `rocket.{body,booster,fin,leg}_color`. The env's variation space
  (`rocket_landing.py:330-340`) contains exactly these five `RGBBox` leaves plus `flame.ignited`, and
  all five are geoms shaded by TinyRenderer's law, so the declaration names exactly the lit leaves and
  no more. `undeclared_leaves_still_judged` is `[]` **because this env has no unlit RGB leaf**, not
  because the reader widened anything; a probe on an undeclared leaf leaves the check measured and
  failing (`rejudge_gate_a.py:_colour_join` "REFUSED: … stays failing").
- **Only one declaration exists in the repo** (grep over `stable_envs/`): `rocket_landing.py:673`. No
  other env gained one. The resolution declarations at `craftax.py:865` / `craftax_full.py:835` are
  pre-existing (Step 7) and unchanged in this delta.
- **Evidence reproduces.** I recomputed the ceiling independently: maximising
  `A + D·max(0,n·l) + S·max(0,r_z)²` with `A=0.6, D=0.35, S=0.05`, `l = normalise(-50,30,100)` over a
  2001×4001 grid gives **0.993567** at normal `(-0.3257, 0.1954, 0.9251)`, and `200→198, 33→32,
  17→16` — matching `colour_law_ceiling.json` to 6 dp. Cited evidence resolves and says what is
  claimed: `step7/rocket-landing/findings.md:251-257` (official PyFlyt frame's brightest body pixel
  `0.991·int(255·c)`), `:1209-1220` (law and defaults, confirmed empirically),
  `review_phase1.md:885-894` (wiring verified to 1.19e-08 against `getVisualShapeData`).
- **Measurement kept.** `check_colors_apply_exactly` now records `best_abs_diff` per probe; the
  re-sweep shard records all ten probes at 4/5/6/10 — **strictly positive**, and
  `tests/gpu/test_conformance.py:274-278` asserts each declared probe's `best_abs_diff is not None`.
- **Nothing else can suppress a failure by declaration.** The only consumers of either declaration are
  `conformance.check_colors_apply_exactly`, `tests/gpu/test_conformance.py`,
  `tests/gpu/test_reproducibility.py`, `examples/fidelity_step4._resolution` and
  `scratch/step8/rejudge_gate_a.py`. The frozen judge `scratch/step4/certify.py:164` still judges
  `not ev["colors_not_applied"]` unchanged (see defect D3).
- **Ruling (a) honoured:** the law-aware successor is labelled a PROPOSAL, not implemented, not
  mandatory (`step8_certification.md:581-593`, `rocket_landing.py` `"successor"` key).

## 3. Resolution applicability

- `conformance.resolution_consistency_applicability` (`conformance.py:1303`): no declaration → every
  state; unresolvable band → **every state** with the reason recorded (never silently narrowed);
  `"all bands": not_applicable` answered before any state is read. The exclusion-only fix is
  `conformance.py:1359-1361` (`if not any(status == "applicable" …): applicable = ~excluded`), covered
  by `tests/gpu/test_reproducibility.py:734-738` (case 3) and the typo case at `:749-755`.
- The CraftaxClassic subset judgment is a recorded join: `oscar_paired_tests.json`
  (`applicability_on_gpu_hardware.cells`, both devices, `judged.exact_frac 1.0` on `n_judged 6`,
  `all_bands.exact_frac 0.816382…`). `rejudge_gate_a.py:_resolution_join` refuses unless the recorded
  bands equal the class's bands and the recorded `all_bands` equals this shard's numbers, and records
  `source_sha256` of the join file and of every shard read. Both conditions held on regeneration
  (`result: JOINED`).
- **Regeneration (into scratch, worktree untouched):**
  - `gate_a_rejudged.json` — identical except `generated_at_head_sha` (`3f59bdd` committed vs `2af1e05` here).
  - `examples/results/step8/support_matrix.json` — identical except `generated_at_head_sha` (`22b27a3` vs `2af1e05`).
  - `docs/research/step8_support_matrix.md` — identical except the same stamp on line 11.
  - `census_class_diff.json` — **byte-identical** (no stamp).
  - `universe8_check.json` — identical except the staging tempdir name (`staged_into`).
  No other difference. The committed stamps are the merge parent `22b27a3`, consistent with the last
  commit regenerating before the final doc edit; content is confirmed current at the tip.
- **Ruling (b) honoured:** night states not applicable per the Step-7 ruling; both corpora kept and
  documented in `corpus_index.json.cells.CraftaxClassic.two_frozen_corpora` (Step-4 64×127 historical
  world-row oracle under zero-extension, Step-7 64×149 full observation), with `not_done: ["no
  retirement","no re-freeze","neither array rewritten"]`. I verified both arrays on disk: 64×127 and
  64×149, and both `array_sha256` (plus the Step-4 `npz_sha256`) match the index.

## 4. Rocket re-sweep shard

`examples/results/step8/certification/gate_a_resweep/cuda_rocket.json`, all verified from the artifact:

- provenance `slurm_job_id 6443480`, `slurm_nodelist gpu2264`, host `gpu2264.oscar.ccv.brown.edu`,
  RTX 3090, driver 575.57.08, torch 2.14.0+cu126, `reference_packages.PyFlyt 0.29.0`,
  `pybullet 3.2.7`, `git_commit 1ca386682a…` (an ancestor of the tip);
- corpus `array_sha256 f4707e54f3007b2e9afb1c236e932626373ee767eaa705161d53faa937d6eedf` — **equal** to
  the closure2 shard's corpus sha; same 64 states, same driver script;
- mandatory **14/14** measured and passing (`mandatory_failing: []`, verdict PASS);
- `check_resolution_consistency` identical to closure2 to every digit (0.9561371527777778 /
  0.9922828052662037 / max_abs 255);
- the rejudged record carries `supersedes = {source: examples/results/step7/closure2/gate_a/cuda_rocket.json,
  source_sha256: 244f375b…, previous_verdict: PASS}` and the closure2 shard is byte-unchanged vs
  43ac952 (`shasum` matches the recorded value).

## 5. Tallies (regenerated, not read)

From my own regeneration of `report.py`:

```
gate_a        {"FAIL": 3, "PASS": 36, "UNMEASURED": 1, "UNSUPPORTED_BY_DESIGN": 2}   # the 2 are the symbolic ids
variation_map {"complete": 21, "partial": 12, "missing": 5, "n/a": 2, "unknown": 2}
rows 42 · pixel_producing 40 · render_covered 40 · symbolic 2
```

FAIL rows: `swm/CheetahDMControl-v0` (`DMCCheetahRun`), `swm/HopperDMControl-v0` (`DMCHopperHop`),
`swm/WalkerDMControl-v0` (`DMCWalkerWalk`). UNMEASURED: `swm/HumanoidDMControl-v0`
(`DMCHumanoidStand`). All 12 `partial` rows are Fetch ids with `partial_leaves: ["background.color"]`.
Step-4 replay: `replay49.json` 0 of 49 changed; corroborated by `universe8 --check` (`changed: []`,
`removed: []`, step4 configs 49).

## 6. Census / classifier, downsample, nvdiff, order independence

- `refresh_census_diff.py` regenerates `census_class_diff.json` **byte-identically**: 100 rows
  (88 in the old census, 12 new), `class_changed 0`, `unknown_resolved 2` (`CraftaxFull`, `MiniHack`),
  `still_unknown 0`. Final classes: S 72 / R 28.
- `downsample_provenance.json`: Rocket `ADDED 0.956137`, Quadruped `ADDED 0.811877`, with the protocol
  (`downsample_protocol_states`: 8 states at `render_worker.STATE_SEED`, default appearance), margin
  0.005 and per-observation `protocol_states_sha256`. I re-read every cited closure/closure2 shard and
  each `exact_frac`/`mean_abs`/sha256 matches; both floors sit at or below the minimum observation.
  CraftaxFull, CraftaxClassic and DMCManipulatorBringBall are correctly recorded as NOT ADDED /
  UNCHANGED with reasons. (No generator script exists for this file; every number is cited and I
  verified them against the shards directly.)
- `nvdiff_ulp_classification.json`: `DOCUMENTED_LIMITATION`, `re_freeze: NONE`,
  `does_any_rendered_pixel_change: NO` with three cited evidence lines; the Step-7 re-freeze is
  recorded as remaining authoritative. Consistent with plan §13 and §19.
- `order_invariance_finding.json`: `NO_OP` — Step 2 froze it as `check_order_independence`
  (`certification.md:121-137`), the harness has it (`conformance.py:1252`, `certify.py:69`,
  `fidelity_step4.py:225`), and the Step-7 "absent in this build" claim is a misspelling false
  negative in `scratch/step7/ogb-manipulation/probe_remeasure_chain.py:189-194`. No new mandatory
  check was created — correct under plan §13.

## 7. Tests (exact tails)

```
$ MUJOCO_GL=glfw se6-venv/bin/python -m pytest tests/unit/test_step8_gate_record.py \
    tests/unit/test_certification.py tests/unit/test_step4.py tests/unit/test_repo_contracts.py -q -p no:cacheprovider
FAILED tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves
1 failed, 102 passed in 1.64s

$ ... -m pytest tests/unit -q -p no:cacheprovider --deselect tests/unit/test_examples.py \
    -k "not test_every_cited_scratch_path_resolves"
978 passed, 186 skipped, 112 deselected, 25918 warnings in 73.97s (0:01:13)
```

The single failure is the test the brief deselects. It already fails at the integration tip for
unrelated citations (`scratch/carracing/logs/`, `scratch/step6/gfxdraw/cost_floor.*`,
`scratch/step8/perf/*` and `scratch/step8/semantic/*` cited from `dashboard.yaml`,
`takeover/stop-integration.txt` and `variations-fetch/findings.md`). See defect D2 for E's
contribution to it.

## Remaining defects

- **D1 (cosmetic)** `scratch/step8/report.py:516` — `main()` raises
  `ValueError: … is not in the subpath of …` when `--json-out`/`--md-out` point outside `ROOT`. Both
  files are written first, so the outputs are correct; only the summary print dies. Reviewers
  regenerating into a temp dir hit it every time. Wrap the two `relative_to` calls.
- **D2 (cosmetic)** `examples/results/step8/certification/oscar/gputests_craftax.txt:1` and
  `.../gputests_vis311.txt:4` add two new entries to the already-failing
  `tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves`: the contract test's
  regex matches `scratch/…` inside the absolute Oscar paths `/oscar/scratch/sboughan/venvs/…`. Same
  false-positive class as the committed Step-6 gfxdraw logs, so this is precedent-consistent, not a
  contract issue — but it is two more lines of noise in a test the coordinator will eventually want
  green.
- **D3 (hand-off, not a defect in this patch)** The colour declaration is honoured by the GPU
  assertion (`tests/gpu/test_conformance.py:254`) and by the recorded join in
  `scratch/step8/rejudge_gate_a.py:185-221`, but **not** by the frozen live judge
  (`scratch/step4/certify.py:164`, deliberately untouched). Consequence: a *fresh* §21 full-universe
  certification sweep of `PFRocketLanding` will record `check_colors_apply_exactly` as FAIL unless it
  is routed through `rejudge_gate_a.py`. The direction is conservative (the frozen judge stays strict)
  and touching it would be a frozen-contract change requiring escalation, so this is correct as
  merged — but the coordinator must route the final pass through the rejudgment or escalate.
- **D4 (provenance nit)** The re-sweep's tree `1ca3866` is **not** byte-identical in `stable_envs` to
  the tip: `git diff --stat 1ca3866 2af1e05 -- stable_envs` shows Workstream B's
  `stable_envs/compat/variation_maps/fetch.py` (new) and one line of
  `stable_envs/compat/variation_maps/__init__.py` (`_MODULES = ("dmc_control", "fetch")`), both merged
  in afterwards. Neither can reach PFRocketLanding's render path, so the measurement stands; the doc
  should say "identical apart from B's Fetch VariationMap registration" rather than implying identity.
- **D5 (record nit)** `oscar_paired_tests.json` itself carries no sha256 of the closure2 shards it
  corroborates; the integrity argument rests on the numeric equality check in `_resolution_join` plus
  the consumer-side `subset_judgment_sha256`. That is sufficient, but recording the shard sha256s in
  the join file would make it self-contained.
- **D6 (observation, no action)** Because every RGB leaf of `PFRocketLanding` is a lit geom, the
  declaration covers all of them, so the "undeclared leaves stay judged" guard has no bite *on this
  cell*; the live regression detector there is the strictly-positive `best_abs_diff` assertion. Worth
  keeping in view if a future unlit leaf (e.g. a sky/background colour) is ever added to this env.

## Commands run

```
git diff --stat fce9036 2af1e05 ; git log --oneline fce9036..2af1e05
git diff 43ac952 2af1e05 -- docs/research/certification.md scratch/certification.json      # empty
git diff --name-status --diff-filter=MDR 43ac952 2af1e05 -- examples/results/step7/ examples/results/step4/   # empty
git diff 43ac952 2af1e05 -- tests/gpu/test_reproducibility.py scratch/step4/certify.py
git diff --stat 1ca3866 2af1e05 -- stable_envs ; git merge-base --is-ancestor 1ca3866 2af1e05
python  (parse DOWNSAMPLE_EXACT_FRAC at both SHAs; recompute the TinyRenderer ceiling; verify corpus sha256s)
python -c "rejudge_gate_a.build()"  -> scratchpad/regen/gate_a_rejudged.json      (diff: head stamp only)
python scratch/step8/report.py --json-out/--md-out <scratchpad>                   (diff: head stamp only)
python -c "refresh_census_diff.build()"                                           (byte-identical)
python scratch/step8/certify/universe8.py --check --out <scratchpad>              (diff: tempdir name only)
pytest tests/unit/{test_step8_gate_record,test_certification,test_step4,test_repo_contracts}.py -q
pytest tests/unit -q --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
```
