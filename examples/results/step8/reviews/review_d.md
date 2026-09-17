# Step 8 — Workstream D (SWM integration / Gate F) — independent review

Reviewed SHA: `579e399` (branch `step8/swm-integration`, worktree
`/Users/samib/stable-envs-step8-swm-integration`), diffed against `2b4b5c3`.

**APPROVE-WITH-CHANGES**

## Ten-item merge-gate table

1. explicit workstream scope — YES. `git diff 2b4b5c3 579e399 --name-only` touches only
   `docs/research/step8_swm_integration.md`, `docs/research/step8_upstream_findings.md`,
   `examples/results/step8/swm/*`, `mkdocs.yml` (nav lines only), `scratch/step8/swm/*`,
   `stable_envs/envs/craftax.py`/`craftax_full.py` (additive only), `tests/swm/*`. No renderer,
   VariationMap, or other workstream's file touched.
2. before/after diagnosis — YES. findings.md states before (D "NOT STARTED", all 42 rows
   `swm_api_gate_f: UNMEASURED`) vs. after (`gate_f_matrix.json`, sourced per column).
3. targeted unit tests — YES, `tests/swm/test_gate_f_matrix.py` (5 tests), all pass.
4. affected Gate-A/Gate-F checks — Gate F is the deliverable; Gate A is read-only joined, not
   mutated. Confirmed no Gate-A input file is in the diff.
5. affected frozen-corpus replay — none required; no renderer/corpus code touched. Confirmed.
6. pixel-movement classification — correct: only two docstring/class-attribute additions in
   `craftax.py`/`craftax_full.py`, no rendering code path changed (see defect D1 below for a
   caveat: the attribute itself is inert data, verified).
7. performance regression check — not applicable, correctly declared (no hot-path code changed).
8. independent reviewer verdict — this document.
9. semantic follow-up decision — none needed; correct, no pixel value changed.
10. dashboard update — pending coordinator action; out of this workstream's scope, correctly
    deferred.

## Commands run and tail output

```
$ cd /Users/samib/stable-envs-step8-swm-integration
$ git diff 2b4b5c3 579e399 --stat
 docs/research/step8_swm_integration.md             |   62 +
 docs/research/step8_upstream_findings.md           |   63 +
 .../results/step8/swm/default_roundtrip_cpu.json   |  396 +++
 examples/results/step8/swm/findings.md             |  186 +
 examples/results/step8/swm/gate_f_matrix.json      | 3621 ++++++++++++++++++++
 .../step8/swm/variation_classification.json        |  907 +++++
 mkdocs.yml                                         |    5 +
 scratch/step8/swm/build_gate_f.py                  |  525 +++
 scratch/step8/swm/probe_default_roundtrip.py       |   86 +
 .../step8/swm/probe_variation_classification.py    |   54 +
 stable_envs/envs/craftax.py                        |   33 +
 stable_envs/envs/craftax_full.py                   |   27 +
 tests/swm/test_gate_f_matrix.py                    |  113 +
 13 files changed, 6078 insertions(+)
```

Regeneration into a fresh in-process run (not the committed file), diffed modulo `baseline_sha`:

```
$ MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -c "... build() vs committed gate_f_matrix.json ..."
IDENTICAL modulo baseline_sha: True
```

Confirmed by reading `build_gate_f.py` end to end: it does a pure dict/JSON join over
`support_matrix.json`, `coverage_matrix.json`, `certification.json`, `gate_a_rejudged.json`,
`reference_arm.json`, `default_roundtrip_cpu.json`, `variation_classification.json`,
`gpu_evidence.json` / `variations-newcells/gpu/*` — no `stable_envs` import, no env construction, no
rendering call anywhere in the file.

Column tally recomputed directly from the committed JSON (independent of the doc's own tally text):

```
registry_resolution              {'MEASURED': 42}
registered_default               {'MEASURED': 38, 'NOT_APPLICABLE': 4}
swm_wrapper_reset_appearance     {'UNMEASURED': 28, 'NOT_APPLICABLE': 2, 'MISMATCH(upstream)': 12}
variation_map                    {'MEASURED': 38, 'NOT_APPLICABLE': 4}
state_observation_api            {'MEASURED': 40, 'NOT_APPLICABLE': 2}
device_behavior                  {'UNMEASURED': 35, 'MEASURED': 7}
wrapper_interaction_path         {'MEASURED': 40, 'NOT_APPLICABLE': 2}
```

All seven counts match the claimed tally in `findings.md` and `docs/research/step8_swm_integration.md`
exactly.

Tests:

```
$ MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/swm/test_gate_f_matrix.py
.....
5 passed in 0.18s

$ MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_repo_contracts.py
FAILED tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves
1 failed, 10 passed in 0.27s
```

`test_every_cited_scratch_path_resolves` is confirmed pre-existing and untouched by this branch:
`git diff 2b4b5c3 579e399 -- tests/unit/test_repo_contracts.py` is empty (the file is not in the
diff at all), and every path in the failure output (`scratch/carracing/logs/`,
`scratch/sboughan/...`, `scratch/step6/gfxdraw/cost_floor...`) is cited from other Step-6/7/8
workstreams' findings files, not this one. No `/oscar` absolute path appears anywhere in
`tests/swm/*`, `scratch/step8/swm/*`, or the two new docs pages (grepped explicitly).

Craftax cross-check against the installed `stable_worldmodel` source:

```
$ grep -n "jit\|_step_fn\|_reset_fn" /Users/samib/stable-worldmodel/stable_worldmodel/envs/craftax/craftax.py
56:        self._reset_fn = jax.jit(_reset, backend=self.backend)
57:        self._step_fn = jax.jit(_step, backend=self.backend)
```

Matches `SWM_WRAPPER_GATE_F["swm_wrapper_source"]` verbatim. The un-jitted 100%/max-diff-0 baseline
the declaration cites is confirmed in `examples/results/step8/semantic/identity_recheck.json`
(`byte_identical: true, exact_frac: 1.0, max_abs: 0` for both Craftax cells at both registered
sizes). The jitted 99.7854%/99.7225% (Classic) and 48/49-states (Full) numbers quoted in the new
`SWM_WRAPPER_GATE_F` dict are pre-existing in `stable_envs/envs/craftax.py` lines 47-48/923/1299 and
`craftax_full.py`'s own `parity_note` — not invented by this workstream, just surfaced into a new
class attribute. The `git diff` for both files is additive only (new docstring paragraph + new class
attribute); no existing line changed.

## Evidence spot-check (10 ids, families)

| id | column checked | evidence cited | verified against source |
|---|---|---|---|
| `swm/FetchPush-v3` | state_observation_api | `.../closure2/replay/certify/certification.json` (config=FetchPush) | `check_state_roundtrip/check_default_style_is_official/render_parity_with_official_renderer` all `pass` in that file — confirmed |
| `swm/OGBCube-v0` | state_observation_api | `gate_a_rejudged.json` (rejudgment) | **DEFECT D1 — see below**: the cited per-cell record has `checks: {}` (empty); status is `MEASURED` regardless |
| `swm/PFRocketLanding-v0` | state_observation_api | `gate_a_rejudged.json` | same DEFECT D1 |
| `swm/CheetahDMControl-v0` (DMC trackcom) | state_observation_api | `reference_arm.json` (Workstream F) | `gate_a: PASS`, `source_within` points at `dmcvis.json`, sha256 present — confirmed real |
| `swm/HumanoidDMControl-v0` (DMC trackcom) | state_observation_api | `reference_arm.json` | same, confirmed |
| `swm/CartPoleControl-v1` (classic-control) | state_observation_api | bulk `certification.json` (config=CartPole) | all 3 checks `pass` — confirmed |
| `swm/CraftaxPixels-v1` | state_observation_api / Craftax declaration | `gate_a_rejudged.json` + `SWM_WRAPPER_GATE_F` | checks non-empty and `pass`; JIT gap cross-checked against upstream source above — confirmed |
| `swm/CraftaxSymbolic-v1` | NOT_APPLICABLE across renderer columns | N/A (symbolic, no cells) | confirmed empty `stable_envs_cells`, all renderer-shaped fields NOT_APPLICABLE, enforced by `test_symbolic_ids_are_not_applicable_across_renderer_columns` |
| `swm/PushT-v1` | state_observation_api | bulk `certification.json` (config=PushT) | all 3 checks `pass` — confirmed |
| `swm/OGBMaze-v0` (maze, PointmazeLargeNavigate/Stitch) | state_observation_api | bulk `certification.json` (config=PointmazeLargeNavigate), noted shared-scene join for Stitch | checks `pass`; the "shares one rendered scene" note is consistent with `support_matrix.json`'s own `join_evidence` |
| `swm/FetchPush-v3` family (12 ids) | swm_wrapper_reset_appearance | `variations-fetch/findings.md` ("swm Fetch declared leaf defaults are not the official appearance") | confirmed by reading Workstream B's cited finding; matches `MISMATCH(upstream)` classification |
| U1/U2/U3 (OGB reset defects) | upstream findings | `ogb-manipulation/findings.md:1026-1038,1041-1046,1048-1053` | opened directly; exact content and line ranges match (`reset() accepts seed and discards it`, `action_space` property rebuild, `reset_variation_space(seed=None)`) |
| U9 (Craftax JIT) | upstream findings | installed `stable_worldmodel` source + `identity_recheck.json` | confirmed above |

## Defects

**D1 (real, narrow — the reason for APPROVE-WITH-CHANGES rather than APPROVE).**
`scratch/step8/swm/build_gate_f.py:294-305` (`state_observation_api_status`), the `rejudged` branch:

```python
if c in rejudged:
    recs = rejudged[c]
    cpu = next((r for r in recs if r["device"] == "cpu"), recs[0])
    checks = cpu["judgment"]["checks"]
    per_cell[c] = {"status": "MEASURED", "checks": {...}, ...}
```

always prefers the `device == "cpu"` record when one exists, and always reports `status: "MEASURED"`
regardless of whether that record's `checks` are populated. For `OGBCube`, `OGBScene`, and
`PFRocketLanding`, `gate_a_rejudged.json`'s **cpu** record is `"gate_a": "BLOCKED"` with `"checks":
{}` (no CPU backend exists for these scenes — confirmed by reading the `why` field: e.g. "OGBCube has
no CPU backend... use device='cuda'"). The **cuda** record for the same three cells, present in the
same file, has real `check_default_style_is_official` / `check_batch_invariance` / etc. statuses
(`pass`/`not_applicable`). The generator picks the empty-checks CPU record over the populated CUDA
one, so `gate_f_matrix.json` rows for these 3 of 42 ids (`swm/OGBCube-v0`, `swm/OGBScene-v0`,
`swm/PFRocketLanding-v0`) show `state_observation_api.status = "MEASURED"` with `checks: {}` —
literally none of `check_state_roundtrip` / `check_default_style_is_official` /
`render_parity_with_official_renderer` is actually attested by the cited record, even though a
`source` path is present. This is exactly the "bare PASS without evidence" failure mode the brief
asked me to check for: the status word says MEASURED, the source path resolves, but the specific
claim (those three checks) is not backed by anything in that record. `tests/swm/test_gate_f_matrix.py`
does not catch this because its evidence check only requires a `source`/`why`/`per_cell` key to
exist, not that a `MEASURED` cell's own `checks` dict be non-empty.
Fix: for GPU_ONLY_CLASSES (already enumerated in the same file, line ~77-83), prefer the `cuda`
record, or more generally prefer whichever device record has non-empty `checks`/is not `BLOCKED`,
falling back to CPU only when CPU actually ran. Scope of fix is confined to
`state_observation_api_status`; does not require touching any other column or re-running anything.

**D2 (report-accuracy, minor).** `examples/results/step8/swm/findings.md:175-179` claims
`tests/unit/test_repo_contracts.py` gives "9 passed, 2 failed" with both failures pre-existing. My
run on this exact commit gives **10 passed, 1 failed** (`test_every_cited_scratch_path_resolves`
only). I confirmed the pre-existing/untouched nature of that one real failure (file not in the
diff), so the substantive claim ("this branch introduces no new failure, and the fix for the
nav-orphan check is the two additive nav entries") holds up — but the specific pass/fail count
printed in the report does not match a live re-run and should be corrected. Does not affect
`gate_f_matrix.json` or any Gate-F claim.

No other defects found: scope boundary is respected, the Craftax declaration is additive and its
factual claims verify against the actual upstream source and existing certified numbers, the build
script is a genuine pure join (no construction/rendering), citation line ranges for the sampled
upstream-findings items resolve exactly, and the seven column tallies recompute identically from the
raw JSON.

## Judgment: do the honest UNMEASURED columns make the 42-id claim untruthful?

**Correctly declared limitations, not an untruthful integration claim.** The workstream's stated
goal (plan section 12 / findings.md) is narrower than "all 42 ids fully verified on every axis": it
is "every one of the 42 SWM ids has an explicit, sourced Gate-F/API status" — and the Step-8 closure
criteria (plan section 23, "SWM integration") ask for exactly that: "all 42 IDs have an explicit
API/Gate-F status" and "upstream defects are separated from stable-envs defects," not "all measured."
Against that bar:

- `device_behavior` UNMEASURED=35 is honestly attributed to a real, named upstream limitation of a
  sibling artifact (`support_matrix.json`'s `supported_devices` populated for only the 7 Step-7-new
  cells `gate_a_rejudged.json` covers) — recorded as STEP9_CLEANUP #2, not hidden, not inferred, not
  rounded up to MEASURED.
- `swm_wrapper_reset_appearance` UNMEASURED=28 is honestly scoped to "no committed Step-8 measurement
  of the wrapper's own reset() exists yet for these families" (as distinct from the mapping
  round-trip, which IS measured) — and the one family where it WAS measured (Fetch, 12 ids) reports
  `MISMATCH(upstream)`, not a false PASS.
- Every cell in the matrix carries a status word from a 4-word vocabulary that has no "PASS" member
  at all — `UNMEASURED` cannot be mistaken for a passing check by a downstream reader, and both docs
  pages state this in their opening line ("never a bare PASS").

The one place this workstream's own honesty claim ("never a bare PASS without evidence") does not
hold up under direct verification is D1 above — 3 of 42 ids' `state_observation_api` is `MEASURED`
with literally empty checks, sourced from a `BLOCKED` record. That is a real defect and the reason
for APPROVE-WITH-CHANGES, but it is narrow (3/42 ids, one column, one function, real evidence for
the same claim exists two keys away in the same already-cited file) and does not change the overall
verdict that the workstream's declared UNMEASURED counts are truthful limitations rather than a
cover for an untruthful "all 42 integrated" claim.

## Re-review at 1c2639c

Delta reviewed: `git diff 579e399 1c2639c` — `scratch/step8/swm/build_gate_f.py` (+17/-3),
`examples/results/step8/swm/gate_f_matrix.json` (regenerated), `docs/research/step8_swm_integration.md`
and `examples/results/step8/swm/findings.md` (wording only). No other files touched.

**D1 fixed.** `state_observation_api_status` now filters `rejudged` records to
`judged = [r for r in recs if r["judgment"].get("checks")]` first, and only then prefers a `cpu`
record among the judged ones; if no record for a cell carries checks the cell is correctly
downgraded to `UNMEASURED` with an explicit `why`, never a bare `MEASURED`. Regenerating in-process
(`build()`) and diffing against the committed `gate_f_matrix.json` modulo `baseline_sha`:
**identical**. Confirmed the three previously-flagged rows now cite the CUDA records with real
checks:

- `swm/OGBCube-v0` → `source_within: examples/results/step7/closure2/gate_a/cuda_ogbmanip.json`,
  `checks.check_state_roundtrip: "pass"` (plus `check_default_style_is_official`/
  `render_parity_with_official_renderer: "not_applicable"`, itself a legitimate status, not a
  fabricated pass — OGBCube declares no style).
- `swm/OGBScene-v0` → same `cuda_ogbmanip.json`, `check_state_roundtrip: "pass"`.
- `swm/PFRocketLanding-v0` → `source_within: examples/results/step8/certification/gate_a_resweep/cuda_rocket.json`,
  `check_state_roundtrip: "pass"`, `check_default_style_is_official: "pass"`.

Tallies recomputed from the regenerated JSON are unchanged: `state_observation_api` is still
`{MEASURED: 40, NOT_APPLICABLE: 2}` (the fix changed *which* record backs 3 rows, not their status,
since all three do have a judged CUDA record). All other six column tallies unchanged from the
original review.

Tests:

```
$ MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/swm/test_gate_f_matrix.py
5 passed in 0.18s
$ MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_repo_contracts.py
1 failed, 10 passed in 0.39s   # test_every_cited_scratch_path_resolves, same pre-existing failure as before
```

**D2 fixed** in substance: `findings.md:176` now reads "10 passed, 1 failed", matching a live rerun.
Trivial residual nit, not worth another round-trip: the same line still says "BOTH failures
pre-exist" (leftover plural from the old wording) directly above a count of one failure — cosmetic
only, does not misstate anything material.

**Verdict: APPROVE**
