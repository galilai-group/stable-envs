# Step 8 — Workstream D: SWM integration / Gate F

**Verdict: `FINDING`.** Every one of the 42 SWM ids now has an explicit, sourced Gate-F/API status
across seven columns. No new `RELEASE_BLOCKING` integration defect was found by this workstream;
the strongest new fact is confirming and dating the Fetch reset-appearance `MISMATCH(upstream)`
against the registry, and the two Craftax classes now carry a stated, bounded (≤1 uint8 step) gap
between what is certified and what a live `stable_worldmodel` caller observes.

- Baseline SHA at start: `2b4b5c3a62b998a1c27918832bfd2311f608c533` (`step8/integration`, seven
  workstreams merged, D "NOT STARTED" per the dashboard)
- Branch: `step8/swm-integration`, worktree `/Users/samib/stable-envs-step8-swm-integration`
- Mode: **evidence-consumption**. The 42 ids were not re-validated from scratch; every column below
  either reads a committed Step-7/Step-8 artifact by path, or runs a cheap, CPU-only, newly
  committed check when no existing artifact covered the claim.
- Local machine only: Apple M5 Mac, no NVIDIA GPU. Interpreters used: `se6-venv` (CPU, 24 of 41
  VariationMap classes construct), `se7-ogb-venv` (adds `PointmazeLargeNavigate/Stitch`, 2 more).
  Nothing was run on Oscar.

## What was generated

| deliverable | path |
|---|---|
| generator | `scratch/step8/swm/build_gate_f.py` |
| supporting probes (this workstream, committed) | `scratch/step8/swm/probe_default_roundtrip.py` -> `examples/results/step8/swm/default_roundtrip_cpu.json`; `scratch/step8/swm/probe_variation_classification.py` -> `examples/results/step8/swm/variation_classification.json` |
| Gate-F matrix (machine-readable) | `examples/results/step8/swm/gate_f_matrix.json` |
| Gate-F matrix (docs table) | `docs/research/step8_swm_integration.md` |
| upstream findings bundle | `docs/research/step8_upstream_findings.md` |
| Craftax SWM-wrapper-JIT declaration | `stable_envs/envs/craftax.py:CraftaxClassic.SWM_WRAPPER_GATE_F`, `stable_envs/envs/craftax_full.py:CraftaxFull.SWM_WRAPPER_GATE_F` |
| tests | `tests/swm/test_gate_f_matrix.py` |
| mkdocs nav (additive) | two entries under the Step-8 section |

## Column tally (42 ids)

| column | counts |
|---|---|
| registry_resolution | MEASURED 42 |
| registered_default (default_style() == style_from_variation(env, {})) | MEASURED 38, NOT_APPLICABLE 4 (2 symbolic + 2 Craftax pixel: swm offers 0 variation leaves for Craftax) |
| swm_wrapper_reset_appearance (SEPARATE from the above: the wrapper's own reset vs. registered default) | MISMATCH(upstream) 12 (all Fetch ids, Workstream B), UNMEASURED 28, NOT_APPLICABLE 2 |
| variation_map (VariationMap.classification() aggregate) | MEASURED 38, NOT_APPLICABLE 4 |
| state_observation_api (check_state_roundtrip / check_default_style_is_official / render_parity_with_official_renderer, per cell, per the Step-7 closure2 gate_a shard) | MEASURED 40, NOT_APPLICABLE 2 |
| device_behavior (from support_matrix.json's `supported_devices`) | MEASURED 7, UNMEASURED 35 |
| wrapper_interaction_path | MEASURED 40 (`swap_renderer`, all pixel ids; +facade/pool noted for TwoRoom/PushT/Piecewise), NOT_APPLICABLE 2 |

`device_behavior` is the one column where the underlying Workstream-E artifact
(`support_matrix.json`) itself only populates `supported_devices` for the 7 Step-7-new cells (the
ones `gate_a_rejudged.json` covers with an explicit `per_device` breakdown); the other 35 pixel ids
have `gate_a.status = PASS` overall (bulk `certification.json`) but no per-device split recorded in
that field. This is reported honestly as `UNMEASURED` rather than inferred -- see **Open items**.

## Registered-default roundtrip: what was actually run here

24 of the 41 VariationMap-carrying classes construct on this Mac's CPU (`se6-venv`); 2 more
(`PointmazeLargeNavigate`/`PointmazeLargeStitch`) need `ogbench`, available in `se7-ogb-venv`. For
these 26, `env.default_style() == style_from_variation(env, {})` was measured directly and is
`True` for every one -- committed in `examples/results/step8/swm/default_roundtrip_cpu.json`.

The remaining 15 classes (12 Fetch, `OGBCube`, `OGBScene`, `PFRocketLanding`) need
`mujoco_warp`/CUDA or `nvdiff`/CUDA and cannot be constructed under `device="cpu"` on this Mac by
design (`missing_backend()` gates their registration). For these, the same claim is read from
already-committed GPU evidence instead of re-measured:

- Fetch (12 ids): `examples/results/step8/variations-fetch/gpu_evidence.json`
  (`default_through_empty_mapping`, `round_trip`, per scene -- Workstream B, GPU).
- `OGBCube`/`OGBScene`: `examples/results/step8/variations-newcells/gpu/default_ogb_{baseline,branch}.json`
  (Workstream C, `gpu2264`, job `6443480`).
- `PFRocketLanding`: `examples/results/step8/variations-newcells/gpu/default_rocket_{baseline,branch}.json`
  (same job).

The exact command that WOULD run this locally if a GPU were available is recorded per row in
`gate_f_matrix.json` (e.g. `se.make('FetchPush', batch=1, size=32, device='cuda')` then compare
`env.default_style()` to `style_from_variation(env, {})`) -- none of these 15 needed to be run
because the cited artifacts already establish the claim.

## State/observation/API status: the per-cell join

Read, per stable-envs cell, from whichever of three committed Step-7/Step-8 artifacts is
authoritative for that cell (same precedence Workstream E's own `_gate_a()` uses):

1. `examples/results/step8/provenance/reference_arm.json` (Workstream F) for the 4 DMC trackcom
   cells with a multi-arm record (Cheetah/Hopper/Walker/Humanoid);
2. `examples/results/step8/certification/gate_a_rejudged.json` (Workstream E) for the 7 Step-7-new
   cells (Craftax x2, `DMCManipulatorBringBall`, `DMCQuadrupedWalk`, `OGBCube`, `OGBScene`,
   `PFRocketLanding`);
3. the bulk `examples/results/step7/closure2/replay/certify/certification.json` for everything else.

Each carries `check_state_roundtrip`, `check_default_style_is_official`, and
`render_parity_with_official_renderer` (the closest existing check to a frame-shape claim), plus
the exact source path and, where the source records it, a `source_sha256`. `PointmazeLargeStitch`
has no separate bulk-cert config -- it shares one rendered scene with `PointmazeLargeNavigate` per
`support_matrix.json`'s own `join_evidence`, and is joined to that cell's record with a note saying
so rather than left blank.

## The Craftax declaration (plan section 12, "Craftax")

Confirmed directly against the installed `stable_worldmodel` source on this Mac
(`/Users/samib/stable-worldmodel/stable_worldmodel/envs/craftax/craftax.py:CraftaxWrapper.__init__`):

```python
self._reset_fn = jax.jit(_reset, backend=self.backend)
self._step_fn = jax.jit(_step, backend=self.backend)
```

So a live `gymnasium.make("swm/CraftaxClassicPixels-v1"/"swm/CraftaxPixels-v1")` workflow always
observes the JIT-produced frame. The certified stable-envs parity claim (100% exact, max |diff| 0)
is measured against the un-jitted renderer, exactly as `reference_env` on both classes already
states for an unrelated reason (the reference itself has no single float value once jitted). Added
as `SWM_WRAPPER_GATE_F`, a class attribute + docstring paragraph, on both `CraftaxClassic` and
`CraftaxFull` -- no equivalent declaration existed before this workstream. It states the bounded gap
rather than hiding it: **<=1 uint8 step**, quoting the numbers already measured in each class's own
`parity_note`/docstring (Classic: 99.7854%/99.7225% exact jitted vs. 100.0000% un-jitted; Full:
48/49 states byte-identical, 1/49 differing on 0.0023% of pixels). Covered by
`tests/swm/test_gate_f_matrix.py::test_craftax_swm_wrapper_jit_declaration_exists_on_both_cells`.

## Ten-item merge-gate table (plan section 18)

| # | item | this workstream |
|---|---|---|
| 1 | explicit workstream scope | plan section 12 -- Gate F / SWM API status for all 42 ids; no renderer/harness/VariationMap edits |
| 2 | before/after diagnosis | before: D "NOT STARTED", all 42 rows `swm_api_gate_f: UNMEASURED` in `support_matrix.json`; after: `gate_f_matrix.json` with an explicit, sourced status per id across 7 columns |
| 3 | targeted unit tests | `tests/swm/test_gate_f_matrix.py` (5 tests: byte-identical regeneration, 42-id agreement with the support matrix, every status has an evidence path, symbolic ids are NOT_APPLICABLE, Craftax declaration exists) |
| 4 | affected Gate-A/Gate-F checks | Gate F is the whole deliverable; Gate A untouched (read-only join) |
| 5 | affected frozen-corpus replay | none -- no renderer or corpus code was touched |
| 6 | pixel-movement classification | **no pixels changed.** `git diff --stat` on `stable_envs/` touches only two docstring/class-attribute additions in `craftax.py`/`craftax_full.py`; no rendering code path changed |
| 7 | performance regression check | not applicable -- no hot-path code changed |
| 8 | independent reviewer verdict | pending (coordinator to spawn) |
| 9 | semantic follow-up decision | none needed -- no pixel changed |
| 10 | dashboard update | pending (coordinator applies; this row documents what to apply) |

## Buckets

- **RELEASE_BLOCKING**: none found by this workstream.
- **UPSTREAM_FINDING**: Fetch `swm_wrapper_reset_appearance = MISMATCH(upstream)` (12 ids,
  Workstream B's finding, now carried into the Gate-F matrix); Craftax `SWM_WRAPPER_GATE_F` bounded
  jit gap (2 ids, this workstream, plan section 12 item U9). Full bundle:
  `docs/research/step8_upstream_findings.md` (19 items, U1-U19, merged from plan section 12 + B/C/F/H).
- **DOCUMENTED_LIMITATION**: none new; the pre-existing ones (nvdiff x86 ULP, Fetch transparent
  arm, Craftax semantic follow-up not testable, mujoco>=3.12 texture re-anchoring) are unaffected
  and unchanged by this workstream.
- **STEP9_CLEANUP**:
  1. `scratch/swm_matrix.json` (legacy Step-3/7 tool) still has 6 stale rows with an empty
     `stable_envs.envs`; not patched here (see `docs/research/step8_upstream_findings.md`'s closing
     note) because `gate_f_matrix.json` already supersedes it as the current, regenerable,
     authoritative registry-resolution source for all 42 ids.
  2. `support_matrix.json`'s `supported_devices` field is populated for only 7 of 42 rows (the
     ones `gate_a_rejudged.json` covers). Extending the same per-device rejudgment to the other 35
     pixel ids' bulk-cert configs would turn 35 `UNMEASURED` `device_behavior` cells into
     `MEASURED` ones -- a Workstream-E-owned change (`gate_a_rejudged.json`'s own generator), not
     made here to respect ownership boundaries.
  3. `swm_wrapper_reset_appearance` is `UNMEASURED` for 28 non-Fetch pixel ids. Establishing it
     properly needs a live `gymnasium.make(swm_id)` reset vs. the registered stable-envs default
     per family -- most of these need a GPU-only `swm_id` (DMC/OGB/Rocket) or PyFlyt, so it was not
     attempted here per the "local CPU only" constraint; the exact command for each is recordable
     the same way the registered-default column records its GPU commands.

## Re-running after T3 (plate texel capability)

The brief notes T3 is in flight and may move DMC/Fetch/OGB variation counts. The generator is built
to re-read `VARIATION_MAPS` through a committed, regenerable intermediate
(`variation_classification.json`) rather than a copy baked into this script, so after T3 merges:

```bash
MUJOCO_GL=glfw PYTHONPATH=. /Users/samib/se6-venv/bin/python scratch/step8/swm/probe_variation_classification.py
MUJOCO_GL=glfw PYTHONPATH=. /Users/samib/se7-ogb-venv/bin/python scratch/step8/swm/probe_default_roundtrip.py   # only if T3 adds/removes a class's constructibility
python scratch/step8/swm/build_gate_f.py
```

regenerates both `gate_f_matrix.json` and its docs table with the new leaf counts, with no other
changes needed.

## Tests run (this Mac)

```bash
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/swm/test_gate_f_matrix.py
# 5 passed
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_repo_contracts.py
# 10 passed, 1 failed -- the failure pre-exist at the step8/integration baseline (verified via
# `git stash`): 11 unrelated dead scratch/sboughan/* citations from other Step-6/7/8 workstreams'
# findings.md files, none touched by this workstream. The nav-orphan failure this workstream WOULD
# have introduced (the two new docs pages) is fixed by the two additive mkdocs.yml nav entries.
```

## Open questions for the coordinator

1. Should `support_matrix.json`'s `supported_devices` be extended to all 40 pixel rows (STEP9_CLEANUP #2), and by whom?
2. Is patching the legacy `scratch/swm_matrix.json`'s 6 stale rows worth doing, given `gate_f_matrix.json` already supersedes it for Gate-F purposes?
3. `swm_wrapper_reset_appearance` UNMEASURED for 28 ids: worth a small follow-up ticket to measure it for the DMC/classic-control ids specifically, since those ARE CPU-constructible (unlike Fetch/OGB/Rocket) and only need a live `gymnasium.make(swm_id)` call, not a GPU?
