# Step 8 — independent review, Workstream G (final performance and pipeline accounting)

**Reviewed SHA:** `fff274fcd238897e9e1f23ec11fdc4506cd95755` (branch `step8/performance`, worktree
`/Users/samib/stable-envs-step8-performance`), diffed against `4210793` (`step8/integration`,
Workstream E merged). Reviewer read `AGENT_COMMON.md`, coordinator-plan sections 15 and 18,
`docs/research/certification.md`, and the author's `examples/results/step8/performance/findings.md`
before touching the diff. No subagents used. No file in the worktree was edited, committed, stashed,
reset, or pushed; a separate `git worktree add` of `4210793` was created under the reviewer's own
scratchpad (outside the target worktree) purely to diff a baseline test run, and was removed with
`git worktree remove` afterward. No Oscar access.

**VERDICT: APPROVE**

## Ten merge-gate items (coordinator plan section 18)

1. Explicit workstream scope — YES. `git diff --stat 4210793 fff274f` touches exactly 7 files:
   `docs/research/step8_performance.md`, `examples/results/step8/performance/findings.md`,
   `examples/results/step8/performance/tiers.json`, `mkdocs.yml` (one additive nav line),
   `scratch/step8/perf/build_performance.py`, `scratch/step8/perf/write_doc.py`,
   `tests/unit/test_step8_performance.py`. No renderer, harness, threshold, or other shared file changed.
2. Before/after diagnosis — N/A, correctly claimed: no code path was altered; every number in
   `tiers.json` carries a source path and sha256, verified below.
3. Targeted unit tests — YES, `tests/unit/test_step8_performance.py`, 10 tests, all read from the
   generated artifact/live regeneration, no hardcoded fixture data.
4. Affected Gate-A/Gate-F checks — NONE, correctly claimed; nothing here can move a Gate-A/F verdict.
5. Affected frozen-corpus replay — NONE, correctly claimed; no step4/step5 corpus touched.
6. Pixel-movement classification — NO PIXELS MOVED; diff contains no rendering code.
7. Performance regression check when hot-path code changed — N/A, correctly claimed; no hot-path code
   in the diff (two `scratch/` scripts, one generated JSON, one generated doc, one test, one nav line).
8. Independent reviewer verdict — this document.
9. Semantic follow-up decision — NONE REQUIRED, correctly claimed; no pixels changed.
10. Dashboard update — PENDING, correctly left to the coordinator; the row content the author supplies
    matches the evidence found here.

## Commands run and tail output

`MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step8/perf/build_performance.py --check`
```
examples/results/step8/performance/tiers.json: byte-identical to a regeneration
```

`MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_step8_performance.py`
```
..........                                                               [100%]
10 passed in 0.08s
```

`MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_repo_contracts.py`
```
FAILED tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves
1 failed, 10 passed in 0.35s
```
Verified pre-existing and improved, not caused, by this branch: a fresh `git worktree add` of
baseline `4210793` run through the same single test shows **13** unresolved cited scratch paths,
including `scratch/step8/perf/*` (cited at `dashboard.yaml:602`, `takeover/stop-integration.txt:147`,
`variations-fetch/findings.md:182` before this workstream's files existed); at `fff274f` the same test
shows **12** — the branch's own files resolve that one citation and add none. `diff before.txt
after.txt` confirms this is the *only* difference in the failing-path list.

`MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit --ignore=tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"`
```
988 passed, 186 skipped, 1 deselected, 25918 warnings in 73.98s (0:01:13)
```

## Verification detail

- **Scope (item 1):** confirmed by `git diff --stat`, above — nothing outside the seven named files.
- **Regeneration (item 2):** `--check` passes; byte-identical. Sampled 10 named sources (including all
  four required: Step-7 closure2 `coverage_matrix.json`, Rocket job `6410105/gpu2106.json`, the OGB
  `m3/fidelity/remeasure.json` W5 re-measurement, and two Step-6 Lance artifacts) and independently
  recomputed each `sha256sum` against the live file on disk — all 10 matched exactly.
- **STEP7_EVIDENCE hand-entered table (item 3):** read `build_performance.py` in full; the table names
  an artifact path and a reader function per cell, and every reader (`read_declaration`,
  `read_manipulator_cpu`, `read_economics`, `read_ogb_remeasure`, `read_ogb_cpu_missing`,
  `read_rocket_gates`, `read_quarantine`) extracts numbers from the JSON/regex-parsed artifact; no
  numeric literal is typed as evidence. Independently re-read 7 rows' underlying artifacts by hand and
  compared every S_cpu/R/cores/fps value against `tiers.json`, all matching exactly:
  Rocket (R=241.51, cores=2.606, fps=3374.4, GPU_PARITY [BATCH_SCALES_FAIL] 14.17x),
  OGBCube (R=3.2008, cores=1.8752, batch_scales=5.362), OGBScene (R=3.6239, cores=2.0518,
  batch_scales=6.498), ManipulatorBringBall CPU (S_cpu=0.6974, fps=15.504) and GPU (R=32.040,
  cores=3.722), CraftaxClassic CPU/GPU declaration fields, CraftaxFull CPU/GPU declaration fields, and
  QuadrupedDMControl's quarantine record (73 polls, load 34.06–43.59, guard 12) and GPU label
  (R=129.936, cores=13.611, GPU_EXCEPTIONAL, correctly recomputed from the underlying numbers because
  the committed label itself reads `tier: "unknown"` from a missing census entry — the recompute is
  flagged `CENSUS_ENTRY_MISSING` and `agrees_with_step7: true`).
- **Sami's rulings (item 4):** `BATCH_SCALES` bar unchanged at `fps(4096) >= 20*fps(1)` (confirmed
  against `docs/research/certification.md` §6.7); Rocket keeps `GPU_PARITY [BATCH_SCALES_FAIL]`;
  Fetch's ratio (8.29x) is reported and its `GPU_PARITY` classification is retained despite an
  otherwise-`GPU_FAST` R/cores pair — both the doc prose and `tiers.json` numbers agree
  (fps_1=622.245, fps_4096=5158.52, ratio=8.290). The revised BATCH_SCALES taxonomy is explicitly
  labelled "PROPOSAL, for Step 9 — not adopted, not applied to any row on this page" and is not
  referenced by any row's tier. Quadruped CPU stays `CPU_UNMEASURED [LOAD_QUARANTINED]`, verified
  against the quarantine artifact. `CPU_MISSING [RAISES_BY_DESIGN]` kept on OGBCube/OGBScene/
  PFRocketLanding, each carrying the `RAISES_BY_DESIGN` flag and status
  `MISSING-UNSUPPORTED_BY_DESIGN`. No projected numbers found anywhere in the diff.
- **Cross-check vs support matrix (item 5):** `tiers.json: cross_checks.disagreements` is `[]` (0 of
  42 ids); `support_matrix.support_matrix.tier_disagreements` is also `[]`. Independently re-read
  `examples/results/step8/support_matrix.json` for the five flagged ids and confirmed the two
  vocabulary differences exactly as described: (a) the two symbolic ids (`CraftaxClassicSymbolic`,
  `CraftaxSymbolic`) read `UNMEASURED`/`unknown` there vs `UNSUPPORTED_BY_DESIGN`/`n/a` here — same
  underlying fact, different words; (b) the three CPU-raise cells read `UNSUPPORTED_BY_DESIGN` there
  vs `MISSING-UNSUPPORTED_BY_DESIGN` here — the tier itself (`CPU_MISSING`) agrees on both sides.
- **Microbenchmark vs pipeline separation (item 6):** the doc keeps the two tables and their
  provenance separate (`docs/research/step8_performance.md`, "The two performance claims, kept
  separate" table). Independently recomputed the Lance/JPEG numbers from
  `examples/results/step6/lance/comparison2.json`'s 36 rows: GPU end-to-end wins 30/36, median 1.44,
  range 0.18–46.21 (Pendulum 46.21, CartPole 44.17, Acrobot 40.84, all reported as "46x/44x/41x");
  GPU-vs-nvJPEG wins 35/36, with `DMCBallInCupCatch` the sole loser at 0.717 (reported as 0.72); CPU
  8-way vs 8 Lance workers wins 1/24 (`CraftaxClassic` 6.96), median 0.111 — every figure in the doc
  matches an independent recomputation from the artifact exactly.
- **Tests (item 7):** all four required runs match the findings.md's claimed exact output, verified
  above; no absolute `/oscar` path found in `docs/research/step8_performance.md`,
  `examples/results/step8/performance/tiers.json`, `examples/results/step8/performance/findings.md`,
  or `scratch/step8/perf/write_doc.py`.

## Minor defect (non-blocking)

- `scratch/step8/perf/build_performance.py:62` defines `SCR = "/oscar/scratch/sboughan"` as a literal,
  used only to scrub that string out of quoted provenance before it reaches `tiers.json` (verified: the
  literal never reaches the committed JSON — `test_no_absolute_oscar_paths_in_the_committed_artifact`
  passes, and a direct `grep '/oscar/' tiers.json` finds nothing). `findings.md`'s sentence "No
  committed file in this workstream contains an absolute `/oscar` path" is therefore not literally true
  of `build_performance.py` itself, only of the artifacts and docs it produces. This matches an
  existing, repo-wide convention (e.g. `scratch/step6/report.py` carries the identical constant for the
  same reason) and is not a new problem introduced here — recommend tightening the sentence's wording
  in a future pass, not blocking on it.

## Summary

Reviewed `step8/performance` at `fff274f` against `4210793`: scope is exactly the seven files claimed,
with no renderer/harness/threshold/shared-file touched. Regeneration is byte-identical and 10 sampled
source artifacts (including all four specifically required) verify by sha256 against the working tree.
Eight-plus hand-entered `STEP7_EVIDENCE` rows were independently re-derived from their named artifacts
and match `tiers.json` to the last decimal, including the one row (`QuadrupedDMControl` GPU) whose
committed label itself says `unknown` and had to be recomputed via the frozen ladder. All of Sami's
rulings (frozen `BATCH_SCALES` bar, Rocket/Fetch classifications, the Step-9-only taxonomy proposal,
Quadruped quarantine, the three CPU-raise cells, no projections) are applied exactly as specified. The
support-matrix cross-check shows 0 tier disagreements and the two described vocabulary differences,
both confirmed against the live file. The Lance/JPEG production-pipeline numbers independently
recompute to the same figures quoted in the doc, and the two performance claims are kept structurally
separate with provenance on every row. All required tests pass with the exact expected outputs,
including the pre-existing (and slightly improved) `test_every_cited_scratch_path_resolves` failure.
One cosmetic wording overclaim in findings.md about `/oscar` paths in source code (not data) does not
affect any committed artifact and is not blocking.
