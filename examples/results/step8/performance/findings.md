# Step 8 — Workstream G: final performance and pipeline accounting

**Classification: `FINDING`.** Reporting and accounting only. No renderer, harness or shared file was
touched; no benchmark was run; no SLURM job was submitted; no Oscar tree was read or written; no
frozen threshold was changed. Nothing here is a patch.

- Branch `step8/performance`, merged from `step8/integration` at `4210793` (Workstream E merged).
- Step-8 baseline SHA recorded in every artifact: `43ac95238f7472137619f3e69e7636e11dd088b1`.
- Host: local Apple M5 Mac, `$HOME/se6-venv/bin/python`. The local CPU is used only to read committed
  artifacts and run unit tests; no number on the page was produced here.

## What was delivered

| file | what it is |
|---|---|
| `scratch/step8/perf/build_performance.py` | generator: reads the committed Step-6/Step-7 artifacts (each named with its sha256 in the output) and emits the tier accounting |
| `examples/results/step8/performance/tiers.json` | per registered swm id (all 42): CPU and GPU tier, status, flags, the numbers the tier rests on, host/job/version provenance, the reference rule applied, and the cross-checks |
| `scratch/step8/perf/write_doc.py` | generates the page from that artifact plus the Step-6 Lance/JPEG comparison |
| `docs/research/step8_performance.md` | the page: generated tables plus the prose that keeps the two claims separate |
| `tests/unit/test_step8_performance.py` | byte-identity of the committed artifact against a regeneration, plus the 42-id status contract |
| `mkdocs.yml` | one additive nav entry |

## Tier tally

| device | MEASURED | UNMEASURED | MISSING-UNSUPPORTED_BY_DESIGN | UNSUPPORTED_BY_DESIGN (symbolic) |
|---|---|---|---|---|
| CPU | 24 | 13 | 3 | 2 |
| GPU | 40 | 0 | 0 | 2 |

The 13 CPU `UNMEASURED` are the twelve Fetch ids (`CPU_BACKEND_MISSING` — no CPU backend exists) and
`swm/QuadrupedDMControl-v0` (`LOAD_QUARANTINED`). The 3 `MISSING-UNSUPPORTED_BY_DESIGN` are
`swm/OGBCube-v0`, `swm/OGBScene-v0` and `swm/PFRocketLanding-v0`, which raise on `device="cpu"`
rather than falling back to CUDA.

## Rulings applied

- **`BATCH_SCALES` not changed.** Rocket keeps `GPU_PARITY [BATCH_SCALES_FAIL]` at 14.17x against
  the frozen 20x bar, Fetch at 8.29x, OGBCube 5.36x, OGBScene 6.50x. The mechanism (a ratio to
  `fps(1)` misses cells whose B=1 number is already high) is documented and a revised taxonomy is
  offered **as a proposal for Step 9 only**, clearly labelled, applied to nothing.
- **Quadruped CPU stays `CPU_UNMEASURED [LOAD_QUARANTINED]`.** The only CPU tier run,
  `examples/results/step7/dmc-quadruped/phase1/tier_cpu/QUARANTINED_LOAD_node2301_6398084.json`,
  measured nothing: 73 polls over 6 h, load1 34.06-43.59, never below the pre-registered guard of 12.
  No trustworthy quiet-node measurement exists in the artifacts and none was taken. A single
  quiet-node `batch`/`normal` job would close it; it is recommended as deferred work, not Step-8 scope.
- **`CPU_MISSING [RAISES_BY_DESIGN]` kept** on the three CPU-unsupported cells, with the measured
  reason on each row.
- **No projections.** Every row is a committed measurement or an explicit `UNMEASURED` that says what
  blocked it.

## Cross-checks and what they found

1. **Against the Step-7 record** (`examples/results/step7/closure2/coverage_matrix.json`): all 42
   ids' CPU and GPU tiers match, and re-judging each row's own artifact with the frozen evaluator
   (`scratch/step3/certify.py`, Step 2's ladder) reproduces every copied tier. **0 disagreements.**
2. **Against Workstream E's support matrix** (`examples/results/step8/support_matrix.json`, sha256
   `55069d7636f4727f...`): 42 of 42 ids compared, **0 tier disagreements**. Two vocabulary
   differences, both recorded in `tiers.json: cross_checks.support_matrix` and explained on the page:
   - the two symbolic ids read `UNMEASURED` / tier `unknown` in the support matrix and
     `UNSUPPORTED_BY_DESIGN` / `n/a` here. **This one is worth acting on**: they produce no frame by
     design, so `UNMEASURED` invites a reader to think a run is owed. Reported to the certification
     workstream; not changed here, because that file is theirs.
   - the three CPU-raise cells read `UNSUPPORTED_BY_DESIGN` there and
     `MISSING-UNSUPPORTED_BY_DESIGN` here — the same claim in the three-word vocabulary the Step-8
     closure criteria name.
3. **Census carryover closed.** Before E merged, four Step-7 cells (`DMCManipulatorBringBall`,
   `DMCQuadrupedWalk`, `OGBCube`, `OGBScene`) could not be re-judged at all: the Step-3 census had no
   entry for them and the evaluator returned `unknown`. E's regenerated census (88 -> 100) assigns all
   four class S and the re-judgement now reproduces their Step-7 tiers exactly.
4. **Superseded measurements recorded, not deleted** (five): OGBCube/OGBScene `m4/m4_tiers.json`
   superseded by the W5 closure re-measurement; two earlier Rocket tier jobs superseded by job
   6410105; and the Step-6 tilemap `CPU_EXCEPTIONAL` / `GPU_EXCEPTIONAL` tiers for CraftaxClassic
   superseded by the Step-7 class declaration measured against the jitted official renderer.
5. **Provenance weaknesses flagged on the row, not hidden**: `REFERENCE_SPLIT_JOB` on both Craftax
   Classic arms (throughput and reference from different jobs on different nodes);
   `HOST_FROM_PROSE` on `DMCManipulatorBringBall` GPU (the committed economics artifact carries the
   three timings but names neither its host nor its SLURM job; the Step-7 prose names gpu2107 and
   job 6395157).

## Tests run — exact output

`MUJOCO_GL=glfw $HOME/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_step8_performance.py`

```text
..........                                                               [100%]
------------------------------ stable-envs safety ------------------------------
resources: 0 EGL contexts created, 0 released, 0 live (peak 0); renderers 0 created / 0 closed
node: cis1b1zs2j3.ad.brown.edu slurm=? reason=none reference-renderer={}
10 passed in 0.09s
```

`MUJOCO_GL=glfw $HOME/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_repo_contracts.py`

```text
------------------------------ stable-envs safety ------------------------------
resources: 0 EGL contexts created, 0 released, 0 live (peak 0); renderers 0 created / 0 closed
node: cis1b1zs2j3.ad.brown.edu slurm=? reason=none reference-renderer={}
=========================== short test summary info ============================
FAILED tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves
1 failed, 10 passed in 0.35s
```

The failing test is `test_every_cited_scratch_path_resolves`, the **pre-existing** baseline citation
failure the Step-7 dashboard already records. It was verified pre-existing by removing this
workstream's files and re-running: the failing set is identical **except** that this workstream
*removes* one entry — `scratch/step8/perf/*` was cited in the Step-8 notes before it existed and now
resolves. No entry in the failing set is cited by any file this workstream adds.

`MUJOCO_GL=glfw $HOME/se6-venv/bin/python scratch/step8/perf/build_performance.py --check`

```text
examples/results/step8/performance/tiers.json: byte-identical to a regeneration
```

No committed file in this workstream contains an absolute `/oscar` path; every provenance string is
scrubbed to `$SCR/...` by the generator and a unit test asserts it.

## Merge gate (coordinator plan section 18)

| # | requirement | status for this workstream |
|---|---|---|
| 1 | explicit workstream scope | **YES.** Workstream G: performance and pipeline accounting. Files owned and touched: `docs/research/step8_performance.md`, `scratch/step8/perf/*`, `examples/results/step8/performance/*`, `tests/unit/test_step8_performance.py`, one additive `mkdocs.yml` nav entry. Nothing else. |
| 2 | before/after diagnosis | **N/A — no implementation change.** There is no "after": no code path was altered, so there is nothing to diagnose. What replaces it is the provenance chain: every number names its artifact and sha256, and every tier is re-judged against the frozen ladder. |
| 3 | targeted unit tests | **YES.** `tests/unit/test_step8_performance.py`, 10 tests: byte-identity against a regeneration; 42 ids each with a valid CPU and GPU status; symbolic ids `UNSUPPORTED_BY_DESIGN`; measured rows carry the numbers their tier rests on; unmeasured rows make no throughput claim and say what blocked them; the three CPU-raise cells; the Quadruped quarantine; the four `BATCH_SCALES` failures keeping `GPU_PARITY`; every source carrying a sha256; no `/oscar` paths. |
| 4 | affected Gate-A / Gate-F checks | **NONE AFFECTED.** No renderer, declaration or threshold changed, so no Gate-A or Gate-F verdict can move. Gate-A status per id is Workstream E's column and is untouched here. |
| 5 | affected frozen-corpus replay | **NONE AFFECTED.** No frozen Step-4/5/6/7 corpus was read for pixels, written or re-frozen; `examples/results/step4/` and `step5/` were not touched. |
| 6 | pixel-movement classification | **NO PIXELS MOVED.** No rendering code exists in this workstream's diff; no frame was produced. |
| 7 | performance regression check when hot-path code changed | **N/A — no hot-path code changed.** The diff is two scripts under `scratch/`, one generated JSON, one generated page, one test and one nav line. |
| 8 | independent reviewer verdict | **PENDING** — for the coordinator to assign. Suggested focus: the hand-entered `STEP7_EVIDENCE` table in the generator (the only hand-entered data; each entry names a committed artifact and a reader, and no number is typed), and the `BATCH_SCALES` section's separation of frozen classification from proposal. |
| 9 | semantic follow-up decision | **NONE REQUIRED.** Semantic follow-ups are triggered by changed pixels; none changed. The Step-5 semantic claims carry forward untouched. |
| 10 | dashboard update | **PENDING** — the dashboard is the coordinator's file. Row content for `performance`: branch `step8/performance`, status `COMPLETE`, classification `FINDING`, affected ids `all 42 (reporting only)`, pixels_changed `no`, semantic_followup `none`, performance_effect `none (no code changed)`, tests as above. |

## Open items handed forward

| item | bucket | note |
|---|---|---|
| Quadruped CPU tier: one quiet-node run under the load-12 guard | `DEFERRED_OPTIMIZATION` (a measurement, not code) | converts the last `UNMEASURED` on a supported CPU path; does not block closure |
| Shared CPU triangle/span backend for OGBCube, OGBScene, PFRocketLanding | `DEFERRED_OPTIMIZATION` | would replace a by-design raise with a real tier |
| Fetch has no CPU backend at all (all twelve ids) | `DOCUMENTED_LIMITATION` | measured and recorded, not chased |
| `BATCH_SCALES` taxonomy proposal | `STEP9_CLEANUP` | a threshold change needs Sami, a re-judgement of every row and a re-freeze |
| Support matrix reports the two symbolic ids' tiers as `UNMEASURED` | `DOCUMENTED_LIMITATION` (reporting) | the certification workstream's file; reported, not edited |
| `DMCManipulatorBringBall` GPU numbers have no host/job in their artifact | `DOCUMENTED_LIMITATION` | flagged `HOST_FROM_PROSE`; the numbers themselves are committed |
