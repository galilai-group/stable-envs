# Adapter cleanup

Work starts at `24fa12f7` on an isolated `adapter-cleanup` worktree while the
repository cleanup proceeds separately. Public environment and stepping APIs
are unchanged. Craftax simulation and performance remain out of scope.

## Decisions

- Share process-selection policy; retain family eligibility, affinity, the Linux
  and daemon checks, B64/four-core thresholds, and the eight-process cap.
- Use one DMC factory for local and worker construction. Quadruped stays local.
- Describe asynchronous submit/receive/abort through `StepPipeline`. Family
  configuration remains separate from the common lifecycle protocol.
- Expose Fetch object masses as owned metadata instead of reading private storage.
- Share identical buffer allocation, views, and snapshot copies across Classic,
  DMC and OGBench. Fetch retains its distinct flat buffers and task shapes.
- Keep family worker loops, validation, and OGBench prepare/commit restoration.
  Their differences do not justify a universal worker framework.
- Group simulator implementation by family; keep common execution and transport
  separate. Move rendering helpers to their owning environment implementation.
- Retain family-specific source space types: precision, sampling order, and
  default/current-value behavior differ from public spaces. No deduplication
  without independent semantic evidence.

## Validation

Execution selection: 251 routing cases passed across all four families.
Initial process/sequence tests: 290 passed, 86 skipped locally; skips are not
acceptance. Fetch's existing Mac environment fails inside Gymnasium Robotics
with MuJoCo 3.12 on both the baseline and candidate. A temporary MuJoCo 3.11
overlay is used for the pinned Fetch checks. Static interface assignments pass. The pinned Fetch run passed all 31 tests;
OGBench routing/process checks passed 73 tests with one CUDA skip.

Completed: relocated imports and registry checks, family process and source
contracts, CUDA output/pipeline checks, and same-host baseline versus candidate
online timings at horizons 1 and 8. The separate saved-state advisor rerun reports: MSE, source and candidate CPU/GPU FPS, time mean/SD, equal
batches and output destinations, explicit CPU-only reference qualifications.
Use fresh output directories; preserve previous measurements and fixture bytes.


After the family moves:

| Check | Passed | Skipped |
|---|---:|---:|
| Classic/DMC/Fetch routing and registry | 238 | 1 |
| Classic/DMC process and sequence masks | 99 | 86 CUDA |
| Fetch process contract, pinned MuJoCo 3.11 | 31 | 0 |
| OGBench routing/process (prior and moved paths) | 73 | 1 |
| PushT public State/replay, pinned source versions | 73 | 16 CUDA |
| Rocket replay/render validation | 23 | 0 |
| Import boundaries, inventory, classic State export | 48 | 5 CUDA |
| Native CPU source profile | 143 | 271 other-profile/CUDA |
| PushT CPU source profile | 75 | 123 other-profile/CUDA |
| Rocket CPU source profile, with shared pygame dependency | 31 | 113 other-profile/CUDA |

The move audit compared executable ASTs, ignoring imports and docstrings, for
all 38 moved modules. They are unchanged. Focused mypy assignments pass across
nine source files; CI runs this target on Python 3.12. Ruff passes.

## Oscar validation and advisor rerun

Scratch is blocked by an expired soft quota. Archives are staged in
`/users/sboughan/se-adapter-cleanup-45c569d6-20261008`; jobs run in node-local
storage and return compact reports to that directory. Previous evidence is
unchanged. The source is an immutable Git bundle at
`176d54532b3c0a251a4f509b74f2fdacc143a63d`.

Candidates:

- `45c569d6`: adapter implementation and family moves.
- `1faa9671`: same production adapter implementation, merged repository cleanup,
  benchmark provenance path fixes, and isolated source-pixel capture.
- Baseline `24fa12f7`: before this adapter refactor.

Initial jobs `7118098` and `7118099` were cancelled while pending after the old
source checkout disappeared. The first general-GPU attempt (`7118176`) could
not establish candidate Git provenance from an archive; the corrected launcher
supplies the verifier's supported file manifest. These are setup failures, not
accepted source results.

The initial DMC/OGBench pixel profiles triggered the context-release guard when
closing source renderers in the candidate process. Source capture now uses one
shared subprocess helper, including Fetch's existing isolation. Pixel thresholds,
source identity checks and the resource guard are unchanged. GPU launch overlays
also pin MJWarp 3.12.0 and Warp 1.17.0; the inherited older renderer API failed
before measuring pixels.

Current campaigns:

| Jobs | Purpose |
|---|---|
| `7118395`, `7118398` | Seven CPU/CUDA source profiles and family process tests |
| `7119515`, `7119516` | DMC/OGBench profiles after source-context isolation |
| `7118396`, `7118397`, `7118709`, `7118710` | Fresh saved-State CPU/RTX3090 advisor table |
| `7119517` | Failed MJWarp table rows with corrected renderer dependencies |
| `7118399`, `7119518`, `7119549`, `7119557` | Same-node baseline/candidate online comparisons |
| `7119892`, `7119893` | PushT table with its recorded corpus dependency setup |

The table retains batches 1, 64, 128 and 256, equal output destinations, independent
CPU/GPU statuses, normalized MSE, and timing mean/sample SD. Most families use 16
saved poses; the separate PushT corpus has 1,024. Setup, State restoration and
upload remain outside render timing. Each source renderer recomputes images.
Craftax's eight device cells are excluded, never filled from historical results.
The historical source-GPU availability audit remains a static audit of the same
source commit, not a newly measured GPU result.

Fetch's first same-node comparison passed all its correctness checks. With all
frames returned, throughput changed by -0.31% to +0.72% across B64/B256 and H1/H8.
The B64/H8 step-only cell was 7.93% lower in the first run; a reversed-order repeat
measured 4,826.8 baseline versus 4,824.0 candidate steps/s (-0.06%).
Do not interpret the stable rendering result as proof that every timing cell is
unchanged. DMC and OGBench before/after comparisons are also required.

Scene OpenGL on one RTX3090 node measured 100.7 FPS for both baseline and
candidate at B256. The much faster historical table value is not reproduced by
either commit; it must not be attributed to this adapter refactor. Both current
MSEs are approximately 0.000339665. The first OGBench pipeline attempt exhausted
its 32 GB allocation during baseline restoration; its retry requests 64 GB.

The old PushT table harness called the retired five-coordinate renderer. Its
replacement witnesses complete candidate diagnostics against each fresh source
owner, writes projected RenderState to disk, reloads it before timing, and calls
the current public `render()`. Both discrete and continuous local smoke runs
passed with MSE zero; a 24-frame/two-seed run also covers reset-retained fields.
PushT currently renders through CPU Pygame on both output configurations; the
CSV must show that CUDA output does not imply GPU rasterization.

Local merged-checkpoint validation: 281 import/registry/routing tests passed with
one profile skip; benchmark-surface tests passed 24/24 and Fetch source-pipeline
helpers passed 32/32 in their matching pinned interpreters. Focused mypy checks
pass for nine files. No new interpreter pins are imposed on other families.

The wheel builds successfully and includes all 48 adapter modules and the three
relocated drawing helpers. All 14 packaging checks pass.

## Adapter acceptance

All seven source profiles pass on CPU and CUDA output: **2,112/2,112 required
checks**, no missing or duplicate required cases. Family process/unit commands
also terminate successfully. All 18 source-result seals, including the retained
failed attempts, verify. [Exact profiles and commits](adapter-cleanup/source-checks.json).

The [before/after timing table](adapter-cleanup/online-before-after.csv) retains
all three DMC runs and both Fetch runs, including the short-sample slowdowns.
DMC's longer run uses 60 samples per mode: candidate changes range from -2.8% to
+9.1%; the earlier B64/H1 loss did not reproduce. This is evidence against a
consistent refactor regression, not a claim of a new speedup. Cube changes are
-0.52% to +0.23% across all six H1/H8 capture cases; sampled peak PSS is 42.31 GiB
baseline and 42.62 GiB candidate. Both complete replay checks pass.

The public environment API, State schema, render algorithms, process thresholds
and eight-process cap are preserved. This work does not promote new environment
or fidelity support. The final advisor-table collection remains separate from
adapter acceptance; failed source replay/stability checks must remain visible.
