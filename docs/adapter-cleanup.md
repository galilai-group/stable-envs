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

## Validation in progress (candidate `45c569d6`)

Execution selection: 251 routing cases passed across all four families.
Initial process/sequence tests: 290 passed, 86 skipped locally; skips are not
acceptance. Fetch's existing Mac environment fails inside Gymnasium Robotics
with MuJoCo 3.12 on both the baseline and candidate. A temporary MuJoCo 3.11
overlay is used for the pinned Fetch checks. Static interface assignments pass. The pinned Fetch run passed all 31 tests;
OGBench routing/process checks passed 73 tests with one CUDA skip.

Required before closure: relocated imports and registry checks, family process
and source contracts, CUDA output and pipeline checks, and same-host baseline
versus candidate online timings at horizons 1 and 8. Rerun the saved-state advisor
table independently: MSE, source and candidate CPU/GPU FPS, time mean/SD, equal
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

Oscar scratch is blocked at 906 GB against a 512 GB expired soft quota. Code
archives are staged under `/users/sboughan/se-adapter-cleanup-45c569d6-20261008`;
jobs extract and compute in node-local temporary storage, then retain compact
result archives. No prior evidence or scratch data was removed. Submitted jobs:

- `7118098`: Native, Fetch, DMC and OGBench CUDA/source gates on 3090-gcondo.
- `7118099`: FetchPushDict/Scene paired CPU/CUDA advisor-table pilots on RTX 3090.
- `7118176`: duplicate correctness gates on general GPUs, capped at two tasks.

The 3090 arrays initially wait for `QOSGrpGRES`. General-GPU correctness results
must not be presented as RTX 3090 performance. A fresh completed advisor table
and the before/after horizon-1/8 timings are still required.
