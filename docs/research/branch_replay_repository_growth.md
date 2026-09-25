# Repository growth at the preceding checkpoint

Measured from `13e51037cdb4afb9ce31aba3a4837ccf5c95adaf` to `31153d38c800f44e5c4c0ae1acdd0edca4c5f656` before new work.

New files: **1,377**; added-file bytes: **50,293,781**; net modified-file bytes: **419**; net tree growth: **50,294,200**.

The worktree's `.git` is a 78-byte pointer file. Its shared Git directory is `/Users/samib/stable-envs/.git`: 273,910,081 logical bytes and 276,948 KiB on disk. This includes other branches/worktree metadata, so it is not the fresh-clone estimate.

```text
count: 2813
size: 30.19 MiB
in-pack: 22311
packs: 4
size-pack: 185.49 MiB
prune-packable: 0
garbage: 0
size-garbage: 0 bytes
```

Read-only single-branch ancestry pack streams are 95,104,642 bytes at the baseline and 98,797,338 bytes at the checkpoint: **3,692,696 bytes additional packed transfer**. Actual network transfer can differ with server packing, refs and negotiation. Checkout bytes remain much larger. No repository repack, deletion, garbage collection or history rewrite occurred.

## Added bytes per new evidence directory

| Directory | Added bytes |
|---|---:|
| pusht_native_feasibility | 1,006,747 |
| rocket_native_feasibility | 28,685,996 |
| state_performance_cold_cpu | 16,980 |
| state_performance_cold_gpu_6694865 | 20,643 |
| state_performance_cpu | 6,705,729 |
| state_performance_cpu_retry | 293,847 |
| state_performance_dmc_native | 454,347 |
| state_performance_final | 612,581 |
| state_performance_gpu_6693308 | 474,516 |
| state_performance_gpu_6694195 | 341,624 |
| state_performance_gpu_submission | 16,196 |
| state_performance_report | 11,366,754 |
| state_performance_start | 169,204 |

## Twenty largest added files

| Path | Bytes |
|---|---:|
| `docs/research/semantic_recovery_evidence/state_performance_report/summary.json` | 11,202,352 |
| `docs/research/semantic_recovery_evidence/rocket_native_feasibility/v2-pre_contacts/run.json` | 511,475 |
| `docs/research/semantic_recovery_evidence/rocket_native_feasibility/v2-activation/run.json` | 511,473 |
| `docs/research/semantic_recovery_evidence/rocket_native_feasibility/v2-baseline/run.json` | 511,220 |
| `docs/research/semantic_recovery_evidence/state_performance_dmc_native/result.json` | 454,269 |
| `docs/research/semantic_recovery_evidence/rocket_native_feasibility/v2-empty_contacts/run.json` | 430,081 |
| `docs/research/semantic_recovery_evidence/pusht_native_feasibility/final/result.json` | 367,372 |
| `docs/research/semantic_recovery_evidence/state_performance_cpu/swm__Piecewise-v0/result.json` | 268,624 |
| `docs/research/semantic_recovery_evidence/state_performance_cpu/swm__TwoRoom-v1/result.json` | 264,596 |
| `docs/research/semantic_recovery_evidence/state_performance_final/whitespace-filtered.log` | 248,237 |
| `docs/research/semantic_recovery_evidence/state_performance_cpu/swm__PendulumDMControl-v0/result.json` | 222,120 |
| `docs/research/semantic_recovery_evidence/state_performance_cpu/swm__AcrobotDMControl-v0/result.json` | 219,626 |
| `docs/research/semantic_recovery_evidence/state_performance_cpu/swm__ReacherDMControl-v0/result.json` | 218,888 |
| `docs/research/semantic_recovery_evidence/state_performance_cpu/swm__HopperDMControl-v0/result.json` | 217,039 |
| `docs/research/semantic_recovery_evidence/state_performance_cpu/swm__BallInCupDMControl-v0/result.json` | 213,945 |
| `docs/research/semantic_recovery_evidence/state_performance_cpu/swm__CartpoleDMControl-v0/result.json` | 212,826 |
| `docs/research/semantic_recovery_evidence/state_performance_cpu/swm__WalkerDMControl-v0/result.json` | 211,518 |
| `docs/research/semantic_recovery_evidence/state_performance_cpu/swm__MountainCarControl-v0/result.json` | 210,769 |
| `docs/research/semantic_recovery_evidence/state_performance_cpu/swm__MountainCarContinuousControl-v0/result.json` | 207,040 |
| `docs/research/semantic_recovery_evidence/state_performance_cpu/swm__AcrobotControl-v1/result.json` | 207,022 |

Before v0, consolidating repeated comparison files and duplicated JSON into small archives would reduce checkout/file-count cost. Git already compresses this checkpoint effectively, so history cleanup is a separate review decision rather than necessary to recover the full logical 50 MB. Preserve provenance and representative failures before any separately authorized compaction. All existing evidence and history remain intact here.

The exact measurement script, `git count-objects -vH`, per-file sizes and packing method are recorded in `semantic_recovery_evidence/branch_replay_start/`.
