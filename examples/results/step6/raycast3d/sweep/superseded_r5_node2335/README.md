# Superseded: the RAYCAST-5 CPU shard pair

*(This file was `PROVENANCE_NOTE.md`. It is a `README.md` so the Step-6 report generator
renders the withdrawal reason beside the withdrawn files instead of listing them as
unexplained artifacts.)*

These three shards were taken on `node2335` in job **6354672** and were correct when they were
written. They describe a build of this branch that no longer exists: RAYCAST-6 changed
`raycast3d.composite` under them, and on the two Pointmaze rows that changes the TIER, not just the
number — `CPU_FASTER` -> `CPU_FAST`.

They are kept rather than deleted, because a measurement that had to be re-taken is part of how the
number that survived was arrived at. They are **renamed** rather than only moved: `report.py`'s
`leg_of` reads the leg out of the deepest path component that says `before` or `after`, and that is
the FILENAME, so leaving `cpu_1thread_after_branch_bounds_on.json` anywhere under this tree would
give the page two after-legs for the same (config, device) and make it print *CPU after-legs
disagree* on `PointmazeLargeNavigate` and `PointmazeMediumNavigate` — the two rows whose tier change
is the whole point. The tokens are gone from the names, so these files now classify as neither leg.

**The current pair is `examples/results/step6/raycast3d/sweep/r6/`** (`node2317`, job 6366607), and
it covers the same eight environments plus `AntmazeGiantNavigate` and
`HumanoidmazeMediumNavigate`, so nothing loses its after-leg in the move. It also has a denominator
the superseded pair does not: the official renderer is measured INTERLEAVED with the sweep (a
reference burst between every batch cell, median with its spread recorded) rather than once in a
block at the end.

| row | R5 after (`node2335`) | R6 after (`node2317`) | tier |
|---|---|---|---|
| `DMCReacherHard` | S_cpu 4.02 | S_cpu 3.40 | `CPU_FASTER` on both |
| `DMCFingerSpin` | 4.01 | 3.70 | `CPU_FASTER` on both |
| `DMCCartpoleSwingup` | 19.00 | 17.30 | `CPU_FASTER` on both |
| `DMCPendulumSwingup` | 6.56 | 5.60 | `CPU_FASTER` on both |
| `DMCAcrobotSwingup` | 9.25 | 8.28 | `CPU_FASTER` on both |
| `DMCBallInCupCatch` | 4.74 | 3.61 | `CPU_FASTER` on both |
| `PointmazeLargeNavigate` | 78.05, cores 0.618 | **314.45, cores 2.703** | `CPU_FASTER` -> **`CPU_FAST`** |
| `PointmazeMediumNavigate` | 43.64, cores 0.472 | **103.11, cores 1.207** | `CPU_FASTER` -> **`CPU_FAST`** |

**No DM Control row changes tier**, which is what makes this move safe: the six control rows agree
across the two shards to within each shard's own reference spread (1.07-1.16x), and the only tier
movement is on the two rows RAYCAST-6 actually changed.

To reverse this, move the three files back up one directory and restore their original names:
`cpu_1thread_before_baseline.json`, `cpu_1thread_after_branch_bounds_on.json`,
`cpu_1thread_branch_bounds_off_control.json`. `scratch/step6/raycast3d/r5_shards.sbatch` still writes
the original names, and its header now says so.
