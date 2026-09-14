# RAYCAST-6's certification pair — what produced it, and from which commit

`cpu_1thread_before_integration.json` and `cpu_1thread_after_branch.json`, job **6366607**,
`node2317`, `--partition=batch --qos=normal`, one torch thread, Intel Xeon Platinum 8268.

| leg | tree measured | commit |
|---|---|---|
| before | `$SCR/se6/integration` | `5b327af` (`step6/integration`, the tree this branch merged) |
| after | `$SCR/se6/raycast3d` | **`a802849`** on `step6/raycast3d` |

**Both shards record `tree_commit: "unknown"` and `git_commit: null`, and this file is why.**
`certify_shard_cpu3d.py` takes the commit from `git rev-parse` in the tree it imported, falling back
to `$STABLE_ENVS_GIT_COMMIT`; the Oscar copies are `rsync`ed without `.git` and the sweep job did not
export the fallback, so both lookups came up empty and the field recorded that honestly rather than
guessing. The commits above are recovered from the job's own `PYTHONPATH` (printed in
`$SCR/logs/se6-raycast3d-r6sweep_6366607.out`) and from this branch's history; nothing is inferred
from the numbers.

Fixed for anything measured after it: the sync writes `$WS/GIT_COMMIT` from `git rev-parse HEAD` and
`scratch/step6/raycast3d/r6_repeat_shard.sbatch` exports `STABLE_ENVS_GIT_COMMIT` from it, so the
repeat pair in `../r6_repeat/` carries the SHA in the shard itself.

The reference (official renderer) is measured **interleaved** with the sweep — one burst between
every batch cell, median reported with its spread — rather than once in a block at the end. Worst
spread in either leg: **1.17x** (`DMCReacherHard`, before), against a 1.25x rejection bound.
