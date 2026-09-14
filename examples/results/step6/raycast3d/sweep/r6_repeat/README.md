# The authorized repeat of the RAYCAST-6 certification pair

Job **6368126**, `node1946`, one torch thread, `--partition=batch --qos=normal`. Two environments
only — `PointmazeLargeNavigate` and `PointmazeMediumNavigate` — because those are the two rows whose
tier was in question. Commit under measurement: **`7fa1856`** on `step6/raycast3d`, carried in the
shard itself this time (`provenance.tree_commit`), not recovered afterwards from a job log.

**One provenance defect in these two files, uncorrected in place on purpose.** Both legs record
`tree_commit: 7fa1856`, but the BEFORE leg measured `$SCR/se6/integration`, which is at `5b327af`.
The cause: the inner producer takes the commit from `git rev-parse` in the tree it imported and falls
back to `$STABLE_ENVS_GIT_COMMIT` when that tree has no `.git` — and `rsync` drops `.git` from both
trees, so one job-level variable got stamped on both legs. `provenance.stable_envs_file` records
which tree each leg actually imported and is correct in both
(`.../se6/integration/...` and `.../se6/raycast3d/...`), so nothing here is ambiguous, only
mislabelled. The JSONs are left exactly as the job wrote them — a measurement artifact edited after
the fact is worth less than one with its defect written down beside it — and the fix is in the
tooling: `r6_certify_shard.py` now takes `$SE6_TREE_COMMIT` **per leg** and writes `null` with a
reason rather than the other leg's answer when it is absent.

**Read `tree_commit` on the before leg as `5b327af`.**

## Why it exists

`sweep/r6/` put `PointmazeMediumNavigate` at `cores_cpu` **1.207** and therefore over the `CPU_FAST`
bar — but only just, and its denominator was suspect: the one-core JPEG-q90 decode rate measured for
that same configuration in the OTHER leg of the SAME job on the SAME node was 2,678 f/s against the
after leg's 2,264, a 15 % swing in the bar itself. This run re-takes the pair on a node that was
actually quiet, with two guards that reject rather than publish:

| guard | bound | observed |
|---|---|---|
| 1-minute load average, checked before anything is timed | 12 | **4.2 -> 4.8** |
| paired reference spread (official renderer, burst between every batch cell) | 1.25x | **1.06x** |

Both passed; `certifiable: true` in both legs. The first pair's worst spread was 1.17x.

## What it found

| | `sweep/r6` (`node2317`) | **this pair** (`node1946`) |
|---|---|---|
| `PointmazeLargeNavigate` | 1,361 -> 6,154 f/s (4.52x), cores **2.703**, `CPU_FAST` | 1,180 -> 5,535 (4.69x), cores **2.152**, `CPU_FAST` |
| `PointmazeMediumNavigate` | 969 -> 2,732 f/s (2.82x), cores **1.207**, `CPU_FAST` | 908 -> 2,479 (2.73x), cores **0.925**, **`CPU_FASTER` + `BELOW_G4`** |

**The speed-up reproduces; the tier does not.** The two shards agree on the change (4.52x vs 4.69x,
2.82x vs 2.73x) and disagree on `PointmazeMediumNavigate`'s tier, because `cores_cpu` divides by a
decode rate that differed by 18 % between the two nodes while the renderer differed by 10 %.

## This pair is meant to win, and the disagreement is meant to show

`report.py` takes the lexicographically last after-leg per (config, device), and
`sweep/r6_repeat/...` sorts after `sweep/r6/...`, so the tier the page shows comes from here — the
better-conditioned measurement. It will also mark *CPU after-legs disagree* on
`PointmazeMediumNavigate`, and that flag is **correct and wanted**: two certification-grade shards
really do straddle the bar on that row, and hiding it by withdrawing one of them would be publishing
a tier this evidence does not support. `PointmazeLargeNavigate` agrees across both and raises no
flag.
