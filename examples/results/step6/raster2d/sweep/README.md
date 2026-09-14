# What is authoritative here, and what is superseded

The Step 6 report generator labels tiers from every Step-3-shaped shard it finds in this directory,
so this directory holds exactly one measurement of each leg and the earlier ones live in
`superseded_*/` subdirectories with their own READMEs. Nothing has been edited: a shard is a record
of a run, and correcting the record is worse than recording that it was wrong.

## Layout, and why the before legs are not in this directory

`scratch/step3/certify.py:merge` keys a run on `(env, device, host, threads)`, and a before shard and
an after shard of the same sweep share all four. Left side by side, the merger would keep whichever
has more measured cells and label a tier from it — possibly the BEFORE tree's numbers, silently. So
the top level holds only the shards a tier should be computed from, which is the current tree, and
everything else is one directory down.

| file | what it is |
|---|---|
| `cpu_1thread_after_{swm,pusht}.json` | **the CPU certification shards.** `raster2d`, `c96f886`, 1 thread, `node2334`, load 40.3, job `6352386` |
| `gpu_after.json` | **the GPU certification shard.** `raster2d`, `c96f886`, `gpu2101`, load 32.6, job `6353295` |
| `before_legs/` | the matching `1103326` legs of the same two jobs — same node, same job, one per file. These are the denominators of every ratio in the findings and are not a second measurement of the same thing. |
| `order_check/` | `gpu_after_first.json` / `gpu_before_second.json`, the GPU pair repeated with the leg order reversed, and `gpu_same_process_ab.json`, the in-process interleaved A/B that settled it. See below. |

`before` is `1103326`, the approved RASTER2D-1c + RASTER2D-2 patch; `after` is this branch's HEAD.
So these shards measure SWMNAV-1 and nothing else. Both legs of each pair ran in ONE job on ONE node,
and each carries its own `git_commit` alongside `stable_envs_file`, which is the field that cannot be
wrong about which tree ran — it is written from the import that actually resolved.

`before` is `1103326`, the approved RASTER2D-1c + RASTER2D-2 patch; `after` is this branch's HEAD.
So these shards measure SWMNAV-1 and nothing else. Both legs of each pair ran in ONE job on ONE node,
and each carries its own `git_commit` alongside `stable_envs_file`, which is the field that cannot be
wrong about which tree ran — it is written from the import that actually resolved.

**Why `order_check/` exists.** A leg that always runs second runs on a warmer card. The spread
within one tree's two runs came out the same size as the difference between the trees —
PiecewiseZones read 1,031 and 1,075 f/s for the same code at batch 1 — so a single ordering is
reporting drift, and reversing it only gives a second biased sample. `gpu_same_process_ab.json`, from
`scratch/step6/raster2d/gpu_same_process_ab.py` (job `6353691`), settles it in one process:
interleaved, one warmed card, worst ratio 0.979 with every frame identical. It is not a Step-3-shaped
shard and is kept out of the top level for that reason as well.

## Superseded

- **`superseded_node2344/`** — the round-1 CPU and GPU shards, job `6346291` / `6346300`. The CPU
  ones were taken on `node2344` at load average 43.79 and their `before` leg is the Step-5 baseline
  rather than `1103326`, so they answer a different question from the shards above and would be
  double-counted beside them. Kept because the Step-5-baseline comparison is the one section 6 of
  the findings quotes.
- **`superseded_round2_node1828/`** — round 2, job `6349273` / `6349274`, `node1828`. Superseded on
  two counts: the code measured (`d862e2a`) regressed the GPU rows and is not what shipped, and
  `STABLE_ENVS_GIT_COMMIT` was exported once for the job so the `before` shards carry the `after`
  tree's SHA. See `PROVENANCE_NOTE.md` in that directory.
- **`superseded_round3_gpu/`** — the round-3 GPU pair, job `6352385`, measured before the leg order
  was cancelled. The CPU half of round 3 is the authoritative set above; only its GPU half is
  superseded.
