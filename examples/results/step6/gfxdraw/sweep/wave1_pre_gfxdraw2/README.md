# Wave 1: the GPU sweep for the planar-canvas patch, before GFXDRAW-2

**Kept as the record of a step, not as certification rows.** These measure a code generation the
tree no longer has: the planar canvas with bounded primitives and no fused kernel, which is what
`6213673` shipped and what the `before` arm of every GFXDRAW-2 measurement now reproduces by
setting `use_fused_render = False`.

Two reasons they are out of the generator's reach.

**They describe superseded code.** `gpu_after_gpu2104.json` gives Acrobot R = 91.4. The same env on
the current tree measures R = 387-439. Both are true of the commit they were taken at, and a page
that lists them side by side without that context reads as a contradiction rather than as two
generations.

**The two arms collide.** `certify.merge` keys runs on `(env, device, host, threads)` with no slot
for an arm, so a `before` and an `after` shard from one host resolve to one row and the loser is
recorded only in `_superseded`. That is silent, and it is silent in the direction of whichever file
happened to have more measured cells. Any before/after pair belongs outside the set the generator
merges; only the arm that reflects the current tree is a certification row.

## What is in them, and what still cites them

| file | what |
|---|---|
| `gpu_before_gpu2104.json` | the pre-Step-6 baseline (`89fa5ca`): Acrobot 16,661 f/s at B=4,096 |
| `gpu_after_gpu2104.json` | the planar-canvas patch (`6213673`): Acrobot 37,221 f/s, 2.2x |
| `tiers_gpu_gpu2104.json` | `scratch/step6/gfxdraw/tiers.py` over the two, not a shard |

`findings.md` cites these for the wave-1 result — 2.2-2.5x, peak VRAM 9.8-10.3 -> 7.6-9.1 GB, and
`BATCH_SCALES` at 891-1,747x before and 1,179-1,890x after — and that is what they are for.

## The current GPU certification rows

`../../sweep2/gpu_paired_after_gpu2264.json` and `..._gpu2107.json`.
