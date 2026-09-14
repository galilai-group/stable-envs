# Rejected: the node2339 CPU shard, on its own evidence

This shard was produced by `scratch/step6/gfxdraw/certify_shard.py` (SLURM 6353689) — the script
written specifically to fix the block-timed-denominator problem, by interleaving a reference burst
between every batch cell and reporting the median with its spread.

**It rejects itself.** The official renderer's throughput across those interleaved readings varied

| config | reference f/s | spread |
|---|---|---|
| CartPole | 44 - 452 | **10.37x** |
| Acrobot | 87 - 394 | **4.52x** |
| Pendulum | 196 - 265 | 1.35x |
| MountainCar | 331 - 442 | 1.34x |
| MountainCarContinuous | 350 - 441 | 1.26x |

against 1.02-1.11x on `node1829` and `gpu2264`. `node2339` was loaded enough during this job that
the reference is not a property of the renderer on it.

**This is the point of the exercise, not a failure of it.** The shard the whole
provenance pass started from — `../superseded_node2338/cpu_1thread_after_node2338.json` — had
exactly this disease and no way to show it, because a single block-timed reading has no spread. Its
denominator had collapsed by 2.3x and nothing in the file said so; the tier it published was wrong
and looked exactly like a tier that was right. Here the same thing happened, the file says so, and
`certifiable: false` is in the JSON with the reason next to it.

The numerator is unaffected and agrees with `node1829`: best 290 / 570 / 482 / 376 / 418 frames/s
for Acrobot, CartPole, MountainCar, MountainCarContinuous and Pendulum. The decode bar it measured
(852-1,210 f/s per core) is also sound, and is consistent with the 0.22-0.50 `cores_cpu` quoted in
`findings.md` from `node2334` and `node2338`.

## What to use instead

`../cpu_1thread_paired_node1829.json`, whose worst spread is 1.11x.
