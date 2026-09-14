# Superseded: the node2338 CPU sweep

**These shards must not be used to label a tier.** They are kept because the renderer throughput in
them is sound and because deleting a measurement is worse than explaining it.

## What is wrong with them

`node2338` was running at a **load average of 47** while this job (SLURM 6346431) was on it, and
`examples/bench_step3.py` times the official renderer **once, in a block, after the whole batch
sweep**. On a node that loaded, that single reading is not a measurement of the reference — it is a
measurement of the node at one moment. Within this one job, on this one node:

| config | official f/s, "before" arm | official f/s, "after" arm | swing |
|---|---|---|---|
| Acrobot | 125.4 | 55.5 | **2.3x** |
| MountainCar | 220.6 | 114.6 | **1.9x** |
| Pendulum | 46.6 | 31.5 | 1.5x |
| CartPole | 217.5 | 218.3 | 1.00x |
| MountainCarContinuous | 139.8 | 141.0 | 1.01x |

`S_cpu` is our throughput over that number, so the after-arm rows in
`cpu_1thread_after_node2338.json` read `S_cpu` **4.64 / 2.94 / 2.77 / 14.60** for Acrobot,
MountainCar, MountainCarContinuous and Pendulum — which would label them `CPU_FASTER` — where the
paired measurement on `node1829` gives **0.94 / 1.12 / 1.12 / 1.92**, i.e. `CPU_REGRESSION` and
`CPU_PARITY`. Pendulum's 14.60 is the clearest tell: a renderer three times slower than CartPole's
does not beat its reference fifteen-fold.

The numerator is unaffected. Our own frames/s in these shards agree with every other host to within
node noise, and the 4.3-7.2x before/after speedup they show is reproduced on `node1829`. It is only
the **denominator** that collapsed, and only `S_cpu` and the tier that depend on it.

## What to use instead

`../paired_cpu_node1829.json` — the same frozen batch sweep with the official renderer measured
**interleaved** with our own, round by round in one process, reporting the median of the per-round
ratio. See `scratch/step6/gfxdraw/paired_cpu.py` for why that is the fix and
`examples/results/step6/gfxdraw/findings.md` for the numbers.

## Why this directory exists at all

`scratch/step6/report.py` labels tiers by running `scratch/step3/certify.py:classify` over every
Step-3-shaped shard it finds under `examples/results/step6/`. It found this one and published
`CPU_FASTER` for four configurations whose own findings file says `CPU_PARITY`. A shard that a
workstream has declared unusable has to be **out of the generator's reach**, not merely
contradicted in prose next to it — moving it here is what makes the page and the findings agree.
