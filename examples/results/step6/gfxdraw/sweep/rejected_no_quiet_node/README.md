# Three attempts, three rejections: the partition could not hold a reference steady

`cores_cpu` for the five gfxdraw configurations is **not measured**, and this directory is why.
Nothing here is a shard to use; each file records an attempt that refused to produce a number.

## The attempts

| # | job | node | mechanism | load at start | reference spread | verdict |
|---|---|---|---|---|---|---|
| 0 | 6346431 | `node2338` | `bench_step3`, reference timed once after the sweep | 47 (48 cores) | not recorded — a single reading has none | the same renderer read 125.4 and 55.5 f/s in the two arms, **2.3x**, and the shard could not say so |
| 1 | 6353689 | `node2339` | `certify_shard.py`, reference interleaved between every cell | — | **4.52x** (Acrobot), **10.37x** (CartPole) | rejected by its own run-spread check |
| 2 | 6356085 | `node2407` | + a 12-burst pre-flight before any renderer work | 133 (192 cores) | **1.62x** in pre-flight | rejected in 13 s, before wasting the sweep |
| 3 | 6356102 | `node2422` | the same, aimed at the **quietest node in the partition** (load/core 0.29, 103 CPUs idle) | 57 (192 cores) | **1.65x** in pre-flight | rejected |

Attempt 3 is the one that settles it. `node2422` was chosen by querying every node in the partition
for `CPULoad` and idle CPUs and taking the best — and its reference still varied 1.65x, a spread
indistinguishable from the busiest node tried. Sole tenancy, not a quieter co-tenant, is what the
measurement needs.

## Why `--exclusive` is not available

Job 6353993 asked for it and pended for two hours with the estimate slipping from 02:03 to 05:34,
finally showing the reason outright: **`QOSMaxCpuPerUserLimit`**. The `normal` QOS caps a user at
`cpu=64`; an exclusive allocation charges the whole node, and this partition's nodes are 48 or 192
cores. On a 192-core node the request is 192 against a cap of 64 and can never be satisfied; on a
48-core node it fits, and those were saturated (632 idle of 12,928 CPUs partition-wide). Cancelled
so the three attempts stayed three.

## What this does and does not mean

**It does not mean the CPU renderer is unmeasured.** `S_cpu` is a ratio of two throughputs measured
against each other, and pairing them round by round is exactly what makes that ratio survive a
noisy node — `../cpu_1thread_paired_after_node1829.json` has a worst spread of 1.11x and is the
CPU certification row. `cores_cpu` is different: it is a ratio to a JPEG decode bar that has to be
measured on the same host in the same conditions, and none of these three nodes could hold either
side still enough to pair them.

**It does not mean `BELOW_G4` is in doubt.** It was measured at 0.22-0.50 on `node2334`, `node2338`
and `node2339`, and the bar is 1. Every CPU row is below G4 by a factor of two to five. What is
missing is a shard entitled to assert it, not knowledge of whether it is true.

**A guessed `cores_cpu` would be worse than a missing one.** Borrowing another host's decode bar to
fill the column is the cross-host ratio this whole provenance pass exists to remove, and it would
be indistinguishable on the page from a measured one.

## What would fix it

An exclusive allocation, which needs either a QOS whose `cpu` cap is at least one node's core count
or a partition with 48-core nodes free. Neither is something this workstream can arrange, and the
decision belongs to whoever owns the certification budget.
