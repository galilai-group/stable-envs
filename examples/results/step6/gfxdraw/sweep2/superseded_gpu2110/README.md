# Superseded: the gpu2110 GFXDRAW-2 sweep

**These shards must not be used to label a tier**, for the same reason as
`../../sweep/superseded_node2338/`: `examples/bench_step3.py` times the official renderer once, in
a block, after the whole batch sweep, and on a busy node that single reading is a measurement of
the node rather than of the reference.

Within this one job (SLURM 6349362) on this one card:

| config | official f/s, kernel OFF | official f/s, kernel ON | swing | R from the ON arm | tier it would publish |
|---|---|---|---|---|---|
| Acrobot | 428.2 | 229.8 | **1.9x** | **700.3** | `GPU_FAST` |
| CartPole | 488.4 | 246.1 | **2.0x** | **756.2** | `GPU_FAST` |
| Pendulum | 192.0 | 263.9 | 1.4x | 667.2 | `GPU_FAST` |
| MountainCar | 448.3 | 439.1 | 1.02x | 333.7 | `GPU_FASTER` |
| MountainCarContinuous | 450.5 | 441.7 | 1.02x | 331.6 | `GPU_FASTER` |

The two rows whose reference halved would be published at `GPU_FAST` — over the class-R bar of 500
— where the paired measurement puts them at **389.8** (`gpu2264`) and **440.1** (`gpu2107`) for
Acrobot and **347.0 / 394.9** for CartPole, i.e. `GPU_FASTER` on both cards. Note that the two rows
whose reference was STABLE are the two that agree with the paired result: that is the tell.

Only Pendulum's `GPU_FAST` survives a paired measurement (691.4 and 785.3 on the two cards), and it
survives because its ratio is genuinely that large, not because its denominator moved.

The numerator is unaffected and is the source for every throughput and VRAM figure in
`findings.md`: our frames/s here (160,896 for Acrobot at B=4,096) reproduce on both other cards to
within node noise, and `peak_gb` in these shards is reset per cell by
`bench_gpu_vs_cpu.gpu_throughput`, so the 8.85 -> 3.10 GB claim rests on them.

## What to use instead

`../gpu_paired_{before,after}_<node>.json` from `scratch/step6/gfxdraw/certify_shard.py`, which is
the same schema and the same cells with the reference measured **between every batch cell** and
reported as the median of those readings with its spread. Cross-checked by
`paired_gpu_B4096_gpu2264.json` and `paired_gpu_B4096_perarmpeak_gpu2107.json`.
