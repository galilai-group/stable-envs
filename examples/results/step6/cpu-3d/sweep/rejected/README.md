Shards the sweep pattern itself refused, kept so a missing file cannot read as a job that never ran:

- `cpu_1thread_paired_after_node2407.json` (job 6356047, commit 5f020fe): every mandatory cell
  measured INCLUDING batch 1024 (Cheetah 209.9 / Hopper 333.6 / Walker 114.9 / Humanoid 81.7 fps,
  flat against 256) but `certifiable: False` -- the interleaved official renderer spread 1.67x on a
  node at load 124-136, over the 1.25x bound, so no tier may be derived from it. The numerator stands.
- `*_6356330.json`, `*_6356449.json`, `*_6356450.json`: load-guarded attempts (`--max-load 12`) that
  aborted at start on nodes at load 39 / 126 / 29 and measured nothing.

The certifiable shard is `../after/cpu_1thread_paired_after_node2333.json` (job 6354767, spread
1.08-1.13; no 1024 cell for Walker/Humanoid at its 12 s per-cell budget, no tree SHA). A shard with
all three -- cells, SHA, spread inside the bound -- needs a quiet batch node or an exclusive
allocation; `scratch/step6/cpu-3d/sweep_cpu.sbatch` produces it when one lands.
