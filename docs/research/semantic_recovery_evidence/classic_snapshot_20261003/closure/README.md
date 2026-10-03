# All five correctness gates and first performance sweep

Array 6970326 completed every registration independently on RTX 3090 hosts.
Each job passed all 147 CPU/CUDA tests without skips and 28 public cases
across both devices. All State/task/RNG/lifecycle/ownership and image checks
match the frozen adapter exactly. The strict collector validates terminal raw
job IDs and source hashes, then reproduces every downloaded CSV from samples.
The joined table retains all 320 paired rows (128 timing cells per registration).

Production is NOT enabled from this closure. Pendulum and Acrobot have slower
cells in the original sweep, including substantial timing spikes; complete
unchanged-code repeats 6970466 and 6970476 investigate repeatability. Original
measurements remain in this table and must not be replaced by the faster run.

At B1024/T8 with intermediate States and frames disabled, CartPole improves
about 1.93x, Pendulum 1.53–1.55x, MountainCar 1.62–1.63x, continuous MountainCar
2.55–2.62x, and Acrobot 1.01–1.05x across output devices. The final complete
State is still returned. These are comparisons with the previous adapter,
not source speedups or renderer-only rates; all-State workloads often change
little. Performance review and activation remain separate from correctness.

The pending implementation and its tests remain in the committed candidate/
overlay and isolated source archive. They have not been inserted into the
production test suite or production adapters. SHA256SUMS.json paths are relative
to the parent classic_snapshot_20261003 directory.
