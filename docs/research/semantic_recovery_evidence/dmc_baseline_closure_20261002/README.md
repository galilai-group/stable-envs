# DMC baseline closure

All twelve baseline tasks in RTX 3090 array `6960514` completed successfully.
The reports contain CPU/CUDA output at B1/64/256, six samples per operation,
complete restore/step replay checks and source-compatible false public done
flags. `comparison.csv` contains 216 component measurements; these are baseline
step/export/restore rates, not new process speedups or renderer FPS.

Frozen source, launcher and reconstruction metadata are in the previously
committed `../dmc_online_20261002/baseline-v2/` directory. Report source hashes
were checked against its manifest. This closure preserves the new reports and
terminal logs without rewriting the earlier checkpoint. `slurm-checkpoint.txt`
records scheduler evidence; report PASS and expected cell coverage, rather than
scheduler success alone, establish completion.

Column-export acceptance `6961949` remains pending. Process acceptance was
recorded separately in `../dmc_process_20261002`; throughput array `6961930`
remains incomplete. Its Acrobot/Pendulum tasks failed opening X11 before the
first renderer proof. Their original reports/logs are under `process-failures/`.
Retry `6964379` runs only those two tasks, with unchanged frozen source and
resource allocation, in a new `benchmark-retry2` output directory. The launcher
is `benchmark-retry2.sh`. This does not select a process default.
