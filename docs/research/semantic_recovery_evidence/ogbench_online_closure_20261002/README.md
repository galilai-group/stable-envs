# OGBench export acceptance and baseline closure

This adds evidence to the committed `../ogbench_online_20261002` checkpoint;
none of its raw reports are overwritten. Production exporter code is unchanged
from `6ce4e82931a629b68218bbe5331296bdc0a946d2`.

Baseline array `6963334`: Maze completed all six B1/16/64 CPU/CUDA cells. Cube
and Scene timed out at the 45-minute allocation, with four completed B1/16
CPU/CUDA cells each. Their JSON status remains RUNNING because SLURM killed
them; terminal TIMEOUT evidence is in `slurm-checkpoint.txt` and the logs.
Do not treat those reports as complete. `profile/run-missing64.sh`, array
`6964316`, measures only Cube/Scene B64, with one CPU/CUDA output task each,
from the identical frozen source. Outputs go to `missing64/`.

Export array `6963669` failed all three tasks during test collection: the
reference-helper package imports the regression support matrix, omitted from
the archive. The failures are preserved in `failed-column-v2/`. V3 adds only
that fixture, leaves production and benchmark code unchanged, and collects all
six tests successfully from the isolated archive. Its manifest and launcher
are in `column-v3/`; use `source_overrides.json` for archived source that differs
from this checkpoint.

Repaired array `6964281`: Maze independently passed CPU/software-GL and
CUDA/EGL exact-export tests and public State/task/frame gates (10 cases each;
MSE 0 versus the old exporter), plus all 16 timing cells per device. Reports
and terminal logs are in `column-v3/results/Maze/`. The actual GPU is RTX 3090.
Cube/Scene remain running at this checkpoint; Maze does not promote siblings.
No new upstream fidelity claim or process default is made.
