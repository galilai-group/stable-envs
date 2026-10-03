# SimplePointMaze bounded native baseline closure

Job 6964296 completed on gpu1503 with RTX A6000. The frozen c39880f9 source
matches every source hash in SimplePointMaze.json. All 54 cells are present:
CPU/CUDA, B1/16/64, nine component operations, three raw samples per cell.
comparison.csv reports component rate means and sample standard deviations.
This completes the bounded baseline alongside the previously committed TwoRoom
and Piecewise profiles, not a full scaling or upstream comparison.

This baseline starts from constructor State and restores it outside timing.
The masking paired tests explicitly reset(seed=17); do not compute speedup by
dividing rates across those different workloads. The source-required goal image
remains part of every step's task output, even with frames=None. No image MSE
or new semantic acceptance is claimed by this timing-only baseline.
