# Local continuous MountainCar mechanism probe

Fresh baseline and candidate interpreters use the frozen isolated-cell source,
B1024/T8/all States/final frame, one Torch thread, three uninstrumented calls
and two cProfile calls. State restoration remains outside profiling/timing;
full output fingerprints match. Source manifest and driver hashes are recorded.
The machine is Apple M5, not the Oscar RTX 3090 host. This is not acceptance
evidence for the cluster performance regression.

The regression does not reproduce here: uninstrumented means are 0.7442 s
baseline and 0.7414 s candidate. In the two profiled calls, rendering consumes
1.336 of 1.545 s baseline and 1.368 of 1.574 s candidate. The native CPU
rasterizer and image copies dominate. This does not prove why the retained
Oscar case is slower; an identical same-host profile is still required.
It gives no justification for a shared scheduler change.

Run this driver from the frozen classic-snapshot-isolated-cells-v1-20261003/code
directory using its Python dependencies and PYTHONPATH=., with --mode baseline
or candidate and a new --output path. Original source_manifest.json must sit
in the parent directory. All frozen source hashes are verified before running.
