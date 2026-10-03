# Cube/Scene production memory diagnostic

Array 6970441 runs one ordinary public population per registration at B64 with
eight CPU process owners and CUDA rendering. Each task requests an RTX 3090,
eight physical cores and 128 GiB. Construction, State capture, held/released
T1/T8 outputs and close receive CPU memory snapshots and Torch CUDA allocator
statistics. PSS apportions shared pages; summed RSS can double-count them.
CPU totals cover the coordinator and eight physics owners, not cluster daemons.
Torch CUDA statistics exclude non-Torch OpenGL/driver allocations. Phase
snapshots are not continuous peak measurements.

Six samples per no/final/all-frame online cell retain all States and check
exact restore/replay. Restoration is excluded from timing but its retained
memory is visible. This estimates a normal production population instead of
the previous simultaneous candidate/reference acceptance workload. It does
not establish a source speedup, full memory peak, or new semantic acceptance.

Production files are unchanged from the pinned integration source. The script
and full source manifest reconstruct this diagnostic. Reports remain pending;
larger batch sizes should follow measured memory headroom, not extrapolation
from the old paired acceptance-test OOM.
