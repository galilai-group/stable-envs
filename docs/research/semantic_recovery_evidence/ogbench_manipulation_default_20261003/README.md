# Cube/Scene default-routing acceptance pending

Array 6967384 tests Cube and Scene independently on RTX 3090 with eight physical
CPUs. Both tasks passed all 60 routing unit tests before starting their CPU and
CUDA public gates. The application reports remain authoritative; this submission
record is not an acceptance result.

The isolated source starts at 81b36915 with the five overrides recorded here.
The constructor override is deliberately absent from the integration checkout
until the required gates pass. The other overrides add the private selection
helper, routing tests and acceptance harness. Small batches, non-Linux systems,
daemon workers and hosts with fewer than four available CPUs retain local owners.
Eligible batches of at least 64 use up to eight persistent process owners.

Each device gate constructs the candidate through unpatched public `se.make`
with its actual default renderer at B64. A forced-local control uses the same
renderer. Twenty-two cases cover one/eight actions, float32/float64 inputs,
device-resident actions, output slices, complete State/task/RNG equality, exact
frames, output ownership, reset, variation, in-flight rendering and rejected
last-shard restore. CPU/software GL and CUDA/NVIDIA rendering run in separate
interpreters. All four reports must independently pass; prior sibling results
cannot satisfy this gate. No timing speedup follows from these correctness runs.

The earlier process sweep and automatic CPU renderer acceptance remain preserved
in `../ogbench_manipulation_process_complete_20261003/` and
`../ogbench_auto_render_20261003/`. This gate combines their accepted changes in
ordinary construction. When all reports are terminal, summarize with
`summarize_ogbench_manipulation_process.py RESULTS NEW_CSV --source FROZEN_SOURCE
--production`; verify the integrated source hashes before enabling constructors.

Remote stage: `/oscar/scratch/sboughan/se-semantic-recovery/ogbench-manipulation-default-v1-20261003`.
Accepted semantic coverage remains 39 VERIFIED / 3 INCOMPLETE.

Scene task 6967384_1 subsequently hit its 64 GiB allocation and ended
OUT_OF_MEMORY after 24:05 (batch MaxRSS 67,075,976 KiB). Its CPU application report
was left RUNNING by the external kill; it is not a passing report. Failed reports
and terminal status are preserved under `failed-scene-v1/`. The earlier profile
measured only 671,040 bytes for a full B64 Scene State, so State payload size
alone does not explain the peak. Repeated simulator construction/restore and
renderer resource ownership require diagnosis before choosing the retry.
Cube task 6967384_0 remains independently live; no restart is inferred from the
Scene failure. No Cube/Scene process default is enabled.

Scene retry 6968217 uses the same frozen source and exact assertions, with
128 GiB and a fresh `ogbench-scene-default-v2-20261003` output directory on
gpu/norm-gpu. Its launcher and unchanged source manifest are in `scene-retry-v2/`.
The [small restore-memory diagnostic](../ogbench_restore_memory_20261003/)
plateaus with a large simulator footprint; it does not prove Linux large-batch
memory closure. The extra allocation is acceptance-test headroom, not a claim
that production memory or restoration is optimized.
