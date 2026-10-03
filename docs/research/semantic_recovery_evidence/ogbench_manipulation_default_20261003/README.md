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
