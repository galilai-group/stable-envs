# Classic direct-column State exporter accepted

All ten independent CPU/CUDA gates in 6967197 passed; all five jobs completed.
The field/ownership suite passed 12 selected test invocations without skips (the
mountain_car filter also selects its continuous sibling). The paired sweep has
160 complete rows. The integrated source/control matches all 171 frozen hashes.

Post-export RTX 3090-host profiles identified one NumPy temporary array per
field per lane, followed by a copy and repeated shape/dtype checks. This
candidate uses the fixed accepted schema to allocate each output column once
and writes the same source values directly into it. The returned tensors still
own fresh storage; no reusable output buffer, cached RNG or approximate physics
is introduced. The shared exporter used by other families is unchanged.

Local semantic tests pass 17 cases, with five CUDA cases reserved for Oscar.
A preliminary paired CPU run for continuous MountainCar passes exact task/State/
frame checks at B64/1024. Its original benchmark source is preserved under local/;
the cluster benchmark additionally asserts that every timed public-step slot is
valid, so padded termination slots cannot inflate throughput.

Array 6967197 independently tests all five registrations on CPU and CUDA. Each
job runs the field/ownership regression test and compares one/eight actions,
slice edge cases, lifecycle/terminal boundaries, RNG caches, unseeded reset,
physical variation and exact native frames. Timings use B1/64/256/1024 and six
samples in reversed-order blocks for get_state and State-retaining public step,
with rendering disabled. Restore, construction, warmup and checks are excluded.
These are adapter-exporter improvements, not source-renderer speedups.

The reference is the COLUMN exporter from the current accepted base 19c3b392,
loaded from a frozen module. Both its get_state and _state_values methods are
bound to reference owners. It is not the earlier, slower per-lane TensorDict
control. Raw benchmark mode labels remain legacy/column; here they mean previous
column exporter/direct column writes. source_overrides.json and source_manifest.json
reconstruct the frozen candidate and reference from that exact base. Read actual
reports and terminal evidence before treating any registration as accepted.

At B1024/T8, State-retaining public-step improvements against the preceding
column exporter are 1.15–1.64x across all five registrations and both output
devices. Export-only improvements across the entire sweep range 1.01–2.14x.
Three B1/T1 public-step cells have lower measured means (CartPole CPU -4.7%,
MountainCar CUDA -0.5%, continuous MountainCar CPU -0.8%), with candidate SD
larger than those differences. Those samples remain in the CSV; this is not a
claim of universal improvement at every batch size. Physics and rendering are
unchanged. No process route is enabled for classic control by this change.

Reproduce comparison.csv with summarize_classic_state_export.py results
NEW_OUTPUT --source FROZEN_CHECKOUT --batches 1 64 256 1024. The reference module
is captured in source_overrides.json, verbatim from the base commit. Semantic
coverage remains 39 VERIFIED / 3 INCOMPLETE. Subsequent process comparisons must
use this faster exporter as their local control.
