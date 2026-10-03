# Sustained independent-process classic diagnostic

Array 6970936 investigates the unresolved Pendulum/Acrobot snapshot candidate
without changing production. Each registration gets one RTX 3090 and eight
physical CPUs. Each strategy/cell runs in a fresh interpreter, with the same
unchanged candidate/reference sources and ordinary renderer. No two populations
coexist. Reconstruct the original candidate overlay at base 405216af, add the
two scripts here under scripts/benchmarks/online_stepping, then verify every
source_manifest.json hash before running run.sh. All 180 prior frozen file
hashes remain unchanged; the two diagnostic scripts are the only additions.

Four CUDA-output cells at B1024: T1/no States/final frame; T8/all States/final
frame; T8/no States/no frames; T8/no States/final frame. Each cell measures six
windows of 128 consecutive transitions per lane. The actions are zero torque
(Pendulum 0, Acrobot discrete 1); validity and elapsed-step counts are checked
without counting terminal padding. Seeded resets are outside timing and their
durations are recorded separately. No restore occurs between step calls.
Normal GC is enabled with unchanged thresholds; collection counts are recorded
outside each window without callbacks, forced collection or sample exclusion.

Untimed full-trajectory fingerprints include every task output, selected State,
mandatory final State and selected image. They must agree across independent
strategy processes. Every timed window counts all returned validity flags and
must reach the same complete final State. The strict summarizer requires six
positive samples, matching trajectory hashes, source hashes, host, allocation,
device, renderer backend and worker counts. Results compare two library paths,
not source performance. These CUDA diagnostics do not replace prior CPU/CUDA
acceptance or establish results for other registrations/batches/workloads.

The small local CPU smoke uses B4/T1, 16 transitions, two windows and final
frames; independent baseline/candidate full-trajectory fingerprints match.
Its rates are not cluster-performance evidence. Earlier complete paired runs,
repeats and the inconclusive GC attribution are preserved unchanged.
