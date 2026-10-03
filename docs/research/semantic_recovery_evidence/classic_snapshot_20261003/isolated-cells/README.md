# Original slow cells, fresh interpreters

Seven cells from the earlier paired regressions are repeated with the original
seed/action/reset workload in separate baseline and candidate interpreters.
Each strategy retains six windows of 32 calls (192 raw durations); restoration
is recorded separately and garbage collection remains enabled. Full State,
task and selected-frame fingerprints agree exactly between strategies.

All seven strict application summaries pass. Continuous MountainCar CPU
B1024/T8/all States/final frame still regresses: 5,739 to 4,960 steps/s, ratio
0.8642. The other six ratios range 0.9925–1.0399. This is evidence against
promoting the candidate unchanged, not a reason to remove slower samples.
The new required-State scheduling experiment remains separate and unaccepted.

Array 6971656 application outputs and source hashes are preserved here. Fresh
terminal accounting and cluster logs remain to be collected after SSH access
returns. No scheduler-terminal acceptance is inferred from JSON alone.
Production is unchanged. Source manifest reconstructs the prior snapshot
candidate plus sustained and isolated drivers from their preserved overlays.
SHA256SUMS.json paths are relative to this directory.
