# Humanoid pipeline harness accepted

Job 6968710 completed in 5:32 on gpu2102 / RTX 3090. Both CPU/software-GL and
CUDA/NVIDIA reports pass six independent complete-State/task/frame comparisons
at B64, T1/8 and no/final/all frames. The strict summarizer validates every
component and pipeline cell, six samples per timing cell, device proof and
source hashes. Measured DMC/dispatcher source still matches integration.
The isolated dependency profile is mujoco-warp 3.12.0 / warp-lang 1.17.0; installed
file hashes are retained without modifying se-step6.

CUDA B64/T8/all gives 5,654 +/- 10 frames/s versus deferred dispatch 5,637 +/- 43:
no material overlap benefit at this batch. Task stepping without State capture
(including packaging/IPC/transfer) gives 9,275 +/- 41 transitions/s, isolated
State export 50,182 +/- 415 states/s, and saved-State all-frame rendering
38,225 +/- 507 frames/s. These independent component timings are not additive.
CPU rendering is the dominant bottleneck at roughly 62–64 FPS. This is a
scheduling comparison within the library, not a same-device upstream speedup
or a new upstream-renderer fidelity certificate. No threshold is relaxed.

The measurement harness can now expand to the eleven accepted DMC process
routes at B64/256/1024. Quadruped remains excluded pending its independent
exact-frame diagnostic. Earlier failed attempts remain preserved separately.
