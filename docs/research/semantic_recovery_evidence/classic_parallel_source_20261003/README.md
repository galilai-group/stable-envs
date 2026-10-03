# Resource-matched classic source pipeline comparison

Array 6971034 benchmarks all five accepted classic registrations independently.
Each task receives one RTX 3090, eight physical CPUs and 64 GiB. The benchmark
compares direct serial pinned source, eight persistent source workers, and the
ordinary production library, on the same allocation in reversed strategy order.
Three array tasks may run concurrently. Source is pinned stable-worldmodel
176d54532b3c0a251a4f509b74f2fdacc143a63d; the manifest records the official
repository URL and seven unchanged source modules. Installed dependency versions
and source runtime file hashes are recorded in every application report.

The source executor uses Gymnasium AsyncVectorEnv with spawn, shared_memory=True,
copy=True and disabled auto-reset. Each worker owns a scalar wrapper shard;
source physics and render calls are unchanged. Images and numeric task outputs
travel in shared observation buffers. Action chunks are forwarded to shards;
requested frames are rendered at their actual intermediate states. Owned parent
arrays are delivered to the selected output device. The source executor is a
benchmark harness, not an upstream native batched simulator or a library route.

Source pygame rendering supports CPU only. CUDA-output rows measure original
CPU rendering plus CUDA delivery against our native CUDA renderer. They must
not be labeled source-GPU rendering. CPU rows compare CPU to CPU. Both serial
and parallel source rates are preserved, and the table also compares ours to
the faster measured source mode. No source superiority claim is made before
these pending reports pass. Ours additionally returns its mandatory final State;
intermediate State history is disabled for this comparison. Source oracle export
and restore stay outside timing; raw task stepping includes output packaging.

B64/256, T1/8 and no/final/all frames receive six samples per strategy. Matched
complete final State, task outputs and images must pass each registration/device
independently. Parallel-source outputs must match serial source exactly and
remain owned after a subsequent shared-buffer write with different observations.
Image MSE versus our renderer is normalized by 255 and bounded by 0.001 per
checked frame. These are bounded seed-17 online trajectories, not a new broad
1000-state renderer corpus. Construction, restore and warmup are outside timing.
Mean, sample SD, actual transition/frame counts and all raw seconds are retained.

All five final local CPU smoke reports pass: B4, two source workers, six cases
per registration. Every checked source image matches exactly (MSE zero). These
are correctness-only reports, not cluster timing evidence. During development,
Torch's extra shared-memory manager was disallowed for untimed oracle snapshots
on the Mac; the harness now sends those untimed snapshots as NumPy columns.
Timed observations/images still use native Gym shared memory.

Reconstruct candidate_manifest.json from base 6b30df6e plus the three benchmark
files committed with this evidence under scripts/benchmarks/online_stepping.
Verify every hash, extract the seven pinned source files into source/, and run
run.sh. Production code and semantic acceptance classifications are unchanged.
