# Sparse inactive-row padding

Parent 69ac94c20d859ffc6d55911654fd85f53169ae63. The previous all-active fast path
still scanned large image tensors when even one lane ended. Replacing boolean
assignment with index_fill_ on the inactive row indices preserves owned outputs
and zero padding while touching only ended rows. CUDA nonzero can synchronize;
GPU cost must be measured, not inferred from the CPU result.

Local public gates independently pass all 16 exact State/task/frame comparisons
for TwoRoom, Piecewise and SimplePointMaze against frozen c39880f9. The strict
54-row CSV covers B1/16/64, T1/8, all/no State history and final-frame pipeline,
with six samples per cell. Shared tests: 226 passed, 76 CUDA cases skipped.
These three CPU runs were sequential. MSE zero is a library regression result,
not a fresh upstream renderer survey.

The profile holds B64/T8, reset seed 17, all States, frames=None. One lane ends
after its first action: 505 valid transitions, not 512. The retained source
info.goal history is 384,000,000 bytes. Before sparse padding, _masked_outputs
cost 92 ms (82 ms direct masking work) of a 139 ms profiled call; afterward it costs 9 ms of a 31 ms call.
These single cProfile calls identify work, not statistically paired speedups.
The separately paired CPU benchmark reports 4,126 to 20,226 valid steps/s for
that workload (4.90x), and 5,522 to 27,831 for T1 (5.04x). See raw samples/SDs.
Stacking/copying remains measurable; the source payload is preserved.

Arrays 6964504 and 6964505 were verified still PENDING and canceled before any
execution. Their previously committed source/evidence was not overwritten.
Replacement RTX 3090 arrays: 6966161 native exact gates, paired timings and all
302 shared/native tests; 6966162 all twelve Fetch CPU/CUDA pipeline gates and
Fetch adapter tests. GPU acceptance remains pending at the snapshot.

The new 173-file bundle passed isolated test collection and remote SHA-256
verification. oscar/ contains source manifest, launchers and overrides. Reproduce
local summary with summarize_step_masking.py local NEW_OUTPUT --source .
--devices cpu from the matching candidate checkout. The default requires both
devices, full coverage, matching hashes and finite positive timing samples.
