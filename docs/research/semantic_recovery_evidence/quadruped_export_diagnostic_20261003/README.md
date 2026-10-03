# Quadruped exporter diagnostic: reproduced, not accepted

Job 6964907 ran on gpu2109 (RTX 3090), reproducing the original exact-frame
assertion failure. Its failed exit is expected from retaining that assertion.
The frozen driver and submission script are preserved in
../dmc_column_closure_20261003/diagnostic/; driver and gate hashes are in the
raw diagnostic report. This run does not relax the gate or enable routing.

For B4, T8, all frames, two uint8 channel values differ by one; normalized MSE
is 7.822011566064334e-11 across 32 images of 64x64 RGB. All captured non-image
outputs, including full State histories, RNG, observations and task outputs,
match byte-for-byte. Re-rendering the same saved states in both reference and
candidate contexts repeats consistently. array_checks.json independently checks
whether those four re-rendered outputs equal the original candidate frames.

This narrows the investigation to image reproducibility or render-path history;
it does not yet identify the cause. The original exporter acceptance remains
open. No semantic promotion or throughput claim follows from this diagnostic.
Raw reports, frames and terminal log are preserved unchanged.
