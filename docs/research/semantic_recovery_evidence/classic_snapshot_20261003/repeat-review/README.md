# Repeat review: candidate remains isolated

Both full unchanged-code repeats completed and pass the same strict CPU/CUDA
report checks. The 128-row table places both runs side by side; no samples
were dropped or replaced, and rates from different hosts are not pooled.

Pendulum repeats its CUDA B1024/T1/no-State/final-frame regression: ratios
0.870 and 0.872. Five candidate samples are about 0.05 seconds, but the fourth
is 0.216 and 0.209 seconds. The T8/all-State/final-frame cell also repeats a
slow fifth sample (0.411 and 0.413 seconds versus roughly 0.28 normally).
This is repeatable; it must not be dismissed as generic measurement noise.
A separate instrumented job records collection events and sample windows to
check whether garbage collection explains those durations. No GC thresholds,
validation, State semantics or production routes are changed.

Acrobot has small gains and several slower cells whose locations vary across
runs; its first repeat was on gpu2110, while the original was on gpu2109.
This does not justify a universal speedup claim or activation without further
review. All five correctness gates remain passed; performance consistency
and production activation are separate outstanding requirements.
