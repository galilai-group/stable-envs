# Native classic CPU thread profile

Array 6971126 completes independently for Pendulum and Acrobot, with exact
State/task/frame gates at B64/256/1024, T1/8, no/final/all frames and 1/2/4/8
parent Torch threads. All intermediate State histories are disabled; mandatory
final State remains. The same ordinary production route and renderer are used.
Acrobot's eight persistent simulation owners stay single-threaded. Six timings
are retained for each of 84 cells per registration, plus untimed render cProfile
at B256 for one and eight threads. Strict summaries verify source hashes,
eight-core allocation, worker counts, exact outputs and process CPU counters.

Linux counters sum process user/system time for the parent and persistent
simulation workers. Average busy cores is CPU seconds / elapsed seconds;
counters are quantized and include slight boundary error. This is measured CPU
use, not GPU utilization or MFU. Every counter and raw sample is preserved.

At B1024, saved-State render FPS rises 590 ± 0.4 → 2,483 ± 26 for Pendulum and
489 ± 1 → 1,700 ± 67 for Acrobot. At T8/all frames, online FPS rises
464 ± 2 → 1,787 ± 37 (3.85x) and 443 ± 17 → 1,601 ± 12 (3.62x).
Saved rendering uses about 7.8–7.9 cores at eight threads versus one at the
one-thread setting. This identifies a benchmark configuration limitation:
prior source comparisons fixed our parent Torch CPU rendering to one thread.

No library-global thread setting or public API is changed. The next source
comparison gives the parent CPU renderer eight threads within the same existing
eight-core allocation; source workers remain bounded to eight. Old one-thread
results are preserved. These thread ratios alone are not upstream speedups.

Local B4/one-versus-two-thread CPU smoke passes for both registrations. The
cluster source is reconstructed from base 061e7a29 plus the two profile scripts
committed with this evidence, then verified against source_manifest.json.
