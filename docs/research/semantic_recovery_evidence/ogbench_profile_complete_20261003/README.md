# Complete bounded OGBench component baseline

The four B64 follow-ups in 6964316 all completed. Combined with the original
Maze B1/16/64 and Cube/Scene B1/16 cells from 6963334, all 18 registration/device/
batch cells now have exact restore/step replay and six samples for four operations.
The strict combined CSV contains 72 component rows. The two original Cube/Scene
reports remain RUNNING with separately preserved terminal TIMEOUT evidence;
their completed cells are reused without relabeling or overwriting those reports.

Frozen source is recorded in ../ogbench_online_20261002/profile/source_manifest.json
and source_overrides.json, based on f0716916. Rebuild from matching committed files
and overrides, verifying every SHA-256. The new benchmark script was first committed
with the profile checkpoint, after the base commit; do not assume every path exists
in that parent. The closure validates identical source maps across every report.

Reproduce with summarize_ogbench_profile.py ORIGINAL_RESULTS missing64 NEW_OUTPUT
--source FROZEN_CHECKOUT. ORIGINAL_RESULTS is
../ogbench_online_closure_20261002/profile/results. Each CSV row records its actual
host/GPU; follow-up cells are not paired optimization comparisons with prior hosts.

This is the original pre-column-export adapter diagnostic. The no-export operation
substitutes an already owned State only to measure physics/task work; it does not
produce a valid transitioned State. It is labeled DIAGNOSTIC and is not a public
throughput claim. Construction and restore remain separate; no renderer FPS,
upstream speedup or new fidelity acceptance is claimed by this baseline.
