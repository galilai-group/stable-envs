# Production memory and initial online measurements

Cube 6970442 and Scene 6970441 independently completed on RTX 3090 gpu2109.
The source hashes, eight-owner route, terminal records, all 32 phase snapshots
and twelve six-sample online cells validate. Complete State/task/frame replay
is exact. This is unchanged production code, not a new optimization gate.

At B64, initial PSS is 18.22 GiB for Cube and 19.03 GiB for Scene. Repeated
restores raise observed phase snapshots to about 41.44 and 42.98 GiB. These
are coordinator-plus-eight-owner PSS snapshots, not a continuous peak. Full
State batches are only 601,408 and 671,040 bytes respectively. After close,
the observed parent PSS is 1.72 and 2.18 GiB. The measurements alone do not
attribute retention to live worlds, allocator arenas or collectible objects.

T8/all-State/no-frame public throughput is 1,570 +/- 65 steps/s for Cube and
1,442 +/- 55 for Scene. T8/all-frame online throughput is 205 +/- 13 and
192 +/- 10 FPS. Final-frame output emits one frame per eight transitions;
its reported FPS and steps/s therefore differ by eight. Construction, restore
and warmup are excluded from timing; no policy inference or source comparison
is included. All cells and SDs remain in pipeline.csv.

The next diagnostic separates full-State validation/projection and GPU transfer
from rendering precomputed RenderState. The full-State validator currently
copies fields to CPU one lane at a time. This is a concrete profiling target,
not yet proof of the dominant cost or authorization to weaken validation.

SHA256SUMS.json paths are relative to the parent evidence directory.
