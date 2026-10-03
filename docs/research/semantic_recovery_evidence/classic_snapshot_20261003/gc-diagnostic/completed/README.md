# GC trace closure

Job 6970565 completed with identical implementation and exact CPU/CUDA gates.
All source hashes and 128 timing cells validate. The instrumented report records
10,568 collection events and 768 sample windows; sample_gc_overlap.csv preserves
every measured window, including those without GC overlap.

One timed generation-2 collection overlaps a CPU B1024/T8/all-State/final-frame
candidate sample by 155 ms. The sample lasts 2.250 seconds. The slow CUDA samples
from the two uninstrumented sweeps do not recur at the same timed locations in
this trace. Callback allocation can shift collection placement. Thus this is
evidence for substantial allocation/collection pauses, but does NOT conclusively
attribute the earlier CUDA spikes to GC or justify discarding those samples.

The candidate remains isolated. The next performance check should use sustained
runs in independent strategy processes and include naturally occurring collection
costs, alongside allocation profiling. Do not force/disable collection or select
the best run to manufacture a gain. No production API or behavior changed.

SHA256SUMS.json paths are relative to this completed directory.
