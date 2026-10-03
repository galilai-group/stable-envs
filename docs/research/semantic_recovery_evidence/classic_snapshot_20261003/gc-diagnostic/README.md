# GC-event diagnostic

6970565 instruments the same frozen Pendulum candidate/control benchmark with
GC start/stop callbacks and timing windows for each measured sample. Production
and candidate implementation bytes are unchanged. Callback bookkeeping can
shift allocation counts and collection placement, so these timings are diagnostic,
not replacement acceptance throughput. No collection is forced or disabled.
Correlate actual event overlap/duration with slow samples before attributing
a cause. The original and both repeat reports remain preserved separately.
