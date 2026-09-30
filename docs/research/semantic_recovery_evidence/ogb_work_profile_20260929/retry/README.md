# Completed Scene diagnostic

Oscar job 6800940_30 completed on gpu2106; downloaded results are in `30/`. The earlier collector status is stale because it stopped while the job was queued. This is an instrumented diagnostic, not a replacement for unprofiled throughput measurements.

Across three batch-256 calls: draw/readback 492.38 ms, mj_forward 179.21 ms (1536 calls, twice per image), context lookup 47.03 ms, update_scene 14.93 ms, and other renderer-boundary work 190.51 ms. The renderer boundary total is 927.92 ms. Warm cache counters show no context churn. Repeat maximum uint8 differences are [1, 1, 0], within the existing one-level tolerance.
