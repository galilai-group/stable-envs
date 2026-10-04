# Rocket final candidate closure evidence

**Final status: VERIFIED.** Job 6996439 completed 0:0; `final/`,
`final-terminal.txt`, `verification/`, and `acceptance.json` close the pending
observations below. Both devices' raw frame pairs were independently recomputed.
The non-Craftax acceptance count is 38/38. Performance remains separately scoped.

This directory is separate from the earlier sealed Rocket investigation.
The earlier pending observations below are retained as submission/retrieval history.

- `submission/`: immutable source manifest and launcher for final job 6996439.
- `superseded/6996062/`: pre-device-guard control, cancelled after both public
  204-checkpoint renderer gates passed; the whole run did not complete.
- `superseded/6996377/`: device-guard control, cancelled after 67 CPU and 3 CUDA
  tests passed, to make the optional Triton helper private to autodiscovery.
- `superseded-terminal.txt`: cancellation evidence, also for the never-started
  3090 duplicate 6996378.
- `thin-fin-comparison.png`: source / old GPU / floating-coverage candidate on
  the saved failing flight frame; diagnostic only, not broad acceptance.

Final job 6996439 has passed 67 CPU tests and 3 CUDA tests with zero skips.
The completed 204-frame CUDA public gate is independently verified under
`completed-public-cuda/`. Both online reports were observed passing (64 cases /
224 frames each), but their raw artifacts and the final CPU/scheduler evidence
still need retrieval after SSH authentication resumes. See
`online-metadata-observation.json`; it is metadata, not a substitute for raw
verification. `local-checks/` preserves the 57-test shared CPU closure and its
earlier missing-dependency retry. No promotion or throughput claim.
