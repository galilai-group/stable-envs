# Rocket final candidate closure evidence

This directory is separate from the earlier sealed Rocket investigation.
The registration remains INCOMPLETE until the final production gate is verified.

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
Full source/renderer gates remain pending at this commit. No throughput claim.
