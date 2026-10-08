# Collection checkpoint

Candidate: `65ec1f387ba30c79929382d2f7181097ef11a5ef` (`sami-dev`).
Source: `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
No production renderer or adapter was changed for this collection.

All requested cells are scheduled. Jobs use RTX 3090 allocations on
`3090-gcondo`, eight CPU cores and 64 GiB host RAM. Temporary environments,
source traces and compilation caches live on node-local storage. Compact
results live in `/users/sboughan/se-presentation-20261008` on Oscar.

| Array | Purpose |
| --- | --- |
| 7122480, 7122737 | B1024/B2048 saved-state rendering, CPU and GPU |
| 7122917, 7123113, 7123155, 7123308 | Forward step/online matrix, H1/H8, all five batches |
| 7123233 | Rocket retry with valid ignition actions |
| 7123441 | Fetch CPU retry with explicit OpenGL constructor backend |
| 7124020 | Native large-batch H8 source transport memory retries |
| 7124430, 7125600 | Full distributions for earlier pixel failures; source GL context check |
| 7123059, 7123084, 7123270 | CUDA kernel and CPU component diagnostics |
| 7124304 | Successful cross-API GPU counter capability probe |
| 7124828, 7125430 | Matched source/ours Nsight Systems GPU counters |

`measurement-tools.tar.gz` preserves campaign scripts, exact online runner
versions, plans, pins, and the staged historical rendering harness. This is
an evidence bundle, not a new maintained benchmark framework. Reproduction
also needs the saved-state corpora at the paths recorded in `plan.json` and
the pinned source bundle. The candidate archive can be regenerated with
`git archive` at the full SHA above. Campaign scripts contain Oscar paths;
they are not a portable install interface.

The local working directory is `/private/tmp/se-presentation-20261008`.
After collecting new archives, run `export.py`, `export_usage.py`,
`summarize.py`, `slow_plan.py`, and `validate_tables.py` in that order.
Refresh `scheduler.tsv` from Slurm before export to distinguish confirmed
host OOM from unexplained process exits. Per-run hashes identify the exact
runner version. Earlier attempts remain preserved even when a retry replaces
that cell in the presentation CSV.

## Benchmark corrections and unresolved cells

- Rocket's original random action generator violated the source ignition bound.
  The retry uses the maintained powered-flight action, without changing physics.
- Fetch CPU requires the public OpenGL backend explicitly. The retry uses that
  constructor option; it does not compare a CUDA-only backend on CPU.
- Source vector observation descriptors initially allocated large redundant
  bound arrays. Lightweight shape/dtype descriptors use Gymnasium's existing
  shared-array transport. B64/H8 source outputs were byte-identical in the
  local before/after check. Large-batch retries independently test that route.
- Pixel failures now retain full timing and MSE distributions with
  `MEASURED_PIXEL_GATE_FAILED`; the threshold remains 0.001. Earlier aborted
  attempts are retried only for their rendered cases.
- Cube/Scene CPU B64 failures contained corrupted source images. A focused
  retry makes each source renderer's GL context current before its environment
  reset, where upstream may destroy/recreate render resources. This is a
  benchmark context-ownership check, not a production parity fix. Its result
  must be verified before attributing those errors to candidate rendering.
- PickAndPlace rendered-online source continuation remains unsupported.
  Saved-state rendering and step-only measurements are separate workloads.
- Saved Rocket replay on GPU and the new large-batch CPU runs and Manipulator CPU source repeat stability remain
  failed gates. Their errors are not replaced with zero FPS.
