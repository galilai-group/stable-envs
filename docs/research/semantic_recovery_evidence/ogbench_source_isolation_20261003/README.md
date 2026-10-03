# OGBench source image ownership — CPU checks complete, GL pending

Base `49309c1c3d810316d308b28de18680120d7bc353`. The prepared image gate now
places both serial (P1) and parallel (P2) source renderers in child processes.
A separate source physics witness in the parent never renders. The public
library renderer remains in the parent. This is benchmark isolation, not a
production renderer, stepping, State or execution-route change.

The previous unsubmitted gate created and closed serial source GL contexts in
the same process as the library renderer. `manipulation_opengl.py` explicitly
retains its contexts for process lifetime because releasing one can invalidate
other live contexts. Separating source ownership removes that conflict by
construction. We have not reproduced a GPU failure or demonstrated its absence
locally; actual source/public GL acceptance still requires Oscar.

The new gate audits the complete source snapshot before and after every source
render. It also compares both rendering source populations' full snapshots and
task sequences to the independent no-render physics witness after a chunk and
a subsequent call. A source renderer that changes continuation or RNG fails the
gate. The independent Cube/Scene engine/RNG witness and Maze full-State witness,
goal-reaching fixtures, held terminal States/frames, ownership and per-frame
normalized MSE <= 0.001 checks remain. Source image identity is read inside its
owner process. P1 here is a correctness oracle over IPC, not a native serial
throughput baseline.

The Oscar-only gate remains B2, T1/T8, five frame selections, zero/one/two reached
lanes: 30 cases per family/device, with two source pools per selection/horizon.
It now checks 20 pool closures per family/device. Each parent physics population
must retain no initialized source renderer. No timing or large-batch P8
acceptance is claimed.

## Executed evidence

- `p1-empty`: 12 real CPU cases for Maze/Cube/Scene, T1/T8, float32/64, B2 in one
  source owner. Empty shared frame columns, source snapshots/tasks, restoration,
  replay, RNG isolation and owned outputs pass. Six workers close cleanly.
- `p2-empty`: the corresponding 12 cases with two owners; twelve workers close.
- `terminal`: 24 real CPU source/P2/adapter-dispatcher cases, one/two reached
  lanes, T1/T8, float32/64. Terminal masks, zero padding, held complete State,
  subsequent-call continuation, source engine/RNG witness and output ownership
  pass. Twelve workers close. This uses real adapters through the unchanged
  dispatcher and does not claim actual public renderer construction.
- `synthetic` and `frozen-bundle`: 58 tests pass, including six new positive/
  negative render-audit cases across the three families. A deliberately changed
  continuation field is rejected. Synthetic pixels do not establish source GL
  fidelity. The frozen run executes from the actual upload code directory.

No local GL or CUDA rendering was run. Local dependencies are recorded per
report. The reports' candidate hashes match this checkpoint. Source snapshots
remain untimed pickled preparation; only tasks/images use shared transport.

## Deployment and remaining work

`candidate_manifest.json` freezes 258 files and `source_manifest.json` pins the
seven actual source dependencies. `prepare.py` reproduces the bundle from matching
candidate/source files and refuses an existing destination:

```bash
python3 docs/research/semantic_recovery_evidence/ogbench_source_isolation_20261003/prepare.py \
  --checkout . --source /path/to/pinned/stable-worldmodel --output /tmp/new-ogbench-source-gate
```

Local bundle: `/private/tmp/ogbench-source-isolation-bundle-v1-20261003`.
`run.sh` uses three eight-core RTX 3090 tasks, one each for Maze/Cube/Scene, with
fresh software-GL CPU and NVIDIA-GL CUDA interpreters. It supersedes the earlier
**unsubmitted** `ogbench-source-frames-v1-20261003` gate. The previous frozen files
and accepted evidence remain unchanged; do not submit both just to duplicate work.

**NOT_UPLOADED / NOT_SUBMITTED.** Oscar SSH still rejects keyboard-interactive
authentication. Per-registration actual images/nonmutation, public CPU/CUDA gates,
GL resource lifetime and terminal accounting remain unverified. Source throughput
and broader batch/resource sweeps remain open. Previously submitted OGBench
optimization job 6970790 has not been observed or restarted in this checkpoint.
No source speedup, optimization promotion or semantic status change follows.
