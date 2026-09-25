# Oscar measurement provenance

All GPU work ran on Oscar RTX 3090 nodes. The source bundles contain the benchmark
and pinned oracle source, with no production changes. `SHA256SUMS` is the original
**remote upload bundle manifest**, including the source/oracle tarballs; it is
not a local directory seal. `cold-bundle-SHA256SUMS` has the same role for the cold
supplement. `EVIDENCE-SHA256SUMS` seals the locally retained submission artifacts.

| Job | Outcome | Use |
|---|---|---|
| 6693308 | FAILED, 1:0, 00:12:00 | Keep 21 correct normal-profile records; preserve 18 affected records |
| 6694195 | COMPLETED, 0:0, 01:01:56 | Corrected isolated-profile retry of those 18 registrations only |
| 6694865 | COMPLETED, 0:0, 00:06:24 | Eight fresh GPU Craftax processes; `afterany:6694195` |

The original dispatcher inherited normal-profile site-packages into child family
profiles. The retry launches the dispatcher with a clean base PYTHONPATH and the
already accepted per-family environments. Its exact script is retained. No
duplicate active runs were submitted. No task jobs remain active.

Each downloaded result directory preserves its remote `RESULT-SHA256SUMS`; the
final closure verifies those against the downloaded files. Local `SHA256SUMS`
files seal the full downloaded directories. Raw `sacct` terminal output and the
machine-readable `terminal-summary.json` are retained here. CPU cold reproduction
is also preserved; it ran separately from CPU benchmark and native-build work.
