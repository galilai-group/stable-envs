# Online direct-source coverage audit — still incomplete

This is a new evidence reconciliation, not a benchmark rerun. Historical
accepted files are preserved. The generator validates the original five classic
reports against their frozen source, successful terminal accounting, hardware
provenance and strict per-cell collector. The 120 measured rows are byte-identical
to `online_source_audit_20261003/measured_online.csv`.

`coverage.csv` contains all 42 registrations. Among the 35 accepted non-Craftax
registrations:

| Status | Registrations | Qualification |
| --- | ---: | --- |
| Measured, qualified | 5 | Classic; one parent CPU Torch thread, eight-thread rerun uncollected |
| Source campaign prepared | 15 | Fetch Reach/Push/Slide nine, DMC first three, Navigation three; not submitted |
| Source campaign prepared, no frames | 3 | PickAndPlace; source render changes continuation, rendered protocol open |
| Source image gate prepared | 3 | OGBench Maze/Cube/Scene; isolated source contexts, actual GL still untested |
| Source comparison missing | 9 | Remaining DMC registrations including Quadruped; bounded source campaign and renderer-history issue remain |

Four Craftax registrations are deferred; PushT, PushT-Discrete and Rocket remain
independently correctness-blocked. The accepted non-Craftax routes remain 27
process-eligible, three native Torch, and five local. The archived routing files
still match the current production routing hashes; no new runtime routing test
or production route change is claimed here. Semantic matrix: 39 VERIFIED /
3 INCOMPLETE, which is not universal rendering or performance acceptance.

Prepared status now requires the archived campaign checksums and deployment
record to resolve. Prepared artifacts never populate measured FPS or MSE cells.
Fetch timing requires its own independent image gates; the three PickAndPlace
campaigns explicitly cannot produce a rendered comparison. The OGBench gate
supersedes its prior unsubmitted version to isolate source GL and audit rendering
against a source population that never renders.

The completed classic source renders only on CPU. Therefore its source-GPU FPS
columns are blank, while CPU rendering plus CUDA delivery is separately labeled.
No-frame rows likewise have blank FPS/MSE. Current GPU-versus-native-source-GPU
online rows remain zero: no such full campaign has completed and been collected.
This says nothing new about the separate offline renderer-only table or earlier
library-control performance sweeps.

Reproduction requires the original frozen classic candidate tree:

```bash
python3 scripts/benchmarks/online_stepping/audit_online_source_results.py \
  --frozen-classic-code /path/to/classic-parallel-source-v1/code \
  --output /tmp/new-online-source-audit
```

Outstanding cluster job IDs (6970790, 6971892, 6971656) are not confirmed live or
terminal now: Oscar rejects SSH authentication. This audit does not restart them,
infer completion from old status files, or substitute local timing. Collect their
actual accounting/logs, then execute the staged bounded source campaigns before
promoting candidates or extending performance conclusions. The full goal remains
incomplete.
