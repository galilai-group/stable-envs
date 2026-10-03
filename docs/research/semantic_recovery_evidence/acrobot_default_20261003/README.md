# Acrobot process default acceptance

Job **6968347**, RTX 3090 host gpu2110, completed with exit 0:0 in 1:36.
The frozen candidate passes **132 tests without skips** and independent CPU and
CUDA public-constructor gates (25 cases each, batch 64, eight process owners).
All 175 frozen files match the integration implementation after applying the
isolated constructor activation. Reproduce the evidence check with:

```sh
python scripts/benchmarks/online_stepping/verify_acrobot_default.py \
  docs/research/semantic_recovery_evidence/acrobot_default_20261003 --source .
```

The selected route is Acrobot only, Linux, batch >=64, at least four affinity
CPUs, and a non-daemon parent. It uses at most eight persistent CPU owners.
Smaller batches, other platforms and the other four classic domains retain the
local implementation. CPU and CUDA here select output/rendering devices;
Acrobot physics remains on CPU in both cases. No public execution knob is added.

The gate covers integer action dtypes, one/eight actions, scalar discrete
batches, independent State/frame selections, reverse/empty slices, mixed early
termination and time limits, cached RNG words, unseeded resets, physical
variation, device actions, exact current-State rendering and owned snapshots
while worker buffers change. Invalid last-shard restore leaves all owners usable
and unchanged. Overlap occurs in 20 checks per device. Native 500x500 frames are
exactly equal to the accepted local path (MSE zero); this is not a fresh upstream
renderer-parity measurement. All five classic domains also have routing and
semantic regression coverage in the 132 tests.

Performance justification is the separately accepted
[process sweep](../acrobot_process_20261003/): P8/local 2.25–6.71x across
B64/256/1024 and T1/8 with all States retained and frames disabled. At B1024/T8,
CPU output is 61,675 +/- 644 steps/s versus 9,197 +/- 145; CUDA output is
57,960 +/- 1,481 versus 9,352 +/- 30. These are adapter comparisons, not source
speedups or online frame rates. Pipeline/source throughput remains outstanding.

`run.sh`, the frozen manifest/overrides, raw report, tests, terminal record and
log preserve the candidate and outcome. `local-tests.xml` records 125 local
passes and seven CUDA-reserved skips before the full cluster gate.
