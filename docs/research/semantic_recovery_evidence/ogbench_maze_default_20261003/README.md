# Maze automatic process route accepted

Job 6966933 passes both CPU/software-GL and CUDA/EGL gates independently on an
RTX 3090 host. Each gate uses the UNPATCHED public constructor at B64, verifies
eight actual process owners, and passes all 22 cases with exact State, RNG,
task and frame outputs. Current sparse termination padding is exercised: lane
zero stops after its first action. Rendering holds owned previous States while
the following transition is submitted. NumPy float32/64 and device-resident
Torch actions, variation, reset continuation, retained outputs and last-shard
restore rejection are covered. The 20 routing tests pass locally and on Oscar,
with no skipped tests. The integrated 141-file snapshot matches the frozen gate.

The public point/large/navigate registration now selects at most eight persistent
processes on Linux for B>=64 with at least four available CPUs. Small batches,
other platforms, daemon processes and constrained CPU allocations remain local.
No other Maze dataset class receives this simulator route. The API is unchanged;
use the same one-action or multi-action step call and slice selectors.

The prior bounded process sweep is the performance basis: P8/local 4.94–7.10x
for State-retaining public steps at B64/256, T1/8. This gate establishes current
integration and default selection, not another throughput claim. Physics is CPU
for either output device. Larger sweeps, component/source baselines and combined
pipeline throughput remain required. Semantic status stays 39 VERIFIED / 3
INCOMPLETE; Cube/Scene process routing has separate prerequisites.

Reconstruct the source from source_manifest.json's base plus source_overrides.json;
check every listed hash. Reproduce gates.csv with summarize_ogbench_maze_process.py
results NEW_OUTPUT --source FROZEN_CHECKOUT --production. Accepted prior evidence
is preserved unchanged.
