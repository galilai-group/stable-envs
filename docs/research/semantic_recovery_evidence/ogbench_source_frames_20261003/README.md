# OGBench source frame transport — CPU prerequisites complete, image gate pending

This checkpoint adds selected-frame transport to the benchmark-only Maze/Cube/Scene
source owners, based on `eaafb3ff0af8bf7e055e65966d877fb9c9d33c57`. Production
adapters, process routing, renderer fidelity and the public API are unchanged.
The upstream pin remains `176d54532b3c0a251a4f509b74f2fdacc143a63d`.

Source owners render only requested transitions, hold a lane's terminal image
within a chunk, preserve requested ordering, and return owned uint8 arrays plus
frame validity. They support disabled and empty selections. Frame columns use
the existing Gym shared-memory transport; source snapshot/restore RPCs remain
untimed, pickled preparation. Benchmark teardown explicitly closes the dedicated
source renderer that upstream `close()` leaves untouched. Sentinel tests cover
this call path; actual GL resource release and memory effects are **not proven**.

Source dimensions remain unchanged by default: Maze/Cube 224x224, Scene 200x200.
The prepared image gate explicitly requests 64x64 for both implementations. No
image resizing, recoloring or output casting is used to reduce error.

## Executed locally

- `nonterminal.json`: 12 passing real CPU source B2/P2 cases, all three families,
  T1/T8, float32/float64. Source tasks/full snapshots, replay, retained outputs,
  RNG and clean worker exits pass. The report records original image defaults.
- `empty-frames.json`: the same 12 cases pass with actual zero-length shared frame
  arrays and explicit 64x64 source configuration, without initializing GL.
- `terminal.json`: 24 passing real CPU terminal cases against the accepted adapters
  and shared dispatcher, with one/both lanes at their goal. Full source snapshots,
  State history, task padding, lifecycle and output ownership pass. These use a
  render-free fixture, not public environment construction.
- `frozen.xml` / `frozen.log`: 67 synthetic helper/report tests pass from the frozen
  candidate. These test frame selection/ordering, held terminal frames, borrowed
  buffer ownership, rejected shapes/dtypes, explicit teardown, public T1 axes,
  strict source-image equality, MSE rejection, and coordinator argument checks;
  they include the 15 existing DMC report/coordinator regressions.

Earlier synthetic logs remain: the first 43-check run passed; v2 has 47 passes
and one failed assertion because scalar `1/255**2` differed by one floating-point
rounding unit from the independently computed array MSE. The formula sanity check
now uses a relative tolerance of 1e-15. Source equality and the renderer's 0.001
acceptance threshold are unchanged. v3 passes 48, v4 passes 67 after adding the
coordinator checks and existing DMC regression suite. No local GL/CUDA result is
claimed. Ruff and shell syntax checks pass.

## Prepared Oscar gate — not uploaded or submitted

`ogbench_source_frame_gate.py` requires a SLURM Linux allocation and `--verify-gl`.
For each registration/device it checks B2/P2, T1/T8, no/empty/final/all/reverse-sparse
frames, and zero/one/both goal-reaching lanes: **30 planned cases per device**.
Each case includes a second call to check retained outputs and post-terminal
source lifecycle. The source worker images must equal serial source exactly;
public image MSE is normalized by 255 squared and must be <=0.001 on every frame.
The gate checks actual source GL identities (software on CPU, NVIDIA on GPU),
CUDA output placement, source snapshots, an independent engine/RNG witness, full
owned public State/history, selected-frame validity, and clean worker exits.

The three-task launcher uses RTX 3090, eight physical CPU cores and 64 GiB per
registration. The existing coordinator creates separate CPU/software-GL and
CUDA/EGL interpreters. This is a correctness gate, **not a timing comparison or
P8 production/scaling acceptance**. No FPS or speedup may be published from it.
Full source timing sweeps and resource-lifetime measurements follow actual gates.

Rebuild without overwriting existing artifacts:

```bash
python3 docs/research/semantic_recovery_evidence/ogbench_source_frames_20261003/prepare.py \
  --checkout . --source /path/to/pinned/stable-worldmodel \
  --output /tmp/new-ogbench-source-frame-bundle
```

The preparer verifies all candidate/source file hashes. `deployment.json` records
the original archive digest and pending status. The fixed remote destination is
`/oscar/scratch/sboughan/se-semantic-recovery/ogbench-source-frames-v1-20261003`.
The `se7-ogb` environment is used without installing or modifying dependencies.

Oscar still denied SSH authentication at this checkpoint. Existing submitted
jobs were not restarted or inferred terminal. No accepted evidence was overwritten,
no candidate optimization was promoted, and the matrix remains 39 VERIFIED / 3
INCOMPLETE (Craftax deferred; PushT and Rocket excluded from this work).
