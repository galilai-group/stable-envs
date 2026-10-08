# Package cleanup checkpoint

Branch: `sami-dev`. The Craftax/PRNG cleanup is committed and pushed at
`334bcc99cee4f8c4bbbad1e793edf874a6f65e57`. This note also records the subsequent
adapter typing and navigation work.

## Completed: Craftax and PRNG ownership

Only the Classic and full Craftax renderers used `render/prng.py`. It implements
JAX-compatible Threefry in Torch; it does not require a general JAX utility
package. The implementation now lives with its callers:

```text
stable_envs/envs/craftax/
    __init__.py
    classic.py
    full.py
    semantics.py
    prng.py
    _textures.py
```

The PRNG introduction is six lines. Function docstrings describe inputs,
outputs and necessary numeric constraints. Historical performance claims and
experiment narration were removed. The retained constraints are partitionable
Threefry, signed storage of unsigned bits, exact key encoding and float32
rounding during night-noise scaling. Incorrect annotations for the counter
return value and uniform shape argument were corrected.

Imports, reference-generation scripts and API documentation use the new paths.
Public registry names are unchanged. The family package now handles its imports,
removing the Craftax exception from shared discovery. Dependency failures remain
isolated and retain their existing diagnostic keys. Frozen evidence was not edited.

Validation at this checkpoint:

- All five moved modules have the same executable AST after excluding imports,
  docstrings and annotations; renderer arithmetic is unchanged.
- All 98 registry names available in the local installation and its import-error
  diagnostics match the starting checkout. This is not an acceptance count.
- The targeted Craftax/import/resource suite reports 206 passed, two skipped and
  six failures. All six failures reproduce on untouched commit `92446d3b`.
- The failures are three tests, each for Classic and full Craftax, that expect
  renderer-only cells to expose simulator State. They need a separate correction
  to the current contract; they were not hidden or converted into passes here.
- The PRNG/repository follow-up reports 24 passed and three skipped. Ruff and the
  strict documentation build pass. GPU kernels were unchanged and were not rerun.

## Adapter typing and navigation

`SimulatorAdapter` is the internal lifecycle/State protocol, not a renderer or
a mandatory parent class. It describes reset, one-step simulation, get/set State
and cleanup. `Env` handles action sequences and rendering above that interface.

The protocol now documents these responsibilities and the action type. Execution
factories have typed arguments and returns. Generic factories preserve the local
family type while allowing a process implementation; they do not erase family
methods behind a broad protocol annotation. Fetch's lifecycle, State and step
methods provide a concrete annotated example. Worker owners keep their concrete
types where family-specific operations are required.

The directory's `README.md` maps the existing files and gives a reading order.
This pass changes annotations and documentation, not process selection, buffer
ownership, simulation or rendering. Type coverage is incremental; the entire
adapter tree is not yet fully annotated or statically checked.

Validation for the typing/navigation pass:

- Execution-selection and import checks: 291 passed. The Fetch process-test
  module skipped because `gymnasium_robotics` is absent from the local profile.
- Acrobot process and repository checks: 16 passed; two CUDA cases require Oscar
  and two repository checks require PyYAML, so those four cases skipped.
- All 76 existing function/method bodies in the edited adapter implementation
  files are unchanged after removing annotations and docstrings.
- Ruff and the strict documentation build pass. This was not a GPU or performance
  acceptance run; no claim is made for the skipped cases.

## Adapter refactor proposal

Status: **proposed, not implemented**. Recorded after `6383c230`. This expands
the earlier folder-grouping plan; only the typing/navigation work above is done.
It is separate from the historical-report and fixture cleanup.

The objective is to make a family straightforward to implement and verify while
reducing duplicated execution code. Directory organization follows those boundaries.

| Observed problem | Proposed direction |
|---|---|
| `SimulatorAdapter` covers lifecycle, while stepping discovers optional methods dynamically. | Define small explicit interfaces for ordinary simulation and optional asynchronous stepping. Keep family-specific capabilities outside the common protocol. |
| Five process adapters repeat worker startup, buffer allocation, State assembly and result retrieval. | Extract only mechanics shown to be common across multiple families. Keep simulator semantics in family implementations. |
| Five routes across four `*_execution.py` files repeat the same process-selection conditions. | Extract the common selection policy before the pilot, preserving family eligibility, Linux/daemon checks, affinity handling, batch/CPU thresholds and the worker cap. Verify all existing routing cases. |
| DMC repeats domain knowledge in `PROCESS_DOMAINS`, worker `make_local`, and the Reacher/Humanoid constructor special case. | Use one family factory for local and worker construction with a consistent call signature. Keep process eligibility explicit and preserve domain-specific defaults and seeding. |
| Fetch environment code reads `_simulator._mass` directly. | Define a family interface for configuration and metadata instead of exposing internal storage. |
| `ClassicControlAdapter` lives under `envs/`, while other simulator owners live under `_adapters/`. | Establish consistent ownership before moving files. |
| `pusht_projection.py`, `pusht_draw.py` and `craftax_render.py` contain family rendering code under `_adapters/`. | Assign rendering to the owning family alongside its render implementation; retain explicit dependencies on simulator State decoding where required. |
| Public spaces, `pusht_spaces.py` and `rocket_spaces.py` have overlapping types with different behavior. | Audit ownership and contracts before deduplicating. Preserve source-specific scalar precision, sampling order, constraints and default/current-value semantics. Keep attribution for derived code. |
| Type hints are present but not systematically checked. | Add a focused static-checking target for adapter interfaces and their callers. Preserve concrete types where family operations require them. |

Responsibilities:

- Family simulation code owns upstream calls, task behavior, RNG and exact State conversion.
- Execution code owns workers, lane partitioning, communication and shared buffers.
- `Env` owns public action shapes, sequences, State/frame selection and rendering.

First consolidate the repeated execution-selection policy as a separate change
with unchanged routing behavior. Then start the interface pilot with Fetch and
check it against DMC and OGBench before extracting worker/buffer machinery.
OGBench's staged restoration must remain explicit:
replacement worlds are prepared before committing the restore. A generic broadcast
of `set_state` is not an equivalent implementation.

Acceptance requirements:

- Preserve public environment names, action shapes and State/frame slice semantics.
- Preserve State, RNG, rewards, termination and inactive-lane behavior.
- Returned snapshots remain independent while workers reuse shared buffers.
- Preserve invalid-restore handling and cleanup after partial dispatch or worker failure.
- Add no unnecessary State captures, full-State copies or synchronization to the stepping loop.
- Compare simulation and step-plus-render throughput at representative batches with
  action horizons 1 and 8, on the same hardware and workload as the baseline.
- Run existing family/source contracts as applicable; document skips separately.
- Keep the implementation small. Do not introduce a universal simulator base class,
  generic framework or compatibility wrappers solely to support the reorganization.

Still undecided: exact optional interface signatures, which buffer/worker functions
can be shared, final package paths, and numeric performance acceptance thresholds.
Resolve these against the pilot and its baseline measurements before expanding the
refactor. The ownership inventory includes classic control and Craftax, even though
Craftax remains outside simulation/performance work. Any Craftax relocation is a
separate mechanical change, not authorization to resume its recovery or tuning.

## Remaining layout work, in order

1. Consolidate execution selection, carry out the adapter pilot above, then group
   simulation code by family: classic control, Fetch, DMC, OGBench, PushT and Rocket.
   Account for Craftax in the ownership inventory with the limited scope above.
   Leave shared process transport separate. Keep mechanical moves distinct from
   behavior changes.
2. Move `render/trackcom_texture.py` and `trackcom_reflection.py` beside DMC,
   their only family consumer.
3. Group rendering engines with their helpers, for example MJWarp with its light
   specialization and gfxdraw with its classic/fused implementations. Avoid
   generic `utils` or `misc` directories and unnecessary compatibility wrappers.
4. Move root `_data.py` corpus utilities under `testing/`. Keep shared public
   types, registry and immediate implementation at the package root.
5. Revisit the six obsolete Craftax tests against the current State/render
   boundary. Do not enable excluded Craftax acceptance or performance work as
   part of this housekeeping.

A leading underscore means internal implementation. It is not a work-in-progress
marker. Directory placement should follow ownership; file size alone does not
justify a split. The documented root API remains the user-facing entry point.

For each move, update maintained imports and documentation, compare registration
and dependency behavior, and run the relevant existing tests. If executable
rendering or simulation behavior changes, run the applicable CPU/GPU source gates
as well. Preserve old measurements as immutable records of the versions measured.
