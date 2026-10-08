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

## Adapter refactor

The adapter refactor has passed its CPU/CUDA source and process checks.
Decisions and validation are in [Adapter cleanup](adapter-cleanup.md).
The directory's `README.md` describes the resulting interfaces.

Family directories now own simulator logic, configuration and State conversion.
Shared execution code selects processes and handles requests and typed buffers.
`Env` continues to own action sequences, capture slices and rendering.

The refactor preserves OGBench's prepare/commit restoration and family-specific
space behavior. Fetch retains its distinct flat buffer layout. Worker loops
remain explicit because their reset, task and restore protocols differ.

Acceptance requires unchanged State/RNG/task/lane behavior, owned snapshots,
failure cleanup, source contracts, CPU/CUDA checks, and same-host simulation and
step-plus-render comparisons at horizons 1 and 8. The advisor renderer table is
a separate saved-state measurement; rollout time must not enter that timer.

## Remaining layout work, in order

1. Move `render/trackcom_texture.py` and `trackcom_reflection.py` beside DMC,
   their only family consumer.
2. Group rendering engines with their helpers, for example MJWarp with its light
   specialization and gfxdraw with its classic/fused implementations. Avoid
   generic `utils` or `misc` directories and unnecessary compatibility wrappers.
3. Move root `_data.py` corpus utilities under `testing/`. Keep shared public
   types, registry and immediate implementation at the package root.
4. Revisit the six obsolete Craftax tests against the current State/render
   boundary. Do not enable excluded Craftax acceptance or performance work as
   part of this housekeeping.
6. Move `render/_raycast_reference.py` to `tests/helpers/`: only tests import it.
   Fold the flat `envs/_classic_semantics.py`, `_fetch_semantics.py` and
   `_rocket_raster.py` into per-family packages with their environments, as
   Craftax, DMC and OGBench already are.

A leading underscore means internal implementation. It is not a work-in-progress
marker. Directory placement should follow ownership; file size alone does not
justify a split. The documented root API remains the user-facing entry point.

For each move, update maintained imports and documentation, compare registration
and dependency behavior, and run the relevant existing tests. If executable
rendering or simulation behavior changes, run the applicable CPU/GPU source gates
as well. Preserve old measurements as immutable records of the versions measured.
