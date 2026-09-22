# First bounded family recovery cohort

The approved Phase 0.5/Reacher checkpoint `5fbfe4f` was fast-forwarded into
`v0-draft` before family work began. This cohort uses that contract unchanged.
Production adapters do not import stable-worldmodel. Physics is delegated to
independent upstream CPU instances; CUDA selects State and renderer placement.

| Historical registration | Verified implementation | Scope |
| --- | --- | --- |
| `swm/AcrobotDMControl-v0` | `DMCAcrobotBalance` | Historical sparse default |
| `swm/PendulumDMControl-v0` | `DMCPendulumSwingup` | Historical default |
| `swm/CartpoleDMControl-v0` | `DMCCartpoleSwingup` | Historical sparse swing-up default |
| `swm/OGBMaze-v0` | `PointMazeLargeNavigate` | Point/large maze, state observations, 224 pixels |

Cartpole Balance and PointMazeLargeStitch are not certified by these mappings.
The canonical [recovery matrix](semantic_recovery_matrix.json) retains their
scope limits and the outstanding registrations.

## Evidence

Both family gates use actual SWM source at
`176d54532b3c0a251a4f509b74f2fdacc143a63d`, asserting the commit and clean tracked
source before execution. Installed SWM package-version metadata differs between
profiles; the pinned source checkout on `PYTHONPATH` is the reference authority.

The gates compare every State leaf and all numeric transition outputs exactly
(`atol=rtol=0`). They cover seeds 0/17/42, batches 1/3, reset, independent
snapshots, roundtrip after different histories, one/32-step trajectories,
eight-step continuation replay, positive reward and lifecycle boundaries,
full-State/private-render equality, rendering without mutation, and rejection
of compact/incomplete snapshots. Reference-only immutable semantic corpora
are additive; the existing renderer corpus and Phase 0 audit are unchanged.

DMC covers the actual 502-public-step implicit-reset sequence and nondefault
gravity. Review reproduced a visual-only light override changing exact nominal
gravity through float32 style storage; a family-local correction and exact
eight-step regression now prevent that. OGBench covers successful termination,
post-termination stepping, 1,001 steps without an invented time limit, and
unseeded reset replay with different action-space RNG histories. Its per-lane
legacy NumPy RNG does not alter the caller's global stream.

CPU and Oscar evidence is stored in [semantic_recovery_evidence/](semantic_recovery_evidence/).
Oscar job/source/profile provenance is in `first_family_oscar_runs.json` there.
Frozen renderer checks compare four existing rows per selected class against
`6d80b68b8b3ccd07d4207be94cbf4f0760758a06` in separate baseline/candidate
processes using the same interpreter. These comparisons establish preservation,
not a new all-environment upstream image-similarity claim. The original Reacher
gate is rerun after DMC integration.

Integrated CPU checks passed: 154 DMC migration/contract/spec/oracle tests
(three expected skips for the unrelated CUDA-only Swimmer cell), and 58
OGBench migration/oracle tests (the existing Scene failure below deselected).
The source audit, frozen regression fixtures, shared contract and parity
harness were checked unchanged. Ruff and Git whitespace checks passed.
Formatting applied during integration preserves the Python AST of every
GPU-validated family source/test file; those hashes are recorded with the jobs.

## Remaining scope

The complete historical Variation surface, nondefault tasks/reset options,
remaining registrations, paired image-similarity corpora and benchmarks remain
uncertified. Pure delegated `transition()`/`unroll()` are not implemented.
See [DMC details](dmc_family_recovery.md) and
[Maze details](ogbench_maze_recovery.md) for the proved State fields.

Broader CPU testing exposed an existing OGBench Scene constructor failure:
`SceneEnv.__init__()` requires `env_type`, which the renderer constructor omits.
The same `_ogbench_make("Scene", {})` failure was reproduced on the untouched
renderer baseline under OGBench 1.2.1. The corresponding manipulation taxonomy
test remains failing; it is outside this Maze-only recovery.
