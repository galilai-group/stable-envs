# OGBench online source prerequisite and cleanup audit

At integration base `4dd9af4f6a24eb65c34f7c20b15dc7c8c328f353`, all three
registrations pass a new bounded CPU source-call/restore prerequisite. This is
not an online speedup, GPU acceptance, renderer parity or process-transport gate.
Production code, process routing and the semantic matrix are unchanged.

## Authoritative call paths

The registration entries in pinned stable-worldmodel
`176d54532b3c0a251a4f509b74f2fdacc143a63d` point directly to these classes,
without registration-specific kwargs or a registered TimeLimit. The seven source
files are hashed in `source_manifest.json`. Imported source/dependency/helper
paths and their hashes are recorded in `local-v2.json`.

| Registration | Source implementation | Stepping | Rendering |
| --- | --- | --- | --- |
| `swm/OGBMaze-v0` | `stable_worldmodel.envs.ogbench.maze_env.MazeEnv`, point/large/states defaults | Scalar OGBench point-maze CPU physics; wrapper adds `env_name`; legacy task RNG is isolated per lane by the accepted oracle | `ogbench.locomaze.maze` custom MuJoCo renderer, lazily constructed |
| `swm/OGBCube-v0` | SWM `cube_env.CubeEnv`, single cube/states defaults | Inherited OGBench `ManipSpaceEnv.step`, CPU controller/IK/physics; SWM task/reset/variation overrides | SWM camera default `front_pixels`, then upstream lazy MuJoCo renderer |
| `swm/OGBScene-v0` | SWM `scene_env.SceneEnv`, scene/states defaults | Inherited OGBench `ManipSpaceEnv.step`, CPU controller/IK/physics; SWM task/reset/variation overrides | SWM camera default `front_pixels`, then upstream lazy MuJoCo renderer |

The installed upstream is [OGBench](https://github.com/seohongpark/ogbench)
**1.2.1**, with MuJoCo 3.12.0, dm-control 1.0.45, Gymnasium 1.3.0, NumPy 2.5.3,
Torch 2.14.0 and TensorDict 0.14.2. The package version is verified; no particular
upstream Git commit is inferred from the installed distribution. Source physics
has no native batch or CUDA step interface in these paths. GPU rendering is
available through hardware OpenGL, but has not been exercised in this prerequisite.

Cube/Scene source seeding requires the previously accepted `OwnedCube` /
`OwnedScene` entropy witness from `ogb_rng_feasibility/ownership.py`. It owns
the action Box, variation-parent and task RNG streams while delegating simulation
and source algorithms. It also forwards Cube's otherwise discarded reset seed
through Gym's seed path. The future source comparison must identify this boundary;
raw wrappers with equal seed arguments alone are not a matched-trajectory oracle.

The source step produces its ordinary observation/info payload even where the
accepted library manipulation API returns only reward/success/termination fields.
The prerequisite calls the actual source step and only then projects the accepted
task fields. A future timed comparison must retain that computation and document
the output contracts. No full oracle State export belongs inside source timing.

## Executed CPU evidence

`local-v2.json` and `local-v2.log` preserve 12 independently checked cases:
three registrations × T1/T8 × float32/float64, each at B2. Each case compares
source and adapter tasks exactly, checks the independent engine/RNG witness after
every step, restores both snapshots and repeats the same actions. Source calls
preserve the caller's global NumPy RNG. Complete candidate State agrees after
restore and at replay endpoints. The selected bounded trajectories are explicitly
required to remain nonterminal; no terminal or early-stop coverage is inferred.

Maze uses its accepted independent complete-State oracle. Cube/Scene use the
existing source engine/cache/model/Python/RNG restore helper plus the accepted
`assert_oracle` witness; candidate State also includes compatibility metadata
not present in the source. The source restore check is scoped to an existing
matching model/reset configuration, not arbitrary geometry-changing checkpoints.
The tested source/candidate hashes, versions and checker hashes are in the report.

MuJoCo renderer construction is guarded with an assertion for the entire run.
No public Env renderer, images, GPU, timing, shared-memory source workers or
large-batch memory measurement is exercised. The initial `local-v1.json` is also
preserved, but `local-v2.json` is authoritative for the current script: V2 adds
the explicit nonterminal check, delegated dependency provenance and cleanup probe.

Reproduce without changing accepted evidence:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/samib/se7-ogb-venv/bin/python \
  scripts/benchmarks/online_stepping/ogbench_source_prerequisite.py \
  --source-root /Users/samib/stable-worldmodel \
  --source-manifest docs/research/semantic_recovery_evidence/ogbench_source_prerequisite_20261003/source_manifest.json \
  --output /tmp/new-ogbench-source-prerequisite.json
```

The output path must be new. Use the recorded dependency versions on another host.

## Cleanup finding and measurement implications

A sentinel placed in each source's dedicated renderer field receives **zero
`close()` calls** when the environment is closed, and the field remains set:

- Cube/Scene resolve `close` to `gymnasium.core.Env.close`, which is a no-op.
- Maze forwards to `MujocoEnv.close`, which does not close its separate
  `custom_renderer` field.

This is a structural cleanup observation, with no GL resource allocated. It
does **not** prove a leak, allocator retention, or the cause of the previously
measured 41–43 GiB production PSS after restores. Production manipulation physics
owners need not instantiate these source renderers. Do not attribute that memory
growth to this finding without an allocation/lifetime measurement.

For the forthcoming online source benchmark, calling source `close()` alone must
not be treated as proof that an oracle's GL contexts have been released. Keep
oracle preparation in a separate process that exits before the timed population,
or establish explicit resource release and reference lifetimes with real GL
measurements. Then independently verify CPU/software GL, GPU/NVIDIA GL, same-state
images, active-lane stopping and output ownership before publishing source FPS.
Use persistent source processes for scalar lanes with an equal CPU budget, and
measure memory before expanding batch size. Existing Oscar array 6970790 and all
accepted renderer evidence remain untouched; its pending validation candidate is
not promoted by this CPU prerequisite.
