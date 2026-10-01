# Fetch observation optimization: accepted

Base integration: `30197dc376d9d5fd706d265ab714e6c1c8d0cbf8`.
The production Fetch adapter now uses cached observation reads. The public API,
State schema, physics/controller/reward/reset implementation, renderer and
existing worker-selection policy are unchanged. No semantic registrations are
promoted by this performance checkpoint.

## Result in the production adapter

Before/after `FetchAdapter.step`, **including full owned State export**. Known
four-action chunks, float32 actions, five repeats with alternating before/after
order. Restoration and exact equality checks are outside timing. Default worker
selection is one thread at batch 1 and four threads at the other batches, all
inside one process. Both implementations use the same CPU allocation on
node1926. Rates are environment transitions/s, mean ± sample SD.

| Batch | Before | After | Speedup |
|---|---:|---:|---:|
| 1 | 447 ± 14 | 554 ± 51 | 1.24× |
| 64 | 1,114 ± 93 | 2,170 ± 158 | 1.95× |
| 256 | 973 ± 32 | 2,472 ± 102 | 2.54× |
| 1,024 | 1,106 ± 30 | 2,268 ± 38 | 2.05× |

This directly measures the production adapter's stepping and State export;
it does not include rendering, GPU transfer, policy inference or the extra
history-stacking work of the public multi-action API. Tiny batch-1 chunks are
particularly susceptible to timing noise. SD is not a confidence interval.

## Change and extensibility

Upstream `generate_mujoco_observations` reads every robot joint position and
velocity, then consumes only the final two scalar joints. The new Fetch-local
mixin resolves that exact model ordering once and reads only those two joints.
It also caches site IDs, reuses per-environment Jacobian scratch buffers and
computes object translation/rotation Jacobians together in one native call.
The same NumPy arithmetic and upstream Euler conversion/output packaging are
retained. Rewards, actions, physics steps and all lifecycle methods remain
inherited from the source classes.

Only topology is cached. `qpos`, `qvel`, positions, orientations, goal and timestep
are read from the live model/data; each simulator owns its scratch storage.
The cache rebuilds if the model object changes, and validates the two scalar
joint layout. The public observation dictionary remains independently owned.
This is a Fetch-local optimization usable under serial/thread/process execution,
not a new cross-family architecture or a new public tuning parameter. New
variants still require source-equivalence evidence before enabling the mixin.
Other families can apply the same principle of caching fixed model structure
without depending on Fetch's fields or layout.

Authoritative arithmetic was traced in Gymnasium Robotics 1.4.2
[`MujocoFetchEnv`](https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/v1.4.2/gymnasium_robotics/envs/fetch/fetch_env.py)
and its `mujoco_utils` helpers. Installed benchmark versions are recorded in
`benchmark/result.json`. Production changes are limited to
`stable_envs/_adapters/fetch_observations.py` and its selection in
`fetch_pickandplace.py`. There is no new runtime dependency or native build.

## Process and hybrid diagnostic

Separate source-task/observation/state-array transport harness, shared buffers,
eight allocated CPU cores on node1955, eight float64 actions per lane. Ten
samples per cell in two reversed configuration-order blocks. Construction,
reset, warmup and exact checks are excluded. Batch 1 is a direct scalar call;
other batches use Gymnasium AsyncVectorEnv. Process/hybrid executors remain
benchmark-only; this checkpoint does not ship a production process executor.
These numbers must not be substituted for the production-adapter table above.

| Batch | Layout | Before | After | Speedup |
|---|---|---:|---:|---:|
| 1 | 1 processes × 1 threads | 646 ± 3 | 849 ± 5 | 1.31× |
| 64 | 4 processes × 2 threads | 3,333 ± 255 | 5,208 ± 425 | 1.56× |
| 64 | 8 processes × 1 threads | 4,385 ± 341 | 5,596 ± 414 | 1.28× |
| 128 | 4 processes × 2 threads | 3,676 ± 377 | 5,264 ± 487 | 1.43× |
| 128 | 8 processes × 1 threads | 4,324 ± 120 | 5,726 ± 244 | 1.32× |
| 256 | 4 processes × 2 threads | 4,299 ± 172 | 5,165 ± 423 | 1.20× |
| 256 | 8 processes × 1 threads | 4,507 ± 240 | 5,978 ± 290 | 1.33× |
| 512 | 4 processes × 2 threads | 4,253 ± 218 | 5,220 ± 309 | 1.23× |
| 512 | 8 processes × 1 threads | 4,715 ± 200 | 6,041 ± 280 | 1.28× |
| 1,024 | 4 processes × 2 threads | 4,219 ± 130 | 5,734 ± 208 | 1.36× |
| 1,024 | 8 processes × 1 threads | 4,667 ± 168 | 5,952 ± 181 | 1.28× |

Every measured cell improves, including batch 1. Hybrid benefits and approaches
the eight-process result at batch 1,024, but does not outperform it in this
sweep. Results vary with node load and contact-workload mix. No post-change GIL
occupancy measurement is claimed; this experiment measures actual throughput.

## Acceptance and provenance

- **Source gate:** all four Fetch tasks × sparse/dense rewards × float32/float64
  actions, seeds 17 and 42, complete 50-step episodes: 1,600 matched transitions.
  Exact observations, rewards, flags, success, qpos, qvel, warmstart, actuator
  control, mocap state and RNG checks. Repeated observations and returned-array
  ownership/mutation checks pass.
- **Benchmark:** all 220 timed chunks match independent serial source outputs.
- **Public CPU and CUDA gates:** all 12 Fetch registrations independently pass
  at batch 64. Full State equality, threaded versus serial replay, owned
  histories, and different per-lane episode endpoints are checked against the
  immutable pre-change production adapter.
- **CUDA frame gate:** FetchPush all-frames output equals repeated stepping,
  MSE 0. Final-frame-only selection exports one snapshot and matches the last
  reference frame. This tests online API/render integration; it is not a new
  all-registration source-renderer MSE survey.
- **Jobs:** CPU benchmark `6926043` (node1955), CPU API gate/adapter timing
  `6926106` (node1926), CPU/CUDA gate `6926105` (NVIDIA L40S on gpu3101).
  All completed with exit 0 and their explicit PASS markers. The original
  pending 3090 gate `6926074` was cancelled after the L40S gate passed; its
  resource request and cancellation are preserved. GPU resource widening is
  recorded in `acceptance/gpu-resource-update.txt`.

[Comparison CSV](comparison.csv) separates the two measurement scopes. Full
samples, scripts, baseline source, candidate source, manifests and terminal
logs are retained under `benchmark/` and `acceptance/`; `collect.py` derives the
CSV. Do not overwrite these accepted outputs. To repeat the standalone task
benchmark, copy the contents of `benchmark/` into a new directory, then run
`python fetch_observation_bench.py --output new-result.json` on eight allocated
cores with the recorded dependencies. The sibling `fetch_observations.py` is
an exact copy of the production helper tested. The adapter comparison additionally
requires the candidate package and preserved `baseline_fetch.py`, as staged by
the acceptance job.
