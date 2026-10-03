# Online execution API catalogue

Current status, 2026-10-02: all 12 Fetch registrations have completed their
process/pipeline acceptance; see [the Fetch checkpoint](fetch_pipeline_checkpoint.md).
The remaining non-Craftax execution work and routing are tracked in
[the stepping backlog](stepping_benchmark_backlog.csv) and
[the DMC audit](dmc_online_execution.md). Persistent pure processes are the
chosen scalar-CPU strategy to evaluate; further hybrid-thread experiments are
not planned. The historical audit and its older updates below remain frozen.

Update, 2026-10-03: eleven DMC registrations, OGBMaze, and classic Acrobot
now also have accepted process defaults. Acrobot's independent CPU/CUDA default
gate passes 132 tests and 25 public cases per device. Navigation retains native
Torch batching; the other four classic registrations retain their optimized local
path. Cube/Scene defaults now also pass all four independent CPU/CUDA gates.
Quadruped retains local execution while its reproduced renderer-history
discrepancy is investigated. These route gates do not complete the required
same-device source/component/pipeline throughput table. See the current backlog
for registration-specific evidence and remaining work.

The current routing tally is **27 eligible process routes, 3 native Torch
routes, and 5 local routes** among the 35 accepted non-Craftax registrations.
The other three non-Craftax registrations remain correctness-blocked. Process
selection requires Linux, B>=64, at least four affinity CPUs and a non-daemon
caller; it uses at most eight owners. Smaller or unsupported configurations
fall back locally. This is routing coverage, not a count of source speedups.

| Current route | Registrations | Scope |
| --- | ---: | --- |
| Persistent processes with local fallback | 12 | Fetch |
| Persistent processes with local fallback | 11 | DMC except Quadruped |
| Persistent processes with local fallback | 3 | OGBMaze, OGBCube, OGBScene |
| Persistent processes with local fallback | 1 | Classic Acrobot |
| Native Torch batching | 3 | TwoRoom, Piecewise, SimplePointMaze |
| Local execution | 5 | Quadruped and the other four classic tasks |
| Correctness-blocked | 3 | PushT, discrete PushT, Rocket |

Craftax's four registrations remain explicitly deferred. The
[current API documentation](../concepts/dynamics.md) explains State/frame slices,
when overlap is possible and how to distinguish steps/s, emitted online FPS and
render-only FPS. The [documentation validation checkpoint](semantic_recovery_evidence/online_api_docs_20261003/)
reconciles all 42 catalogue rows with the semantic matrix and records 251 passing
CPU-only routing selection checks plus executed public API examples. It provides
no new GPU, source-performance or semantic promotion.

Update, 2026-10-01: public `step` accepts `[B,A]` or `[B,T,A]` with independent
`states` and `frames` slices. Fetch State export is column-wise and large batches
use a bounded persistent thread pool. The new [profiling checkpoint](semantic_recovery_evidence/fetch_scaling_slices_20261001/README.md)
measures a substantial gain from upstream process scheduling; this remains an
experimental execution route until exhaustive State/lifecycle acceptance.
See [the API](../concepts/dynamics.md). Other execution strategies below remain
the original audit/proposals.
The original audit and its frozen companion outputs follow unchanged.

Source audit, 2026-09-29, `repair/render-config-fixes` at
`df80c9c54590afa9fc0482a4dcb241d21f6b63a3` plus the existing working-tree changes.
The companion CSV lists all 42 historical registrations individually. The JSON
records source hashes and the dependency sources actually inspected. This is a
capability audit and proposed routing, not a concurrency benchmark or permission
to change accepted semantics. No production API, implementation or acceptance
status changed; Craftax, PushT and Rocket received catalogue entries only.

There are **32 accepted registrations currently looping over scalar CPU
environments**, **4 calling scalar JAX kernels in a Python loop**, and **3 using
native batched Torch dynamics**. The remaining **3 are semantically incomplete**.
An array-shaped public action argument alone does not establish parallel execution.

| Family | Registrations | Current execution | Available route | Adapter responsibility |
|---|---:|---|---|---|
| Classic control | 5 | Loop over scalar Gymnasium environments | Generic Gymnasium process vectorization; CartPole also has a native NumPy vector implementation | Preserve each lane's RNG, TimeLimit, post-terminal behavior, physics Variation and complete State; measure process overhead for these cheap tasks |
| DMC | 12 | Scalar dm_control calls; task-specific action repeats and outputs; full-State export | Parallel independent environment owners; MuJoCo releases the GIL during C calls, so threads are a candidate | Retain dm_control task/wrapper logic; benchmark threads versus processes; own lane ordering, snapshots and resets |
| Fetch | 12 | Scalar Gymnasium Robotics/MuJoCo calls, TimeLimit, then State export | Generic process vectorization or independently owned threaded environments after validation | Keep action dtype/scaling, goals, reward variant, TimeLimit and mass/reset rules; transport State as well as observations |
| OGBMaze | 1 | Scalar OGBench step inside process-global NumPy RNG swap | Persistent processes are the conservative candidate | Isolate global RNG; serialize lanes within each process; preserve goal and integration state |
| OGB Cube/Scene | 2 | Scalar compatible upstream OGBench classes, then State export | Independent CPU owners; processes or validated threads | Own model/data, task state, RNG streams, variations and restore staging; renderer contexts remain a separate ownership concern |
| Navigation | 3 | Batched Torch dynamics on configured CPU/CUDA device | Keep existing tensor batching | Avoid a worker pool for the step itself; reset/restore and SimplePointMaze's Matplotlib goal-cache reconstruction are separate costs |
| Craftax | 4 | Python loop over individually JIT-compiled upstream JAX kernels | `jit(vmap(...))` over explicit State/key/action pytrees; CPU or GPU | Vectorize the existing function, preserve per-lane PRNG splitting and explicit-reset policy, reduce host conversions; defer implementation as requested |
| PushT / discrete PushT | 2 | Unaccepted approximate native implementation; source Pymunk is scalar | No accepted execution route to optimize yet | Exact restore/contact history remains unresolved; a pool cannot fix correctness |
| Rocket | 1 | Existing replay adapter loops over CPU PyFlyt/Bullet clients; incomplete semantic acceptance | Independent process-owned clients are a future candidate | Preserve replay/lifecycle rules; source can manage drones in a world, which is not a batch of independent Rocket task instances; no promotion |

## Which upstream APIs can actually be delegated to?

**Gymnasium scalar API:** `reset`, `step`, and `render` operate on one task
instance. `SyncVectorEnv` collects a batch but loops serially. `AsyncVectorEnv`
provides persistent processes, asynchronous dispatch/wait and shared-memory
observation transport. In the inspected Gymnasium 1.3.0 source, it creates one
process per factory. Its `call` broadcasts the same arguments to all workers;
shared observation storage does not automatically transport our exhaustive State.
Full-State access therefore needs an adapter-aware facade or worker protocol.
Choose the established vector wrapper when it meets the contract; do not build a
second scheduler merely because upstream scalar environments lack batching.
[Official vector API](https://gymnasium.farama.org/main/api/vector/async_vector_env/).

**CartPole special case:** the installed package has `CartPoleVectorEnv`, and the
official registration offers `vectorization_mode="vector_entry_point"`. Its
implementation uses one vector RNG, automatic next-step resets, and shared
physics attributes. Our accepted scalar adapter retains lane RNGs, explicit
TimeLimit state, post-terminal fields and per-lane physics configuration.
Consequently it is a real native-batching candidate, but is not a drop-in exact
replacement. Other Gymnasium `phys2d` JAX registrations are different tasks,
not the accepted classic-control registrations.
[Official CartPole vector entry point](https://gymnasium.farama.org/environments/classic_control/cart_pole/#vectorized-environment).

**MuJoCo native API:** `mj_step` advances an engine world. The inspected MuJoCo
3.12.0 `rollout.Rollout(nthread=...)` exposes a reusable native thread pool for
batched engine trajectories. This does not execute dm_control task callbacks,
Fetch goal/reward/TimeLimit logic or OGBench action processing. Replacing the
whole task step with raw rollout would change behavior. Native batching may
eventually replace a carefully isolated physics portion, subject to a separate
semantic proof. Python bindings release the GIL during native calls, which makes
threads worth testing but does not establish thread safety of every wrapper.
[Official MuJoCo Python API](https://mujoco.readthedocs.io/en/stable/python.html).

**JAX functional API:** Craftax exposes key/State/action functions that can be
mapped across a batch and compiled. Our current `kernel` compiles a scalar
function, and `CraftaxAdapter.step` invokes it once per lane. Thus upstream
capability exists without adding multiprocessing. Pixel variants also construct
pixel observations during upstream step; an additional render may duplicate
image work and must be identified explicitly in online measurements.
[JAX vectorization](https://docs.jax.dev/en/latest/automatic-vectorization.html),
[upstream Craftax](https://github.com/MichaelTMatthews/Craftax).

## Public operations and what a parallel adapter must retain

| Operation | Existing contract / behavior | Consequence for execution |
|---|---|---|
| Construction and close | Persistent native resources and lane ownership | Build worlds once inside their owning workers; do not pickle live engines or recreate them each step; cleanly close workers/resources |
| `reset` | Family-specific seeds, RNG streams, options and lifecycle | Preserve existing reset/autoreset rules; do not silently adopt a vector wrapper's default autoreset; masked reset is not currently a universal public contract |
| `step` | Batched public inputs; most accepted families return a TensorDict containing `state` and outputs; Craftax returns the full State directly with lifecycle fields | Parallelize independent lanes, retain action order/dtype/repeats and output lane order; do not omit the full State return to claim speed or assume a universal Gym tuple |
| `get_state` | Independent exhaustive snapshot, including task/RNG/lifecycle | Export from the owning worker at a consistent step boundary; returned arrays cannot alias later simulation changes |
| `set_state` | Validated full restoration under persistent Variation | Route each lane to its owner; preserve validation-before-mutation and existing staging guarantees, rather than committing some workers before discovering invalid input elsewhere |
| `to_render_state` / rendering projection | Compact image inputs; explicit API exists in some families | Treat projection and transfer as measured work; compact image inputs do not replace exhaustive simulator snapshots |
| `render(state)` | Uses an explicit snapshot; renderer has its own batching/contexts | Keep GPU rendering in an appropriate owning process; do not turn each CPU simulation worker into a competing GPU renderer |
| `render()` with omitted State | Shared base obtains `get_state()` first; family overrides must be checked | Reuse `step_result['state']` for the StepResult families, or Craftax's directly returned State, instead of requesting a second snapshot |

The current DMC, Fetch, OGBench, classic-control and Craftax adapters already
export State inside `step`. Removing an extra caller-side `get_state` avoids a
duplicate snapshot; it does not remove the export inside the existing contract.
SimplePointMaze's ordinary step is batched, but reset/restore rebuilds its cached
source goal image using per-lane Matplotlib work. Whole-API performance therefore
cannot be inferred from the step implementation alone.

OGBMaze is a concrete thread constraint: `_in_legacy_rng` saves the process RNG,
installs a lane RNG, calls upstream, captures its result and restores the caller's
RNG. Those sequences can interleave across threads. A lock would retain correctness
but serialize that section; process isolation avoids cross-worker RNG sharing.
This finding does not establish that every other wrapper is thread-safe.

## Routing decision and the next bounded experiment

1. Preserve already batched Torch execution for the three navigation tasks.
2. Prefer compatible native batching where present. Craftax vectorization is
   catalogued but deferred. CartPole's vector entry point first needs a lifecycle,
   RNG and Variation compatibility decision.
3. For scalar CPU tasks, reuse a suitable existing vector executor, or make the
   adapter own persistent workers when State transport/ownership needs more than
   that executor supplies. Test MuJoCo wrapper threads against processes rather
   than assuming either wins. OGBMaze needs process isolation for its current RNG
   mechanism if those calls are to run concurrently.
4. Separate `num_envs` from worker count. A batch of 4,096 should not automatically
   create 4,096 processes. A bounded pool can own multiple lanes per worker and
   return one packed shard per call. Decide the number of workers from measured
   CPU, memory and communication costs; native library threads must be accounted
   for as well.
5. Start with one Fetch registration: compare current serial, persistent threads
   and persistent processes at equal total CPU resources, then sweep workers.
   Measure engine step, complete State export, projection, IPC/transfers and
   rendering separately and as a composed loop. Preserve the same-device source
   comparison and rerun trajectory/RNG/reset/restore/nonmutation gates. This is a
   proposed experiment, not a performance result from this audit.

The evidence distinguishes the accepted profile metadata from this Mac's
installed inspection profile (Gymnasium 1.3.0, MuJoCo 3.12.0, dm-control 1.0.45).
Craftax's current source explicitly pins its own profile; the old matrix's
`DELEGATED_CPU` label is not evidence that its JAX implementation lacks GPU
support. No version upgrade or matrix rewrite is implied by this catalogue.
