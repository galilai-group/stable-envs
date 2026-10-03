# Fetch shared-memory source executor — CPU prerequisite complete

Base `30a590a4343df1d55654ed6498f797640e582fa2`. This adds benchmark-only source
owners, using the same Gym spawn/shared-memory scaffold as the other family
source comparisons. Production adapters, execution routing, shared State,
renderer and user-facing API are unchanged. This is neither throughput nor
rendering acceptance.

The helper loads the actual pinned source registration module and constructs each
historical ID with `gym.make`. A package namespace avoids importing unrelated
policy/data dependencies; the real Fetch registration, wrapper and upstream
Robotics classes still execute unchanged. Every lane resets with `variation=[]`
and seed 17 plus its global lane index. Shards preserve action dtype and order.
No cached-observation candidate mixin is used by the source.

Each call starts with all lanes active. Once a lane terminates/truncates, its
remaining actions in that chunk are not consumed. Task columns are padded with
zero and accompanied by validity. A new call retains ordinary source behavior,
including post-terminal stepping; the container never invents an autoreset.
The source still computes its complete native observation/info; the harness
projects the numeric task fields exposed by the accepted library API.

Only task outputs use shared-memory arrays. `copy=True` provides owned parent
buffers before workers can write the next result. Independent source snapshots
and their restoration are **untimed, pickled preparation**, not a measured State
transport optimization. Restoration copies explicit source continuation fields,
rebuilds derived MuJoCo values, then restores warmstart/controller caches using
the already demonstrated source restoration procedure. It is scoped to this
controlled default-variation benchmark, not arbitrary model/variation migration.

## Executed checks

`local-v2.json` / `local-v2.log` are authoritative: **96 passing real CPU cases**,
12 registrations x T1/T8 x float32/float64 x ordinary/mixed time limits, B4/P2.
Each source owner contains two lanes, so this exercises within-shard ordering
and masks as well as process boundaries. Source is compared against independent
serial source and real accepted Fetch adapters through the unchanged shared
step dispatcher. The latter is a render-free fixture, not a public constructor.

The checks cover:

- Exact source numeric snapshots/tasks and independent full candidate State
  extraction at reset, continuation endpoints and restored replay.
- Default 50-step limits and mixed limits `[1,3,2,50]`, zero task padding, explicit
  validity and held terminal State history. T8 executes 32 or 14 transitions,
  respectively; T1 executes four. Padding is never counted as extra transitions.
- Returned task arrays and complete public State/history remain owned after a
  subsequent call with changed actions. Physical qpos actually changes.
- Different seeded source RNG history, full restoration, then actual unseeded
  reset agree with the independent adapter and serial source.
- Two distinct workers per pool, one Torch thread each, correct global lane
  ranges, no initialized GL viewers, and disabled container autoreset.
- **24 pools close; all 48 workers exit with code zero.**

`local-v1` is retained as an earlier passing run before the explicit unseeded-reset
RNG exercise was added. It is not the final candidate. The final report's candidate
hashes match the current benchmark code and unchanged production dependencies.
Source hashes refer to stable-worldmodel
`176d54532b3c0a251a4f509b74f2fdacc143a63d`.

The local profile uses the isolated MuJoCo 3.11.0 wheel and Robotics 1.4.2 from
the preceding [source prerequisite](../fetch_source_prerequisite_20261003/).
The source wrapper's separate rendering behavior is not invoked or repaired here.
The PickAndPlace render-callback continuation issue and source internal-versus-output
raster dimensions remain material constraints for the future rendered comparison.
No source FPS, GPU output acceptance, public process-default acceptance or broader
registration promotion follows from these CPU checks.

## Reproduction and next gate

```bash
PYTHONPATH=/path/to/isolated-mujoco311:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /path/to/fetch-python -m scripts.benchmarks.online_stepping.fetch_source_transport_gate \
  --source-root /path/to/pinned/stable-worldmodel \
  --source-manifest docs/research/semantic_recovery_evidence/fetch_source_transport_20261003/source_manifest.json \
  --output /tmp/new-fetch-source-transport.json
```

The output must not already exist. No production process route or sibling result
is substituted for independent source checks. Next comes actual source CPU/GPU
frame transport for the nine callback-stable registrations, then equal-allocation
serial/parallel source timing and memory measurements at justified batch sizes.
PickAndPlace no-frame source measurements can use this executor; its rendered
online comparison still requires explicit lifecycle accounting. Oscar currently
rejects SSH authentication; no new cluster job was submitted or old job restarted.
