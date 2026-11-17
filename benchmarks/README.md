# stable-envs benchmarks

Two scripts measure rendering cost through the public `stable_envs` API:

- `render.py` measures synchronized `env.render(state, out=buffer)` throughput at given batch sizes.
- `pipeline.py` measures the copy of a host State to the device plus `env.render(state, out=buffer)`. It
  can also time JPEG decoding of a directory of files as a baseline.

They do not test pixel correctness, fidelity or dynamics. The test suite covers those.
Neither script imports `stable_worldmodel` or renderer internals.

`speed.py`, `render_mse.py` and `rollout_parity.py` compare stable-envs with stable-worldmodel. See
[Comparison with stable-worldmodel](#comparison-with-stable-worldmodel).

`online_ladder.py` and `offline_sweep.py` measured the speedup tables of the paper. See
[Paper measurements](#paper-measurements).

## Run

```bash
python -m benchmarks.render \
  --envs PushT,CartPole --device cpu --batches 1,16,64 \
  --warmup 5 --repeats 10 --threads 1 --output render.json --pretty
```

```bash
python -m benchmarks.pipeline \
  --envs PushT --batch 256 --device cuda \
  --warmup 5 --repeats 10 --output pipeline.json --pretty
```

The scripts refuse a device that is not available. They never fall back to another device.

`pipeline.py` can read a full State from a file instead of a seeded reset. Pass an `.npz` file with one
array per State field with `--state-file`. Each key is the field path joined by `/`, for example
`agent/position`. The file must hold at least one full batch. Classes that have `to_render_state`, such as
PushT, copy only the fields that `render` reads.

`pipeline.py --jpeg-dir frames/` decodes `batch` distinct `.jpg`/`.jpeg` files on every timed call.
With several `--envs`, put the files under `frames/<EnvName>/`. This needs Pillow. The JPEG timing
runs in one Python process. It is not a multi-worker loader benchmark.

## Method

- Construction, reset, the first render and output allocation happen before timing.
- `--warmup` calls are not timed. `--repeats` synchronized calls are timed.
- `--threads` sets `torch.set_num_threads`, and `OMP_NUM_THREADS` and `MKL_NUM_THREADS` unless they are
  already set.
- CUDA and MPS calls are synchronized. CUDA peak memory is reset after warmup.
- Omitting `--size`, `--supersample` and `--backend` uses each environment's own defaults.
- `--max-output-mib` skips points whose frame buffer would be too large.
- Every requested row is written, including errors. Any row error makes the command exit nonzero.

To compare two revisions, run both on the same idle host and device, in fresh processes, and compare
the distributions. Do not compare across GPU models, thread counts, resolutions or dependency versions.

## Output

The schema identifier is `stable-envs-benchmark/v1`. JSON is written atomically to `--output`.
`--output -` writes to stdout; diagnostics go to stderr.

The median is the headline value. Each point also keeps minimum, quartiles, maximum, all raw samples,
process CPU seconds per wall second, host load, versions, resolved size and supersampling, device,
configuration and the Git revision.

- `render.py` writes one point per `(environment, batch)`: frames/s, µs/frame, output GB/s, frame
  shape, settings, timing distribution and optional CUDA peak memory.
- `pipeline.py` writes one point per environment, with State and frame byte sizes. The JPEG
  result and the throughput ratio are nested under that environment.

## Comparison with stable-worldmodel

These scripts compare stable-envs with stable-worldmodel (swm) at commit `21446f1`. Each class uses its swm id
from `tests/helpers/reference/crosswalk.py`. `swm_source.py` builds the swm envs, writes a stable-envs State into
an swm env, and reads it back. Install the `bench` dependency group, and the `pin-*` group and the extra of each
family:

```bash
uv sync --group bench --group pin-pusht --extra pusht --group torch-cuda
```

| File | Function |
|---|---|
| `speed.py` | Times stable-envs and swm on the same cores in one run, and writes one CSV row per cell |
| `render_mse.py` | Draws the same States with swm and stable-envs, and writes the frame MSE per env |
| `rollout_parity.py` | Plays the same actions in both from the same seeded reset, and from swm's own state loaded into ours, and writes the state error per env. It never writes a state into swm |
| `swm_source.py` | The swm source: ids, the swm env calls, and the State mapping |
| `backend_switch.py` | Times both CUDA simulation backends of a class at batches 1 to 4096 and prints its switch point |

Run the scripts from the repository root.

```bash
python -m benchmarks.speed --cpu --gpu --step --render --step-and-render --env PushT TwoRoom \
  --batch 1 8 16 64 --actions-per-call 1 16 --out speed.csv
python -m benchmarks.render_mse --env PushT TwoRoom --device cpu cuda --out render_mse.csv
python -m benchmarks.rollout_parity --env PushT TwoRoom DMCReacher OGBCube --steps 20 --out rollout_parity.csv
```

### Method of `speed.py`

- The script runs every combination of device (`--cpu`, `--gpu`), workload, env, batch B and actions per call T.
- Each cell times stable-envs ("ours") first and then swm. Each side runs in a new process. Both sides use the
  same CPU cores (`--cores`, default: the first 8 cores of the process).
- Ours uses the default keyword arguments of `snv.make`. Ours imports `stable_envs` before `torch`.
- Workloads:
  - `step`: physics steps without frames. Ours steps B instances with T actions per `step` call. swm steps B
    envs in min(B, cores) worker processes, one core each. The parent sends step t+1 before it takes step t.
  - `step-and-render`: the same, with a frame after each step.
  - `render` (the default): frames of stored States. Ours draws them with `snv.FrameLoader`, B frames at a time.
    swm writes each State into an env and draws it, in 1 process and in one process per core. The CSV keeps
    the faster of the two.
- Each episode starts with a seeded reset, and the stored actions are uniform random actions.
- Each side warms up for `--warmup` seconds and at least `--warmup-steps` steps per instance (default 20).
  Then it times `--repeats` windows (default 3), or `--repeats-small` windows (default 7) at B up to
  `--small-max-batch` (default 16). A window holds `--timed-steps` steps per instance (default 100), rounded up
  to whole episodes. With `--timed-steps 0`, a window lasts `--window` seconds. The CSV gives the median rate.
- On the GPU device, swm also draws with MuJoCo EGL when the env draws with MuJoCo. The CSV keeps the faster
  swm result and names it in `source_kind`.
- A short probe times the first steps of each swm step cell. When the cell would take more than
  `--source-cap` seconds, the script estimates the rate from a partial run: `--repeats` windows of
  `--partial-window` seconds, counted in steps. The CSV sets `estimated` to true for these cells.
- The CSV has the host, the CPU model, the GPU, the driver, the cores, the stable-envs commit and the swm
  commit and source library versions of each cell. A `.jsonl` file next to the CSV holds every window of
  every cell.

## Paper measurements

These scripts measured the speedup tables and the LeWM training numbers of the paper:

| File | Function |
|---|---|
| `online_ladder.py` | Times seeded episodes with fixed actions: physics steps, and steps with a frame after each step |
| `offline_sweep.py` | Saves frame inputs, times stable-envs and the source when they draw the same inputs, and writes the frame MSE |
| `swm_dmc.py` | Draws DM Control, `OGBCube` and `OGBScene` sources with the stable-worldmodel envs, and applies `SNV_VARIATION` |
| `examples/lewm/throughput.py` | Trains LeWM and measures training steps/s and GPU use for three frame inputs |

`online_ladder.py` and `offline_sweep.py` are the fixed record of the method of the paper. `speed.py` is the
general tool that the project maintains.

The source is stable-worldmodel at commit `21446f1` where it has the class. Fetch uses Gymnasium Robotics.
Other classes use the library of their `reference_id`.

### Setup

Install the `bench` group, and the extra and the `pin-*` group of the family, as in
[Comparison with stable-worldmodel](#comparison-with-stable-worldmodel). The paper used torch 2.14.1+cu126.
Run every command from the repository root with these settings:

- `PYTHONPATH=.` and `SDL_VIDEODRIVER=dummy`. Ours calls `env.step(actions)` with its default `states` argument.
  `STEP_STATES=none` passes `states=None` instead. The paper did not use it.
- `WARMUP_MAX_S=300`. This stops the warm-up step count of a slow source after 300 s.
- `taskset -c 0-7`. Each process, and each source worker that it starts, uses the same 8 cores.
- CPU cells: software OpenGL, that is `xvfb-run -a` with `LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=glfw PYOPENGL_PLATFORM=glx`.
- GPU cells: `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_DEVICE_ORDER=PCI_BUS_ID`, and
  `MUJOCO_EGL_DEVICE_ID` set to the value of `CUDA_VISIBLE_DEVICES`.
- Rendering-only and both cells of `OGBCube`:
  `SNV_VARIATION='{"OGBCube": {"agent.opacity": [1.0], "target.opacity": [1.0]}}'`. Ours then draws the look
  of the stable-worldmodel env. Physics cells use no variation.

The cells use B = 1, 16 and 128 and frames of 224 x 224. In the commands below, `E` is the class, `P = min(B, 8)`,
`K = ceil(B / P)`, and `R` is 5 at B = 1 and 3 at other batches.

### Timing

- Each command below is one cell and runs in its own process.
- Ours uses the default keyword arguments of `snv.make` and one action per `step` call (`--chunks 1`).
- Each cell warms up for 2 s and at least 20 steps per environment (rendering only: 20 batches).
- Then the script times `R` windows of at least 3 s and reports the median rate. All window rates are in the JSON
  file.
- After the first window, no new window starts when the windows took 60 s in total. The offline source in
  several processes always times `R` windows. `repeats` in the JSON file gives the number of windows.
- Online windows hold whole 64-step episodes. Each episode starts with a seeded reset, so the reset cost is part
  of the rate.
- The source runs B envs in P worker processes, one core each. The parent sends step t+1 before it takes step t.
- The paper marks two kinds of cells:
  - `*` (OGBCube, CPU, both, source at B = 16 and 128): the source rate is an estimate from a partial run. The
    partial run has a 180 s warm-up and then 3 windows of about 100 s, and gives their median. That script is
    not in this directory. A full run of the command below gives the rate over whole episodes.
  - `‡` (PushT physics): the value is the median of 5 pairs of the commands below, ours and the source in
    turn.

### Physics and both

```bash
python -m benchmarks.online_ladder dataset --envs E --out episodes --episodes 128 --length 64
COMMON="--env E --data episodes --procs P --source-envs K --num-envs B --threads 8 --repeats R --seconds 3 \
  --warmup 2 --warmup-steps 20 --timed-steps 0 --episodes B --no-frame-check --chunks 1"
python -m benchmarks.online_ladder run $COMMON --measure src-pipelined-step --output src-step-LB.json
python -m benchmarks.online_ladder run $COMMON --measure snv-step --output snv-step-cpu-LB.json
python -m benchmarks.online_ladder run $COMMON --measure snv-step-gpu --output snv-step-gpu-LB.json
python -m benchmarks.online_ladder run $COMMON --measure src-pipelined-render --src-label cpu --output src-render-cpu-LB.json
python -m benchmarks.online_ladder run $COMMON --measure src-pipelined-render --src-label gpu \
  --output src-render-gpu-LB.json
python -m benchmarks.online_ladder run $COMMON --measure snv-render-cpu --output snv-render-cpu-LB.json
python -m benchmarks.online_ladder run $COMMON --measure snv-render-gpu --output snv-render-gpu-LB.json
```

Run the `src-render-gpu` cell only where `reference_gpu_render` in `tests/helpers/reference/sources.py` gives a
GPU renderer for the class.

Each file has one entry in `measurements`, with the name of the measurement and `-T1` for ours. The entry has
`steps_per_s`, which is the median of `rates`. It also has `window_s`, `items`, `repeats`, `warmup_items`,
`warmup_s`, `num_envs` and `status`.

- Physics speedup: `steps_per_s` of `snv-step` (CPU table) or `snv-step-gpu` (GPU table), divided by
  `steps_per_s` of `src-pipelined-step`. The source has no GPU physics, so both tables use its CPU rate.
- Both speedup: `steps_per_s` of `snv-render-cpu` or `snv-render-gpu`, divided by the source rate. The CPU table
  uses `src-pipelined-render-cpu`. The GPU table uses the higher rate of `src-pipelined-render-cpu` and
  `src-pipelined-render-gpu`.

### Rendering only

```bash
python -m benchmarks.offline_sweep states --envs E --out states --n 128 --devices cpu cuda
TIMING="--repeats 3 --seconds 3 --warmup 2 --warmup-steps 20 --timed-steps 0"
python -m benchmarks.offline_sweep ours --envs E --states states --device cpu --threads 8 --batches 1 16 128 \
  $TIMING --repeats-small 5 --small-max-batch 1 --output snv-cpu.json
python -m benchmarks.offline_sweep ours --envs E --states states --device cuda --threads 8 --batches 1 16 128 \
  $TIMING --repeats-small 5 --small-max-batch 1 --output snv-gpu.json
python -m benchmarks.offline_sweep source --envs E --states states --procs 1 $TIMING --output src-cpu-1p.json
python -m benchmarks.offline_sweep source --envs E --states states --procs 8 $TIMING --output src-cpu-8p.json
python -m benchmarks.offline_sweep source --gpu --envs E --states states --procs 1 $TIMING --output src-gpu-1p.json
python -m benchmarks.offline_sweep source --gpu --envs E --states states --procs 8 $TIMING --output src-gpu-8p.json
python -m benchmarks.offline_sweep mse --envs E --states states --device cpu --output mse-cpu.json
python -m benchmarks.offline_sweep mse --envs E --states states --device cuda --output mse-gpu.json
```

`states` saves rows from `sample_states` where the class has a sampler. For other classes, it saves full States
from random-action rollouts. Ours draws the inputs with `snv.FrameLoader`, B frames per batch. The source draws
the same inputs in 1 and in 8 processes. The `--gpu` cells run only where the source draws with MuJoCo EGL.

- `ours` files: `envs.E.points` has one entry for each batch, with `fps` (the median of `rates`), `input`
  (`rows` or `state`) and `frame_shape`.
- `source` files: `envs.E.fps`.
- Rendering-only speedup: `fps` of ours at batch B, divided by the highest source `fps`. The CPU table uses the two
  CPU files. The GPU table uses the four source files.
- `mse` files: `mse_mean`, `mse_p95` and `mse_max` of the per-frame MSE on the 0-255 scale.

### LeWM training speed and GPU use

```bash
python examples/lewm/throughput.py --env E --input source --data E_state.h5 --steps 1000 --warmup-steps 200 \
  --out runs --model-overrides loader.num_workers=8
python examples/lewm/throughput.py --env E --input ours --data E_state.h5 --steps 1000 --warmup-steps 200 \
  --out runs --model-overrides loader.num_workers=8
```

`E` is `tworoom`, `pusht`, `reacher` or `cube`. The data file holds the state rows of the LeWM dataset. Each run
gives one GPU, 16 CPU cores and 8 DataLoader workers. Add `seed=S` to `--model-overrides` for more seeds.
Each run writes `summary-<env>-<input>.json` and appends one row to `throughput.csv` in `--out`. The
docstring of the script lists the fields. The steady state is steps 201 to 1000.

- Training speedup: `steps_per_s` of `ours` divided by `steps_per_s` of `source`.
- Training-step GPU busy fraction: `steps_per_s` multiplied by `step_gpu_ms` / 1000 of the `ours` run of the same
  class. The model does the same work in each step for both inputs.
