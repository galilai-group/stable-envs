# stable-envs benchmarks

Two scripts measure rendering cost through the public `stable_envs` API:

- `render.py` measures synchronized `env.render(state, out=buffer)` throughput at given batch sizes.
- `pipeline.py` measures the copy of a host State to the device plus `env.render(state, out=buffer)`. It
  can also time JPEG decoding of a directory of files as a baseline.

They do not test pixel correctness, fidelity or dynamics. The test suite covers those.
Neither script imports `stable_worldmodel` or renderer internals.

The paper harness, `paper/run.sh`, compares the speed and the frames of stable-envs with the source libraries.
See [Paper harness](#paper-harness).

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

## Paper harness

`paper/run.sh` measures the speed of stable-envs against the source library ("the source") and the frame
MSE between them. It pins every arm to the same CPU cores. It uses only the public `stable_envs` API.
Facts about the source libraries, such as which source draws on a GPU, are in
`tests/helpers/reference/sources.py`.

| File | Function |
|---|---|
| `paper/compare.sh` | Runs the whole comparison on one machine and writes `HEADLINE.md` |
| `paper/run.sh` | Runs the arms for each family, env, device and batch, and writes the results |
| `offline_sweep.py` | Saves frame inputs, and times stable-envs and the source on the same inputs. Also writes the frame MSE |
| `online_ladder.py` | Times seeded episodes with fixed actions: steps, and steps with frames |
| `draw.py` | The one call that draws stable-envs frames from stored inputs (`env.render`) |
| `jax_source.py` | The batched JAX source of the Craftax classes: upstream Craftax under one `jit(vmap(...))` |
| `paper/summarize.py` | Writes Markdown tables from one results directory |
| `paper/sanity.py` | Checks the spread and the status of each cell before you use its numbers |

### One command

On a Linux machine with NVIDIA GPUs, clone the repository and run one command from its root:

```bash
bash benchmarks/paper/compare.sh results-h100                   # H100 node: all cells
bash benchmarks/paper/compare.sh results-l40s --missing-only    # L40S node: the cells that rt4 does not have
```

The cells are steps, frames+steps and offline, on CPU and GPU, for stable-envs and the source, at 1, 16, 256 and 1024
envs, for TwoRoom, PushT, DMCReacher and OGBCube. The script installs what it needs, uses all
GPUs, and times the stable-envs cells first, then the source cells. A cell stops after 20 minutes (status `timeout`).
Run the command again to continue. It writes `HEADLINE.md`; `--summary OUT` shows it, `--dry-run` shows the cells.

### Run

Run the harness from the repository root, on Linux, with `taskset`, `timeout` and `git` installed.
For CPU arms of sources that use OpenGL, install `xvfb-run`.

```bash
SWM=/path/to/stable-worldmodel DEVICES=cpu ARMS="offline frames" \
  bash benchmarks/paper/run.sh results/cpu classic=CartPole,Pendulum dmc=DMCCartpoleSwingup
```

Each argument is `FAMILY=ENV,ENV`. The script builds one virtual environment for each family with `uv`, unless
`VENV_PYTHON` gives an interpreter that has the family installed. Under Slurm, submit the script with one task,
8 CPUs and, for CUDA arms, one GPU.

The header of `paper/run.sh` lists all the environment variables. The most important are:

| Variable | Default | Function |
|---|---|---|
| `DEVICES` | `cpu cuda` | Devices of the stable-envs arms |
| `ARMS` | `offline step frames` | Arms to run |
| `SWEEP_CPU`, `SWEEP_CUDA` | `1:8:64:256:1024`, `1:256:512:1024:2048` | Batches and lane counts |
| `QUICK` | `0` | `1` gives 3 windows of 5 s and lane counts `8:256:1024` |
| `REPEATS`, `WINDOW`, `WARMUP` | `3`, `10`, `2` | Timed windows per cell, seconds per window, untimed seconds before them |
| `CPU_SOURCES` | `1` | `0` does not run the source arms on CPU |
| `VENV_PYTHON` | none | An interpreter to use instead of a new family environment |
| `STATES_DIR` | a temporary directory | Keeps the stored inputs and episodes between runs |
| `SRC_FROM` | none | A results directory from the same machine to copy the source cells from |
| `FRAME_CHECK` | `1` | `0` does not compare the source workers' frames with one-env replays |
| `SWM`, `UV`, `NVDIFFRAST_SITE` | `$HOME/stable-worldmodel`, `uv`, `$HOME/nvdiffrast-site` | Used only to build a family environment |

A file that exists in the output directory is not measured again. Thus, a second run with the same directory
measures only the cells that are missing.

### Arms

- `offline`: stable-envs draws stored inputs with `env.render` at each batch of the sweep. The inputs are rows
  from the family sampler (`sample_states`). For a class without a sampler, the inputs are full States from
  random-action rollouts. If `env.render` refuses rows on a device (PyFlyt Rocket on CPU), stable-envs draws
  rollout States on that device. Each cell records the form that stable-envs drew in `input` (`rows` or
  `state`). The source draws the same inputs in 1 process and in one process for each core. The frame MSE
  compares the two on the 0-255 scale. An input that a device cannot draw gives the status `unsupported`.
- `step`: physics steps with L lanes for each L of the sweep, and the source stepping L envs on CPU.
- `frames`: steps with a frame after each step. The source steps and draws in worker processes, one core each.
  The parent sends step t+1 before it reads step t.
- The Craftax classes have two sources. The stable-worldmodel `CraftaxWrapper` runs in the worker processes as
  above. It draws no frames for the Symbolic classes. The batched JAX source runs upstream Craftax with
  `auto_reset=False` as one jitted, vmapped call over all lanes, on the device of the cell. It splits the keys inside
  the call, and the actions are on the device before timing. Without frames, the Pixels step drops its observation,
  as stable-envs does, and the Symbolic step keeps it. With frames, each step returns uint8 frames; the Symbolic
  classes draw them with the pixel env of their game. Offline, this source draws the stored States at each batch.
  It is also the reference of the Craftax frame MSE. Each cell compares stable-envs with the faster of the two sources.

### Output

- `run.json`: the commit, host, cores, GPU and settings of the run.
- `offline/<Env>/snv-{cpu,gpu}.json`: stable-envs frames/s at each batch.
- `offline/<Env>/src-{cpu,gpu}-<P>p.json`: source frames/s in P processes.
- `offline/<Env>/mse-{cpu,gpu}.json`: frame MSE against the source (mean, 95th percentile, maximum).
- `online/<Env>/{snv,src}-{step,render}-*-L<L>.json`: steps/s at L lanes.
- `offline/<Env>/src-jax-{cpu,gpu}.json` and `online/<Env>/src-jax-{step,render}-{cpu,gpu}-L<L>.json`: the batched
  JAX source (Craftax).
- Each timed cell records `cpu_util`: CPU seconds of the timing process, all threads, per wall second.

Each timed cell keeps the median rate and the rate of each window (`rates`). A cell whose rates spread by more
than 10% is not reliable. Measure it again before you use it.

```bash
python benchmarks/paper/summarize.py results/cpu results/gpu
python benchmarks/paper/sanity.py results/cpu
```
