# stable-envs benchmarks

Two scripts measure rendering cost through the public `stable_envs` API:

- `render.py` measures synchronized `env.render(state, out=buffer)` throughput at given batch sizes.
- `pipeline.py` measures the copy of a host State to the device plus `env.render(state, out=buffer)`. It
  can also time JPEG decoding of a directory of files as a baseline.

They do not test pixel correctness, fidelity or dynamics. The test suite covers those.
Neither script imports `stable_worldmodel` or renderer internals.

`speed.py`, `render_mse.py` and `rollout_parity.py` compare stable-envs with stable-worldmodel. See
[Comparison with stable-worldmodel](#comparison-with-stable-worldmodel).

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
