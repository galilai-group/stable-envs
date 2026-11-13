# stable-envs benchmarks

Two scripts measure rendering cost through the public `stable_envs` API:

- `render.py` measures synchronized `env.render(state, out=buffer)` throughput at given batch sizes.
- `pipeline.py` measures the copy of a host State to the device plus `env.render(state, out=buffer)`. It
  can also time JPEG decoding of a directory of files as a baseline.

They do not test pixel correctness, fidelity or dynamics. The test suite covers those.
Neither script imports `stable_worldmodel` or renderer internals.

`offline_sweep.py` and `online_ladder.py` compare the speed and the frames of stable-envs with the source
libraries. See [Comparison with the source libraries](#comparison-with-the-source-libraries).

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

## Comparison with the source libraries

`offline_sweep.py` and `online_ladder.py` measure the speed of stable-envs against the source library ("the
source"), and `offline_sweep.py` also writes the frame MSE between them. They use only the public
`stable_envs` API. Facts about the source libraries, such as which source draws on a GPU, are in
`tests/helpers/reference/sources.py`.

| File | Function |
|---|---|
| `offline_sweep.py` | Saves frame inputs, and times stable-envs and the source on the same inputs. Also writes the frame MSE |
| `online_ladder.py` | Times seeded episodes with fixed actions: steps, and steps with frames |
| `draw.py` | The one call that draws stable-envs frames from stored inputs (`env.render`) |

`python benchmarks/offline_sweep.py --help` and `python benchmarks/online_ladder.py --help` list the options.
Pin each measurement to the same CPU cores, for example with `taskset`, so that stable-envs and the source
use the same hardware.

- Offline: stable-envs draws stored inputs with `env.render` at each batch. The inputs are rows from the
  family sampler (`sample_states`). For a class without a sampler, the inputs are full States from
  random-action rollouts. The source draws the same inputs. The frame MSE compares the two on the 0-255 scale.
- Online: physics steps with L lanes, and steps with a frame after each step, against the source stepping L
  envs on CPU.
