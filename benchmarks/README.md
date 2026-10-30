# stable-envs benchmarks

The general-purpose scripts measure rendering cost through the public `stable_envs` API:

- `render.py` measures synchronized `env.render(state, out=buffer)` throughput at given batch sizes.
- `pipeline.py` measures host-state transfer plus rendering. It can also time JPEG decoding of a
  directory of files as a baseline.

They do not test pixel correctness, fidelity or dynamics. The test suite covers those.
Neither script imports `stable_worldmodel` or renderer internals.

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

`pipeline.py` can read States from a file instead of a seeded reset. Pass an `.npy` array, or an
`.npz` file with a two-dimensional array, with `--state-file` and `--state-key`. The file must hold at
least one full batch.

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
- `pipeline.py` writes one point per environment, with packed-state and frame byte sizes. The JPEG
  result and the throughput ratio are nested under that environment.

## Object-permanence replay

`python -m benchmarks.object_permanence_recordings --recordings <dataset> --output <json>`
measures the shared GPU renderer with and without static caching. Timings include
CPU frame selection and pose upload, with startup/JIT/cache construction reported
separately. `--modes rgb,depth,ids` measures combined outputs. Both G34 and G62
use the same recorded-trajectory and shared-renderer path.

Use `--clips g13-nominal g146-nominal g01-nominal` for other recordings; their
manifest selects the explicit orthographic, scale, alpha or optical contract.
Both cache settings upload only dynamic nodes, including optional scales or
exact optical matrices. Specialized diagnostic commands remain available.

The same command can build native-source/shared GIFs from source previews.
See [the recording guide](../docs/adding/object-permanence-recordings.md) for the
import, independent oracle, validation and preview workflow.
