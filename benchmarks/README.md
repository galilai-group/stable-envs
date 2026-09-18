# stable-envs benchmarks

These scripts answer performance questions through the public `stable_envs` API:

- `render.py` measures synchronized `env.render(public_state, out=reused_buffer)` throughput over
  explicit batch sizes.
- `pipeline.py` measures packed host-state ingress plus rendering. It can also time a directory of
  real JPEG files as a separate, single-process decode baseline.

They do **not** test pixel correctness, fidelity, reproducibility, or environment dynamics. Run the
pytest certification suite for those claims. Benchmark output deliberately contains measurements and
errors, not pass/fail tiers, release gates, badges, or a selected "best" row.

Canonical environment names are constructed through the public class-first API (`snv.PushT(...)`).
SWM aliases and other registry strings go through `snv.make(...)`. Neither script imports renderer
internals, `stable_envs.testing`, or `stable_worldmodel`.

## Quick start

From a repository checkout:

```bash
python -m benchmarks.render \
  --envs PushT,CartPole \
  --device cpu \
  --batches 1,16,64 \
  --warmup 5 \
  --repeats 10 \
  --threads 1 \
  --output render.json \
  --pretty
```

On CUDA, use the same command with `--device cuda`. The script refuses an unavailable requested
device; it never falls back to another one.

Measure the host-state pipeline:

```bash
python -m benchmarks.pipeline \
  --envs PushT \
  --batch 256 \
  --device cuda \
  --warmup 5 \
  --repeats 10 \
  --output pipeline.json \
  --pretty
```

An `.npy` array or an `.npz` file containing a two-dimensional `states` array can replace the
seeded reset:

```bash
python -m benchmarks.pipeline \
  --envs PushT \
  --batch 256 \
  --state-file data/pusht_states.npz \
  --state-key states
```

The file must hold at least one full requested batch. The script does not duplicate a short corpus
silently.

## Optional JPEG comparison

`pipeline.py --jpeg-dir frames/` reopens and decodes `batch` distinct `.jpg`/`.jpeg` files on every
timed call. With multiple `--envs`, place files under `frames/<EnvironmentName>/`. Pillow is imported
only when this option is used and a missing installation produces an explicit error.

This comparison is intentionally narrow: one serial Python process, real files, normally warm OS page
cache after warmup. It is not a cold object-store, sharded-loader, multi-worker, or energy benchmark.
Use files and a loader representative of your production system before making a deployment decision.

Neither script imports or requires `stable_worldmodel`. Integration with that package is a separate
concern from the cost of turning stable-envs states into frames.

## Methodology

- Environment construction, reset/corpus loading, the first lazy render, and output allocation happen
  before timing.
- `--warmup` calls are untimed. `--repeats` synchronized samples are the timed region; CUDA peak
  allocated memory is reset after warmup so it describes that region only.
- `--threads` sets `torch.set_num_threads` and, unless already exported, `OMP_NUM_THREADS` and
  `MKL_NUM_THREADS`. The budget is recorded on every device, including CUDA.
- Every timed device call is synchronized on CUDA and MPS. CPU calls are already synchronous.
- A requested CUDA or MPS device that is not visible is refused; the scripts never fall back.
- The median is the headline statistic. Minimum, quartiles, maximum, every raw sample, process CPU
  seconds per wall second, host load, versions, resolved size/supersampling, resolved device, and the
  Git revision are retained in JSON.
- `native` size, native supersampling, and `backend=auto` are represented by omitting those
  constructor keywords. This preserves each environment's actual defaults instead of substituting a
  cheaper picture.
- `--max-output-mib` records oversized points as skipped before allocating their frame buffer.
- Every requested row is written, including unsupported environments, missing optional dependencies,
  and runtime errors. A row error makes the command exit nonzero.
- `--seed`, `--warmup`, `--repeats`, `--threads`, `--device`, `--backend`, `--size`, and
  `--supersample` are recorded as configuration.

For a before/after regression claim, run both revisions on the same otherwise-idle host and device,
alternate the revisions in fresh processes, and compare distributions rather than one selected run.
Do not compare numbers across GPU models, thread budgets, resolutions, supersampling factors, or
dependency versions as though they were the same experiment.

## Output

The machine-readable schema identifier is `stable-envs-benchmark/v1`. JSON is written atomically to
`--output`; `-` writes JSON to stdout while diagnostics remain on stderr.

`render.py` reports one point per requested `(environment, batch)`, including frames/s,
microseconds/frame, output GB/s, frame shape, resolved settings, timing distribution, and optional
CUDA peak allocated memory.

`pipeline.py` reports one point per environment, including packed-state and frame byte sizes. When
requested, the JPEG result and the direct throughput ratio are nested under that environment so the
two differently scoped measurements cannot be mistaken for one benchmark.
