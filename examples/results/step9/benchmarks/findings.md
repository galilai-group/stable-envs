# Step 9 — Wave 2, Workstream N: OSS benchmarks

**Workstream:** `step9/benchmarks`
**Step-8 baseline:** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`
**Implementation:** `b879a4074577e28b15be368cc7a028b161005189` (`benchmarks: add OSS performance harnesses`)
**Integration merges:** `d3cb4a8` as `3ff917b`, then latest `4feceba` as `71e7d92`
**Audit/evidence:** `ff30e8f997d07b9e777c477ca8e1116f4d8350c9`
**Host:** Apple M5 Mac (`cis1b1zs2j3.ad.brown.edu`), CPU only — no CUDA device in this ticket
**Interpreter:** `/Users/samib/se6-venv` (Python 3.12.11, torch 2.14.0 CPU, numpy 2.5.3, tensordict, loguru)
**MUJOCO_GL:** `glfw`

**Result: PATCH.** The public `benchmarks/` harnesses measure renderer throughput and an optional
host-state pipeline through the Step-9 `snv` API. They do not certify pixels, do not import
`stable_worldmodel` or renderer internals, and write machine-readable JSON with errors rather than
pass/fail tiers.

---

## 1. Files moved / created / removed

Created on `b879a40`, then audited and tightened on this branch:

| path | role |
|---|---|
| `benchmarks/README.md` | OSS-facing methodology and CLI |
| `benchmarks/__init__.py` | package marker; performance-only disclaimer |
| `benchmarks/_common.py` | parsers, device/thread pinning, timing, provenance, JSON |
| `benchmarks/render.py` | class-first `snv.<Class>` (dynamic aliases through `snv.make`) → `reset` → `render(public State)` |
| `benchmarks/pipeline.py` | packed host-state ingress + `render`; optional JPEG arm |
| `tests/unit/test_benchmark_surface.py` | CLI/config/JSON/API/error/certification-separation guards |
| `examples/results/step9/benchmarks/render_cpu_local.json` | preserved CPU evidence (tree-identical to `b879a40`) |
| `examples/results/step9/benchmarks/pipeline_cpu_local.json` | preserved CPU evidence (tree-identical to `b879a40`) |
| `examples/results/step9/benchmarks/findings.md` | this file |
| `scratch/step9/benchmarks/run_cpu.sh` | reproducible command for the preserved local CPU take |

Removed: nothing. The historical `examples/bench_*.py` scripts and `tests/unit/test_benchmarks.py`
remain for Workstream O; N does not replace their published tables.

Shared files not edited: `stable_envs/*`, `pyproject.toml`, `docs/`, `dashboard.yaml`.

---

## 2. API changes

None to the library. The new CLI is:

```bash
python -m benchmarks.render --envs PushT,CartPole --device cpu --batches 1,16,64 \
    --warmup 5 --repeats 10 --threads 1 --output render.json --pretty
python -m benchmarks.pipeline --envs PushT --device cuda --batch 256 \
    --warmup 5 --repeats 10 --output pipeline.json --pretty
```

Canonical names construct as `snv.PushT(num_envs=..., device=..., seed=...)`. SWM aliases go through
`snv.make`. Native size, native supersample, and `backend=auto` omit those keywords so the picture is
the environment default. `batch=` is not used.

---

## 3. Step-8 IDs affected

None. This workstream does not change dynamics, pixels, variation maps, or registry rows.

---

## 4. Targeted tests

```bash
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider \
    tests/unit/test_benchmark_surface.py
```

**23 passed** in 14.73s.

Guards: parser rejection, native constructor keywords omitted, CUDA/MPS refusal without fallback,
CUDA index bounds, warmup untimed vs repeats retained, CUDA peak reset after warmup, OpenMP/MKL
setdefault, class-first construction, CPU CLI smoke + schema, pipeline public `reset`/`pack` contract,
unknown env recorded with exit 1, short corpus not duplicated, oversized skip, stdout JSON with
stderr diagnostics, optional JPEG error and success paths, no worldmodel/testing/render-internal
imports, preserved evidence JSON schema.

---

## 5. Certification checks

None by design. Benchmarks are not a Gate, not a support-matrix row, and not a pixel oracle. JSON
contains `correctness_claim: null` and no `tier` / `passed` / `certification` / `badge` keys. Run
pytest for those claims.

---

## 6. Default-frame comparison

Not applicable. The scripts time `render`; they do not compare frames.

---

## 7. Variation / state comparison

Not applicable. `render.py` times `env.render(public State)`. `pipeline.py` times
`torch.as_tensor(host packed rows).to(device)` then `env.render(rows, out=...)`, with the packed
rows coming from `env.pack(env.reset(seed))` unless `--state-file` is supplied.

---

## 8. Performance (CPU evidence, not a library regression)

These numbers are **local CPU measurements on a loaded laptop**, not certification and not a
before/after of a hot path. Load average around the preserved take was ~6.5–7.1. `--threads 1`,
`--warmup 2`, `--repeats 7`. Median wall time; raw samples retained in the JSON.

Preserved renderer throughput (`render_cpu_local.json`; measured before the attribution-only history
reconstruction, with a tree identical to `b879a40`):

| env | batch | frame | fps | µs/frame | cpu s / wall s |
|---|---|---|---|---|---|
| PushT | 1 | 96×96 | 691 | 1446 | 1.00 |
| PushT | 16 | 96×96 | 1879 | 532 | 1.00 |
| PushT | 64 | 96×96 | 2994 | 334 | 0.99 |
| CartPole | 1 | 400×600 | 194 | 5166 | 1.00 |
| CartPole | 16 | 400×600 | 1130 | 885 | 1.00 |
| CartPole | 64 | 400×600 | 1416 | 706 | 0.99 |
| DMCReacherHard | 1 | 84×84 | 147 | 6808 | 0.98 |
| DMCReacherHard | 16 | 84×84 | 295 | 3385 | 0.99 |
| DMCReacherHard | 64 | 84×84 | 253 | 3950 | 1.00 |
| DMCWalkerWalk | 1 | 84×84 | 78 | 12754 | 0.90 |
| DMCWalkerWalk | 16 | 84×84 | 176 | 5694 | 1.00 |
| DMCWalkerWalk | 64 | 84×84 | 155 | 6442 | 0.97 |

Preserved pipeline (`pipeline_cpu_local.json`, same host/revision, batch 64, no `--jpeg-dir`):

| env | packed state | fps | µs/frame |
|---|---|---|---|
| PushT | (64, 5) / 20 B | 2727 | 367 |
| DMCReacherHard | (64, 4) / 16 B | 253 | 3952 |

Post-audit smokes (warmup 1, repeats 3, not overwriting the evidence files) stayed in the same
order of magnitude: PushT 1/16 ≈ 677 / 1823 fps; CartPole 1/16 ≈ 202 / 1130 fps; pipeline PushT
batch 16 ≈ 1645 fps.

`resolved_supersample` was 1 for every preserved row, which matches this tree's class defaults
(`DMControlSuite.default_supersample = 1`). Native keywords were omitted, so this is not the old
`--ss 1` substitution bug.

---

## 9. Resource lifetime

Each point/environment is constructed, used, and `close()`d in `finally`. Correctness does not depend
on `__del__`. No EGL/MJWarp/nvdiff resources were created on this CPU ticket
(safety plugin: 0 EGL contexts, 0 renderers).

---

## 10. Unexplained oracle differences

Zero. No oracle was run; none is owed by this workstream.

---

## Audit of `b879a40` and fixes retained

What already held:

- CLI writes `stable-envs-benchmark/v1` JSON atomically (or to stdout with diagnostics on stderr).
- Warmup is untimed; repeats are synchronized on CUDA/MPS; CPU is already synchronous.
- Requested CUDA/MPS is refused when unavailable; no silent device fallback.
- Optional JPEG comparison is nested under the environment, separately timed, Pillow imported only
  then, and a missing dir/install does not erase the render row.
- Unknown/optional-family failures become error rows and a nonzero exit.
- No certification keys, no SWM import, no renderer-internal import.

Gaps closed on this branch:

1. Canonical names now construct through `snv.<Class>(...)`; `snv.make` remains the string/alias path.
2. `backend=auto` is omitted from the constructor, matching native size/supersample.
3. `--threads` also `setdefault`s `OMP_NUM_THREADS` / `MKL_NUM_THREADS`.
4. CUDA peak memory is reset **after** warmup so it describes the timed region.
5. A requested `cuda:N` above `device_count()` is refused.
6. `resolved_device` is recorded. Pipeline skip/error rows carry `batch`.
7. Surface tests cover the contract above, including the preserved JSON files.

---

## Methodology (binding for these scripts)

- Time the public caller-visible path, never a private kernel.
- Exclude construction, reset/corpus load, the first lazy render, and output allocation.
- Headline statistic is the median of synchronized samples; keep every sample.
- Record host load, CPU seconds per wall second, versions, git revision, and resolved size/ss.
- Do not pick a "best" row. Do not compare across GPU models, thread budgets, resolutions, or
  supersample factors as if they were one experiment.
- JPEG decode, when requested, is one serial Python process over real files, normally warm in the OS
  page cache after warmup. It is not a cold object-store or sharded-loader measurement.

---

## Limitations

- **No GPU numbers.** This host has no CUDA. CUDA synchronization, peak-memory, and device-refusal
  paths are unit-tested; throughput on a GPU is owed to whoever runs `--device cuda` on Oscar.
- **Local CPU is not certification.** Load ~6–7; numbers do not travel. DMCWalkerWalk batch 64 is
  slower per frame than batch 16, and the raw samples show the spread.
- **JPEG comparison** is optional and was not part of the preserved evidence take; it is unit-tested.
- **`import_errors`** in the JSON is interpreter provenance from `snv.import_errors()`. Some messages
  mention third-party install paths; they are library text, not N-owned.
- **Families that still only accept `batch=`** will record a constructor error if selected. N uses
  `num_envs=` only (Wave-0 contract). That is a family leftover, not a benchmark bug.
- **History hygiene:** the attribution-only reconstruction preserved all three original trees and
  merge topology under `b879a40` / `3ff917b` / `ff30e8f`. Every reachable Workstream-N commit is
  authored by `sami-bg` and has zero AI attribution trailers.

---

## Review readiness

Ready for independent review. Suggested verdict: **APPROVE** if the reviewer agrees that (a) the CLI
is performance-only and separated from certification, (b) construction uses the public `snv` API,
and (c) the CPU JSON is evidence of the harness, not a Step-8/9 performance gate.

Requests for the coordinator: none. Dashboard row `N_benchmarks` is coordinator-owned.
