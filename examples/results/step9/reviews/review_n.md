APPROVE

# Step 9 Wave 2 — independent review of Workstream N (OSS benchmarks)

**Reviewed** `step9/benchmarks` at **`71e7d928d36dddbc21025b537a086adcd0449ec8`** (`Merge latest
step9/integration into benchmarks`). N-attributable delta: `git diff step9/integration...HEAD` =
9 files, +1,963 / −0. Step-8 behavioural baseline
`21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`. Second parent of HEAD is integration
`4fecebade7dd53d814e593be68074abb36985d30` (approved MuJoCo merge). Current `step9/integration` tip
`ab355147b220ed66e9e462967a75fa54df46172d` (docs merge) is **not** an ancestor of HEAD; no N file
overlaps M.
**Reviewer host** Apple M5 Mac `cis1b1zs2j3.ad.brown.edu`, local CPU only, no CUDA, no Oscar.
`MUJOCO_GL=glfw`, `-p no:cacheprovider`, interpreter `/Users/samib/se6-venv`. Nothing was edited,
stashed, reset, rebased or pushed except this review file. Untracked
`scratch/step9/benchmarks/run_cpu.sh` was left untouched and is not part of the delta.

## Verdict

**APPROVE-WITH-CHANGES.** The harness matches plan §14/N and stays out of §21's certification
performance gate. Public construction is class-first, warmup/repeat/thread/device logic is honest,
CUDA peaks reset after warmup, JPEG is optional and nested, JSON is `stable-envs-benchmark/v1` with
errors instead of tiers, and the committed CPU numbers are labelled local / non-certification. I
re-ran the 23 surface tests and cheap CLI smokes; both are green. The coordinator's reconstruction
of the three original branch commits succeeded: the reachable N delta has **zero** git AI trailers.

Two documentation updates are owed before merge (N-1, N-2 below). Neither is a product-logic defect.

## Attribution (reachable delta)

The coordinator reconstructed the three original N commits onto identical trees and dropped the
prohibited trailer. None of the originals is an ancestor of HEAD:

| role | original | reconstructed | trees identical | trailer |
|---|---|---|---|---|
| add harnesses | `4d2e0da` | `b879a40` | `fcc8643` = `fcc8643` | original had `Co-authored-by: Cursor <cursoragent@cursor.com>`; reconstructed body is the subject line only |
| merge `d3cb4a8` | `89a7ab0` | `3ff917b` | `fdcb34e` = `fdcb34e` | neither |
| CPU evidence + audit | `3680547` | `ff30e8f` | `587ef0c` = `587ef0c` | neither |

`git merge-base --is-ancestor 4d2e0da HEAD` is false (same for `89a7ab0` / `3680547`). Unique
commits vs merge-base `4feceba` are `b879a40`, `3ff917b`, `ff30e8f`, `71e7d92`. Author and
committer on all four: `sami-bg <samibghanem@gmail.com>`. `git log --format=%B 4feceba..HEAD`
contains **zero** `Co-authored-by`, `Cursor`, `cursoragent`, `Generated with`, Claude, ChatGPT, or
Copilot strings.

The **tree** still contains one stale sentence in `findings.md` claiming the implementation commit
carries that trailer. That is now false for every reachable commit. That is N-1, not a remaining
git trailer.

## N-attributable delta

`git diff --name-only 4feceba...HEAD`:

```
benchmarks/README.md
benchmarks/__init__.py
benchmarks/_common.py
benchmarks/pipeline.py
benchmarks/render.py
examples/results/step9/benchmarks/findings.md
examples/results/step9/benchmarks/pipeline_cpu_local.json
examples/results/step9/benchmarks/render_cpu_local.json
tests/unit/test_benchmark_surface.py
```

`git diff --name-only 4feceba...HEAD -- stable_envs/ pyproject.toml tests/unit/test_benchmarks.py
examples/bench_*.py docs/` is empty. Historical `examples/bench_*.py` + `tests/unit/test_benchmarks.py`
are untouched (O keeps those tables). Latest merge `71e7d92` does not retouch N files vs `ff30e8f`.

## The §17 items

1. **Files moved/removed/created.** Nothing moved, nothing removed. Created: the five `benchmarks/`
   paths, `tests/unit/test_benchmark_surface.py`, three artifacts under
   `examples/results/step9/benchmarks/`. Confirmed against the diff, not the prose.

2. **API changes.** None to the library. New CLI is `python -m benchmarks.render` /
   `python -m benchmarks.pipeline`. Canonical names use `snv.<Class>(num_envs=..., device=...,
   seed=...)`; SWM aliases go through `snv.make`. Native size/supersample and `backend=auto` omit
   those keywords. `batch=` is not passed.

3. **Step-8 IDs affected.** None. No dynamics, pixels, variation maps, or registry rows.

4. **Targeted tests.** Re-run by me at `71e7d92`:

   ```
   MUJOCO_GL=glfw se6-venv pytest -q -p no:cacheprovider tests/unit/test_benchmark_surface.py
   -> 23 passed, 1 warning in 13.37s
      resources: 0 EGL contexts created, 0 released, 0 live
   ```

   Non-interference (not owned by N): `tests/unit/test_benchmarks.py` → **25 passed** in 0.08s.

5. **Applicable certification checks.** None by design. JSON `methodology.correctness_claim` is
   `null`. Keys `tier` / `passed` / `certification` / `badge` / `best` are absent from record keys
   (surface tests assert this). Import-error *values* may mention a third-party "VISUAL tier"; that
   is library text, not a harness claim.

6. **Default-frame comparison.** n/a. The scripts time `render`; they do not compare frames.

7. **Variation/state comparison.** n/a. `render.py` times `env.render(public State)`. `pipeline.py`
   times `torch.as_tensor(host packed rows).to(device)` then `env.render(rows, out=...)`.

8. **Performance.** Not a §21 paired hot-path check and not a library regression. Preserved CPU JSON
   matches the findings table on every rounded fps / µs/frame / cpu-s-per-wall-s cell (12 render
   rows + 2 pipeline rows). My cheap smokes on this loaded host stayed in the same order of
   magnitude and are slower, which is what "local, not certification" means:

   | arm | preserved | this review (warmup 1, repeats 3) |
   |---|---|---|
   | PushT b=1 / 16 | 691 / 1879 fps | 583 / 1302 fps |
   | CartPole b=1 / 16 | 194 / 1130 fps | 175 / 867 fps |
   | pipeline PushT b=16 | (post-audit ~1645) | 1291 fps |

   `resolved_supersample` is 1 on every preserved row. CUDA throughput remains owed to Oscar.

9. **Resource lifetime.** Each point/environment is constructed, used, and `close()`d in `finally`.
   My surface run: 0 EGL / 0 renderers. Correctness does not depend on `__del__`.

10. **Unexplained oracle differences.** Zero. No oracle is owed.

## Contract checks I verified in code, tests, and live CLI

**Performance-only vs certification / §21.** `benchmarks/__init__.py`, README, and both CLIs state
performance-only. Live and committed JSON carry `correctness_claim: null` and no pass/fail tier.
§21's 10 % hot-path bar is a later paired comparison on 2D / raycast / MJWarp / nvdiff / moved
dynamics — not this ticket. N does not replace `examples/bench_*.py`.

**Public class-first `snv` API.** `public_env` constructs `snv.PushT` (an `Env` subclass) and must
not call `snv.make` for that name — unit-tested. Live: `snv.PushT(num_envs=2, device="cpu", seed=0)`
and `snv.make("swm/PushT-v1", ...)` both yield `PushT` on cpu. `env_kwargs` omits `size` /
`supersample` / `backend` when native/`auto`. Wave-0 `num_envs=` only.

**Warmup / repeats / threads / device.** Warmup is untimed; repeats are the samples (3 warmup + 4
repeats → 7 calls, 4 samples — tested). `--threads` calls `torch.set_num_threads` and
`setdefault`s `OMP_NUM_THREADS` / `MKL_NUM_THREADS` without clobbering an exported OpenMP value —
tested. Requested CUDA/MPS is refused with no fallback; `cuda:N` above `device_count()` is refused.
Live: `--device cuda` → argparse exit **2**, no JSON, stderr names `torch.cuda.is_available()`.

**CUDA peak reset.** `measure()` warmup-runs, synchronizes, then `torch.cuda.reset_peak_memory_stats`
before the timed loop. Test `test_cuda_peak_reset_happens_after_warmup` pins that order. Methodology
string on a live run: `"CUDA max_memory_allocated of the timed region after warmup"`.

**Optional JPEG.** Nested under the environment, separately timed, Pillow imported only then. Missing
dir leaves the render row `ok` and records `jpeg_comparison.status=error` with exit 1. Real files
decode and write `render_fps_over_jpeg_fps`. Both paths unit-tested. Preserved pipeline JSON has
`"jpeg_dir": null` and no comparison object.

**Machine-readable schema.** `schema = stable-envs-benchmark/v1`. Atomic replace (or stdout JSON /
stderr diagnostics). `allow_nan=False`, `sort_keys=True`. Live smoke at HEAD writes
`resolved_device: cpu` and records warmup/repeats/threads. Committed evidence files are the 4d2e0da
take: same schema and `correctness_claim`, **no** `resolved_device` key (added in the audit commit;
intentionally not overwritten).

**Honest failures.** Unknown env → error row + exit 1 + stderr. Oversized point → `skipped`, exit 0.
Short state corpus → refuses silent duplication. One optional-family error does not erase other
rows. KeyboardInterrupt/SystemExit are re-raised.

**CPU numbers labelled local / non-certification.** Filenames `render_cpu_local.json` /
`pipeline_cpu_local.json`. Findings §8 / limitations: "local CPU measurements on a loaded laptop,
not certification". JSON: `config.device=cpu`, host `cis1b1zs2j3.ad.brown.edu`,
`correctness_claim: null`, no `/oscar/` path. Load averages ~6.5–7.1 are retained. There is no
separate `scope` field; filename + findings + `correctness_claim` are the label, and they are
honest.

## Cheap smokes (this review)

```
python -m benchmarks.render --envs PushT,CartPole --device cpu --batches 1,16 \
    --warmup 1 --repeats 3 --threads 1 --output $TMP/render.json --pretty
python -m benchmarks.pipeline --envs PushT --device cpu --batch 16 \
    --warmup 1 --repeats 3 --threads 1 --output $TMP/pipeline.json --pretty
```

Both wrote `stable-envs-benchmark/v1` with `resolved_device=cpu`, `correctness_claim=null`,
`stable_envs_git_revision=71e7d92`, PushT packed source
`env.reset(seed) -> env.pack(public State)`, no JPEG object, `--help` ok. CUDA refusal as above.
Did not overwrite the committed evidence files.

## Defects (documentation; required before merge)

**N-1.** `examples/results/step9/benchmarks/findings.md` still names implementation `4d2e0da`,
integration merge `89a7ab0`, and says that implementation commit **carries**
`Co-authored-by: Cursor`. After reconstruction those SHAs are not on HEAD and no reachable N
commit has that trailer. JSON `stable_envs_git_revision` may keep `4d2e0da` as the original take
(same tree as `b879a40`); the findings header and Limitations bullet must say that, and must stop
claiming a live trailer. Point the branch at `b879a40` / `3ff917b` / `ff30e8f` / `71e7d92`.

**N-2.** `benchmarks/render.py` module docstring still times the public
``snv.make -> reset -> render`` surface. Construction is class-first (`snv.PushT(...)`); `snv.make`
is the alias path only. README and `public_env` already say that. One-line docstring fix.

## Nits (do not block once N-1/N-2 land)

- Preserved JSON lacks `resolved_device` and the `peak_device_memory` methodology key that live
  CLI now emits. That is the documented 4d2e0da take; do not regenerate it as if it were a GPU
  certification run.
- Untracked `scratch/step9/benchmarks/run_cpu.sh` is not in the delta. Drop or commit under O's
  scratch rules; do not mix it into this merge.
- Coordinator: merge current `step9/integration` (`ab35514`, Workstream M) before landing if that
  tip should be an ancestor; N files do not overlap.

No other defects found. Dashboard row `N_benchmarks` remains coordinator-owned.

---

## Re-review at `f170cd0` (delta `e23c975..f170cd0`)

**APPROVE.** I reviewed only `git diff e23c975..f170cd0` (one commit,
`benchmarks: correct reconstructed provenance`, author/committer `sami-bg <samibghanem@gmail.com>`,
empty body, **no** AI trailer). Three paths: `benchmarks/render.py`,
`examples/results/step9/benchmarks/findings.md`, `scratch/step9/benchmarks/run_cpu.sh` (+49/−11).
No library, test, JSON, or dashboard edit.

**N-1 closed.** Findings header now names reachable `b879a40` / `3ff917b` / `ff30e8f` / `71e7d92`.
Those four are ancestors of HEAD. The Limitations bullet states the reconstruction preserved the
original trees and that every reachable N commit is `sami-bg` with zero AI trailers. `findings.md`
no longer contains `4d2e0da`, `89a7ab0`, or `Co-authored-by` / `Cursor`. Evidence JSON is described
as tree-identical to `b879a40`, not as a live trailer-bearing commit.

**N-2 closed.** `benchmarks/render.py` docstring now times the class-first
``snv.<Environment> -> reset -> render`` surface for canonical names and `snv.make` only for
dynamic aliases.

**`run_cpu.sh`.** The previously untracked local reproducer is now committed under
`scratch/step9/benchmarks/`. Header and flags match the preserved take (`cpu`, warmup 2, repeats 7,
threads 1, same envs/batches) and label the numbers as performance samples, not certification.

**Tests (this re-review, at `f170cd0`):**

```
MUJOCO_GL=glfw se6-venv pytest -q -p no:cacheprovider tests/unit/test_benchmark_surface.py
-> 23 passed in 13.94s
   resources: 0 EGL contexts created, 0 released, 0 live
```

Reachable `4feceba..HEAD` commit messages still have zero `Co-authored-by` / `cursoragent` /
`Generated with`. Product and findings were not edited in this re-review.

Nothing above reopens N-1 or N-2. **APPROVE.**
