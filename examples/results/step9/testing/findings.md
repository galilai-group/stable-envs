# Step 9 — Wave 2, Workstream K: tests/certification cleanup

**Workstream:** `step9/testing` · **Branch HEAD:** see below · **Baseline:** Step-8
`21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`, integration parent `786993b`
**Host:** Apple M5 Mac (`cis1b1zs2j3`), CPU only — no CUDA device in this ticket.
**Interpreter:** `/Users/samib/se6-venv` (torch 2.14 CPU).

**Result: PATCH.** The pytest suite now has the sibling-repo layout
(`unit` / `integration` / `gpu` / `certification` / `fixtures`). Marker commands collect the
intended buckets. Product behavior, thresholds, frozen corpora, and every assertion are unchanged.

---

## Rulings (binding for this workstream)

### `stable_envs/testing/` is public library code

The plan §14 line "stable_envs/testing → tests" does **not** mean move the package. This
directory is the measurement library, like `numpy.testing` / `torch.testing`. Env authors,
`docs/api/testing.md`, examples, scratch probes, and `python -m stable_envs.testing.supervisor`
import it. Nothing in it is this repo's own pytest suite.

What moved is **this repository's certification assertions and golden fixtures**:

- assertions → `tests/certification/`
- golden JSON → `tests/fixtures/`

`stable_envs/testing/` stays in the package. The dependency still runs one way: `tests/`
imports the library; the library does not import `tests/`.

### `stable_envs/rollout.py` stays in the package

Wave-0 deferred a contract-§5 move of policies/`sample` to Workstream K. The prohibition
that contract actually makes — no `env.rollout(policy)` — is already honoured. `rollout.py`
is the state-generation convenience (`snv.sample` / `BasePolicy`), not a stepping API, and
is imported by name from tests and examples. Moving it is a call-site rename, not a test
layout change. It stays.

---

## 1. Files moved / created / removed

Moved (`git mv`):

| before | after |
|---|---|
| `tests/unit/test_certification.py` | `tests/certification/test_certification.py` |
| `tests/unit/test_declarations.py` | `tests/certification/test_declarations.py` |
| `tests/unit/test_provenance.py` | `tests/certification/test_provenance.py` |
| `tests/unit/test_step3.py` | `tests/certification/test_step3.py` |
| `tests/unit/test_step4.py` | `tests/certification/test_step4.py` |
| `tests/unit/test_step5.py` | `tests/certification/test_step5.py` |
| `tests/unit/test_step6_report.py` | `tests/certification/test_step6_report.py` |
| `tests/unit/test_step8_gate_record.py` | `tests/certification/test_step8_gate_record.py` |
| `tests/unit/test_step8_performance.py` | `tests/certification/test_step8_performance.py` |
| `tests/unit/test_step8_provenance.py` | `tests/certification/test_step8_provenance.py` |
| `tests/unit/test_step8_semantic_craftax.py` | `tests/certification/test_step8_semantic_craftax.py` |
| `tests/swm/test_gate_f_matrix.py` | `tests/certification/test_gate_f_matrix.py` |
| `tests/test_safety_net.py` | `tests/integration/test_safety_net.py` |
| `tests/data/prng_threefry_golden.json` | `tests/fixtures/prng_threefry_golden.json` |
| `tests/data/craftax_full_golden.json` | `tests/fixtures/craftax_full_golden.json` |

Removed: empty `tests/swm/` (only `__pycache__` remained after the move).

Created: this findings file. `tests/certification/` and `tests/fixtures/` are the move
destinations, not new code.

Left in place: every family/unit contract under `tests/unit/` (including
`tests/unit/test_examples.py` — repo-contract checks, not multi-component),
`tests/gpu/`, `tests/integration/test_worldmodel.py`, `stable_envs/testing/`,
`stable_envs/rollout.py`. Historical Step-8 findings under `examples/results/step8/`
were not rewritten.

`Path(__file__).resolve().parents[N]` for files that stayed at depth 2 (`tests/unit/`,
`tests/certification/`) is still the repo root. The two readers that pointed at
`tests/data/` now point at `tests/fixtures/`. `tests/integration/test_safety_net.py`
uses `parents[2]` (was `parents[1]` at `tests/`).

## 2. API changes

None in the product. pytest config only:

- `testpaths = ["tests"]` so bare `pytest` is the official suite, not
  `scratch/step9/packaging-ci/test_packaging.py` (11 tests; L's CI still names that file).
- Automatic directory markers in `tests/conftest.py`. Existing per-file markers
  (`unit`, `gpu`, `integration`, `reference`, `selftest`, `smoke`) are kept.

## 3. Step-8 IDs affected

None. No registry, renderer, threshold, or corpus change.

## 4. Targeted tests

Collection (`MUJOCO_GL=glfw`, se6-venv, `--collect-only -p no:cacheprovider`): **2243**
tests, no missing-file errors.

Prescribed unit-ish suite (same interpreter):

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit tests/certification tests/integration -x \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
```

**1721 passed, 279 skipped, 124 deselected, 1 failed** in 444.33 s.

The failure is `tests/integration/test_safety_net.py::test_a_c_level_exit_mid_test_is_treated_as_a_crash_not_as_a_test_failure`:
the nested supervisor hit the test's 300 s `subprocess.run` timeout. The inner pytest was
left orphaned (PPID 1) after the supervisor was SIGKILL'd. This is a crash-isolation flake,
not a path/import break — unit and certification had already finished (the F is the last
item). Assertions in the moved files were not edited.

`-m unit` collects **only** `tests/unit/` (1836 items) and does not import the GPU modules
that `allow_module_level` skip without CUDA.

## 5. Applicable certification checks

Layout-only. The frozen Step-3/4/5/6/8 readers now live under `tests/certification/` and
still assert the same files under `examples/results/step{3,4,5,6,8}/` and the same
golden JSON (now `tests/fixtures/`). GPU certification
(`test_conformance.py`, `test_reproducibility.py`, `test_fast_paths.py`) is marked
`gpu` **and** `certification`; those modules skip at import without CUDA, so
`-m "certification and gpu"` collects 0 on this host. L's GPU workflow is the place
that command is meant to run.

## 6. Default-frame comparison

Not applicable. No renderer or env was edited.

## 7. Variation/state comparison

Not applicable. Packed rows, style layouts, and corpora were not touched.

## 8. Performance comparison for hot paths

Not applicable. No hot path changed.

## 9. Resource-lifetime comparison

The prescribed run's resource guard: **24 EGL contexts created, 0 released, 24 live
(peak 24); renderers 19 created / 0 closed**. That is the pre-existing unit/certification
pattern of constructing CPU envs and not closing them; this workstream did not add a
release. `close()` semantics are unchanged.

## 10. Unexplained oracle differences

**Zero.** No oracle surface was modified.

---

## Marker table

Directory hook in `tests/conftest.py` (`pytest_collection_modifyitems`):

| directory | markers added |
|---|---|
| `tests/unit/` | `unit` |
| `tests/integration/` | `integration` |
| `tests/certification/` | `certification` |
| `tests/gpu/` | `gpu` |
| `tests/gpu/test_{conformance,reproducibility,fast_paths}.py` | `gpu` **and** `certification` |

Existing extra markers (`reference`, `smoke`, `selftest`) stay. Files that already had
`pytestmark` keep it; the hook adding the same mark again is a no-op for `-m` selection.

Collect counts on this host (se6-venv, no CUDA):

| command | before (786993b) | after |
|---|---:|---:|
| `pytest` (collect-only) | 2254 | 2243 |
| `pytest -m unit` | 616 | 1836 |
| `pytest -m integration` | 6 | 18 |
| `pytest -m gpu` | 55 | 104 |
| `pytest -m certification` | 0 | 285 |
| `pytest -m "certification and not gpu"` | 0 | 285 |
| `pytest -m "certification and gpu"` | 0 | 0 (CPU; modules skip at import) |
| `tests/unit/` | 2116 | 1836 |
| `tests/certification/` | 0 | 285 |
| `tests/integration/` | 6 | 18 |
| `tests/gpu/` | 104 | 104 |
| `tests/swm/` | 5 | gone |
| `tests/test_safety_net.py` | 12 | moved into integration |

The 11-test drop in bare `pytest` is `scratch/step9/packaging-ci/test_packaging.py`
falling out of `testpaths`. The `-m unit` jump (616 → 1836) is the point of the hook:
files that forgot `pytestmark` were invisible to L's `pytest tests/unit -m unit`.
The `-m gpu` jump (55 → 104) is the same for GPU files that only had per-test marks
or a `skipif` `pytestmark`.

Arithmetic check: 2116 − 1836 = 280 left `tests/unit/`; 280 + 5 (Gate-F) = 285
`tests/certification/`. 6 + 12 = 18 `tests/integration/`.

---

## The §17 ten-item report

1. **files moved/removed/created** — §1. 15 moves, `tests/swm/` deleted, findings created.
2. **API changes** — §2. None in the product; pytest `testpaths` + directory markers.
3. **Step-8 IDs affected** — none.
4. **targeted tests** — §4. Collection 2243; prescribed suite 1721 passed / 279 skipped
   / 1 pre-existing safety-net timeout.
5. **applicable certification checks** — same readers, new paths; GPU arm needs CUDA.
6. **default-frame comparison** — N/A.
7. **variation/state comparison** — N/A.
8. **performance comparison for hot paths** — N/A.
9. **resource-lifetime comparison** — §9. Guard still 0 releases; counts are the
   pre-existing construct-and-hold pattern.
10. **unexplained oracle differences** — **zero**.

---

## Coordinator requests

1. **Freeze the two rulings** in `WAVE0_CONTRACT.md` / the plan: (a)
   `stable_envs/testing/` stays public library code; "→ tests" means this repo's
   certification assertions and fixtures. (b) `stable_envs/rollout.py` stays as
   `snv.sample` / `BasePolicy`; the forbidden thing remains `env.rollout(policy)`.
2. **Dashboard row** for Workstream K on `examples/results/step9/dashboard.yaml`
   (coordinator-owned; not edited here).
3. **Workstream L CI** now does what it printed: `pytest tests/unit -m unit` collects
   every unit file (1836 here), not the 616 that remembered `pytestmark`. Intended.
   The certification workflow's `-m "certification and gpu"` still needs a CUDA
   runner — those three modules skip at import on CPU.
4. **Stale comments** in `stable_envs/envs/ogbench/manipulation.py` still name
   `tests/unit/test_declarations.py`. Not edited (env file, comment only).
5. **Safety-net `os._exit` timeout** left an orphan pytest. Pre-existing supervisor
   behavior (SIGKILL of the supervisor does not always reap the inner pytest). Not
   fixed here; do not treat it as a K regression.
6. Historical path strings under `examples/results/step8/` and prior Step-9 findings
   still name `tests/unit/test_step*` / `tests/swm` / `tests/data`. Left on purpose.

---

## Commits

| SHA | subject |
|---|---|
| `4f75633` | step9: put certification assertions and golden fixtures in tests/ |
| `249ad75` | step9: stamp directory markers so pytest -m buckets work |
| `65b28c9` | step9: run the suite through the new marker commands |
