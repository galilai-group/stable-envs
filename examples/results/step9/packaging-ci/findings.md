# Step 9 — Wave 2, Workstream L: packaging / dependencies / CI

**Workstream:** `step9/packaging-ci` · **Parent merge:** `step9/integration` `d3cb4a8`
(`d21b1f3`, no rebase) · **Step-8 baseline:** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`
**Host:** Apple M5 Mac (`cis1b1zs2j3`), CPU only.
**Interpreters:** isolated uv venvs for 3.10/3.12 installs; `/Users/samib/se6-venv` for nav-compat tests.

**Result: PATCH.** The package builds, installs, and imports on core dependencies. Optional families
fail per-module through `import_errors()` instead of breaking `import stable_envs`. GitHub Actions
cover unit/integration/package/docs/pre-commit, tag-only trusted publishing, and a manual
certification matrix. **Unexplained oracle differences: zero** (no renderer or environment semantics
changed).

### Independent-review response

`review_l.md` initially returned `APPROVE-WITH-CHANGES` because the post-F CI environment did not
install MuJoCo, the integration job repeated the Fetch assertion without the unit-job deselect, and
the packaging checks skipped wholesale on Python 3.10. The corrected workflow installs
`.[dev,mujoco-locomotion]` (plus `interop` in the integration job), so the same test command has the
dependency profile its collected locomotion cells require and no Fetch exception is hidden by a
deselect. The `dev` extra now supplies `tomli` on Python 3.10 and the packaging test falls back to it.

Reproduced in the workflow's clean editable venv after the fix:

```text
pytest tests/unit -m unit                         403 passed, 88 skipped
pytest scratch/step9/packaging-ci/test_packaging.py    11 passed
```

---

## 1. Files moved / created / removed

| before | after |
|---|---|
| `mkdocs.yml` | canonical `mkdocs.yaml` (site_url / edit_uri / directory URLs) plus a compatibility symlink `mkdocs.yml -> mkdocs.yaml` so existing K tests keep reading one config |
| `pyproject.toml` | PEP 639 license, dynamic version, two ordered authors, core deps, family extras, `dev`/`docs` |

Created:

- `.github/workflows/{testing,pre-commit,docs,release,certification}.yml`
- `.pre-commit-config.yaml`
- `scratch/step9/packaging-ci/{audit_dist,audit_optional,smoke_installed,test_packaging}.py`
- `examples/results/step9/packaging-ci/{validation.json,optional_import_errors.json,findings.md}`

Removed: **nothing importable.** Direct-URL `nvdiffrast` metadata and the unsatisfiable `all` extra
are gone from `pyproject.toml` on purpose.

---

## 2. API changes

Packaging/metadata only:

- `pip install -e ".[dev]"` / `pip install .` / wheel / sdist
- `snv.__version__` stays `0.1.0` from `stable_envs.__about__`
- `__about__.__author__` is the integration spelling `Randall Balestriero, Sami BuGhanem`
- `pyproject` authors are exactly that pair, in that order

No environment class, registry, or render signature changed.

---

## 3. Step-8 IDs affected

None. Discovery still records a missing optional stack as that family's `import_errors()` entry.

---

## 4. Targeted tests

```text
scratch/step9/packaging-ci/test_packaging.py          11 passed
tests/unit -m unit (clean .[dev,mujoco-locomotion])   403 passed / 88 skipped
tests/unit/test_repo_contracts.py mkdocs nav           2 passed via the yml symlink
```

Archive: Twine PASSED; 104 package modules in wheel+sdist; Author metadata
`Randall Balestriero, Sami BuGhanem`; no `Requires-Dist` direct URLs; no `all` extra.

---

## 5. Applicable certification checks

None on this workstream. Certification CI is `workflow_dispatch` only, split by incompatible
profiles (`core`, `dmcontrol`, `ogbench`, `fetch`; GPU self-hosted profiles including nvdiff).

---

## 6. Default-frame comparison

Not applicable. Core-only smoke: `PushT(num_envs=2, size=32)` renders `(2, 32, 32, 3)` uint8 on
Python 3.10 (wheel) and 3.12 (sdist).

---

## 7. Variation / state comparison

Not applicable.

---

## 8. Performance comparison

Not applicable.

---

## 9. Resource-lifetime comparison

Not applicable. `mjwarp` / `nvdiff` still import without their CUDA implementations.

---

## 10. Unexplained oracle differences

Zero.

---

## Validations

| check | result |
|---|---|
| editable `.[dev]` on Python 3.12 | PASS; `stable_envs` resolves to the worktree |
| wheel on Python 3.10 | PASS; 63 envs registered, 13 optional modules recorded |
| sdist on Python 3.12 | PASS |
| docs build (`mkdocs.yaml`) | PASS, 76 HTML pages |
| `mkdocs --strict` | FAIL, 23 warnings in current research notes (Workstream M) |
| pre-commit | PASS (syntax/safety hooks; no research-history reformat) |
| PEP 639 `license = "MIT"` | PASS (the License-classifier combination was rejected) |

---

## Limitations

1. **No `all` extra.** Fetch needs `mujoco<3.12`, Overcooked needs its own interpreter, and
   nvdiffrast cannot be expressed as PyPI metadata.
2. **`metaworld` extra is not resolver-clean.** Upstream pins `mujoco==3.3.0`; install
   `metaworld` with `--no-deps` as before. `humanoid_bench` remains git/`--no-deps`.
3. **`mkdocs --strict` is withheld** until M rewrites research-note links into `examples/results`
   and `scratch/`.
4. **CI `pytest -m unit` deselects** `test_envs_fetch.py::test_the_module_still_refuses_to_import_without_the_raytracer`
   because a core-only interpreter reports `No module named 'mujoco'` rather than `mujoco_warp`.
   Request K to treat both as the same graceful miss.
5. **GitHub Pages / PyPI / GPU runner** need maintainer environments `github-pages`, `release`,
   and label `stable-envs-gpu`. The workflows do not store API tokens.
6. **Compatibility symlink** `mkdocs.yml -> mkdocs.yaml` stays until K updates
   `tests/unit/test_repo_contracts.py` and `tests/unit/test_step8_semantic_craftax.py`.

---

## Coordinator requests

| id | owner | request |
|---|---|---|
| R1 | K | Read `mkdocs.yaml` (or follow the symlink and then drop it). |
| R2 | K | Fetch "missing backend" unit test should accept a missing `mujoco` as well as `mujoco_warp`. |
| R3 | M | Make research-note links `--strict`-safe, then CI can add `--strict`. |
| R4 | coordinator/maintainer | Create GitHub environments and the GPU runner label before the first real release. |

---

## Review readiness

Ready for independent review. No Step-8 behavior change. Remaining items are documented
limitations, not unfinished packaging work.
