APPROVE-WITH-CHANGES

# Step 9 — independent review of Workstream L (packaging / dependencies / CI)

**Reviewed** `step9/packaging-ci` at **`eca6eb283241e9198b04888c171f91f76cce0a9a`** (author delivery
`3e52386`, then the coordinator merge of `step9/integration` `4feceba`). L-attributable delta:
`git diff 4feceba HEAD` (16 files, +1298/−409). Step-8 behavioural baseline
`21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`.
**Reviewer host** Apple M5 Mac `cis1b1zs2j3.ad.brown.edu`, local CPU only. Interpreters:
`/tmp/se9-review-l/build` (build/twine only), `/tmp/se9-l-py310` (Python 3.10.18, core wheel),
`/tmp/se9-l-py312` (Python 3.12.11, sdist), `/tmp/se9-l-dev` (editable `.[dev]`), `/tmp/se9-l-docs`
(mkdocs), `/Users/samib/se6-venv` (nav-contract tests + F locomotion cells with mujoco present).
Nothing was edited except this review file. No push.

## Verdict

**APPROVE-WITH-CHANGES.** The package itself is right: PEP 639 MIT, dynamic version from
`stable_envs.__about__`, authors exactly `Randall Balestriero, Sami BuGhanem` in that order, core
deps are the Wave-0 five, optional extras are PyPI-installable with no `all` extra and no
nvdiffrast/direct-URL metadata, wheel/sdist audit, Twine, and clean 3.10/3.12/editable smokes all
pass on this tip. Release is tag-only trusted publishing behind the `release` environment — stricter
than the sibling `stable-*` tag workflows. Docs build (76 HTML pages). No AI trailers in
`4feceba..HEAD`.

What is not merge-ready is **unit/integration CI at this post-F tip**. L's `testing.yml` installs
`.[dev]` (core + test tools, no mujoco) and runs `pytest tests/unit -m unit`. After merging
`4feceba`, that job is red:

1. **Integration job omits the fetch deselect** the unit job already documents. On a core-only
   interpreter `fetch.py` fails at `import mujoco` (`No module named 'mujoco'`), so
   `test_the_module_still_refuses_to_import_without_the_raytracer` fails its `mujoco_warp` substring
   check. I reproduced that failure on `/tmp/se9-l-dev` and on the 3.10 wheel install.
2. **Sixteen F locomotion construction tests fail on the same core-only extra set.**
   `mujoco_locomotion` registers without mujoco; construction then raises
   `gymnasium.error.DependencyNotInstalled`, which `_constructible` does not skip. Same `. [dev]`
   run: **17 failed, 386 passed, 92 skipped**. Those four cells pass in `se6-venv` where mujoco is
   present (**20 passed**).

Packaging/metadata is not the problem. Do not merge until `pytest -m unit` is green on the workflow's
own install command at this HEAD.

## L-attributable delta

`git diff --name-status 4feceba HEAD` is exactly:

```text
A .github/workflows/{certification,docs,pre-commit,release,testing}.yml
A .pre-commit-config.yaml
A examples/results/step9/packaging-ci/{findings.md,optional_import_errors.json,validation.json}
A mkdocs.yaml
T mkdocs.yml          (120000 symlink -> mkdocs.yaml)
M pyproject.toml
A scratch/step9/packaging-ci/{audit_dist,audit_optional,smoke_installed,test_packaging}.py
```

No `stable_envs/` product file is in the L delta. `__about__.__author__` was already the integration
spelling (`d3cb4a8`); L only asserts it. `git diff 4feceba HEAD -- stable_envs/` is empty.

Three commits on `4feceba..HEAD`, all `sami-bg <samibghanem@gmail.com>`, empty `%(trailers)`:
`d21b1f3` (earlier integration merge), `3e52386` (L patch), `eca6eb2` (merge `4feceba`). No
`Co-Authored-By`, no `Generated with`, no AI attribution in those messages or in the 16 files.

## §14/L and §22 against this tip

Owned surfaces match the brief: `pyproject.toml`, `.github/`, `.pre-commit-config.yaml`,
`mkdocs.yaml` (plus the K-compat symlink). Sibling alignment holds: PEP 639 `license = "MIT"` like
`stable-worldmodel`, ordered `{name=}` authors, `requires-python = ">=3.10"`, loguru+torch core,
tag-only trusted publish on environment `release`. L's release job is *stricter* than
`stable-datasets` / `stable-pretraining` (those publish after `uv build` with smoke steps commented
out). L adds version-tag equality, Twine, `audit_dist.py`, and wheel/sdist smokes before publish.

§22 package/API items L can finish: clean install, build, optional-deps fail per family,
`snv.import_errors()`, small core Requires-Dist. §22 tests/docs leftover: unit CI not green on the
workflow's own extra set; `mkdocs --strict` withheld (Workstream M); certification/GPU are
`workflow_dispatch` + self-hosted (K / maintainer). Changelog is Wave 3 / M, not L.

## The ten §17 items

1. **Files moved/removed/created.** Confirmed against the diff, not the findings prose. Canonical
   docs config is `mkdocs.yaml`; `mkdocs.yml` is a git symlink (`120000`). Direct-URL
   `mujoco-locomotion-visual` extra and the unsatisfiable `all` extra are gone from metadata on
   purpose. Nothing importable was removed from `stable_envs/`.
2. **API changes.** Packaging only. `snv.__version__` stays `0.1.0` via
   `tool.setuptools.dynamic.version.attr = stable_envs.__about__.__version__`. Public constructors
   and registry semantics unchanged.
3. **Step-8 IDs affected.** None. Discovery still records a missing optional stack as that family's
   `import_errors()` entry. Core-only smoke still registers 63 envs including PushT.
4. **Targeted tests.** Re-run by me at `eca6eb2` (see Commands). Packaging module **11 passed**.
   Mkdocs nav contracts **2 passed**. Core-only `pytest tests/unit -m unit` is **not** green (17
   failed). Fetch assertion reproduced independently.
5. **Applicable certification checks.** None on this workstream. `certification.yml` is
   `workflow_dispatch` only, split by incompatible extras; GPU uses label `stable-envs-gpu`.
6. **Default-frame comparison.** N/A for semantics. Core smoke: `PushT(num_envs=2, size=32)` renders
   `(2, 32, 32, 3)` uint8 from the 3.10 wheel and the 3.12 sdist, imported from `/tmp` with
   `PYTHONPATH=` empty so site-packages is what ran.
7. **Variation / state comparison.** N/A.
8. **Performance comparison.** N/A.
9. **Resource-lifetime comparison.** N/A. `mjwarp` / `nvdiff` still import without CUDA
   implementations on the clean installs.
10. **Unexplained oracle differences.** Zero (no renderer or env file in the L delta).

## What I re-ran

Rebuild (fresh `/tmp/se9-review-l/build`, `setuptools==84.0.0`):

```text
python -m build --outdir /tmp/se9-review-l/dist
twine check                          PASSED (wheel + sdist)
audit_dist.py                        104 package modules; unexpected wheel paths []
METADATA Author                      Randall Balestriero, Sami BuGhanem
License-Expression                   MIT
Requires-Python                      >=3.10
core Requires-Dist                   gymnasium>=1.0, loguru>=0.7, numpy>=1.24,
                                     tensordict>=0.6, torch>=2.2
Provides-Extra                       no `all`; includes dev, docs, mjwarp, robotics, …
no Requires-Dist with ` @ ` or git+
```

Installs, from `/tmp` so the repo is not on `sys.path`:

```text
3.10 wheel   -> site-packages/stable_envs; 63 envs; 13 optional errors; PushT [2,32,32,3]
3.12 sdist   -> same counts and shape
editable .[dev] -> /Users/samib/stable-envs-step9-packaging-ci/stable_envs/__init__.py
                   version 0.1.0; author Randall Balestriero, Sami BuGhanem
```

Optional-miss list on both clean installs (matches the committed JSON):

```text
carracing dmcontrol fetch griddly humanoidbench jumanji lunarlander
metaworld overcooked pgx procgen robocasa sokoban
```

`import stable_envs` succeeds; PushT stays registered; fetch's recorded error is
`ModuleNotFoundError: No module named 'mujoco'` — graceful, family-scoped.

Docs / contracts / YAML:

```text
mkdocs build --config-file mkdocs.yaml --site-dir /tmp/se9-review-l/site
  -> 76 HTML pages; exit 0
mkdocs --strict
  -> 23 warnings, all research-note links into examples/results/ or scratch/
     (step4_fidelity_baseline, step7_coverage, step8_variations_*, swm_coverage, swm_matrix).
     Pre-existing content; not introduced by L's config. Withhold --strict until M (R3).
test_repo_contracts mkdocs nav       2 passed (symlink keeps K's mkdocs.yml readers working)
workflows parse with yaml.BaseLoader contents: read on all five
no PYPI_API_TOKEN / password in workflow text
ruff --no-fix scratch/step9/packaging-ci   clean
```

`test_packaging.py` uses `pytest.importorskip("tomllib")` at module scope. CPython 3.10.18 has no
`tomllib`, so the **entire 11-test module skips on the 3.10 CI cell**. 3.11/3.12 still run it. Not
the reason CI is red, but the 3.10 matrix does not actually check authors/workflows.

## CI / release gates

`testing.yml` unit job: Python 3.10/3.11/3.12, `pip install -e ".[dev]"`, then
`pytest tests/unit -m unit` with one fetch deselect, plus the packaging module. That deselect is
necessary and correct for the current assertion. **The integration job** installs `.[dev,interop]`
and runs `pytest tests/unit [tests/integration] -m "unit or integration"` **without** the deselect.
`tests/integration/` does not exist yet. The fetch test is `pytestmark = unit`, so integration CI
fails today.

`release.yml` does **not** publish on `pull_request` or `workflow_dispatch`. Gates before
`pypa/gh-action-pypi-publish`:

- tag matches `vX.Y.Z` or `X.Y.Z`
- tag equals `stable_envs/__about__.py.__version__`
- build + Twine + `audit_dist.py`
- wheel smoke + sdist smoke
- job `environment: release` + `id-token: write` only on publish
- no API token stored in the workflow

That is enough not to publish by accident. The `release` environment, Pages `github-pages`
environment, and self-hosted label `stable-envs-gpu` are **explicit external GitHub setup** (author
R4). I did not verify those environments exist on the remote; the YAML does not ship credentials.

`docs.yml` deploy is `push && refs/heads/main` only, after a successful build. `certification.yml`
is manual and profiled; GPU `max-parallel: 1` on `[self-hosted, linux, x64, stable-envs-gpu]`.
nvdiffrast is checked as a pre-provisioned import, not a PyPI extra.

Pre-commit is syntax/safety only (`check-ast` … `detect-private-key`, ruff E9/F63/F7 `--no-fix` on
`stable_envs/|tests/|scratch/step9/packaging-ci/`). It does not reformat research history.

## Required changes (L-owned, before APPROVE)

**C1. Make `testing.yml` green on `.[dev]` at this HEAD.** Minimum:

- copy the fetch deselect onto the integration job (or, better, stop re-running all `unit` tests
  there once `tests/integration/` exists);
- do not collect the four `MujocoInverted*` / `MujocoPusher` / `MujocoReacher` construction cases on
  a core-only interpreter. Either add a `mujoco-locomotion` profile job, or deselect those nodeids,
  or (F/K, preferred long-term) treat `gymnasium.error.DependencyNotInstalled` like `ImportError` in
  `_constructible`.

**C2. Optional but cheap:** stop `importorskip("tomllib")` from skipping every packaging test on
3.10 (`tomli` fallback, or `tomllib` only on 3.11+ with an explicit 3.10-safe parser).

Do not turn on `mkdocs --strict` in L. The 23 warnings are research-note paths outside `docs/`; that
is Workstream M (author R3). Do not restore an `all` extra or a nvdiffrast Requires-Dist.

## Non-blocking limitations (already disclosed)

- No `all` extra: Fetch wants `mujoco<3.12`, Overcooked wants its own interpreter, nvdiffrast is not
  expressible as PyPI metadata.
- `metaworld` / `humanoidbench` still need `--no-deps` / git. Documented.
- No `robocasa` extra (was already absent on integration). RoboCasa cannot share an interpreter with
  MJWarp (`mujoco==3.3.1` assert).
- `mkdocs.yml` symlink stays until K reads `mkdocs.yaml` (author R1).
- Fetch "missing backend" unit test should accept missing `mujoco` as well as `mujoco_warp` (author
  R2 / K).

## Commands and tails

```text
$ git rev-parse HEAD
eca6eb283241e9198b04888c171f91f76cce0a9a
$ git merge-base --is-ancestor 4feceba HEAD && echo yes
yes
$ python -m twine check /tmp/se9-review-l/dist/*
Checking .../stable_envs-0.1.0-py3-none-any.whl: PASSED
Checking .../stable_envs-0.1.0.tar.gz: PASSED
$ python scratch/step9/packaging-ci/audit_dist.py /tmp/se9-review-l/dist
audited stable_envs-0.1.0-py3-none-any.whl and stable_envs-0.1.0.tar.gz: 104 package modules

$ (cd /tmp && PYTHONPATH= /tmp/se9-l-py310/bin/python scratch/.../smoke_installed.py)
{"distribution_version": "0.1.0", "optional_import_errors": [13 names],
 "package": "/private/tmp/se9-l-py310/lib/python3.10/site-packages/stable_envs/__init__.py",
 "pusht_shape": [2, 32, 32, 3], "registered_envs": 63}

$ (cd /tmp && PYTHONPATH= /tmp/se9-l-py310/bin/python -c "… import_errors()['fetch'] …")
fetch_err= ModuleNotFoundError: No module named 'mujoco'
unit_assertion FAIL

$ /tmp/se9-l-dev/bin/python -m pytest tests/unit -m unit --deselect …fetch raytracer…
17 failed, 386 passed, 92 skipped, 1326 deselected in 5.81s
  (1 fetch + 16 MujocoInvertedDoublePendulum/Pendulum/Pusher/Reacher construction)

$ MUJOCO_GL=glfw se6-venv pytest tests/unit/test_envs_mujoco.py -k "MujocoInverted or MujocoPusher or MujocoReacher"
20 passed, 109 deselected in 0.44s
```
