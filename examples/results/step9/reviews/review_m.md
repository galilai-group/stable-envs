APPROVE

# Step 9 — independent review of Workstream M (docs + examples)

**Reviewed** `step9/docs-examples` at **`1f23f518e4c81605fc1237e507b481f490759812`** (author delivery
`fbf776e`, then `git merge step9/integration` at `4feceba`). M-attributable delta:
`git diff step9/integration...HEAD` (33 files, +1,296/−1,206). Step-8 behavioural baseline
`21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`. Latest integration merge is an ancestor of HEAD
(`INT_exit=0`); Workstream J `c9e94dc` is **not** (`J_exit=1`).
**Reviewer host** Apple M5 Mac `cis1b1zs2j3.ad.brown.edu`, local CPU only, no CUDA, no Oscar.
`MUJOCO_GL=glfw`, `-p no:cacheprovider`, interpreter `/Users/samib/se6-venv`. Nothing was edited,
stashed, reset, rebased or pushed except this review file.

## Verdict

**APPROVE.** After the latest integration merge the OSS-facing README, eight new concept/listing
pages, and four named examples describe the frozen Wave-0 public API accurately: class-first
`snv.PushT(num_envs=..., device=...)`, stateful `step` vs functional `transition`/`unroll`,
`close`/context manager, registry `make`/`spec`/`list_envs`, TensorDict State + specs, SWM ownership
split, and the honest native-`World(env=...)` limitation. Authors on the user-facing pages are
**Randall Balestriero** and **Sami BuGhanem**. The four examples run and print `--help` using only
`snv.*` and `stable_envs.interop`. Library code, tests, and packaging were not rewritten. The one
ownership exception is a nav-only `mkdocs.yml` edit, already disclosed as coordinator/L additive
conflict; it is not a product defect and does not block.

Nits below do not require a resubmission.

## M-attributable delta

`git diff --name-status step9/integration...HEAD` is 33 paths, all under `README.md`, `docs/`,
`examples/`, plus `mkdocs.yml` and this workstream's `examples/results/step9/docs-examples/findings.md`.

Created (eight docs pages + four examples, matching plan §14/M):

- `docs/quickstart.md`, `docs/environments.md`
- `docs/concepts/{state,dynamics,lifecycle,registry,worldmodel,limitations}.md`
- `examples/{render,batched_env,functional_dynamics,worldmodel}.py`

Rewritten / path-refreshed: `README.md`, `docs/index.md`, `docs/api/{env,envs,compat,style}.md`,
`docs/adding-an-env.md`, `docs/concepts/{physics,fidelity-tiers,variation-spaces,rasterizer}.md`,
`docs/benchmarks.md` (methodology prepended; measured tables kept), `docs/testing.md`,
`examples/README.md`, `examples/swm_compat/README.md`, four research notes named in findings.

`git diff --name-only step9/integration...HEAD -- stable_envs/env.py stable_envs/dynamics.py
stable_envs/specs.py stable_envs/registry.py stable_envs/_config.py stable_envs/__init__.py
pyproject.toml tests/ .github setup.py setup.cfg MANIFEST.in` is empty.

The sole shared/config file is `mkdocs.yml` (nav entries for the eight new pages + three stale API
titles `core.env` → `env`, `core.style` → `_style`, `compat.worldmodel` → `interop`). Theme, plugins,
validation, CI, and `pyproject.toml` are untouched. Findings R1 already hands this file to L.

Both M commits are `sami-bg <samibghanem@gmail.com>` author and committer. `git log --format=%B
step9/integration..HEAD` has **no** `Co-Authored-By`, `Generated with`, or other AI trailers.

## The §17 items — my own numbers

1. **Files moved/removed/created.** Nothing moved, nothing removed. Created/rewritten as the delta
   above. Confirmed against `git diff --name-status`, not the prose.

2. **API changes.** None in library code. Docs/examples now describe the Wave-0 contract. Deprecated
   names (`StableEnv`, `REGISTRY`, `batch=`, `env.rollout(policy)`) are named as not-the-happy-path
   and, for `rollout`, as belonging to `stable-worldmodel`.

3. **Step-8 IDs affected.** None. Documentation only.

4. **Targeted tests.** Re-run by me (see Evidence). Four new scripts: **12 passed**. Nav existence +
   reachability: **2 passed**. Full argparse `--help`: **35 collected**; the four new scripts pass;
   the only persistent failure is pre-existing robocasa (below). Citation scanner: **18** unresolved
   `scratch/` paths, none cited from M's eight new pages.

5. **Applicable certification checks.** None. No renderer, dynamics, or registry behaviour changed.

6. **Default-frame comparison.** n/a. CPU smoke of `examples/render.py` produced `(2, 96, 96, 3)
   uint8` and is not an oracle.

7. **Variation / state comparison.** n/a.

8. **Performance comparison.** n/a. Methodology is documented in `docs/benchmarks.md`; harnesses
   remain Workstream N / existing `examples/bench_*.py`.

9. **Resource-lifetime comparison.** n/a. Lifecycle/`close`/context manager are documented and
   exercised: `examples/batched_env.py` printed `closed=True` after the `with` block.

10. **Unexplained oracle differences.** Zero (no behavioural patch).

## Checklist against plan §14/M, §22 Docs, WAVE0, AGENT_COMMON

### Public API (WAVE0 §1–2, plan §2)

README / quickstart / `docs/index.md` use only `import stable_envs as snv`, `snv.PushT(num_envs=...,
device=...)`, `snv.make("swm/PushT-v1", ...)`, `snv.list_envs` / `snv.spec` / `snv.set` /
`snv.get_config`. Concrete classes are canonical; strings are secondary. I executed the README
snippet against live `snv.PushT`: `action_spec.sample(4, env.rng)` returns a bounded `(4, 2)`
action, `transition` does not mutate `env.get_state()`, `step` returns `['state']` only (matches
`docs/concepts/state.md`: PushT omits fabricated reward/done). `snv.__all__` is `Env, State,
StepResult, Spec, CompositeSpec, Dynamics, make, spec, list_envs, list_families, import_errors,
set, get_config, SWM_IDS, __version__`. Nothing from `render/`, `testing/`, or `interop.py` is
advertised at top level. `get_config()` keys are exactly `verbose`, `default_device`,
`warn_on_deprecated`.

### Authors

User-facing pages name **Randall Balestriero** and **Sami BuGhanem** (`README.md` line 13,
`docs/index.md` line 9). `pyproject.toml` has no `authors` field; packaging is Workstream L, not M.

### Stateful vs functional (WAVE0 §5, plan §2)

`docs/concepts/dynamics.md` and `examples/functional_dynamics.py` match the contract: `step` writes
`env.state`; `transition`/`unroll` do not; `unroll` is known actions with time first, length `T`,
initial state omitted. Live CPU run on PushT:

```
transition left env.state untouched
transition(state, a) == step(a)['state']  (bit-identical on this device)
unroll actions (4, 2, 2) -> traj batch [4, 2]
unroll == repeated transition
```

CraftaxClassic (renderer-only) raises the WAVE0-style refusal (`PushT has dynamics;
CraftaxClassic does not`) and exits 0. There is no `env.rollout(policy=...)` in the four examples
or the eight new pages.

### Lifecycle (WAVE0 §5, plan §2 Resource ownership)

`docs/concepts/lifecycle.md` documents `close()`, idempotence, context manager, `num_envs`,
`torch.device`, short batches, and `reset(seed=)` re-seeding. `env.py` `_check_open` is called from
`reset` and from `render_states_arg` (so `render` refuses a closed env). `examples/batched_env.py`
constructed `num_envs=4`, rendered a short batch of 2, and printed `closed=True`. Correctness is
not documented as depending on `__del__`.

### Registry (WAVE0 §6)

`docs/concepts/registry.md` / `docs/environments.md` document `make` / `spec` / `list_envs(family=)`
/ `list_families` / `import_errors`, auto-discovery skipping `_`-prefixed modules, SWM aliases, and
the family-filter caveat (`list_envs` keys on the last module segment: `"_craftax_full"`,
`"ogb_manip"`, not `"ogbench"` for manipulation). Live se6-venv: **81** registered classes, **14**
families including `_craftax_full` / `ogb_manip` / `navigation`; Fetch is in `import_errors`.
`len(snv.SWM_IDS) == 42`. `snv.spec("swm/PushT-v1")["name"] == "PushT"`. Symbolic
`swm/CraftaxSymbolic-v1` raises `KeyError` with the two symbolic ids named.

### State / specs (WAVE0 §3–4)

`docs/concepts/state.md` matches `env.py`: public TensorDict grouped by entity, `pack`/`unpack` as
the unchanged Step-8 float32 row, `unpack` copies, hidden fields come back as zeros, one `Spec`
dataclass plus `CompositeSpec`, `variation_spec` flat, `variation_space` kept for the harness.
PushT smoke: state keys `['agent', 'block']`, `pack -> (4, 5)` with
`state_names=['agent_x','agent_y','block_x','block_y','block_angle']`,
`observation_spec` `(96,96,3) uint8`, `action_spec` `(2,) [0, 512]`, `variation_spec.shape (29,)`.

### SWM ownership and native `World(env=...)` (plan §2 Interoperability, §22)

`docs/concepts/worldmodel.md`, `docs/api/compat.md`, `docs/concepts/limitations.md`, and
`examples/worldmodel.py` all state that native `swm.World(env=snv.PushT(...))` is **not** a public
API. That remains true at this HEAD: `git merge-base --is-ancestor c9e94dc HEAD` exits 1. The
script prints `NATIVE WORLD LIMITATION` and does not construct a `World`. Frozen surface exercised:
`snv.make("swm/PushT-v1")`, `snv.SWM_IDS`, `interop.StableEnvGym("PushT", device="cpu")` (reset obs
`(5,)`, render `(96,96,3)`), `swap_renderer` skipped honestly without `gym_pusht`. Public interop
names used (`StableEnvGym`, `swap_renderer`, `pusht_state`, `VariationMap`,
`style_from_variation`, `variation_from_style`, `check_mapping`) all exist on
`stable_envs.interop`.

### Device / backend limitations (plan §22, AGENT_COMMON device honesty)

`docs/concepts/limitations.md` documents optional extras, Overcooked's own interpreter,
`STABLE_ENVS_HUMANOIDBENCH=1`, Fetch no CPU backend, OGBCube/OGBScene/PFRocketLanding refusing
`device="cpu"`, DMC CPU3D vs MJWarp, nvdiff x86 sparse-stage, Fetch `background.color` partial,
certification pins (`mujoco==3.11.0` + `dm_control==1.0.44`), and the
`HumanoidmazeGiantNavigate` `DOCUMENTED_LIMITATION`. README says CUDA is required for 3-D / most
MuJoCo families and that PushT/classic/navigation run on CPU. Examples refuse a requested CUDA
device when `torch.cuda.is_available()` is false rather than silently falling back.

### Benchmark methodology (plan §14/N, §22)

`docs/benchmarks.md` opens with a methodology section: paired same-job/same-card, native
supersample, both sides the same job, host-bound `cpu%`, generated file as source, Step 9 not an
optimisation phase, `benchmarks/` owed to N, `examples/bench_*.py` until then. Measured tables
kept.

### Current module paths

User-facing current-source citations were refreshed: `navigation.py` (was `swm_nav.py` /
`simple_point_maze.py`), `_craftax_full.py` + `craftax_full` alias, `stable_envs.env` (was
`stable_envs.core.env`), `stable_envs.interop` (was `compat.worldmodel`), `stable_envs._style`.
Generated research pages (`swm_matrix.md`, `step7_coverage.md`, …) still contain historical
`compat.worldmodel` / `se.make` strings; findings R6 already says rewriting them by hand would
drift from the generator. That is not a Wave-0 API lie on the getting-started path.

### Examples use only public `snv` APIs

Module-level imports of the four scripts are `argparse`/`sys` only; `stable_envs` is imported
inside `main`. No `se.make`, no `batch=`, no `StableEnv`, no `REGISTRY`, no `env.rollout`, no
`stable_envs.core` / `compat` / `envs` / `render`. `worldmodel.py` imports `stable_envs.interop`
(WAVE0: not at `snv.*` top level; this is the documented public face). `--help` does not construct
an env.

## Evidence — tests and smokes I ran

Host `cis1b1zs2j3.ad.brown.edu`, `/Users/samib/se6-venv`, `PYTHONPATH` = this worktree,
`MUJOCO_GL=glfw`, `SDL_VIDEODRIVER=dummy`, `-p no:cacheprovider`.

```
pytest tests/unit/test_examples.py \
  -k "render.py or batched_env.py or functional_dynamics.py or worldmodel.py"
-> 12 passed, 111 deselected in 0.29s

pytest tests/unit/test_repo_contracts.py::test_every_page_in_the_mkdocs_nav_exists \
       tests/unit/test_repo_contracts.py::test_every_docs_page_is_reachable_from_the_nav
-> 2 passed in 0.34s

pytest tests/unit/test_examples.py::test_every_argparse_script_can_print_its_own_help
-> 35 collected. Four new scripts PASSED. See pre-existing failures below.

CPU smokes (all exit 0):
  examples/render.py --env PushT --num-envs 2 --device cpu
    state batch [2] keys ['agent', 'block']; pixels (2, 96, 96, 3) uint8 cpu
  examples/batched_env.py --num-envs 4 --short 2 --device cpu
    81 envs, closed=True, short batch frames (2, 96, 96, 3)
  examples/functional_dynamics.py --env PushT --num-envs 2 --horizon 4 --device cpu
    bit-identical transition==step; unroll == repeated transition
  examples/functional_dynamics.py --env CraftaxClassic --num-envs 2 --device cpu
    renderer-only refusal, WAVE0 wording
  examples/worldmodel.py --mode ids
    SWM_IDS 42; make('swm/PushT-v1') -> PushT; symbolic refused; prints NATIVE WORLD LIMITATION
  examples/worldmodel.py --mode gym --device cpu
    StableEnvGym reset+render; prints the limitation; does not call World(env=)
  examples/worldmodel.py --mode swap --device cpu
    exit 0: "gym_pusht is optional ... NamespaceNotFound"
```

Did not run the full unit suite (K owns tests). Did not run GPU certification.

## Pre-existing failures, distinguished from M

**Robocasa `--help` (pre-existing, not M).**
`examples/robocasa_visual/render_mjwarp.py --help` exits 1 with
`FileNotFoundError: ... '/mnt/vast/home/randall.balestriero/code/stable-envs/stable_envs/render/mjwarp.py'`.
That path is not in the M delta. Handoff §8 already names this as a Step-9-predating unit failure.

**Citation scanner (pre-existing, not M).**
`test_every_cited_scratch_path_resolves` failed with **18** unresolved `scratch/` paths (Step-6
Oscar paths, `scratch/carracing/logs/`, `scratch/robocasa/out/corpus`, brace-glob cites in other
workstreams' findings, `docs/research/step9_handoff.md` `$SCR` prose). None of the 18 are cited
from M's eight new pages or four examples. Same deselect the coordinator unit-suite command already
uses.

**`bench_gpu_vs_cpu.py --help` timeout in the contended sweep is not a M defect.** The full
argparse sweep, run in parallel with other work, timed out that one script at 300 s. A standalone
retry printed usage and exited 0 in 2.6 s. M did not touch the file. Effective help sweep: 34/35
pass, the remaining failure is robocasa.

## Ownership exception: `mkdocs.yml`

M **did** edit `mkdocs.yml`. Plan §14/M owns `README.md`, `docs/`, `examples/` — not the MkDocs
config. Handoff procedure 5.5: additive conflicts (mkdocs nav) are the coordinator's. The diff is
nav-only (eight new pages + three stale API titles). Without those nav keys,
`test_every_docs_page_is_reachable_from_the_nav` would fail on M's new pages. Findings R1 already
asks L to take the file back. I am not treating this as APPROVE-WITH-CHANGES: reverting it would
make the new pages unreachable until the coordinator copies the same lines, which is what merge
already does for additive nav.

M did **not** edit core, tests, or packaging (`pyproject.toml` / setup / CI).

## Nits (do not block)

1. `docs/api/rollout.md` still presents `env.sample(steps=..., batch=..., policy=...)` as a current
   API page. Primary pages correctly forbid `env.rollout(policy)` and
   `docs/concepts/worldmodel.md` already says `stable_envs.rollout` remains a Step-8 convenience
   until Wave 3. A one-line banner on `api/rollout.md` would make that consistent; K/O can add it.
2. `examples/worldmodel.py --mode gym` triggers the one-per-process `batch=` deprecation warning
   from `StableEnvGym`'s constructor. The example does not pass `batch=`; that spelling lives in
   interop (J).
3. Historical `examples/README.md` sections still document `se.make` / `env.sample` for the
   measurement harnesses they describe. Findings R4 already scoped those out of M's four named
   scripts.
4. Generated research pages still say `compat.worldmodel` / `se.make` (findings R6).

## Suggested coordinator merge note

On merge: take M's `mkdocs.yml` nav keys as the additive conflict (or copy them onto L's branch).
When J lands native `World(env=...)`, replace the limitation block in
`docs/concepts/worldmodel.md`, `docs/api/compat.md`, and `examples/worldmodel.py` (findings R2) —
do not invent it before that merge. Do not treat the robocasa `--help` or citation-scanner red as
M's.
