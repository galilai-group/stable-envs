# Step 9 — Wave 2, Workstream M: docs + examples

**Workstream:** `step9/docs-examples` · **Parent:** `step9/integration` @ `9cb3261`
(Workstream F MuJoCo/non-42 wrappers merged APPROVE; Workstream J interop **not** merged) ·
**Step-8 baseline:** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`
**Host:** Apple M5 Mac, CPU. Interpreter: `/Users/samib/se6-venv`.
**Result: PATCH.** OSS-facing README/quickstart and concept docs on the Wave-0 public API;
four examples that use only `snv.*` / `stable_envs.interop`; research history preserved with
stale module paths refreshed. Workstream J native `swm.World(env=…)` is **not** documented as
shipped.

Authors named in the user-facing pages: **Randall Balestriero**, **Sami BuGhanem**.

---

## 1. Files moved / created / removed

Created:

- `docs/quickstart.md`
- `docs/environments.md`
- `docs/concepts/{state,dynamics,lifecycle,registry,worldmodel,limitations}.md`
- `examples/{render,batched_env,functional_dynamics,worldmodel}.py`
- `examples/results/step9/docs-examples/findings.md` (this file)

Rewritten / refreshed (owned): `README.md`, `docs/index.md`, `docs/api/{env,envs,compat,style}.md`,
`docs/adding-an-env.md`, `docs/concepts/{physics,fidelity-tiers,variation-spaces,rasterizer}.md`,
`docs/benchmarks.md` (methodology section prepended; measured tables kept), `docs/testing.md`
(markers note), `examples/README.md`, `examples/swm_compat/README.md`.

Stale-path refreshes in research (history kept, current source path named):

- `docs/research/swm_coverage.md` — `swm_nav.py` / `simple_point_maze.py` → `navigation.py`
- `docs/research/step8_upstream_findings.md` — `envs/craftax_full.py` → `_craftax_full.py` + alias
- `docs/research/step8_reference_provenance.md` — `stable_envs.core.env` → `stable_envs.env`
- `docs/research/step8_certification.md` — classifier table keeps both `craftax_full` spellings

Removed: **nothing.**

`mkdocs.yml` **nav only**: entries for the eight new pages so
`tests/unit/test_repo_contracts.py::test_every_docs_page_is_reachable_from_the_nav` is green.
Theme, plugins, validation, CI, and release docs remain Workstream L. L should restyle section
order if sibling `stable-*` convention differs. See Request R1.

## 2. API changes

None in library code. Docs and examples now describe the frozen Wave-0 contract:

```python
import stable_envs as snv
env = snv.PushT(num_envs=64, device="cpu")
state = env.reset(seed=0)
out = env.step(action)                 # stateful StepResult
nxt = env.transition(state, action)    # functional
traj = env.unroll(state, actions)      # (T, num_envs, …) known actions
pixels = env.render(state)
env.close()
```

Not documented as public: `env.rollout(policy)`, `swm.World(env=snv.PushT(...))`, top-level
`snv.interop`, `batch=` (deprecated, still works), `StableEnv` / `REGISTRY`.

## 3. Step-8 IDs affected

None. Documentation only. All 42 ids keep their frozen semantics.

## 4. Targeted tests

- `tests/unit/test_examples.py` — 12 passed for the four new scripts (parse, main-guard, `--help`).
  Full argparse `--help` sweep: **132 passed**, 1 failed (`examples/robocasa_visual/render_mjwarp.py`
  FileNotFoundError on a `/mnt/vast/.../mjwarp.py` path — pre-existing, not M).
- `tests/unit/test_repo_contracts.py` — nav existence + reachability **passed**. Citation scanner
  **failed** (18 pre-existing unresolved `scratch/` paths, including Step-6/Step-9 research cites
  M did not author). Deselected for the M green path.
- Example smokes (CPU, se6-venv, `PYTHONPATH` = this worktree): `render.py` PushT 2 envs;
  `batched_env.py` short batch + close; `functional_dynamics.py` bit-identical `transition==step`
  and `unroll`; `worldmodel.py --mode ids` (42 SWM_IDS + symbolic refusal + native-World limitation);
  `--mode gym` StableEnvGym CPU; `--mode swap` skips without gym_pusht; CraftaxClassic renderer-only
  refusal.

Did not run the full unit suite (K owns tests; this workstream does not edit them). Did not run GPU
certification.

## 5. Applicable certification checks

None. No renderer, dynamics, or registry behaviour changed.

## 6. Default-frame comparison

n/a (no pixels produced by this workstream except the CPU smoke of `examples/render.py`, which is
not an oracle).

## 7. Variation / state comparison

n/a.

## 8. Performance comparison

n/a. Benchmark **methodology** is documented in `docs/benchmarks.md`; harnesses remain Workstream N
(`benchmarks/`) and the existing `examples/bench_*.py` scripts.

## 9. Resource-lifetime comparison

n/a. Lifecycle/`close`/context manager are documented and exercised in `examples/batched_env.py`.

## 10. Unexplained oracle differences

Zero.

---

## Limitations (honest)

1. **Native `swm.World(env=snv.PushT(...))` is not a public API.** Rechecked after merging
   `step9/integration` @ `9cb3261` (F APPROVE): `step9/interop` is still **not** an ancestor of
   integration (`c9e94dc` vs `9cb3261`). Docs and `examples/worldmodel.py` describe only the frozen
   contract (`StableEnvGym`, `BatchedEnvPool`, `swap_renderer`, `VariationMap`, `SWM_IDS`) and print
   the limitation. Do not invent the native constructor.
2. **`list_envs(family=)` keys on the last module segment**, so `"ogbench"` / `"dmcontrol"` /
   `"_craftax_full"` do not mean what a reader might type. Documented on `docs/environments.md`.
3. **Optional families.** `snv.list_envs()` is per-interpreter. After merging F, se6-venv smoke
   saw **81** registered classes (families include `_craftax_full`, classic_control, craftax,
   crafter, dmcontrol, highway, maze2d, minigrid, minihack, mujoco_locomotion, navigation,
   ogb_manip, ogbench, pusht). Fetch is still an `import_errors` entry on that machine.
4. **`mkdocs.yml` nav.** Entries for M's eight new pages so the nav-reachability unit test is
   green, plus three stale API nav titles (`core.env` → `env`, `core.style` → `_style`,
   `compat.worldmodel` → `interop`). Theme, plugins, validation, CI remain L.
5. **Historical examples** (`train_from_states.py`, `rollout_with_policy.py`, …) still use Step-8
   spellings (`se.make`, `batch=`, `set_state`). They are measurement harnesses, not the OSS
   quickstart; Wave 3 / K can migrate them. M did not rewrite them (user: own the four named
   scripts).
6. **Generated research pages** (`swm_matrix.md`, `step7_coverage.md`, …) still contain historical
   `compat.worldmodel` / `se.make` strings. They are regenerated from frozen artifacts; rewriting
   them by hand would drift from the generator. User-facing current-source citations were updated.
7. **Pre-existing unit failures** (citation scanner on some Step-6 paths; robocasa example `--help`)
   predate this workstream and were not in scope.

## Requests

**R1 — L (packaging / mkdocs).** Theme, plugins, `mkdocs.yml` CI, and nav styling. M added nav keys
for `quickstart.md`, `environments.md`, and the six new concept pages, and renamed three API nav
titles to the current modules. Please take the file back; do not treat that diff as a product
redesign.

**R2 — J.** When native `World(env=…)` merges, replace the limitation block in
`docs/concepts/worldmodel.md`, `docs/api/compat.md`, and `examples/worldmodel.py` with the real
call. Do not invent it before that merge.

**R3 — N.** `docs/benchmarks.md` methodology points at `benchmarks/` once that workstream lands.
Until then the `examples/bench_*.py` scripts remain the measurement path.

**R4 — K / O.** Historical `examples/*.py` still say `se.make(..., batch=)` and
`stable_envs.REGISTRY`. Out of M's named ownership.

## Review readiness

Ready for independent review on docs/examples. Not a behaviour change. Suggested verdict inputs:

- README / quickstart use only `snv.PushT` / `snv.make` / `num_envs=` / `transition` / `unroll`
- `examples/worldmodel.py --mode ids` prints the native-World limitation and does not call it
- `--help` green on the four new scripts
- research notes still exist; module-move citations point at current files
