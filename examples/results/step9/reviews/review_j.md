APPROVE

# Independent review — Step 9 Workstream J

Reviewed clean `step9/interop` HEAD `c9e94dc389632b8a9cfacdfe6c55881516c7dde0`
against integration `d3cb4a81cced123d767661bf485f0c5059a0dd73` and the frozen Step-8
baseline `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`.

## Scope and delta

The integration-to-HEAD delta is exactly the claimed three files:

- added `examples/results/step9/interop/findings.md`;
- modified `stable_envs/interop.py`;
- added `tests/integration/test_worldmodel.py`.

`git diff --shortstat d3cb4a8..c9e94dc` reports 323 insertions and 13 deletions.
`git diff --check` is clean. No implementation under `_interop`, environment, renderer, rollout,
policy, planner, MPC, packaging, or documentation ownership changed in this delta.

## Binding-contract review

### Frozen SWM IDs

I independently loaded `examples/results/step8/swm/gate_f_matrix.json` rather than relying on the
findings. It has exactly 42 rows: 40 `pixel` and 2 `symbolic`. The complete
`swm_id -> tuple(stable_envs_cells)` mapping is exactly equal to `stable_envs.registry.SWM_IDS` and
the public facade exports that same dictionary object. `canonical_names()` returns the frozen tuple
for every row. The symbolic tuple is exactly:

- `swm/CraftaxClassicSymbolic-v1`
- `swm/CraftaxSymbolic-v1`

For every frozen class available in the active SWM interpreter, the top-level `stable_envs` class,
`REGISTRY` entry, and `_resolve()` result were the same class object. Multi-cell family IDs remain
honest tuples rather than silently selecting one cell. Missing optional families remain explicit
registry/import errors, not altered aliases.

### Dependency direction

`stable_envs.interop` imports only the pure registry mapping at import time. A fresh-process probe
confirmed that importing `stable_envs` and the five core modules loads neither
`stable_envs._interop.worldmodel` nor any `stable_worldmodel` module. Requesting an adapter symbol
loads the private adapter lazily. The AST dependency-direction suite also passed. There is no SWM
dependency leak into core and no circular dependency.

### VariationMap and compatibility imports

The public facade and the Wave-3 compatibility paths expose the same `VariationMap` class and
`VARIATION_MAPS` dictionary objects. Direct import probes confirmed that all five old paths alias
their private replacement module objects:

- `stable_envs.compat.worldmodel`
- `stable_envs.compat.variation_maps`
- `stable_envs.compat.variation_maps.dmc_control`
- `stable_envs.compat.variation_maps.fetch`
- `stable_envs.compat.variation_maps.newcells`

The focused PushT style-to-variation-to-style round trip is byte-identical, and the broader existing
VariationMap suites pass. The compatibility imports therefore remain available until Wave 3.

### `swap_renderer`

The live test constructs the official `swm/PushT-v1`, retains the original unwrapped SWM type and
render method, installs only the stable-envs renderer, obtains a `(32, 32, 3)` `uint8` frame, and
then completes an official environment step with scalar reward and valid termination flags. This is
an honest renderer replacement; official dynamics and task semantics remain owned upstream.

### Registered-gym SWM path

A real `stable_worldmodel.World` successfully constructs the registered
`stable_envs_step9/PushT-v0`, resets it, reaches a `StableEnvGym` instance through the SWM wrapper
stack, and returns `(1, 1, 32, 32, 3)` `uint8` pixels. This verifies the supported installed-SWM
entry path rather than only testing `gymnasium.make` in isolation.

### Native `World(env=...)` limitation

Installed stable-worldmodel is commit `176d54532b3c0a251a4f509b74f2fdacc143a63d`. Direct inspection
shows:

- `World` requires positional `env_name: str`;
- its public signature has no `env` parameter;
- `World.__init__` builds `partial(_make_env, env_name, ...)`;
- `_make_env` unconditionally calls `gym.make(env_name, ...)`;
- directly calling `World(env=snv.PushT(...), ...)` raises
  `TypeError: World.__init__() missing 1 required positional argument: 'env_name'`.

The single native-path skip is therefore narrow and truthful for the installed upstream. It is not
masking a reachable stable-envs failure. If upstream exposes an explicit `env` parameter, the test
stops skipping and exercises construction, reset, pixel shape, and cleanup.

### Ownership boundary

The reviewed delta adds facade exports and integration tests only. It does not add or move policy
rollouts, MPC, CEM/MPPI, planning, dataset collection, evaluation, or policy semantics into
stable-envs. References to SWM planning/evaluation describe consumers of the adapter; they do not
implement those responsibilities.

## Re-run evidence

Apple CPU, Python 3.12:

- `se8-vm-venv`: `tests/integration/test_worldmodel.py` — **5 passed, 1 skipped**; the only skip is
  the proven native `World(env=...)` upstream limitation.
- `se6-venv`: `tests/integration/test_worldmodel.py` — **3 passed, 3 skipped**; all three skips are
  because stable-worldmodel is not installed, while the pure ID and VariationMap facade tests pass.
- `se8-vm-venv`: `tests/swm/test_gate_f_matrix.py` — **5 passed**.
- `se8-vm-venv`: foundation import and registry tests — **71 passed, 1 skipped**; the skip records
  the interpreter's optional-family set mismatch.
- `se8-vm-venv`: partial/new-cell VariationMap tests — **81 passed, 8 skipped**; skips are explicit
  local OGB/PyFlyt dependency or GL gaps.

Resource accounting after the focused integration runs reports zero live renderers and zero live EGL
contexts.

## Verdict

The Workstream J delta satisfies plan §14/J and the Wave-0 dependency and compatibility contracts.
No corrective changes are required before integration.
