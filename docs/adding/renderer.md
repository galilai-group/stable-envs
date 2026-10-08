# Adding a renderer

Use this guide for pixels, cameras, a visual backend or rendering implementation
that preserves simulator/task semantics. A collision-shape or physics change is
an [environment](environment.md) change even if it also changes the image.

## Responsibility and minimum interface

The rendering path is:

```text
full State → private RenderState → Frame → downstream learner packaging
```

`Env.render(state=None, ...)` is the public entry point. Families supply
`_pack_render_state`, the relevant private decode/project helpers, and `_render`
or a `scene()` consumed by the existing renderer. `_get_render_state` and
`_set_render_state` serve renderer internals. Compact column labels/width live in
`state_names`/`state_dim` (readable as `render_state_names`/`render_state_dim`).
These describe renderer inputs, not the full simulator schema.

A Frame is uint8 `(B,H,W,3)` on the declared device. Honor the existing size,
Variation, batch and `out=` contracts where supported; reject unsupported backends
explicitly. `_unpack_render_state` may expose a partial named projection; it must
not invent velocities/RNG to impersonate exhaustive State. No public
`pack_render_state`/`unpack_render_state` methods are part of the current API.

Reuse the authoritative renderer or the accepted native rendering path. CartPole's
accepted implementation in `stable_envs/envs/classic_control/envs.py` uses the native
`gfxdraw` path. Other families delegate to MuJoCo/MJWarp. Scene2D is optional;
forcing an upstream tile or mesh renderer through a second scene representation
is not required. Renderer internals and fidelity declarations are documented in
[the Env API](../api/env.md) and [fidelity tiers](../concepts/fidelity-tiers.md).

## Complete example: accepted full/private equality and nonmutation

This contributor-level example deliberately uses private renderer entry points
to check the boundary. Applications should call `render(full_state)`.

```python
import torch
import stable_envs as se

with se.CartPole(num_envs=2, device="cpu") as env:
    saved = env.reset(seed=17)
    held, live = saved.clone(), env.get_state()
    rows = env._pack_render_state(saved)
    full = env.render(saved)
    compact = env._render(rows)
    assert full.dtype == torch.uint8
    assert full.shape[0] == 2 and full.shape[-1] == 3
    assert torch.equal(full, compact)
    assert torch.equal(full, env.render(saved))
    assert (saved == held).all()
    assert (env.get_state() == live).all()
    for incomplete in (rows, env._unpack_render_state(rows)):
        try:
            env.set_state(incomplete)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("lossy projection restored dynamics")
```

For a new renderer, keep this test and substitute its accepted source projection
and renderer implementation. The required baseline comparison is additional:
full/private equality alone could mean two identically incorrect frames.

## Identify authority

Pin the official rendering code, engine/assets, camera configuration, resolution,
color conversion, draw order and backend dependencies. Read source before
implementing, then use runtime comparisons to verify the interpretation. Preserve
existing frozen renderer definitions, corpus layouts and baselines. If an actual
renderer defect is found, propose a reviewed fix with evidence; do not silently
change fixtures or thresholds to obtain a pass.

## Fast development checks

- Full State and private RenderState produce equal frames under the same Variation.
- Repeated rendering is deterministic and changes neither live simulator State nor
  supplied snapshots, RNG, task goals or subsequent transition results.
- Shapes, uint8 dtype, device, short batches, chunking, `out=` and invalid inputs
  obey the supported contracts; unsupported combinations explicitly refuse.
- Batch permutation and chunk-size invariance hold where promised.
- Visual-only Variation affects pixels without changing physics. Render, then
  step against an unrendered control to detect hidden simulator mutation.
- Context/buffer creation is bounded and ownership is correct. `close()` is
  idempotent. Do not destroy shared process-lifetime EGL contexts or renderers
  from a per-environment close path; obey the existing resource guard.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/unit/test_native_classic_semantics.py tests/unit/test_full_state_contract.py
python -m pytest -q tests/unit/test_raster2d_torch_backend.py
ruff check stable_envs tests
ruff format --check stable_envs tests
git diff --check
```

Use the new family's targeted tests as well. Existing CartPole tests are a worked
example, not coverage of an unrelated renderer.

## Final certification

Compare against both the official reference and the established frozen baseline.
Use exact pixels unless the existing fidelity contract explicitly allows another
criterion. Report the criterion and its scope; do not broaden an interior-pixel
or visual-tier result into all-pixel equality. Include representative tasks,
contact poses, cameras and supported appearance configurations.

Run each claimed CPU/GPU backend separately. Exact CPU semantic physics acceptance
and GPU rendering acceptance can use separate dependency profiles, but their
claims must stay separate. GPU rendering does not certify physics on that GPU
profile. A CPU-backend refusal is an honest limitation, not a CPU rendering pass.

```bash
python -m pytest -m 'regression and not gpu'
# On a compatible allocated GPU machine:
python -m pytest -m 'regression and gpu'
```

Use the repository's crash-isolated runner for suites that need it, as described
in [Testing](../testing.md). Also run the family-specific full/private/nonmutation
and frozen-baseline gate. A generic suite cannot certify untested task defaults.
Retain raw records and an exact terminal marker; hash the immutable remote source
stage and downloaded evidence when executing remotely. A queued job or skipped
required GPU test leaves the registration incomplete.

## Evidence, definition of done, and review

Follow the [common evidence requirements](index.md#evidence-definition-of-done-and-review).
Record source/assets/profile hashes, device/backend, resolutions, tasks, Variation,
comparison criteria, batch/chunk sizes and resource counts. Keep reference,
frozen-baseline and candidate outputs identifiable. Report unsupported scope.

Done means all claimed renderer paths meet their established fidelity and
nonmutation contracts and the simulator/task acceptance is independently valid.
Common failures are changing physics while rendering a snapshot, assuming a
compact pose is exhaustive State, releasing shared contexts, introducing batch
order dependence, or making a new physics claim from a pixel result.

PR-review checklist:

- [ ] State projection and renderer input layout are explicit and independently tested.
- [ ] Full/private equality, determinism and live/snapshot nonmutation pass.
- [ ] Official/frozen baseline results preserve the established criteria.
- [ ] Task/camera/Variation coverage matches the claim.
- [ ] Batch/chunk/resource lifecycle and backend refusals are covered.
- [ ] CPU/GPU rendering and physics claims are separately scoped.
- [ ] No frozen definition or threshold changed merely to make a gate pass.
