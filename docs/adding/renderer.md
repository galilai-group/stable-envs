# Adding a renderer

Use this guide for pixels, cameras or a rendering backend that keeps the simulator and task unchanged.
A change to collision shapes or physics is an [environment](environment.md) change, even if the image
also changes.

## Interface

```text
full State → private RenderState → Frame → downstream learner packaging
```

`Env.render(state=None, ...)` is the public entry point. A family supplies:

- `_pack_render_state` and the private decode helpers
- `_render`, or a `scene()` that an existing renderer consumes
- `state_names` / `state_dim` (also `render_state_names` / `render_state_dim`), the renderer input
  layout. This is not the full State schema.

`_get_render_state` and `_set_render_state` are renderer internals. `_unpack_render_state` may return a
partial named projection. It must not invent velocities or RNG.

A Frame is uint8 `(B,H,W,3)` on the declared device. Support the size, Variation, batch and `out=`
contracts where possible. Refuse unsupported backends explicitly.

Reuse the upstream renderer or an existing rendering path. CartPole in
`stable_envs/envs/classic_control/envs.py` uses `gfxdraw`. MuJoCo families use MuJoCo or MJWarp.
`Scene2D` is optional. See [the Env API](../api/env.md) and [Fidelity tiers](../concepts/fidelity-tiers.md).

## Example: full/private equality and no mutation

This example uses private renderer methods to check the boundary. Applications call
`render(full_state)`.

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
            raise AssertionError("a compact projection restored the simulator")
```

Keep this test for a new renderer. You also need a comparison with the reference, because two equal
frames can both be wrong.

## Identify authority

Pin the official rendering code, engine and assets, camera, resolution, colour conversion, draw order
and backend dependencies. Read the source, then check your reading with runtime comparisons. Do not
change frozen renderer definitions, corpus layouts or baselines. If you find a real defect, propose a
reviewed fix with evidence.

## Local checks

- Full State and private RenderState give equal frames under the same Variation.
- Rendering is deterministic and changes neither the live State, the supplied snapshot, RNG, goals nor
  later transitions.
- Shapes, dtype, device, short batches, chunking, `out=` and invalid inputs follow the contract.
- Batch order and chunk size do not change frames, where the renderer promises this.
- Visual-only Variation changes pixels and not physics. To detect hidden mutation, render, then step,
  and compare with an unrendered control.
- Context and buffer creation is bounded. `close()` is idempotent. Do not destroy shared EGL contexts
  or renderers from a per-environment close path.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/unit/test_native_classic_semantics.py tests/unit/test_full_state_contract.py
python -m pytest -q tests/unit/test_raster2d_torch_backend.py
ruff check stable_envs tests
ruff format --check stable_envs tests
git diff --check
```

## Final acceptance

- Compare against the official reference and the frozen baseline.
- Use exact pixels unless the class's fidelity tier allows another criterion. State the criterion.
- Include representative tasks, contact poses, cameras and appearance settings.
- Run each claimed CPU and GPU backend separately.

GPU rendering results do not certify physics under the GPU profile. If a backend refuses CPU, record
that as a limit. It does not count as a CPU pass.

```bash
python -m pytest -m 'regression and not gpu'
# on a CUDA machine:
python -m pytest -m 'regression and gpu'
```

Run suites that initialize native renderers through the supervisor. See [Testing](../testing.md) and
[Definition of done](index.md#definition-of-done).

Common mistakes:

- changing physics while rendering a snapshot
- treating a compact pose as full State
- releasing shared contexts
- making frames depend on batch order
- claiming physics from a pixel result

## Review checklist

- [ ] The State projection and renderer input layout are explicit and tested.
- [ ] Full/private equality, determinism and no-mutation tests pass.
- [ ] Reference and frozen-baseline comparisons keep the existing criteria.
- [ ] Task, camera and Variation coverage matches the claim.
- [ ] Batch, chunk, resource lifecycle and backend refusals are tested.
- [ ] CPU and GPU rendering claims and physics claims are stated separately.
- [ ] No frozen definition or threshold changed to make a test pass.
