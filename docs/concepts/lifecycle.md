# Lifecycle

Use an environment as a context manager or call `close()` explicitly. Close is
idempotent; correctness must not depend on a finalizer. Closed instances refuse
further operations rather than silently creating a new backend.

```python
import stable_envs as snv

with snv.CartPole(num_envs=2, device="cpu") as env:
    state = env.reset(seed=17)
    frame = env.render(state)
assert env.closed
```

An instance releases only resources it owns. Shared process-lifetime rendering
contexts/caches must not be destroyed by a single instance's close path. Follow
the resource guard and backend ownership rules in the [renderer guide](../adding/renderer.md).

`num_envs` is the simulator batch size; `len(env) == env.num_envs`.
Restoration must satisfy that backend's batch contract. Supported renderers can
accept smaller batches; this does not imply that full-State restoration accepts
any batch size. `out=` requires the documented contiguous uint8 output layout.
Public rendering accepts full State or the current live State; compact rows are
private renderer inputs.

Reset seeding is source-specific. Do not assume `reset(None)` always continues a
single generator: some sources reseed reset/Variation generators with fresh
entropy. State and reset anchors must capture the promised replay scope.

A registration's own TimeLimit belongs to its lifecycle State. A caller-supplied
training/collection cap belongs to that caller's wrapper. Trace construction to
decide ownership, retain elapsed steps where needed, and compare termination and
truncation separately. See the [task guide](../adding/task.md).
