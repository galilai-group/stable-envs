# Lifecycle and resource ownership

Resource lifetime is explicit. Correctness must not depend on `__del__`.

```python
env = snv.PushT(num_envs=8, device="cpu")
state = env.reset(seed=0)
pixels = env.render(state)
env.close()
env.close()                            # idempotent
```

Or:

```python
with snv.PushT(num_envs=8, device="cpu") as env:
    state = env.reset(seed=0)
    pixels = env.render(state)
```

`close()` releases renderer resources (EGL contexts, atlases, device buffers, MJWarp/nvdiff runners). A closed env refuses `reset` / `render` / `step` with a message that says to build a new one, rather than drawing through a backend that no longer exists.

`env.closed` is the flag. There is no "reopen".

## Constructor

```python
snv.PushT(num_envs=64, device="cuda", size=None, supersample=1, backend="auto", seed=0)
```

- `num_envs` replaces Step-8 `batch` (`batch=` still works and warns once per process until Wave 3).
- `env.device` is a `torch.device`.
- `env.size` is the frame size; `None` means the class default (`default_size`).
- `len(env) == num_envs`.

Short batches (`n <= num_envs`) keep working exactly as `set_state` documents: the last batch of an epoch is rendered, not dropped.

## What `reset` does

`reset(seed=0)` draws `num_envs` states from the env's own reset distribution, writes them into the held buffer, and returns the public `State`. Passing `seed` re-seeds `env.rng`, so `reset(seed=0)` twice is the same draw. Omitting `seed` continues the generator.

## Render

`render` accepts a `State`, packed rows, or nothing (the currently held state). The output is `(n, H, W, 3)` uint8 on `env.device`. `out=` must be a contiguous uint8 buffer; a strided destination is refused rather than filled, because the renderer writes linearly through a flat view.
