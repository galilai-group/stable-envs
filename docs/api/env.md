# Env

Concrete classes such as `snv.CartPole` are the main API. The shared `Env` defines the simulator and
rendering interface:

- `reset`, `get_state` and `state` return independent full State snapshots.
- `set_state(full_state)` restores every required field and returns `None`.
- `step(actions, *, states=slice(None), frames=slice(-1, None))` accepts `[B,A]` or `[B,T,A]` and returns
  State and the task's reward, success and done values. See [Stepping](../concepts/dynamics.md).
- `render(x=None, **options)` returns uint8 Frames. `x` is one of these forms:
  - None: the frames of the live environments.
  - A full State from `reset`, `step` or `get_state`.
  - Compact rows `(B, state_dim)`, numpy or torch, as `sample_states` returns them.

  Cube, Scene, PushT and PyFlytRocketLanding also accept the RenderState that `to_render_state` returns.
  A State draws with its own appearance; sampled rows draw with the env's current Variation, or with
  `variation=`. Classes without compact rows (`state_dim` 0) and Rocket on CPU refuse rows. The error
  gives the forms that the class accepts.
- `native_size` is the only `size` that the class draws, or None when the class draws any size. See
  [Environments](../environments.md).
- `close` and the context manager release owned resources.
- The State, action, frame and Variation specs describe shapes and dtypes. Family validators check
  full schemas before restoring.

State is a TensorDict checkpoint. It can be privileged and is not the learner observation. Variation
can affect physics, reset, task or appearance.

## OGBench Cube and Scene backends

- On CPU, `backend="auto"` selects `backend="opengl_fast"`: software OpenGL with simplified arm meshes
  and a 2048 shadow map. `backend="opengl"` draws the source's frames. Configure Mesa software OpenGL
  before importing MuJoCo. CPU rendering rejects a hardware GL context.
- On CUDA, `backend="auto"` selects NVIDIA OpenGL at 1× sampling. Its fidelity record compares two
  stable-envs OpenGL instances with the same compiled appearance, not upstream-default OGBench frames.
  With `supersample=2` or more, `backend="auto"` selects the two-pass MJWarp renderer, as
  `backend="mjwarp_fidelity"` does; its reference is that same-style stable-envs OpenGL path.
- The default `supersample` is 1 on every device. An explicit `supersample` argument keeps its value.
  The sampling setting stored in a supplied RenderState also keeps its value.
- On Linux, a batch draws in up to `ogb_gl_workers` OpenGL worker processes
  (see [Performance tuning](../performance-tuning.md)). On CPU, the workers start when the batch has
  one state for each worker. On CUDA, they start when the batch has 8 states for each of 2 or more
  workers. The frames do not change.
- An explicit `backend="mjwarp"` requires CUDA.
- Physics runs on CPU with every backend.

## Internal interfaces

Contributor interfaces include `_pack_render_state`, `_unpack_render_state`, `_get_render_state`,
`_set_render_state`, `_render`, `state_dim` / `state_names`, `scene` and the style helpers. Compact
RenderState layouts are not State schemas. Renderer input checks do not replace State validation.

See [State](../concepts/state.md), [Lifecycle](../concepts/lifecycle.md) and
[Adding a component](../adding/index.md).

::: stable_envs.env
