# Env

Concrete classes such as `snv.CartPole` are the main API. The shared `Env` defines the simulator and
rendering interface:

- `reset`, `get_state` and `state` return independent full State snapshots.
- `set_state(full_state)` restores every required field and returns `None`.
- `step(actions, states=slice(None), frames=slice(-1,None))` accepts `[B,A]` or `[B,T,A]` and returns
  State and the task's reward, success and done values. See [Stepping](../concepts/dynamics.md).
- `render(full_state)` returns a uint8 Frame.
- `close` and the context manager release owned resources.
- The State, action, frame and Variation specs describe shapes and dtypes. Family validators check
  full schemas before restoring.

State is a TensorDict checkpoint. It can be privileged and is not the learner observation. Variation
can affect physics, reset, task or appearance.

## OGBench Cube and Scene backends

- On CPU, `backend="auto"` selects software OpenGL. Configure Mesa software OpenGL before importing
  MuJoCo. CPU rendering rejects a hardware GL context.
- On CUDA, `backend="auto"` selects the two-pass MJWarp renderer with 2× sampling. Scene aligns its
  four samples with NVIDIA OpenGL, including its cached static background.
- An explicit `supersample` argument keeps its value. The sampling setting stored in a supplied
  RenderState also keeps its value. To render a 1× RenderState with the default sampling, pass `supersample=2`.
- An explicit `backend="mjwarp"` requires CUDA.
- Physics runs on CPU with every backend.

## Internal interfaces

Contributor interfaces include `_pack_render_state`, `_unpack_render_state`, `_get_render_state`,
`_set_render_state`, `_render`, `state_dim` / `state_names`, `scene` and the style helpers. Compact
RenderState layouts are not State schemas. Renderer input checks do not replace State validation.

See [State](../concepts/state.md), [Lifecycle](../concepts/lifecycle.md) and
[Adding a component](../adding/index.md).

::: stable_envs.env
