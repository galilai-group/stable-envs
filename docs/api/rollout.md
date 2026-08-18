# `stable_envs.rollout`

`env.sample(steps=100, batch=256, policy=...)`: batched trajectories generated on device, never touching
disk. The policy interface mirrors `stable_worldmodel.policy.BasePolicy`, with the one deliberate change
that `get_action` receives a batch and returns `(B, action_dim)`.

**This is a convenience for when you have no states, not the route to a training set.** If you already
have a corpus — an offline dataset, a rollout of the official CPU env, anything — feed it to the
renderer directly: your own `DataLoader` over those states plus `env.set_state(batch).render()` in the
training step. Most envs here ship a renderer and no simulator, so for most of the registry `sample` is
not available at all, and that is by design rather than a gap.

Rolling out needs `env.step`, a batched GPU simulator, which most envs here deliberately do not ship — a
renderer can be bit-exact against the official reference and a reimplemented contact solver cannot. On
those envs `sample` says so precisely and names the alternative rather than failing somewhere deep.

::: stable_envs.rollout
