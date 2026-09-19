# `stable_envs.interop`

Cross-package glue only. Nothing here is imported by `env`, `dynamics`, `specs`, `registry` or `_config`, and nothing is advertised at `import stable_envs as snv` top level.

**Ownership.** stable-envs owns `reset` / `step` / `transition` / `unroll` / `render` and state semantics. stable-worldmodel owns policy rollouts, MPC, planning, dataset collection and evaluation. There is no `env.rollout(policy=...)`.

**Native `swm.World(env=snv.PushT(...))` is not a public API.** Installed `stable-worldmodel` takes `env_name: str` and always `gym.make`s it. The shipped contract is the adapters below. See [stable-worldmodel](../concepts/worldmodel.md).

::: stable_envs.interop
