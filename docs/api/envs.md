# `stable_envs.envs`

The registered environments. The package auto-discovers its modules, so adding a file here is the whole registration step, and `snv.import_errors()` names any module that did not load — a missing optional reference package costs that family and not the package.

List what this interpreter actually imported with `snv.list_envs()` / `snv.info(name)`; the narrative index is [Environments](../environments.md).

Each module's docstring is where its measured constants live: the reference id, the state layout, and every value that was recovered by pixel-diffing rather than read off the reference's source. Those are reproduced below, one per family, without their class members. PushT is rendered in full because it is the worked example the [porting guide](../adding/index.md) follows.

## PushT

::: stable_envs.envs.pusht.semantics

## The other families

::: stable_envs.envs.classic_control
    options:
      members: false

::: stable_envs.envs.navigation
    options:
      members: false

::: stable_envs.envs.ogbench
    options:
      members: false

::: stable_envs.envs.ogbench.maze
    options:
      members: false

::: stable_envs.envs.ogbench.manipulation
    options:
      members: false

::: stable_envs.envs.dmcontrol
    options:
      members: false

::: stable_envs.envs.fetch
    options:
      members: false

::: stable_envs.envs.craftax.classic
    options:
      members: false

::: stable_envs.envs.craftax.full
    options:
      members: false

::: stable_envs.envs.rocket
    options:
      members: false
