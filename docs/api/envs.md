# `stable_envs.envs`

The registered environments. The package discovers these modules automatically, so adding a file is
the registration step. `snv.import_errors()` lists modules that did not load. A missing optional
package disables only that family.

Use `snv.list_envs()` and `snv.info(name)` to see what this interpreter registered. The overview is
[Environments](../environments.md).

Each module docstring records its reference id, state layout and measured constants. They are shown
below, one per family, without class members. PushT is shown in full.

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
