# `stable_envs.envs`

The registered environments. The package auto-discovers its modules, so adding a file here is the whole registration step, and `snv.import_errors()` names any module that did not load — a missing optional reference package costs that family and not the package.

List what this interpreter actually imported with `snv.list_envs()` / `snv.info(name)`; the narrative index is [Environments](../environments.md).

Each module's docstring is where its measured constants live: the reference id, the state layout, and every value that was recovered by pixel-diffing rather than read off the reference's source. Those are reproduced below, one per family, without their class members. PushT is rendered in full because it is the worked example the [porting guide](../adding-an-env.md) follows.

## PushT

::: stable_envs.envs.pusht

## The other families

::: stable_envs.envs.classic_control
    options:
      members: false

::: stable_envs.envs.navigation
    options:
      members: false

::: stable_envs.envs.minigrid
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

::: stable_envs.envs.highway
    options:
      members: false

::: stable_envs.envs.craftax.classic
    options:
      members: false

::: stable_envs.envs.craftax.full
    options:
      members: false

::: stable_envs.envs.rocket_landing
    options:
      members: false

::: stable_envs.envs.minihack
    options:
      members: false

::: stable_envs.envs.mujoco_locomotion
    options:
      members: false

::: stable_envs.envs.crafter
    options:
      members: false

::: stable_envs.envs.sokoban
    options:
      members: false

::: stable_envs.envs.lunarlander
    options:
      members: false

::: stable_envs.envs.procgen
    options:
      members: false

::: stable_envs.envs.carracing
    options:
      members: false

::: stable_envs.envs.jumanji
    options:
      members: false

::: stable_envs.envs.pgx
    options:
      members: false

::: stable_envs.envs.griddly
    options:
      members: false

::: stable_envs.envs.overcooked
    options:
      members: false
