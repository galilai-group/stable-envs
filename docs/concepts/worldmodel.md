# Sibling projects and observation ownership

stable-envs owns the simulator and the task: full State, `reset`/`get_state`/`set_state`/`step`,
actions, reward, success, termination, truncation, RNG, bounded Variation and rendering. It calls the
upstream simulator for dynamics.

stable-worldmodel owns learner observations: which pixels, proprioception and goals a model sees, how
they are packaged, and dataset filtering. Full State can contain privileged information. It is not
the default learner input. stable-envs has no Observation class and no State-filtering layer.

```text
stable-worldmodel    observation selection, models, data collection
        ↓
   stable-envs      simulator and task State, actions, Variation, rendering
        ↓
 upstream simulator dynamics, controllers, task behaviour
```

The stable-envs runtime does not import stable-worldmodel. There is no `snv.make("swm/...")` lookup.

Policy rollouts, planning, datasets and training belong downstream. `step` takes actions, never a
policy.

- [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
- [stable-datasets](https://github.com/galilai-group/stable-datasets)
- [stable-pretraining](https://github.com/galilai-group/stable-pretraining)
- [Contributor responsibilities](../adding/index.md)
