# Sibling projects and observation ownership

Stable-envs owns the simulator and task: exhaustive State, reset/get/set/step,
actions, reward/success/termination/truncation, RNG, bounded Variation,
State-to-RenderState projection and rendering. Delegating to the authoritative
upstream simulator is the intended implementation.

Stable-worldmodel owns learner-visible observations: pixel/proprioception/goal
selection, flat/dict packaging, model input keys and dataset filtering. An
exhaustive State can contain privileged information and is not the default learner
input. Stable-envs does not add an Observation class or a State-filtering layer.

```text
stable-worldmodel    learner observation selection, models, collection
        ↓           observation integration remains deferred
   stable-envs      simulator/task State, actions, Variation, rendering
        ↓
 upstream simulator authoritative dynamics/controllers/task behavior
```

The stable-envs runtime does not import stable-worldmodel. There is no
`snv.make("swm/...")` lookup API. Historical registration correspondence and
evidence do not establish a runtime observation-container integration.

Policy-driven rollouts, planning, datasets and training belong downstream.
`unroll`, where supported, consumes known actions rather than a policy.

- [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
- [stable-datasets](https://github.com/galilai-group/stable-datasets)
- [stable-pretraining](https://github.com/galilai-group/stable-pretraining)
- [Contributor responsibilities](../adding/index.md)
