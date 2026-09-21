# Sibling projects

`stable-envs` is the canonical environment layer. It owns Env / State / Action / Variation / Frame,
with stateful `reset` / `step` and functional `transition` / `unroll` / `render`.

`stable-worldmodel` is a sibling that may later depend on this package. The installed
`stable-envs` runtime has **zero knowledge** of `stable-worldmodel`: no SWM ids in the registry,
no interop adapters, no `snv.make("swm/...")`.

```text
stable-worldmodel          (policy, planning, collection — later)
        ↓  may depend
   stable-envs             (environments; zero runtime knowledge the other way)

stable-datasets            dataset/storage API
stable-pretraining         training API
```

Do not look for `env.rollout(policy, …)` here. Planning, policy rollouts, dataset storage and
training are deliberately outside this API. Other [stable-*](https://github.com/galilai-group)
projects provide tools in those areas.

The packages may be used together by ordinary Python and data conventions. There is no runtime
interop contract. Historical correspondence (the 42-id inventory, VariationMaps) lives in
regression fixtures and the research archive, not in the shipped lookup.

- [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
- [stable-datasets](https://github.com/galilai-group/stable-datasets)
- [stable-pretraining](https://github.com/galilai-group/stable-pretraining)
