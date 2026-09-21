# Historical research artifacts

These directories are **Step-5 paired-eval harnesses**, not getting-started examples.

They import `stable_worldmodel`, drive `gymnasium.make("swm/...")`, and freeze renderer-swap
task-success evidence. The public `examples/` tree is for using stable-envs as stable-envs.

| path | what it is |
|---|---|
| [`step5_lewm_pusht/`](step5_lewm_pusht/README.md) | PushT + LeWM: official vs stable-envs pixels under the same planner |
| [`step5_downstream/`](step5_downstream/README.md) | Step-5 ingest / paired-gate / Reacher arm on top of that evidence |

Do not delete them. They are certification history. New user examples belong under `examples/`.

Replacement-readiness of the 42 historical cells: [step9_replacement_readiness.md](../step9_replacement_readiness.md).
