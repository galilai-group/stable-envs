# stable-envs and stable-worldmodel

Two packages, a deliberate split.

**stable-envs owns**

```text
reset
step                 (stateful, where a cell ships dynamics)
transition / unroll  (functional, known actions)
render
state / action / variation semantics
```

**stable-worldmodel owns**

```text
policy rollouts
MPC, CEM / MPPI, planning
dataset collection
evaluation
async orchestration
auto-reset policy semantics
```

Do not look for `env.rollout(policy, …)` here. `stable_envs.rollout` remains `snv.sample` / `BasePolicy` for existing tests; it is not `env.rollout(policy)`.

## Frozen interop (what is shipped today)

Workstream J merged the public `stable_envs.interop` facade (not advertised at `snv.*` top level). Native `swm.World(env=snv.PushT(...))` is still **unsupported** by the installed `stable-worldmodel` (`World` takes `env_name: str` and always `gym.make`s it). The shipped contract is:

```python
from stable_envs import interop

snv.SWM_IDS                               # 42-id table (public)
snv.spec("swm/PushT-v1")                  # resolves aliases without constructing on GPU
interop.StableEnvGym("PushT", device="cpu")          # gym.Env over a batch-1 renderer
interop.BatchedEnvPool("PushT", num_envs=8, device="cpu")
interop.attach_batched_pool(world)                   # replace a swm.World's env list
interop.swap_renderer(gym_env, "PushT", read_state=interop.pusht_state, device="cpu")
interop.variation_map / style_from_variation / variation_from_style / check_mapping
```

`gymnasium` is imported when an adapter is first used; `stable_worldmodel` is imported only by operations that inspect a live SWM env. A user with neither installed pays nothing.

### What each adapter is for

| adapter | keeps | replaces | pixel-comparable? |
|---|---|---|---|
| `StableEnvGym` | caller code (`swm.World` + wrappers) | the env id string | no — different simulator, different reset draw |
| `BatchedEnvPool` | `swm.World` loop shape | N single envs with one batched renderer | no |
| `swap_renderer` | official dynamics, reward, termination | `render()` only | **yes** — both arms visit the same states |

`swap_renderer` is the honest path when the simulator is the reference and only the renderer is ours. `StableEnvGym` does **not** carry an official reward function: a reward is a task, and this package defines what a frame is.

The Step-8 `stable_envs.compat.worldmodel` import path still resolves to the same module objects.

## Native `World(env=…)` — not a public API

```python
# NOT shipped. Do not write this against current stable-envs:
world = swm.World(env=snv.PushT(num_envs=64, device="cuda"))
```

That remains the long-term direction (interop shrinks as SWM consumes `Env` natively). Installed SWM still requires `env_name: str`. `examples/worldmodel.py` prints this limitation rather than inventing the call.

Worked evidence that the frozen adapters run an unmodified SWM workflow: [`examples/swm_compat/`](https://github.com/galilai-group/stable-envs/tree/main/examples/swm_compat).
