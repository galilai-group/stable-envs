# Stateful step and optional functional dynamics

`step(actions)` advances the live simulator. For accepted delegated backends it
calls the authoritative upstream environment/controller directly. Normal sequential
steps must not internally restore State on every call.

```python
import numpy as np
import stable_envs as snv

with snv.CartPole(num_envs=1, device="cpu") as env:
    saved = env.reset(seed=17)
    action = np.array([[1]], dtype=np.int64)
    expected = env.step(action)
    env.set_state(saved)
    actual = env.step(action)
    assert (actual == expected).all()
```

`transition(state, action)` and `unroll(state, actions)` are optional functional
capabilities. Where supported, they leave live State untouched, and unroll takes
known actions with a leading time dimension. It is not a policy rollout.
Delegated classic control, for example, explicitly refuses functional transition
rather than falling back to incomplete historical equations. Use an independent
simulator and `set_state`/`step` for branching.

`has_dynamics` identifies a simulator capability, not a promise that every
functional operation exists. A renderer-only cell refuses reset/get/set/step
operations that would claim full simulator semantics. Private renderer corpus
helpers remain separate.

Renderer and physics acceptance are independent. Calling the authoritative
simulator is the intended path to exact semantic behavior. GPU pixels do not
establish GPU-profile physics parity. PushT is parked precisely because
restoration across contact histories failed; an approximate solver or matching
image cannot close that gate.

See the [environment guide](../adding/environment.md),
[task guide](../adding/task.md) and [sibling ownership](worldmodel.md).
