# Physics and rendering have separate acceptance gates

A registration can have an accurate renderer and an incomplete simulator State.
The rendering claim concerns pixels from a fixed input; the simulation claim
concerns reset, actions, task behavior and faithful continuation from a checkpoint.
Neither result implies the other.

Calling the authoritative upstream simulator/controller is the intended path.
A new environment does not need a second implementation of its physics. Exact
restoration still requires all relevant engine, controller, task, lifecycle and
RNG state, in the correct restore order.

PushT illustrates the failure boundary: its historical semantic recovery remains
incomplete because matching exposed body fields and inspected numeric contact
caches did not reproduce different-history contact continuation. Approximate
solver agreement and matching frames do not satisfy that requirement. A safe,
reviewed engine snapshot mechanism is needed before that work resumes.

Renderer-only capability remains useful, but those cells must refuse operations
that would claim a full simulator checkpoint. Functional dynamics are optional,
and an accepted delegated stateful backend may refuse transition/unroll.

A GPU renderer can coexist with authoritative CPU physics. Record the separate
dependency profiles and device claims; passing GPU pixel comparisons does not
certify physics under the GPU profile. Do not relax semantic comparisons because
a reimplemented contact solver diverges.

See the [environment](../adding/environment.md), [task](../adding/task.md) and
[renderer](../adding/renderer.md) guides for independent tests and promotion gates.
