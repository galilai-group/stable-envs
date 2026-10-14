# Physics and rendering have separate acceptance gates

A class can have an accurate renderer and an incomplete simulator State. The two claims are tested
separately:

- The rendering claim is about pixels from a fixed input.
- The simulation claim is about reset, actions, task behaviour, and exact continuation from a
  checkpoint.

Neither result implies the other.

Call the upstream simulator and controller. A new environment does not need its own copy of the
physics. Exact restoration still needs every relevant engine, controller, task, lifecycle and RNG
field, restored in the right order.

Matching frames and close solver agreement do not prove restoration. For example, PushT body
positions and velocities alone do not restore its contact history. The released PushT State
therefore stores the action history and replays it.

Renderer-only classes refuse operations that would need a full simulator checkpoint. Simulating
classes expose `step` with single actions or action sequences.

A GPU renderer can sit on top of CPU physics. Keep their dependency profiles and device claims
separate. A GPU pixel comparison does not certify physics under the GPU profile. Do not relax a
semantic comparison because a reimplemented contact solver diverges.

See the [environment](../adding/environment.md), [task](../adding/task.md) and
[renderer](../adding/renderer.md) guides.
