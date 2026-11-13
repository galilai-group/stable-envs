# Matching the upstream library

Each class is compared with an environment in its [upstream library](../terms.md#upstream-library). This page shows how to make a class match it. [Adding an environment](environment.md) says where the code goes, and [Tests](testing.md) covers the suite that runs the comparisons.

## Read the upstream source

Read the full construction path: registration, constructor defaults, wrappers, reset and goal sampler, controller, physics, reward and success, `TimeLimit`, RNG and rendering. Pin the upstream versions and record hashes of the source and assets you use, including dependency versions that change arithmetic. Keep and document unusual upstream behavior, such as dtype conversions, knobs that do nothing or odd task rules: users compare against what the upstream library does, not what it should have done. Then check your reading with [runtime comparisons](#measure-constants-against-the-upstream-library).

## Find the upstream renderer

Pin the upstream library's rendering code, engine and assets, camera, resolution, color conversion, draw order and backend dependencies. Read its source code, then check your reading with runtime comparisons. Frozen renderer definitions, corpus layouts and baselines stay as they are. If you find a real defect, propose a reviewed fix with the evidence.

## Measure constants against the upstream library

Set each rendering constant from a measurement against the upstream renderer. Reading the upstream source gives you a hypothesis, and comparing pixels decides it. `stable_envs.testing.conformance.check_source_mse` renders the same states through both renderers and returns the per-frame mean squared error (MSE) on the 0-255 scale:

```python
import numpy as np
import torch
import stable_envs as snv
from stable_envs.testing import conformance as C

with snv.CartPole(num_envs=8, device="cpu") as env:
    result = C.check_source_mse(env, n=8)
    print(result["mse_mean"], result["mse_max"])  # 0.0 0.0
```

Constants that read plausibly from the source can still be wrong:

| What the source suggested | What was true | How it showed |
|---|---|---|
| PushT flips y (`512 - y`) | no flip: `positive_y_is_up` is False | agent at row 348 instead of 162 |
| the T rotates about its center of gravity, `cog = (0, 45)` | it rotates about the body origin | 44.6 px error |
| block fill `(142, 163, 183)`, truncating a 1.2× color scale | `(143, 163, 184)`, because `as_int()` rounds | off by one on every interior pixel of the block |
| walls are `draw_fat_segment(radius=2)`, half-width 2.0 | half-width 2.5, because `r > 2` adds caps | 8,012 wall pixels instead of 10,009 |
| Fetch's render state is `qpos + qvel` | it also needs `mocap_pos`, `mocap_quat` and the goal site | 29–138/255 error instead of 0 |
| MJWarp textures are the limit | MJWarp ignores alpha and paints invisible markers | mean error from 79.34 to 7.47 after a model pre-pass |

- Document each constant next to it, with the measurement that fixed it. "8,012 wall pixels against the upstream renderer's 10,009" helps the next reader, and "Matches pygame" tells them nothing.
- Time a speedup on the code path a caller actually reaches.
- Test on states where the question comes up. Draw order, for example, only matters where two bodies overlap.
- Look at the worst frame and the region that differs, not only the mean. A wrong corner or a small part at the wrong pose changes few pixels and barely moves the mean MSE.
- Record the environments you rejected, with the check that rejected them.

The [GPU conformance tests](testing.md#gpu-conformance) compare frames with the upstream library, and [frame regression checks](testing.md#frame-regression-checks) hold each class to its recorded MSE.

## Rendering and physics are tested separately

A class can have an accurate renderer and an incomplete simulator state. A rendering test is about the pixels drawn from a fixed input, and a simulation test is about reset, actions, task behavior and exact continuation from a checkpoint. Two CartPole states that differ only in velocity draw the same frame but step differently:

```python
with snv.CartPole(num_envs=1, device="cpu") as env:
    state = env.reset(seed=17)
    moving = state.clone()
    moving["x_dot"] += 1.0  # same pose, different velocity
    print(torch.equal(env.render(state), env.render(moving)))  # True

    push = np.array([[1]], dtype=np.int64)
    env.set_state(state)
    still = env.step(push)["state"]["x"]
    env.set_state(moving)
    sliding = env.step(push)["state"]["x"]
    print(torch.equal(still, sliding))  # False
```

Call the upstream simulator and controller instead of giving a new environment its own copy of the physics. Exact restoration still needs every engine, controller, task, lifecycle and RNG field that affects the future, restored in the right order. Matching frames and close solver agreement don't prove restoration: PushT body positions and velocities alone, for example, leave out its contact history. Renderer-only classes refuse operations that would need a full simulator checkpoint.

A GPU renderer can sit on top of CPU physics. Keep their dependency profiles, and the devices you say they support, separate. A GPU pixel comparison doesn't prove the physics under the GPU profile. A reimplemented contact solver that diverges is a bug to fix, not a reason to loosen a simulation comparison.

## Source contracts

`tests/contracts/` holds shared source-contract tests for representative registrations. Each profile compares state, simulation, task and upstream pixels against a pinned stable-worldmodel checkout, with the pins in `tests/helpers/reference/profiles.py`. The Fetch profile builds the upstream environment at the contract frame size.

```bash
python -m pytest tests/contracts --contract-profile native \
  --contract-source /path/to/pinned/stable-worldmodel \
  --contract-output /path/to/new-run
```

`--contract-device cuda` runs the CUDA output path. `--contract-output` must be a new directory, which receives the plan, hashes, per-case progress and a `SHA256SUMS` seal. The run computes the required case set before it starts. Missing, skipped, duplicated or deselected cases fail the profile, which leaves `-k` for development only. Run one profile per interpreter without xdist, and after a crash rerun the whole profile into a new directory. The generators for frozen corpora are in `scripts/testing/`, and its README lists them.

## Final acceptance

Write a driver for your registration with pinned dependencies, and produce its expected values with the upstream library, never with the code you're testing. Compare upstream and new reset, next state, reward, success and both done flags for a fixed seed set, field by field, across contacts, controller history, random draws and episode boundaries. For a renderer, record the source MSE with `scripts/testing/record_source_mse.py` and capture golden frames with `scripts/testing/capture_golden.py`. If a different rasterizer draws the frames, copy the recorded MSE into `visual_fidelity` and say where they differ in `visual_note`. Check seeded and unseeded reset wherever you promise replay, test physics and task variation, and cover representative tasks, contact poses, cameras and appearance settings. Run each CPU and GPU backend you support separately. A backend that refuses CPU is a limit, not a CPU pass.

```bash
uv run pytest -q tests/unit/test_replacement_readiness.py
uv run pytest -m 'regression and not gpu'
uv run pytest -m 'regression and gpu'   # CUDA only
```
