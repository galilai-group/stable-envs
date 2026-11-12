# Roadster

Roadster is five planning tasks for a car with nine cold-gas jets, simulated in MuJoCo. The car drives on its wheels and flies on its jets, and each class fixes one task of the simulator in `stable_envs.envs.roadster.source`. The [state](../terms.md#state) holds everything that determines the future, the [variation](../terms.md#variation) sets wind and rain, and MuJoCo OpenGL draws the [frames](../terms.md#frame) from a world-model camera.

```python
# needs the roadster extra
import stable_envs as snv

with snv.RoadsterTrafficJam(num_envs=4, device="cpu") as env:
    options = {"variation_values": {"weather.wind": [8.0], "weather.rain": [0.5]}}
    state = env.reset(seed=0, options=options)
    print(env.render(state).shape)  # torch.Size([4, 96, 96, 3])
    out = env.step(env.action_spec.sample(4, env.rng))
    print(out["success"].shape)     # torch.Size([4])
```

## Install

The classes need the `roadster` extra (MuJoCo and Pillow): install `stable-envs[roadster]`.

## Task

| Class | Task | Horizon (steps) |
|---|---|---:|
| `RoadsterLoop` | drive the half-pipe loop, leave it inverted and roll upright | 60 |
| `RoadsterMission` | the full course: loop, three rings, pad landing | 450 |
| `RoadsterTrafficJam` | hop a traffic jam and land on the road beyond | 150 |
| `RoadsterRooftop` | fly from a tower roof around a taller tower to a helipad | 200 |
| `RoadsterCarHop` | jump sideways over a parked car with a barrel roll | 120 |

The action has four values in [-1, 1]. The environment applies each action at 10 Hz through a 50 Hz autopilot:

| Index | Control |
|---|---|
| 0 | forward acceleration in the heading frame (wheel torque on the ground, jets in the air) |
| 1 | sideways acceleration (the car banks into it) |
| 2 | vertical acceleration relative to hover; on the ground a positive value lights the lift jets |
| 3 | yaw rate |

The task's cost is the distance to the current subgoal plus the path through the remaining ones. The reward is the decrease of that cost, plus 10 on success and minus 10 on failure. The step output carries `success`, `failed`, `terminated`, `truncated` and the current `cost`. A [batch row](../terms.md#batch-row) terminates on success or failure and truncates at the task's horizon, and the elapsed steps are part of the state.

`stable_envs.envs.roadster.source.GoalExpert` is a scripted expert that reads the simulator, for demos and data collection.

## Simulation

Physics always runs on CPU, and `device="cuda"` stores state and frames on the GPU.

### State

The state is made of:

- MuJoCo's integration state, split into `sim.time`, `sim.qpos`, `sim.qvel`, `sim.warmstart`, `sim.ctrl`, `sim.qfrc_applied`, `sim.xfrc_applied`, `sim.mocap_pos` and `sim.mocap_quat`. The two mocap fields place the traffic and the parked cars.
- `sim.qpos_pre`, the pose that MuJoCo's derived quantities refer to after a step.
- The autopilot commands and the task bookkeeping: status, target and route index, fuel, counters, flags and the previous cost.
- The goal, the route waypoints, the traffic jam, and the weather with its gust RNG.
- The exhaust history that the smoke is drawn from (`fx.count` and `fx.history`, the last 75 steps of pose, velocity, jet commands and wind).
- `lifecycle.elapsed_steps`, and the physics leaves the episode was reset with (`physics.weather.wind` and `physics.weather.rain`).

`set_state` validates the whole batch before it changes any environment, and it refuses render rows, partial dictionaries and states of another task.

Large batches step and render in [worker processes](../performance-tuning.md#worker-processes), with the same results as one process. The [knobs](../api/tuning.md#knobs) of the `roadster-sim` and `roadster-render` paths set the batch size that starts workers and the number of workers. Workers write only the state fields a step changes. A step that needs a frame but no state draws it from the render-row fields, and the frame equals what `render(state)` draws.

## Rendering

`render(state)` draws the `wm` camera, which follows the car with a fixed world orientation, at 96 px by default, with MuJoCo's OpenGL renderer (`MUJOCO_GL=egl` or `osmesa`). It rasterizes on the GPU when the EGL driver is a GPU's (NVIDIA's, for example) and on the CPU otherwise (Mesa).

Frames show the car, its jet flames, exhaust smoke and blob shadow, the scenery and the rain, exactly as MuJoCo draws the live simulation at the same state. The smoke is regenerated from the exhaust history rather than simulated.

`rendering.cinematic=1` draws the same state from the same `wm` camera on a dark-space scene, with 8× MSAA, real shadows, fog to black and post-processing (bloom, filmic grade, vignette). It leaves out grain and motion blur, and a state always gives the same pixels. It looks best at a larger `size` such as 512. Set it per environment with `set_variation({"rendering.cinematic": [1]})`, or per call with `render(state, variation={"rendering.cinematic": [1]})`. The installed variation records which look drew a frame, and the default, 0, is the raw world-model frame.

## Variation

| Leaf | Kind | Values | Effect |
|---|---|---|---|
| `weather.wind` | physics | 0 to 20 m/s | mean wind speed, with gusts of about 30% of it |
| `weather.rain` | physics | 0 to 1 | drops, haze, and tire grip down to 0.55× in the heaviest rain |
| `rendering.cinematic` | visual | 0 or 1 | draws the dark-space look instead of the world-model frame |

The [variation](../tutorials/variation.md) has two physics leaves and one visual leaf. The simulator sets the weather at reset, and `set_variation` and the reset options for the two physics leaves apply from the next reset. The defaults (0, 0) give the simulator's physics without weather, and the weather of a restored state shows in `get_variation()` for that episode.

## Classes

### RoadsterLoop

<!-- env: RoadsterLoop -->

### RoadsterMission

<!-- env: RoadsterMission -->

### RoadsterTrafficJam

<!-- env: RoadsterTrafficJam -->

### RoadsterRooftop

<!-- env: RoadsterRooftop -->

### RoadsterCarHop

<!-- env: RoadsterCarHop -->

## Limitations

The drawn pose is the one MuJoCo's derived quantities refer to, which is one substep behind `sim.qpos` after a step.
