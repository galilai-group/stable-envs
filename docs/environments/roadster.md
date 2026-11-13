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

The classes need the `roadster` extra (MuJoCo 3.17 and Pillow): install `stable-envs[roadster]`. MuJoCo Warp physics on CUDA needs the `roadster-gpu` extra, which adds MuJoCo Warp 3.17 and Warp.

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

`simulation_backend` selects the physics:

| Value | CPU device | CUDA device |
|---|---|---|
| `"default"` (default) | CPU batch | [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) |
| `"auto"` | CPU batch | CPU batch below `mjwarp_min_envs` (64) environments, MuJoCo Warp from there |
| `"native"` | CPU batch | CPU batch |
| `"mjwarp"` | error | MuJoCo Warp |

Both backends step every environment in one batch. The source's autopilot, drag and wind forces, fuel, ring and route bookkeeping, crash checks, cost, reward and success run on arrays of environments. MuJoCo steps the 10 substeps of each 50 Hz step one at a time, because the drag and wind force of a substep depends on the velocity after the previous one. The physics model is the source's world with MuJoCo contact sensors for the contacts that the task reads (tires on anything, on the pad, on the ground and on the goal roof, the hull on anything, the car on each obstacle). The visual geoms outside the car are not in it. Sensors and these geoms do not change the dynamics.

The CPU batch runs the task logic in NumPy and MuJoCo in one native `mujoco.batch` in float64. It steps bit for bit as the source environment: the same `mj_step` calls with the same applied forces, and the same float64 operations in the same order. NumPy computes a stack of small matrix products and least-squares solves with the same BLAS and LAPACK routines as one. A `device="cuda"` environment with the CPU batch returns its state on the GPU.

The MuJoCo Warp backend runs the task logic in float64 Torch on the GPU, without host reads in a step, and MuJoCo in float32, so a rollout follows the source approximately. Each 50 Hz step is three CUDA graph launches: the Torch phase before the substeps, one MuJoCo Warp graph of the 10 substeps with a Warp kernel for the drag and wind force before each substep, and the Torch phase after them. The autopilot's bounded least-squares jet allocation uses a precomputed pseudo-inverse for each set of free jets. With wind, the gusts of an action are drawn on the host from the generator of each environment, as the source does. An environment that ends within an action has then drawn the gusts of the whole action. Replays of the source's scripted expert under wind and rain (`tests/gpu/test_roadster_mjwarp.py`, 8 environments, 300 steps) keep the outcome of every environment and end within 3 steps of the source. The car stays within 0.01 m of the source's path in TrafficJam and CarHop, 0.3 m in Rooftop and 3.4 m in Loop until the episode ends. In Mission the paths separate after the loop, where the source itself moves 13 m by step 300 when its start velocity changes by 1e-9 m/s.

Resets draw the start, goal, traffic and parked cars of each environment with the source's random draws on the host, and write all environments to the physics batch in one call.

### State

The state is made of:

- MuJoCo's integration state, split into `sim.time`, `sim.qpos`, `sim.qvel`, `sim.warmstart`, `sim.ctrl`, `sim.qfrc_applied`, `sim.xfrc_applied`, `sim.mocap_pos` and `sim.mocap_quat`. The two mocap fields place the traffic and the parked cars.
- `sim.qpos_pre`, the pose that MuJoCo's derived quantities refer to after a step.
- The autopilot commands and the task bookkeeping: status, target and route index, fuel, counters, flags and the previous cost.
- The goal, the route waypoints, the traffic jam, and the weather with its gust RNG.
- The exhaust history that the smoke is drawn from (`fx.count` and `fx.history`, the last 75 steps of pose, velocity, jet commands and wind).
- `lifecycle.elapsed_steps`, and the physics leaves the episode was reset with (`physics.weather.wind` and `physics.weather.rain`).

`set_state` validates the whole batch before it changes any environment, and it refuses render rows, partial dictionaries and states of another task.

Large batches draw frames in render [worker processes](../performance-tuning.md#worker-processes), with the same frames as one process. The [knobs](../api/tuning.md#knobs) of the `roadster-render` path set the batch size that starts workers and the number of workers. A step that needs a frame but no state draws it from the render-row fields of the live environments, and the frame equals what `render(state)` draws.

## Rendering

`render(state)` draws the `wm` camera, which follows the car with a fixed world orientation, at 96 px by default, with MuJoCo's OpenGL renderer (`MUJOCO_GL=egl` or `osmesa`). It rasterizes on the GPU when the EGL driver is a GPU's (NVIDIA's, for example) and on the CPU otherwise (Mesa). `render` also accepts compact render rows `(N, state_dim)` from `sample_states`, on each device.

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
