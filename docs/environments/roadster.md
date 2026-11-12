# Roadster

The Roadster classes are five planning tasks for a Roadster-style car with nine cold-gas jets in MuJoCo,
ported from roadster-jets. The car drives on its wheels and flies on its jets: around a half-pipe loop, along
the full course (loop, three rings, pad landing), over a traffic jam, between rooftops, and sideways over a
parked car with a barrel roll. Each class fixes one task of the source env
(`stable_envs/envs/roadster/source/`), whose autopilot, reward, success rule and horizon apply unchanged.

The action has four values in [-1, 1], applied at 10 Hz by a 50 Hz autopilot:

| Index | Control |
|---|---|
| 0 | forward acceleration in the heading frame (wheel torque on the ground, jets in the air) |
| 1 | sideways acceleration (the car banks into it) |
| 2 | vertical acceleration relative to hover; on the ground a positive value lights the lift jets |
| 3 | yaw rate |

**Reward** is the decrease of the source's cost (distance to the current subgoal plus the path through the
remaining ones), +10 on success and −10 on failure. The step output carries `success`, `failed`,
`terminated`, `truncated` and the current `cost`. A lane terminates on success or failure and truncates at
the task's horizon; the elapsed steps are part of State.

**State** holds everything that determines the future: MuJoCo's integration state split into its
components (`sim.time`, `sim.qpos`, `sim.qvel`, `sim.warmstart`, `sim.ctrl`, `sim.qfrc_applied`,
`sim.xfrc_applied`, `sim.mocap_pos`, `sim.mocap_quat`, the last two placing the traffic and the parked
cars), the pose MuJoCo's derived quantities refer to after a step (`sim.qpos_pre`), the autopilot commands, the task bookkeeping (status, target and route index, fuel, counters, flags, the
previous cost), the goal, the route waypoints, the traffic jam, the weather with its gust RNG, the exhaust
history the smoke is drawn from (`fx.count`, `fx.history`: the last 75 steps of pose, velocity, jet commands
and wind), `lifecycle.elapsed_steps`, and the physics leaves the episode was reset with
(`physics.weather.wind`, `physics.weather.rain`). `set_state` validates the whole batch before changing any
lane and refuses render rows, partial dictionaries and States of another task.

**Variation**: `weather.wind` (mean wind speed in m/s, gusts of 30 % of it) and `weather.rain` (0-1:
drops, haze and tyre grip down to 0.55×) are physics leaves. The source sets the weather at reset, so
`set_variation` and the reset options apply from the next reset. The defaults (0, 0) give the source's
original physics. The weather of a restored State shows in `get_variation()` for that episode.

**Rendering**: `render(state)` draws the source's `wm` camera, which follows the car with a fixed world
orientation, at 96 px by default, with MuJoCo's OpenGL renderer (`MUJOCO_GL=egl` or `osmesa`). It
rasterizes on the GPU when the EGL driver is a GPU's (for example NVIDIA's) and on the CPU otherwise
(Mesa). Frames show the car, its jet flames, exhaust smoke and blob shadow, the scenery and the rain,
exactly as MuJoCo draws the live simulation at the same State (the smoke is regenerated from the exhaust
history, not simulated). The pose drawn is the one MuJoCo's derived quantities refer to (one substep behind
`sim.qpos` after a step).

Large batches draw frames in render workers (`roadster-render` tuning path); they give the frames of one
process. A step that needs a frame but no State builds no snapshot: the frame is drawn from the render-row
fields of the live environments and equals what `render(state)` draws.

The visual Variation leaf `rendering.cinematic` (0 or 1; use a larger `size`, e.g. 512) draws the same State
and the same `wm` camera on a dark-space scene: 8x MSAA, real shadows, fog to black, and post-processing
(bloom, filmic grade, vignette) without grain or motion blur, so a State always gives the same pixels. Set it
per lane with `set_variation({"rendering.cinematic": [1]})`, or per call with
`render(state, variation={"rendering.cinematic": [1]})`; the installed Variation records which look drew a
frame. The default, 0, is the raw world-model frame.

The classes need the `roadster` extra (MuJoCo and Pillow): `pip install "stable-envs[roadster]"`.
`stable_envs.envs.roadster.source.GoalExpert` is a scripted expert that reads the simulator, for demos and
data collection.

```python
import stable_envs as snv

env = snv.RoadsterTrafficJam(num_envs=4)
state = env.reset(seed=0, options={"variation_values": {"weather.wind": [8.0], "weather.rain": [0.5]}})
frames = env.render(state)                       # (4, 96, 96, 3) uint8
out = env.step(env.action_spec.sample(4, env.rng))
```

## Physics backends

| `simulation_backend` | CPU env | CUDA env |
|---|---|---|
| `"default"` (default) | CPU batch | [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) |
| `"auto"` | CPU batch | CPU batch below `mjwarp_min_envs` (64) environments, MuJoCo Warp from 64 |
| `"native"` | CPU batch | CPU batch |
| `"mjwarp"` | not supported | MuJoCo Warp |

Both backends step every environment in one batch. The source's autopilot, drag and wind forces, fuel, ring
and route bookkeeping, crash checks, cost, reward and success run on arrays of environments. MuJoCo steps the
10 substeps of each 50 Hz step one at a time, because the drag and wind force of a substep depends on the
velocity after the previous one. The physics model is the source's world with MuJoCo contact sensors for the
contacts that the task reads (tyres on anything, on the pad, on the ground and on the goal roof, the hull on
anything, the car on each obstacle). The visual geoms outside the car are not in it. Sensors and these geoms
do not change the dynamics.

The CPU batch runs the task logic in NumPy and MuJoCo in one native `mujoco.batch` in float64. It steps bit
for bit as the source env: the same `mj_step` calls with the same applied forces, and the same float64
operations in the same order (NumPy computes a stack of small matrix products and least-squares solves with
the same BLAS and LAPACK routines as one). A `device="cuda"` env with the CPU batch returns State on the GPU.

The MuJoCo Warp backend runs the task logic in float64 Torch on the GPU, without host reads in a step, and
MuJoCo in float32, so a rollout follows the source approximately. The autopilot's bounded least-squares jet
allocation uses a precomputed pseudo-inverse for each set of free jets. With wind, the gusts of an action are
drawn on the host from each environment's generator, as the source does; an environment that ends within an
action has then drawn the gusts of the whole action. Replays of the source's scripted expert under wind and
rain (`tests/gpu/test_roadster_mjwarp.py`, 8 environments, 300 steps) keep the outcome of every environment
and end within 3 steps of the source. The car stays within 0.01 m of the source's path in TrafficJam and
CarHop, 0.3 m in Rooftop and 3.4 m in Loop until the episode ends. In Mission the paths separate after the
loop, where the source itself moves 13 m by step 300 when its start velocity changes by 1e-9 m/s.

Resets draw each environment's start, goal, traffic and parked cars with the source's random draws on the host
and write all environments to the physics batch in one call.

## RoadsterLoop

<!-- env: RoadsterLoop -->

## RoadsterMission

<!-- env: RoadsterMission -->

## RoadsterTrafficJam

<!-- env: RoadsterTrafficJam -->

## RoadsterRooftop

<!-- env: RoadsterRooftop -->

## RoadsterCarHop

<!-- env: RoadsterCarHop -->
