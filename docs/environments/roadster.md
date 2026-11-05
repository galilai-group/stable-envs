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
`sim.qpos` after a step). Physics runs on CPU; `device="cuda"` stores State and frames on the GPU.

Large batches step their lanes in worker processes (`roadster-sim` tuning path) and draw frames in render
workers (`roadster-render`); both give the frames and State of one process. Workers write only the State
fields a step changes. A step that needs a frame but no State builds no snapshot: the frame is drawn from just
the render-row fields (in-process batches read them from the live lanes) and equals what `render(state)` draws.

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
