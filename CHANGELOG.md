# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Classic control `simulation_backend`: `CartPole`, `Pendulum`, `Acrobot`, `MountainCar` and
  `MountainCarContinuous` step all lanes in one batched float64 operation. With `"auto"`, the default,
  the lanes step in NumPy on the host, and 4096 or more CUDA lanes step in Torch. Below 2 lanes
  (CartPole, Acrobot) or 8 lanes (MountainCarContinuous), `"auto"` keeps the Gymnasium loop because it
  is faster there. Gymnasium still resets the lanes, owns the PCG64 streams and supplies the State.
  NumPy steps give State and outputs equal to the Gymnasium steps, and the native contract tests check
  this. Torch steps agree with Gymnasium to rounding. `"gym"`, `"numpy"` and `"torch"`
  select one backend on any device.
- Roadster: `RoadsterLoop`, `RoadsterMission`, `RoadsterTrafficJam`, `RoadsterRooftop` and `RoadsterCarHop`:
  five planning tasks for a jet-powered car in MuJoCo, ported from roadster-jets into
  `stable_envs/envs/roadster/source/`, with a complete State, the `weather.wind` / `weather.rain` physics
  leaves and the world-model camera drawn by MuJoCo's OpenGL renderer, on the GPU with a GPU's EGL driver and on
  the CPU otherwise (`roadster` extra: MuJoCo, Pillow). The visual Variation leaf `rendering.cinematic` draws the same State on a dark-space scene with
  shadows and MSAA. The exhaust smoke is drawn from the exhaust history in State (`fx`), so `render(state)`
  shows it. Large CPU batches step their lanes in worker processes and draw frames in render workers
  (`roadster-sim` and `roadster-render` tuning paths); steps that need frames but no State build no
  snapshot.
- Miniworld: `MiniWorldMovingBlockWorld` runs the 3D Dynamic Blockworld of Flow Equivariant World Models
  (`MovingBlockWorld` from the hlillemark/Miniworld fork's `flowm_release` branch) with a complete State: room and block textures,
  block kinds, sizes, poses and velocities, the agent and its carried block, and the env's generator.
  `preset="flowm"` (default) is the paper's dynamic dataset, `preset="flowm-textured"` its textured split and
  `preset="upstream"` the registered env. Each lane draws 6 to 10 blocks per episode, as the dataset script does;
  State pads to 10 blocks with a `blocks.present` mask. Frames are Miniworld's own OpenGL agent view
  (`miniworld` extra). Large batches step in worker processes and draw `render(state)` in render workers
  (`miniworld-sim` and `miniworld-render` tuning paths).
- MS-Human-700: `MSHumanLocomotion` wraps the whole-body musculoskeletal locomotion model (36 DoF,
  100 muscles) on the MuJoCo Warp backend (`mshuman` extra). Physics and State run on the env device;
  frames come from MuJoCo's OpenGL renderer. The Apache-2.0 model is fetched from LNSGroup/MS-Human-700
  at a pinned commit on first use, overridable with `STABLE_ENVS_MSHUMAN_HOME`.
- DM Control on CPU: one Numba kernel draws the frames of all 17 classes. Each tile of 8 × 8 pixels
  tests its camera, shadow and mirror rays only against the moving geometry that its bounds can reach.
  The kernel then shades the pixels in the float32 operation order of the Torch renderer.
  `backend="auto"` (the default) uses the kernel when Numba imports, and `backend="torch"` selects the
  Torch renderers. The frames are the same as the Torch frames, except for rare pixels, because Torch's
  CPU square root can round through MKL. The `dmcontrol` extra installs Numba, and the `pin-dmc` group
  pins it.
- `Pendulum(simulation_backend=...)`: CUDA lanes step Pendulum-v1 in batched float64 Torch on the GPU
  (`"auto"`, the default), and CPU lanes keep the Gymnasium simulator. Resets, the PCG64 streams and
  State export still come from Gymnasium. Torch steps match Gymnasium to rounding; `"gym"` or `"torch"`
  selects one backend on any device.
  at a pinned commit on first use, overridable with `STABLE_ENVS_MSHUMAN_HOME`. The extra needs
  MuJoCo and MuJoCo Warp 3.17.
- Classic control on CPU: a Numba kernel draws the same primitive records as the Triton kernel, in
  parallel over frames. `backend="auto"` and `backend="port"` use it when Numba imports; otherwise the
  CPU paths below apply. Frames are byte-identical to the Torch port. The `classic` extra installs
  Numba, and so do the `navigation` and `pusht` extras. `backend="auto"` draws batches of fewer than 8 rows of the default appearance with the source.
  The `pin-classic`, `pin-navigation` and `pin-pusht`
  groups pin Numba.
- Navigation on CPU: a Numba kernel draws the dots of TwoRoom, TwoRoomTarget, Piecewise and
  PiecewiseZones, in parallel over frames. `backend="auto"` uses it when Numba imports. Frames are
  byte-identical to the Torch path, and `backend="torch"` selects the Torch path.
- Craftax: `CraftaxClassicPixels`, `CraftaxClassicSymbolic`, `CraftaxPixels` and `CraftaxSymbolic`
  run the upstream JAX simulator with all lanes in one jitted `vmap` (`craftax` / `craftax-gpu` extras).
  With Craftax, the extras provide 75 classes. Fetch and `OGBCube`/`OGBScene` pin different MuJoCo versions,
  so they need separate Python environments.
- `snv.tuning`: chunk sizes, Triton block sizes, worker counts and memory budgets are named knobs with
  `snv.tuning.set`, `snv.tuning.override` and `STABLE_ENVS_<KNOB>` variables. Some defaults are derived from the GPU's free memory or the CPU's L3 size.
  Knobs never change frames or State. See docs/performance-tuning.md.
- `stable-envs tune <Env>` and `snv.tune(env)` measure the fastest knob values for one render path on
  this machine and cache them; `--show`, `--clear` and `--all` manage the cache.
- A large render on an untuned path logs one `INFO` hint to run `stable-envs tune`
  (to hide it, set `STABLE_ENVS_TUNING_HINT=0`). `STABLE_ENVS_AUTOTUNE=1` tunes such a path on first use.
- docs/environment-variables.md lists every `STABLE_ENVS_*` variable.
- `Env.describe()` reports `state_fields`: the shape and dtype of each State field.

### Changed

- `OGBCube` and `OGBScene` step faster on CPU. A step runs the OGBench step without its observation
  and info dictionaries, because the step outputs do not use them. The State leaves that only reset,
  restore and `set_variation` change (the PCG64 generators other than the environment generator, the
  task, the Variation values and the compiled model values) are kept from the last capture until one of
  these calls. State and task outputs do not change, and a test compares them with the OGBench step.
  On 8 cores of an AMD EPYC 9554 with 1024 lanes, `OGBCube` steps 7,398 times per second with one
  action per call (before: 5,773) and 8,759 times with 16 actions per call (before: 7,289). `OGBScene`
  steps 4,828 and 5,592 times per second (before: 3,755 and 4,712). The speed of steps with frames does
  not change, because drawing the frames takes most of the time.
- DM Control resets run on the `mujoco.batch` threads (`dmc_threads`). The reset of each lane is
  dm_control's `Environment.reset` without its first observation. MuJoCo releases the GIL while it computes,
  so lanes reset in parallel. Quadruped searches its start height with direct MuJoCo calls that leave the data
  exactly as dm_control's search does. At 256 lanes on 8 cores, a reset with 64 steps is 2.4x faster
  for CheetahRun and 3.2x faster for QuadrupedWalk.
- TwoRoom, TwoRoomTarget, Piecewise and PiecewiseZones on CPU: one Numba kernel moves all lanes. The
  same call returns the State, the observation, the reward and the termination flags. The kernel does
  each float32 operation of the Torch step with the same rounding, so State and task outputs do not
  change. `simulation_backend="auto"` (the default) selects the kernel on CPU when Numba imports.
  `"torch"` selects the Torch physics, and TwoRoom also accepts `"jax"`. On CUDA, TwoRoom uses
  `jax_min_envs` to select Torch or JAX, as before. `execution_mode` is `NATIVE_NUMBA` for the kernel.
- Seeded resets of the navigation classes seed only the nodes of the source reset tree that a lane
  reads. A bounded float leaf draws its values with one `uniform` call, as `Box.sample` does. The reset
  values do not change.
  On 8 cores of a Xeon Platinum 8268, episodes of physics steps (64 steps after each seeded reset) of
  the four classes are 2 to 6 times faster than before, from 1 to 1024 lanes.
- Envs that step in worker processes (OGBench, dm_control, Fetch, PyFlyt, PushT and classic control) on a
  CUDA device keep the step bookkeeping on the CPU. At the end of `step`, the outputs move to the device.
  The outputs do not change.
- `PushT` accepts continuous actions outside [-1, 1], as stable-worldmodel does. Such an action moves the target
  further from the agent. Discrete `PushTDiscrete` indices must still be in [0, 16).
- `PyFlytRocketLanding` on CPU: the physics worker processes draw the step frames of their running
  lanes with TinyRenderer, in parallel. They also draw `render` batches of 2 or more States. Before, the
  main process drew every frame. A step that returns only frames does not export a State. The frames are
  byte-identical. One process now holds at most 511 Rocket lanes, because the frames use one more
  PyBullet client.
- `SimplePointMaze` draws the `info.goal` images faster. Each distinct background is drawn once with
  Matplotlib Agg. Then only the agent disc is drawn over a copy of it, in the render worker processes on
  Linux. The image bytes do not change. Steps return the goal image without a copy: it is the same
  tensor at each step, and a `[B, T]` step returns a view of it along the time axis. Do not write into
  `info.goal`.
- OGBench point mazes, except `teleport`: the lanes of a worker step in one loop. The loop does not swap
  the NumPy legacy RNG for each lane, because a point step draws no random numbers. A step sets the
  ball velocity to zero, so a ball that does not touch a wall stays where it is. For such a lane the
  step advances only the clock and does not call MuJoCo. Workers send only the State leaves that a step
  changes. State and task outputs do not change.
- `PointMazeLargeNavigate` on CPU: the spotlight shadow rays of a frame, and of the goal and agent
  samples, test only the walls near them. The goal and the agent are drawn in two small tiles when their screen bounds do not meet.
  Frames do not change.
- Reacher with CUDA output now uses native CPU physics below 32 environments. Larger CUDA batches
  keep JAX/MJX physics. `simulation_backend` selects either implementation and `mjx_min_envs` changes
  the provisional cutoff. TwoRoom uses Torch below `jax_min_envs` and JAX at or above it. Its cutoff
  defaults to 1, which keeps JAX at every batch size. Both environments allow explicit backend selection.
- `PiecewiseZones` and `TwoRoomTarget` draw seeded resets from the same distribution tree as `Piecewise`
  and `TwoRoom`, so `reset(seed=k)` starts the agent (and `TwoRoomTarget`'s target) where stable-worldmodel
  does for the same seed. Before, they drew from `sample_states` and matched for no seed.
- PushT on CPU: a Numba rasterizer draws `render` batches of 8 or more States from the same primitive
  records as the GPU path. Frames are byte-identical to pygame. `backend` takes `"auto"`, `"source"`
  (pygame, also `"pygame"`) or `"port"`. CPU `render` of 1024 States is approximately 1.5 times faster
  than the pygame worker pool.
- On CPU, classic control draws frames with the default appearance with the Gymnasium renderer when
  pygame is installed. The `backend` argument of `_render` selects `"source"` or `"port"` (the gfxdraw
  port); the default `"auto"` selects one of them. On Linux, batches of at least `render_worker_min_rows` rows draw in up to
  `render_workers` worker processes (new knobs). Frames do not change. CPU rendering is approximately
  2 to 7 times faster and reaches the speed of the source in 8 processes.
- The PushT error for a bad `render(out=)` buffer tells that the buffer must be contiguous and match
  the frames in shape, dtype and device.
- `OGBCube` and `OGBScene` on CUDA: `backend="auto"` draws with NVIDIA OpenGL, and the default
  `supersample` is 1 on every device. The source-MSE regression compares this path with another
  stable-envs OpenGL instance using the same compiled appearance. `supersample=2` or
  `backend="mjwarp_fidelity"` keeps the MJWarp renderer.
- `OGBCube` and `OGBScene` draw OpenGL batches of 2 or more states in up to `ogb_gl_workers` worker
  processes on Linux, and read many frames back in one call. Frames do not change.
- Fetch on CPU: on Linux, batches of at least `render_worker_min_rows` states draw with software OpenGL
  in up to `render_workers` worker processes. Each worker has its own OpenGL context and writes its frames
  into one shared buffer. Frames do not change. The knobs use the new tuning path `fetch-opengl`.
  Together with the native frame size below, on 8 CPU cores rendering stored States at 224 px is
  approximately 3.5 times faster, and `step` with frames is approximately 4 times faster.
- Fetch with OpenGL (CPU, or `backend="opengl"` on CUDA) draws each frame at the requested size, as
  Gymnasium Robotics does when it is built at that size, and draws the translucent mocap markers at the
  gripper. Before, frames were drawn at 480 px and then resized, and the markers were not drawn. On CPU,
  frames now equal the Gymnasium Robotics frames byte for byte at 64, 224 and 480 px (64 States of each
  task). Per-frame MSE at 224 px against Gymnasium Robotics was 14 to 17 and is now 0.
- Fetch with OpenGL uploads each sky colour once for each OpenGL context and keeps the textures
  for later renders. A batch with a different sky colour in each lane renders as fast as a batch with
  one sky colour. Frames do not change.

- The `dmcontrol` and `dmcontrol-gpu` extras need MuJoCo 3.17 and dm_control 1.0.50, and `dmcontrol-gpu`
  installs MuJoCo Warp and MJX 3.17. The `ogbench` and `mjwarp` extras accept MuJoCo 3.17.
- The `ogbench-manipulation` extras pin MuJoCo 3.17.0 (and MuJoCo Warp 3.17.0 for `-gpu`) with dm_control
  1.0.50 or later, so they install next to `dmcontrol`.
- DM Control steps every lane in one `mujoco.batch` call on CPU threads instead of worker processes. The
  `dmc_threads` knob sets the number of threads (8 by default), at most one per lane and one per CPU in
  the process affinity. The `workers` and `worker_min_lanes` knobs no longer apply to DM Control. Reacher
  on CUDA still steps with MJX. With 1024 lanes on 32 cores and `dmc_threads=32`, physics-only stepping is
  approximately 2.4 (Humanoid) to 8 (Cartpole) times faster than with 32 worker processes. Lanes copy one
  compiled model instead of each compiling their own, so 1024 lanes build in approximately 12 s. Outputs do
  not change.
- A mid-episode physics `set_variation()` on DM Control now takes full effect from the next step. Before,
  the first step after the change still used quantities computed before it.
- One process adapter (`stable_envs._execution.process.ProcessAdapter`) runs the worker processes of
  classic Acrobot, Fetch, OGBench Maze and OGBench Cube/Scene. It replaces four copies of
  the worker loop. Outputs do not change.
- With worker processes, `set_state` checks every lane in every worker before any worker restores.
  A State that is not valid in one lane raises and changes no lane in classic Acrobot and Fetch too.
- PushT runs in worker processes on Linux for batches of at least `worker_min_lanes` lanes (tuning
  path `pusht-sim`).
- `PyFlytRocketLanding` runs in worker processes on Linux for batches of at least `worker_min_lanes`
  lanes (tuning path `rocket-sim`). A batch of more than 512 lanes uses one process per 512 lanes or
  fewer on every platform, so batches of 1024 lanes or more work.
- `visual_fidelity` records the per-frame MSE against the source as `mse_mean`, `mse_p95` and `mse_max`
  on the 0-255 scale. Only `measured_at` is required.
- The tests compare frames with a recorded source MSE for each class (`tests/fixtures/source_mse.json`)
  and with golden frames of our own renderers (`tests/fixtures/golden/`).
- TwoRoom uses JAX dynamics on CPU or CUDA and exchanges Torch tensors through DLPack.
  CUDA requires the `navigation-gpu` extra.
- TwoRoom and Piecewise calculate target distances on the selected device, preserving the source's
  success boundary without copying positions to CPU during stepping.
- Navigation steps avoid redundant State validation and copies of owned task outputs. On CUDA,
  supported Torch compilers fuse the shared TwoRoom/Piecewise task outputs, and TwoRoom packs
  physics inputs into one DLPack handoff.
- `stable-envs doctor` reports how many render paths are tuned on this machine, lists the
  `STABLE_ENVS_*` variables that are set and flags unknown names, and checks torch's compiled GPU
  architectures with binary-compatibility rules.
- A `STABLE_ENVS_*` variable with a value that does not parse raises `ValueError` naming the variable.
- `cpu3d.render(chunk_size=None)` and `raster2d.render_triton(block=None)` read the tuning knobs.
- `PushT.step` and `PushTDiscrete.step` do not export a State when the caller requests no State. Physics-only
  stepping is approximately 2.9 times faster.
- `PushT` and `PushTDiscrete` keep the State fields that change only at reset, restore or `set_variation` as
  cached batch columns, record the action history in one growing array, and draw step frames from the running
  scenes. Resets copy the engine without a second scene build. Frames, State and task outputs do not change.
- PushT worker processes draw step frames and copy only the State leaves that a step changes. The parent
  keeps the other leaves and copies them for the next State while the workers step. Process adapters
  declare this with `process_static_leaves`, `process_step_state`, `render_live` and `frame_shape`.
- `PushT.render` on CUDA is faster. On an RTX A6000 with 224 px frames, `render(state)` draws 463 frames/s
  at batch 1, 29,500 at batch 64 and 473,000 at batch 4096. The pygame worker pool draws about 3,000
  frames/s on 8 processes. Frames do not change.
- With `device="cuda"` and Triton, `PushT` and `PushTDiscrete` draw `render` and step frames on the GPU.
  A Triton kernel builds the shape records from the State tensors, and a second kernel draws them and
  resizes the canvas. The frames are byte-identical to the pygame frames. Frames with action markers,
  shapes that are not supported, or a coordinate within 1e-9 of a rounding midpoint use pygame.
- `PushT.render` and `PushTDiscrete.render` draw a batch of at least `worker_min_lanes` States in up to `workers`
  spawned processes on Linux (render path `pusht-render`). The frames do not change.
- `PyFlytRocketLanding` reads its live colours directly for the default render style and copies each State
  leaf to the host once during validation. Rendering of supplied pose rows is approximately 70 times faster.
  Frames and State do not change.

### Fixed

- DMCFingerTurnHard draws the target sphere and the tip site by default, as dm_control's `finger-turn_hard`
  does. `visibility.target` now defaults to 1, the same as DMCFingerTurnEasy. Set it to 0 for
  stable-worldmodel's default, which hides both sites.
- Classic control frames of the gfxdraw port and the Numba kernel are equal to the Gymnasium frames at
  angles within 1e-6 rad of a multiple of pi/2. The port rotates as pygame's `Vector2.rotate_rad` does: it
  wraps the angle into [0, 2 pi), rotates such angles exactly, and computes cos and sin with the C library on CPU.
- `PyFlytRocketLanding` on CUDA draws the fins, legs and fuel tank at the current body pose, as PyFlyt
  does, and does not draw the ground when the camera is below it. The mean per-frame MSE against PyFlyt
  is 4.2 on 480 px frames.
- `PyFlytRocketLanding.render(state)` on CUDA picks ground texels as TinyRenderer does, so frames that look
  down at the far ground match PyFlyt when a ground corner is close to the camera plane.
- `PyFlytRocketLanding` changes the flame colour at ignition, as the PyFlyt pixel env does. The State
  `render/rgba` flame row and the CPU and CUDA frames show the red ignition flame. The source makes this
  change only when it has a render mode. Supplied pose rows carry no ignition, so they render with the flame
  off. States saved after ignition with an earlier version do not restore.
- `PyFlytRocketLanding` reset saves and restores the engine as the PyFlyt pixel env does around its goal
  render. The restore recomputes the cached link poses at the reset pose. The reset frame, the first
  steps and the observations are then equal to the pixel env after a reset that moves the start pose.
  States saved with an earlier version do not restore.
- `PointMazeLargeNavigate` renders batches of 2048 and more on CUDA. Its spotlight pass casts wall shadow
  rays in one launch row per frame and lights a bounded number of rows at a time.

### Removed

- `MJW_CHUNK` in the DMC and OGBench packages. The `dmc_min_worlds` and `ogb_mjw_worlds` tuning knobs
  set the worlds per MJWarp launch.
- `Env.fidelity_tier`, `stable_envs.env.FIDELITY_TIERS` and the `fidelity_tier` key of `snv.info()`.
- `Env.parity_note` and `stable_envs.envs.dmcontrol.PARITY_NOTE`. `visual_fidelity` and `visual_note` describe
  how frames differ from the source.

## [0.1.0] - 2026-10-14

Initial release.

### Added

- Batched environments in seven families: classic control, DM Control, Fetch, navigation,
  OGBench, PushT and PyFlyt Rocket Landing. 70 environments register when every extra is
  installed. A missing extra disables only its family; `snv.import_errors()` names it.
- Class-first API: `snv.CartPole(num_envs=..., device=...)`, plus `snv.make`, `snv.info`,
  `snv.list_envs` and `snv.list_families`.
- Batched stepping on CPU or GPU: one `step` call takes `[B, A]` actions for one transition or
  `[B, T, A]` for a sequence, and returns a validity mask instead of resetting finished lanes.
- Batched rendering: `render(state)` draws frames from a State without advancing the simulation.
- Save and restore: `get_state` / `set_state` with a complete TensorDict `State`.
- Variation spaces: `sample_variation` / `set_variation` and per-lane reset options, compatible
  with stable-worldmodel.
- Source-comparison tests: each environment is checked against the library it ports and against
  stable-worldmodel, for State, simulation, tasks and pixels.
- `stable-envs doctor` checks the driver, Warp and MuJoCo OpenGL setup per env family.
- `stable_envs.testing` measurement helpers, the `benchmarks/` harness and MkDocs documentation.

See [Limitations](https://galilai-group.github.io/stable-envs/concepts/limitations/) for known gaps.

[Unreleased]: https://github.com/galilai-group/stable-envs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/galilai-group/stable-envs/releases/tag/v0.1.0
