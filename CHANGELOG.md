# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Shared batched MuJoCo backend (`stable_envs/_execution/mjbatch.py`) for the MuJoCo families: native
  `mujoco.batch` on CPU, bit-exact with one float64 `MjData` per environment, and MuJoCo Warp on CUDA, with the
  substeps of one step in one captured CUDA graph. It writes and reads the integration State of a subset of
  environments, batches per-environment model parameters, raises `BufferOverflowError` when the MuJoCo Warp
  contact or constraint buffers are too small, and runs per-environment source resets in a pool of threads or
  processes.
- MuJoCo Warp backend (`WarpBatch`): the captured graphs compute subtree centers of mass, composite inertias and
  RNE internal forces with one kernel launch per pass instead of one launch per body tree level
  (`stable_envs/_execution/mjwarp_fused.py`), up to `fused_tree_max_envs` environments (default 4096). This
  applies to MuJoCo Warp 3.12.0 and 3.17.0; other versions run MuJoCo Warp unchanged. The results are equal up to
  float32 summation order. Public steps/s at 64 environments on an L40S: OGBench Cube +15%, MS-Human +16%,
  Fetch Push +9%, OGBench Scene +7%, AntMaze +6%.
- MuJoCo Warp backend (`WarpBatch`): the captured graphs run kinematics, body velocities and RNE with one thread
  block per environment and one thread per body, with a block barrier between body tree levels
  (`stable_envs/_execution/mjwarp_tree.py`). The kinematics launch also computes the geom and site poses, the
  centers of mass, the composite inertias and the joint motion axes. This removes about 12 launches per substep,
  and the three tree walks in which one thread follows a branch body by body. Same MuJoCo Warp versions and
  `fused_tree_max_envs` limit as the fused tree passes; models with more than 1024 bodies keep those passes. The
  results are equal up to float32 rounding order. Public steps/s at 64 environments on an A100: OGBench Cube +14%,
  AntMaze +14%, DMC Walker +11%, Fetch Push +8%, DMC Hopper +7%.
- DMC native/MJWarp benchmarks offer an opt-in extended batch matrix through 8192 lanes and
  a `--dry-run` preview. Reports include kernel peak host RSS through construction and worker
  exit, including failed workers on supported platforms, alongside existing construction timing.
  Seeded random-action rollouts default to the PR's three seeds, 20 warmups, a checkpoint restore,
  and 100 timed calls. Reports verify matching native/MJWarp action and checkpoint hashes.
  A focused measurement profile retains stepping, raw physics, construction and memory costs
  while skipping repeated reset/restore and rendering measurements.
- All 17 DMC classes accept `simulation_backend="mjwarp"` for direct CUDA physics on MuJoCo Warp 3.17.
  The shared adapter retains native resets, RNG, targets, physics variation and checkpoints.
  DMC-local capsule-contact corrections and ordered constraint/solver reductions support repeatable
  checkpoint replay. CUDA regressions cover all 17 classes, with focused 100-call replay checks.
  See the DMC documentation for the 3.17 measurements and numerical limitations.
- `PyFlytRocketLanding` `simulation_backend`: `"numba"` (the default on CPU) steps every environment in one call of
  a Numba port of the PyFlyt Rocket forces and the Bullet rigid-body step, in parallel over environments. Resets
  sample the start as the source does and settle in the same kernel. Each environment draws its thrust noise from
  its own seeded PCG64 generator, so a seed gives the source's start state and noise. The State holds physical
  values and the generator states; `set_state` copies them without a replay. With `"numba"` on CUDA the kernel runs
  on the host and the outputs move to the device (`execution_mode="HOST_NUMBA"`). `"numba_cuda"` (the default on
  CUDA) steps the CUDA build of the kernel, one GPU thread per environment, with each step and each reset settle in
  one CUDA graph; `"auto"` selects it from `numba_cuda_min_envs` environments (default 1). `"pybullet"` keeps the
  PyFlyt and PyBullet engines with the replay-backed State that is exact to the source. The `rocket` extra installs
  Numba.
- PushT `simulation_backend`: `PushT` and `PushTDiscrete` step every environment in one batched call of a
  port of the Pymunk contact solver. `"numba"` (the default on CPU) runs it in Numba in `float64`, in
  parallel over environments. `"triton"` (the default on CUDA) runs it in one Triton kernel in `float32`, one
  GPU thread per environment, and a second kernel records the actions and computes the observation, reward
  and success. `"jax"` selects the JAX solver, which is the CUDA default when Triton is not installed.
  `"auto"` steps CUDA batches below `triton_min_envs` (default 1) in Numba and returns CUDA tensors. The
  source reset still runs in Pymunk, in worker processes for large batches. On an L40S with 8 CPU cores,
  with a reset every 196 steps, CPU PushT runs at 7,100, 40,000 and 103,000 steps/s with 8, 64 and 1024
  environments and one action per call, and at 19,000, 85,000 and 186,000 steps/s with 16 actions per
  call. CUDA PushT runs at 5,800, 43,000 and 253,000 steps/s, and at 10,400, 66,000 and 278,000 steps/s.
  Before, CPU PushT ran at 3,700, 10,800 and 9,000 steps/s and CUDA PushT at 835, 6,000 and 69,000
  steps/s with one action per call. The source runs at 14,100, 23,400 and 28,200 steps/s in 8 worker
  processes. A State takes fewer copies to build, so a CPU step with 8 environments takes about 1 ms.
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
  shows it. Large CPU batches draw frames in render workers (`roadster-render` tuning path); steps that need
  frames but no State build no snapshot.
- Miniworld: `MiniWorldMovingBlockWorld` runs the 3D Dynamic Blockworld of Flow Equivariant World Models
  (`MovingBlockWorld` from the hlillemark/Miniworld fork's `flowm_release` branch) with a complete State: room and block textures,
  block kinds, sizes, poses and velocities, the agent and its carried block, and the reset generator.
  `preset="flowm"` (default) is the paper's dynamic dataset, `preset="flowm-textured"` its textured split and
  `preset="upstream"` the registered env. Each lane draws 6 to 10 blocks per episode, as the dataset script does;
  State pads to 10 blocks with a `blocks.present` mask. Frames are Miniworld's own OpenGL agent view
  (`miniworld` extra). Large batches draw `render(state)` in render workers (`miniworld-render` tuning
  path).
- MS-Human-700: `MSHumanLocomotion` wraps the whole-body musculoskeletal locomotion model (36 DoF,
  100 muscles; `mshuman` extra). Envs step every lane in one native `mujoco.batch`; CUDA envs of 256 or more
  lanes step on MuJoCo Warp; State is `float32` on the env device. Frames come from MuJoCo's OpenGL renderer. The
  Apache-2.0 model is fetched from LNSGroup/MS-Human-700 at a pinned commit on first use, overridable with
  `STABLE_ENVS_MSHUMAN_HOME`. The extra needs MuJoCo and MuJoCo Warp 3.17.
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

- DM Control, `OGBCube` and `OGBScene`: `render(state)` checks only the State leaves that the frame reads.
  DM Control gives `Env.render` the drawn fields: `qpos`, the episode model entries of the task (Reacher:
  `task.target_pos`) and the physics leaves that move pixels, such as the Ball-in-Cup gravity and ball size, the
  masses that move the tracking camera and the floor tilt. Their presence, shape and finite values are checked. Cube/Scene
  checks that the projected leaves are present and then validates the RenderState: dtype, shape, finite values,
  camera, target alpha, unit quaternions, colour, camera-angle and cube-size ranges and button bits. On CUDA
  these checks use one host synchronization. `set_state` keeps the checks of the leaves that only restore
  simulation (velocities, RNG, lifecycle, task and replay leaves). Cube/Scene `render(state)` also accepts a
  State with a batch size different from the env. Frames do not change.
- `PyFlytRocketLanding` with `simulation_backend="numba_cuda"` captures and stacks States with fewer copies. The
  leaves of a State are views of one device buffer, which one `torch.cat` for each packed `physics/*` leaf and
  each dtype fills, and `env.step` stacks the States of a step sequence with one `torch.stack`. Other States
  take the leafwise stack. The State values do not change. On 8 cores of an EPYC 7763 with an A100, `env.step`
  with States and one reset for each 64-step episode makes 1.36 to 1.42 times the steps/s before for 1 to 1024
  environments, and 1.15 times at 4096.
- `PyFlytRocketLanding.render(state)` and `to_render_state(state)` check only the State leaves that the frame
  reads: the body poses, the camera link pose, the pose rows and the colours. They check presence, shape, dtype,
  finiteness and the colour range on these leaves, with one copy to the host. `set_state` keeps the replay,
  history and RNG checks. A State of the Numba backends and a State of the PyBullet backend pass the same
  checks. On CUDA with `simulation_backend="pybullet"`, `render(state)` of 256 or 1024 environments is about 3.4
  times faster on an RTX 4090. `PushT.render(state)` and `to_render_state(state)` select and check only the
  leaves that the renderers read; `render(state)` on CUDA is 25% to 35% faster. The frames do not change.
- `Env.render(state)` checks only the State fields that the renderer reads: presence, shape and finite values,
  with one host synchronization. Classic control also checks their dtype. Piecewise also checks the target
  position, which it draws. `set_state` keeps the check of the full State. A family that overrides
  `_validate_state` and does not override `_validate_render_state` keeps its full check in `render`. Frames do
  not change.
- PushT on CPU (`"numba"`): a step does its Torch operations on one thread, as the OGBench mazes do, so idle
  Torch threads do not spin on the cores of the Numba threads. A step that returns one frame hands over the
  rendered frames without a copy. Frames and State do not change. Public CPU steps/s with frames and a reset
  every 64 steps, 8 cores of an AMD EPYC 7713: 13,440 at 256 environments (9,015 before) and 15,034 at 1024
  (11,167 before); with one of the 8 cores busy, 5,761 at 256 (2,808 before) and 8,555 at 1024 (4,407 before).
- Fetch CPU batch on MuJoCo 3.11: the threads take environments from one shared queue instead of an equal
  share each, so a thread that other processes delay steps fewer environments. Public CPU steps/s of
  `FetchPush` with a reset every 64 steps, 8 cores: +4% at 8, +86% at 64 and +29% at 256 environments on one
  host (AMD EPYC 7K62), and +25%, +45% and +21% with a busy process on one of the 8 cores; equal on a second
  host (Intel Xeon Gold 6240R). State and outputs do not change.
- OGBench mazes on CPU: a step does its Torch operations on one thread, and the render draws batches of fewer
  than 256 frames on one thread when the scene has no tracking light. With 8 Torch threads, idle Torch threads
  kept the cores busy while the `mujoco.batch` threads waited, and the time of an AntMaze episode of 64 steps with
  frames changed between runs by up to 10x. Frames and State do not change.
- `PointMazeLargeNavigate` on CPU draws the tracking light with Numba kernels: the lit plates of all the frames of
  a render in one pass, and the light on the body and goal samples of each composite. The Torch path did about a
  hundred elementwise operations for each group of 16 frames, and each operation waited for all the Torch threads.
  The square roots and the powers stay Torch operations on the same tensors, thus the frames are equal to the
  frames of the Torch path, byte for byte. The `ogbench` extra adds Numba. Without Numba, the Torch path draws the
  light.
- DMC CPU batch resets with a seed reseed the task RNG of each environment in place, which gives the state of a
  new `RandomState(seed + i)` without its entropy draw, and the State columns after a reset come from the
  `mujoco.batch` data in array operations. `DMCCheetahRun` resets call MuJoCo directly instead of through the
  dm_control wrappers, and read the task RNG keys in place. The data, State and RNG are bitwise those of
  dm_control's reset. Public CPU steps/s with a reset every 64 steps, 8 cores of an EPYC 7532: 78,783 at 1024
  environments (58,211 before; 47,489 for 8 dm_control processes), 68,723 at 256 (51,565), 37,386 at 64 (32,763).
- `DMCQuadrupedWalk` CPU resets search the start heights of all resetting environments together: each attempt
  computes the position stage, which includes collision, of the environments still in contact on the
  `mujoco.batch` threads. Every environment finds the same height as dm_control's search, and then repeats
  dm_control's last attempt. The data, State and RNG do not change. Public CPU steps/s with a reset every 64
  steps, 8 cores of a Xeon Platinum 8470: 33,057 at 256 environments (27,631 with the search in one thread),
  33,346 at 1024 (28,272).
- `DMCCheetahRun` CPU resets run the 200 settling steps of dm_control's reset in the `mujoco.batch` of the
  environments, on its threads. dm_control's legacy step starts with `mj_step2` on the stages of the state before
  the random joint draw: each environment does that call, and the batch runs the other 199 `mj_step` calls and the
  closing `mj_step1`. The data, State and RNG are bitwise those of dm_control's reset. The CPU batch adapter has no
  reset threads. Public CPU steps/s with a reset every 64 steps, 8 cores of a Xeon Platinum 8470: 27,863 at 64
  environments (17,212 before), 61,314 at 256 (24,204), 83,657 at 1024 (29,463).
- `DMCQuadrupedWalk` resets search the start height with MuJoCo kinematics and collision only, as only the
  contact count is read, and then repeat dm_control's last attempt with both forward passes. The data and State
  do not change. Quadruped resets run in one thread on CPU and in one reset owner on CUDA: many short MuJoCo calls
  on several threads wait for the GIL. Public CPU steps/s with a reset every 64 steps, 8 cores of a Xeon
  Platinum 8470: 13,621 at 64 environments (9,045 before), 27,631 at 256 (16,175), 28,272 at 1024 (19,839).
- Fetch: on CPU, `backend="auto"` draws frames with a compiled Numba rasterizer that follows MuJoCo's
  OpenGL renderer: the meshes, the primitive tessellations, per-vertex lighting, 4 samples per pixel, the
  skybox and the translucent markers. It is faster than the source at every batch size. The frames differ
  from the source frames only at some edges and by rounding; the mean squared error per frame is about 0.15
  on average and less than 0.4 (0-255 scale). `backend="opengl"` keeps MuJoCo OpenGL and the exact source
  frames. `backend="numba"` selects the rasterizer. The `robotics` extra now installs Numba.
- `OGBCube` and `OGBScene` copy the State in one array per dtype and make each leaf a view of it. A CUDA env on
  `mujoco.batch` (`simulation_backend="auto"` or `"native"`) copies one array per dtype to the device, not one
  per leaf. The inverse kinematics of the CPU controller skips collision detection, which it does not read.
  An all-active CPU step passes no index list to `mujoco.batch`. State and outputs do not change.
- `OGBCube` and `OGBScene` frames have the look of the OGBench pixel tasks (`visual-cube-single-v0`,
  `visual-scene-v0`): the arm is see-through and the goal cube is not drawn. The `reference_env()`
  builds OGBench with `ob_type="pixels"`, so it draws the same `front_pixels` camera. Before, the arm was
  opaque, the goal cube was drawn with alpha 0.2, and the reference drew the `front` camera of
  `ob_type="states"`. The RenderState field `appearance/target_alpha` is now 0.
- The `OGBScene` default of `light.intensity` is now 0.7, which is the OGBench light. Before, it was 0.6.
- `OGBCube` and `OGBScene` compare with the original OGBench environments (`CubeEnv`, `SceneEnv`):
  `reference_env()` builds the OGBench environment and `reference_render()` draws compact rows with its
  own renderer. Before, the reference was the class itself with `backend="opengl"`.
- MuJoCo Warp steps (`WarpBatch` and the DMC runtime) compute sensors on the last substep of a step only. The
  values that a step returns do not change, and the other substeps launch fewer kernels.
- DM Control on CUDA steps through the shared `WarpBatch` backend with the MuJoCo Warp 3.17 step as it is. A
  control round is dm_control's `legacy_step` (`mj_step2`, `n - 1` calls of `mj_step`, then `mj_step1`) in one
  captured CUDA graph, with the acceleration-stage sensor values of the last integration, and the task formulas
  run in a second captured graph. The step reads no values from the device: nonfinite controls or physics
  raise `PhysicsError` one or two steps later, and `check_buffers()` on the simulator raises at once. Before,
  every step read the action check and a physics result from the device, and a failed control round kept the
  last valid checkpoint. On an RTX 4090 with 8 CPU cores and a reset every 64 steps, Walker, Hopper, Swimmer,
  Humanoid, Quadruped, Cartpole, Manipulator, Reacher and Cheetah step 1.5 to 2.1 times as fast with 1 to 8
  environments, 1.3 to 1.8 times with 64 and 1.0 to 1.5 times with 256. With 1024 and 4096 environments the
  native resets set the time; there the step alone is faster (Cheetah, 1024 environments: 1.80 ms against
  2.67 ms; Quadruped, 4096: 8.4 ms against 9.8 ms). Over 300 public calls of smooth actions from the same
  reset, the returns differ from native CPU physics as much as before; the DMC page gives the numbers.
  Humanoid restored from a State does not repeat the original rollout bit for bit, and
  `tests/gpu/test_reproducibility.py` bounds it as it bounds the other bodies in contact.
- `WarpBatch` accepts `step_fn(m, d)`, which runs in place of the `frame_skip` calls of `mjw.step` in the step
  graph, for a family whose source steps in another call order. `batched_fields` also takes options as
  `"opt.<name>"`, with one value per environment, and `model_field(name)` is a live view of a batched model field
  or option for direct per-environment writes.
- `WarpBatch` accepts `exclude_pairs`, geom pairs that MuJoCo Warp does not test for contact. OGBCube and
  OGBScene on MuJoCo Warp exclude the pairs of Robotiq gripper links, which make no contacts. The Cube oracle
  replay has the same outcomes and first-success steps.
- Fetch on MuJoCo Warp: the step of the blocked-gripper tasks does not synchronize with the host.
- PointMaze on MuJoCo Warp integrates with Euler instead of the source's RK4. The step is about four times
  faster; positions differ from OGBench only after a wall contact, and the subgoal oracle reaches the goal at
  the same steps.
- `WarpBatch` captures its graphs with the temporary arrays of MuJoCo Warp in retained memory
  (`WarpBatch.capture`), so the graphs have no allocation nodes.
- Fetch: the twelve Fetch classes step all environments in one batched call. On CPU, the Gymnasium Robotics
  step runs on arrays and MuJoCo steps every environment in `mujoco.batch`, or on up to 8 threads when the
  installed MuJoCo has no `mujoco.batch`. State and outputs are equal to Gymnasium Robotics bit for bit, and
  the existing parity tests do not change. CUDA envs step on MuJoCo Warp at every batch size by default
  (`simulation_backend="default"`), with the 20 substeps in one captured CUDA graph and no host
  synchronization. `"auto"` keeps the CPU batch below `mjwarp_min_envs` (256) CUDA environments, and
  `"source"` keeps the Gymnasium Robotics `step` per environment in worker processes. A scripted controller
  replayed on MuJoCo Warp for 200 steps reaches the source task outcome in 124 of 128 episodes.
- `Env.step` costs less per call. It builds the output TensorDict and stacks the `states` leaf by leaf without
  TensorDict validation, keeps the action layout and the step pipeline check of each env, and allocates the
  `valid`, `state_valid` and `frame_valid` flags of one action batch in one tensor. The outputs do not change.
- `PyFlytRocketLanding` State capture is faster. A step reads from PyBullet only the leaves that a step
  can change (4 queries for each environment instead of about 60) and stacks them in the worker processes.
  The other leaves come from the last reset, configuration or restore. A zero leaf that a step does not
  change, such as `current/wrapper/current_goal` (480 x 480 x 3 bytes for each environment) without a
  render mode, is a new zero tensor and is not copied. On CUDA, the outputs of the worker processes go to
  the device when they arrive, and the unchanged leaves are copied on the device. Batch actions are checked
  in one array operation. In the worker processes, freed memory stays allocated for the next PyBullet
  clients, so a reset of 1024 environments takes 2.8 s instead of 3.4 s. The State is bit-identical to a
  full read of the engine. On 8 cores of an EPYC 9354 with an L40S, `env.step` with State and one reset
  for each 183-step episode makes 2.7k to 3.1k steps/s for 64 to 1024 environments, 3.0 to 3.5 times the
  steps/s before, on CPU and on CUDA.
- `PyFlytRocketLanding` uses less memory. A reset or a restore replaces the PyBullet client of one
  environment at a time, so a worker process holds at most one client more than its environments. Before, a
  worker made a new client for each environment before it closed the old clients, and its memory stayed at
  two clients (about 54 MB) for each environment. A large State leaf that is all zero, such as
  `current/wrapper/current_goal` without a render mode, is not stored for each environment. Reset checks
  the arguments of every environment before it changes one. If PyBullet fails during a reset, the
  environments before the failed one can have a new episode. A restore keeps the State of each environment
  that it replaced and replays these States if it fails, so no environment changes. One process holds at
  most 1022 Rocket lanes. With frames on CUDA and 8 cores of an L40S pod, the peak host memory is 37.0 GB
  for 1024 environments (before: 67.3 GB) and 70.0 GB for 2048 environments (before: out of memory on a
  125 GB pod). The PyFlyt source in 8 worker processes uses 37.5 GB and 70.6 GB.
- With worker processes and `process_static_leaves`, a step that only needs a frame from `frame_fields`
  does not export the State. On CUDA without host outputs, the static State leaves are cached and copied
  on the device.
- OGBench mazes: `env.step` stacks the `states` of a CPU env leaf by leaf with the shared core stack. On CUDA,
  MuJoCo Warp copies the State into one buffer with one call, and the `states` of a step are one copy of these
  buffers. The State values do not change.
- Classic control on CPU: a step does not copy the State into the render rows of the env. The rows are copied
  from the simulator when a reader needs them (`render()` without a State, the live rows). The Gymnasium
  backend builds its State and outputs without TensorDict validation. The outputs do not change.
- MS-Human-700 on CPU: `env.step` stacks the `states` with the shared core stack on one Torch thread, instead of
  a NumPy stack per leaf. The State values do not change.
- MS-Human-700: `MSHumanLocomotion` steps on the shared batched MuJoCo backend. CUDA envs step on MuJoCo Warp at
  every batch size by default (`simulation_backend="default"`); `"auto"` keeps `mujoco.batch` below
  `mjwarp_min_envs` (256) CUDA environments. The MuJoCo Warp step does not synchronize the host.
- Classic control on CUDA: `CartPole`, `Pendulum`, `Acrobot`, `MountainCar` and `MountainCarContinuous`
  step on the GPU at every batch size by default. The State stays on the GPU. One Triton kernel per class
  steps all environments, writes the outputs and the render rows, and a CUDA graph replays it, so a step does not
  read data back to the host. Resets draw the initial states on the GPU from the Gymnasium distributions;
  the `rng` State fields hold each environment's key and reset count. Steps agree with Gymnasium to rounding.
  `simulation_backend="auto"` uses the NumPy steps on the host below `torch_min_envs` CUDA environments
  (default 1). On the CPU the default does not change.
- Miniworld: `MiniWorldMovingBlockWorld` steps all environments in one batch: a Numba kernel on CPU
  (`simulation_backend="numba"`) and a Triton kernel in a CUDA graph on CUDA (`"triton"`); `"torch"` runs
  on any device, and `"auto"` steps CUDA batches below `triton_min_envs` (default 1) with Numba. The Numba step
  gives the fork's values from the same State, bit for bit. Resets sample the fork's reset distribution
  from a counter-based generator; State replaces the PCG64 fields with `rng.seed` and `rng.episode`. The
  worker processes and the `miniworld-sim` tuning path are removed. The `miniworld` extra installs Numba.
- Roadster: the five classes step every environment in one batch on the shared batched MuJoCo backend
  (`simulation_backend`, `mjwarp_min_envs`). On CPU the source's autopilot, drag and wind, fuel, course
  bookkeeping, crash checks, cost and reward run in NumPy and MuJoCo runs in one native `mujoco.batch`, bit for
  bit as the source env. On CUDA the same logic runs in Torch without host reads and MuJoCo runs on MuJoCo Warp
  in float32 (approximate; replays of the source's scripted expert keep every outcome). Contact sensors give the
  contacts the task reads; the visual geoms outside the car are left out of the physics model. Resets draw the
  source's per-environment start and goal on the host and write all environments in one call. The worker
  processes and the `roadster-sim` tuning path are removed. The `roadster` extra needs MuJoCo 3.17;
  `roadster-gpu` adds MuJoCo Warp. On an RTX 4090 pod with 8 CPU cores, with a reset every episode,
  `RoadsterTrafficJam` steps at 320, 1,400, 3,200 and 5,700 steps/s on CPU with 1, 8, 64 and 1024
  environments (before: 230, 260, 1,370 and 1,300; the source env: 260, 1,120, 1,260 and 1,330 in 8 worker
  processes), and at 80, 620, 3,300, 31,000 and 78,000 steps/s on CUDA with 1, 8, 64, 1024 and 4096.
- `OGBCube` and `OGBScene` step on the shared batched MuJoCo backend. A CPU env steps every environment in one
  `mujoco.batch` call. The controller and its inverse kinematics run for all environments together, with MuJoCo's
  kinematics and Jacobians and the operations of the OGBench step in the same order. The State and the outputs
  are equal to OGBench, bit for bit, and tests compare them with the per-environment OGBench step. A CUDA env
  steps on MuJoCo Warp (`float32`) at every batch size by default (`simulation_backend="default"`). One Warp
  kernel computes the controller in `float64`, and the step does not synchronize the host. When a Scene cube
  leaves the workspace, the device puts it at a position that the host drew before the step from the
  environment generator, as OGBench draws it. `"auto"` keeps `mujoco.batch` below `mjwarp_min_envs`
  (256) CUDA environments. On the scripted OGBench Cube oracle, MuJoCo Warp gives each episode the success
  and the first-success step of OGBench.
  On an L40S with 8 EPYC 9374F cores, with uniformly random actions and a reset every 200 steps, a CPU
  `OGBCube` runs 8,634 steps/s with 64 environments and 13,383 with 1024 (OGBench in 8 worker processes:
  3,045 and 4,452; before: 3,071 and 4,440). A CUDA `OGBCube` runs 12,493 steps/s with 256 environments,
  37,071 with 1024 and 78,671 with 4096 (before: 3,840 and 18,117 with 256 and 1024). A CUDA `OGBScene`
  runs 17,989 steps/s with 1024 environments and 41,221 with 4096 (before: 5,248 and 5,479). Below 64
  environments, MuJoCo Warp is slower than `mujoco.batch` and than OGBench. Worker processes reset the
  environments of batches of 64 or more in parallel.
- `PyFlytRocketLanding` on CUDA draws step frames from the body poses, camera link pose and colours of the
  running lanes. The physics workers write these fields after each step, and the main process draws them on
  the GPU while the workers run the next step. A step with frames no longer builds, copies and checks a full
  State for each frame. The frames do not change.
- DMC MJWarp keeps compact model columns and independent per-environment RNG checkpoints, avoiding a
  full native environment and static-asset copy per environment. Cheetah and Quadruped reset their
  environments on one native owner per `dmc_threads` thread; other tasks use one owner. Reset preparation
  failures leave the installed records intact.
- Ordinary DMC MJWarp steps capture both physics rounds and task formulas in one guarded graph,
  with one input-validation read and one combined physics-result read. A failed first round skips
  the second; a failed second round preserves the first checkpoint. Observations and info are built
  once, both rewards are retained, and returned outputs own their storage. Lifecycle counters update
  on device; native reset boundaries retain their success checks and RNG behavior.
- DMC MJWarp reuses temporary GPU storage across the ordered control rounds. Its CUDA graphs stay
  captured through episode boundaries: a mixed reset advances the full batch in the same runtime and
  writes the reset States afterward, and no step clears the CUDA allocator cache.
- DM Control on a CUDA device steps with MuJoCo Warp by default at every batch size, including
  Reacher, which no longer defaults to native physics below 32 environments or to MJX above. On CPU the
  default stays native. `simulation_backend="auto"` is opt-in: it uses native physics below
  `mjwarp_min_envs` and MuJoCo Warp from there, with a measured cutoff per task. `"native"` runs physics
  on CPU with State and outputs on CUDA, and `"mjx"` remains Reacher's explicit MJX choice. The
  `mjx_min_envs` argument is removed.
- The OGBench mazes (`PointMaze*`, `AntMaze*`, `HumanoidMaze*`) step all environments in one batched call on the
  shared MuJoCo backend. On CPU, `mujoco.batch` steps them in float64, and the State, observations, task
  outputs, resets and RNG streams are equal to OGBench, bit for bit. Resets sample the start, the goal and the
  task from the RNGs of each environment, and do not run the OGBench reset. On CUDA, MuJoCo Warp steps them in
  float32 by default, with no host synchronization in the step; teleport destinations come from each
  environment's own RNG. `simulation_backend="auto"` steps CUDA batches smaller than `mjwarp_min_envs` on
  `mujoco.batch`, and `"native"` and `"mjwarp"` select one backend. Before, each environment called the OGBench
  `step` in a Python loop in worker processes, on both devices.
- `PushT` and `PushTDiscrete` on CPU step the Numba port of the Pymunk contact solver, not Pymunk. The
  port is not bit-identical to Pymunk. The CPU State also holds the solver-cache leaves (`current/jax`), so
  `set_state` restores without replaying the action history, and a CPU env restores a CUDA State. A State
  from an earlier release (`recipe/version` 2) still restores by replay in Pymunk.
- The PushT physics ports (Numba, Triton and JAX) follow Munk2D, the Chipmunk2D fork in pymunk 7.2.0. The
  walls are segments of radius 2 with the Munk2D segment-polygon contacts, so a block pushed hard into a
  wall passes through it, as in the source. Before, the walls were half-planes. An arbiter keeps its
  contacts for 3 steps after its shapes separate, and a circle contact keeps its cached impulse when the
  closest polygon face changes. The solver visits the shape pairs in the order that a replay of the Munk2D
  bounding-box tree reports them. The collision slop and bias are the Munk2D `float` literals, and the
  agent PD control runs in `float32` for `float32` actions, as stable-worldmodel computes it. A reset
  copies the bias velocities, the cached impulses and the broadphase tree from the Pymunk reset. Contact
  segments in frames start at the wall-side contact point, as Munk2D draws them. The State has two new
  leaves: `current/jax/tree` and `current/jax/tree_bounds`. Over 200 steps of a scripted pushing policy
  from 256 source resets, replayed on the CPU port, the block position gap at step 200 is at most 0.001 px
  in all 256 episodes, including the 70 where the source block passes through a wall. Before, the median
  gap was 0.19 px, the 90th percentile 383 px, and 112 blocks were more than 1 px away. With a sampling
  planner that reaches the goal in 57 of 256 source episodes, the replayed success flags are equal at every
  step, and 255 episodes stay within 2e-6 px. Triton, in `float32`, has a median gap of 0.0004 px at step
  200; in 32 of the 256 pushing episodes its rounding grows to a different trajectory near a wall. The JAX
  step on CUDA runs the broadphase on the CPU through a host callback and takes about 45 ms, and the first
  Triton step compiles for about 50 s. On an L40S with 8 CPU cores and a reset every 196 steps, the Numba and
  Triton steps/s change by -0.5% to -8.5% with 1 to 1024 environments (median of 3 runs).
- `OGBCube` and `OGBScene` draw CPU frames with the new `opengl_fast` backend by default. It is MuJoCo
  software OpenGL with simplified arm meshes and a 2048 shadow map. The arm meshes keep one vertex per grid
  cell of 3 mm, so they keep 22 % of their triangles.
  On 8 CPU cores, offline frames are 2.1 to 2.9 times as fast as OGBench in 8 processes (before: 0.3 to
  1.0 times), and steps with frames are 2.3 to 2.7 times as fast with 8 and 64 environments (before: 0.4
  to 0.9 times). The mean frame MSE against OGBench is 1.3 (Cube) and 3.0 (Scene) on the 0-255 scale. `backend="opengl"` draws the source's frames. CUDA frames do not change.
  A CPU batch with one state for each OpenGL worker (8 by default) draws in the workers. Before, a batch
  needed 16 states.
- Classic control CPU frames are faster at small batches. The frames do not change. A step with an action
  sequence keeps the render rows of each step and draws all the frames in one render call. The frames go
  into memory that the env uses again when no tensor holds the frames of an earlier step. Torch uses one
  thread during the Numba draw, so idle Torch threads do not take CPU time from the Numba threads. The
  Numba renderer also clips the polygon edges in Numba.
- `PointMazeLargeNavigate` CPU frames are faster at large batches. The frames do not change. On CPU, the
  lit passes draw 16 rows at a time, so each shadow ray tests fewer walls.
- `OGBCube` and `OGBScene` use the OGBench cube size. The default `cube.size` is now 0.02, as in OGBench.
  Before, the default was the stable-worldmodel value `float32(0.02)`, which compiles to
  0.019999999552965164. That size changed the cube mass and inertia, so contacts moved the physics away
  from OGBench after about 100 steps. Now the default physics model, the reset and the physics are equal
  to OGBench, bit for bit, and a test compares 200 steps for each lane. The frames do not change. The
  default model is different from the stable-worldmodel model in the cube size, mass and inertia. A State
  that you saved before this change keeps its cube size when you restore it.
- `OGBCube` and `OGBScene` set `reference_id` to the OGBench pixel tasks, `ogbench/visual-cube-single-v0`
  and `ogbench/visual-scene-v0`.
- `OGBCube` and `OGBScene` reset as OGBench 1.2.1 resets. Before, the reset used the stable-worldmodel
  rule: the arm started at the `agent.ee_start_position` Variation leaf, and the task came from a
  separate generator. Now the end effector starts at the position that OGBench samples from the
  environment generator. OGBench samples the task from NumPy's global RNG, so each lane keeps its own copy
  of that RNG in the `rng/task` leaves. A seeded reset seeds the copy as `np.random.seed(seed + lane)`
  does, and an unseeded reset continues it. OGBench samples the two settling actions of a reset from an
  action space that has no seed. The lanes sample them from the seeded `rng/action` generator, as before.
  With these two inputs set, each reset gives the State of the OGBench reset, and a test compares the
  two for each lane, for repeated resets and after a restore. The arm starts at `agent.ee_start_position`
  only when the reset options name this leaf (in `variation` or `variation_values`), or when
  `set_variation` sets its anchor to a value that is not the default. In these resets, the arm uses the
  stable-worldmodel rule, and OGBench samples only the yaw. The State leaves do not change. A State that
  you saved before this change restores correctly, but the next reset uses the new rule and gives a
  different start.
- `PushT` and `PushTDiscrete` on CUDA reset faster. On Linux, a batch of at least `worker_min_lanes` lanes
  resets in up to `workers` worker processes (tuning path `pusht-sim`). JAX steps the physics on the GPU,
  as before. The workers write the State columns and the body geometry of the new episodes into shared
  memory. The State, the frames, the rewards and the success flags do not change. On an L40S with 8 EPYC
  cores, `PushT` runs 64-step episodes with `reset(seed)` at 64,000 steps/s with 1024 lanes and at 98,000
  steps/s with 2048 lanes. Before, it ran at 22,000 and 25,000 steps/s. The CPU source runs at 51,000 and
  55,000 steps/s on the same machine. A 1024-lane reset takes 0.3 s. Before, it took 2.5 s.
- `OGBCube` and `OGBScene` step faster on CPU. A step runs the OGBench step without its observation
  and info dictionaries, because the step outputs do not use them. The State leaves that only reset,
  restore and `set_variation` change (the PCG64 generators other than the environment generator, the
  task, the Variation values and the compiled model values) are kept from the last capture until one of
  these calls. State and task outputs do not change, and a test compares them with the OGBench step.
  On 8 cores of an AMD EPYC 9554 with 1024 lanes, `OGBCube` steps 7,398 times per second with one
  action per call (before: 5,773) and 8,759 times with 16 actions per call (before: 7,289). `OGBScene`
  steps 4,828 and 5,592 times per second (before: 3,755 and 4,712). The speed of steps with frames does
  not change, because drawing the frames takes most of the time.
- DM Control steps on a persistent `mujoco.batch`, which keeps an `MjData` for each environment. A step
  continues from the position and velocity stages that the last call computed, as dm_control's step does,
  and does not compute them again. A physics Variation and a restored State compute the stages again with the
  changed models. State and outputs do not change.
- DM Control batches choose the number of `mujoco.batch` threads from the MuJoCo work of one step call. The
  env times one MuJoCo step of the task when you construct it. Each thread then gets about 0.1 ms of work per
  call and at least two environments, up to `dmc_threads`. Small batches of light tasks thus use one or two
  threads: each thread that a call wakes adds a delay, and a thread for each CPU slows the Python thread that
  calls the batch. With 8 environments on 8 cores, `DMCCheetahRun` used 8 threads and stepped at 0.6 times
  the speed of dm_control in 8 worker processes. It uses one or two threads and steps at 1.9 times that
  speed. An explicit `threads` argument to `make_batch` sets the exact number. State and outputs do not change.
- DM Control CPU steps build their outputs and State without the per-leaf checks of `TensorDict`, and read the
  State fields of the batch through views that they bind once. Each step returns new output tensors, so the
  step dispatcher does not copy them again. State and outputs do not change.
- DM Control resets skip dm_control's first observation, which the batch task computes. CheetahRun and
  QuadrupedWalk reset their lanes on the `mujoco.batch` threads (`dmc_threads`): their resets are mostly MuJoCo
  work (200 settling steps; the start-height search), and MuJoCo releases the GIL while it computes. The other
  domains reset in one thread, which is faster for their short resets. Quadruped searches its start height with
  direct MuJoCo calls that leave the data exactly as dm_control's search does. On the Hydra EPYC 9554 with
  8 cores, CheetahRun steps 3.4x faster and QuadrupedWalk 3.2x faster, resets included.
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
  lanes with TinyRenderer, in parallel. Before, the main process drew every frame. A step that returns
  only frames does not export a State. The frames are byte-identical. One process now holds at most 511
  Rocket lanes, because the frames use one more PyBullet client.
- `PyFlytRocketLanding` `render` of full States on CPU is faster:
  - One check of the whole batch replaces the State check of each row. The row check took 2 to 7 ms
    for each State with a long action history.
  - On Linux, render worker processes draw batches of 2 or more States, for every number of lanes.
    The main process checks the State while the workers draw. `render_worker_min_rows` is 2 for Rocket
    when it is not set. Before, only batches of 64 or more lanes drew in parallel, in the physics workers.
  - TinyRenderer pixels are read with `bytes` instead of `np.asarray`, which takes 0.6 ms instead of
    3.9 ms per 224 px frame. Step frames get this speed-up too.

  The frames are byte-identical. Invalid State gives the same errors as before.
- CPU render worker pools wake the worker that shares the CPU of the main process last. Before, that
  worker preempted the main process while it sent the other requests.
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
  keep JAX/MJX physics. `simulation_backend` selects native, MJX or MJWarp explicitly, and
  `mjx_min_envs` changes the provisional cutoff. TwoRoom uses Torch below `jax_min_envs` and JAX at or above it.
  Its cutoff defaults to 1, which keeps JAX at every batch size. Both environments allow explicit backend selection.
- `PiecewiseZones` and `TwoRoomTarget` draw seeded resets from the same distribution tree as `Piecewise`
  and `TwoRoom`, so `reset(seed=k)` starts the agent (and `TwoRoomTarget`'s target) where stable-worldmodel
  does for the same seed. Before, they drew from `sample_states` and matched for no seed.
- PushT on CPU: a Numba rasterizer draws `render` frames from the same primitive records as the GPU
  path. Frames are byte-identical to pygame. `backend` takes `"auto"` or `"source"` (pygame, also
  `"pygame"`). With 64 threads, CPU `render` of 1024 States at 224 px is approximately 16 times faster
  than the pygame worker pool (`benchmarks/render.py`).
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
- Native DM Control steps every lane in one `mujoco.batch` call on CPU threads instead of worker processes. The
  `dmc_threads` knob sets the number of threads (8 by default), at most one per lane and one per CPU in
  the process affinity. The `workers` and `worker_min_lanes` knobs no longer apply to DM Control.
  With 1024 lanes on 32 cores and `dmc_threads=32`, native physics-only stepping is
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

- MS-Human: `state_names` and `describe()["render_state_names"]` name the 36 render state columns after the
  model joints. They were empty.
- Miniworld: the first `render` in a process compiles the room display list once. The render env records the
  room that its constructor compiled, so its first frame does not compile the same room again.
- `WarpBatch` builds its CUDA graphs when MuJoCo Warp frees temporary arrays at other points in the captured
  call than in the first call. MuJoCo Warp frees some of them only when the garbage collector runs, and the
  capture then raised `RuntimeError` ("requested temporary memory that its first call did not"), for example for
  a second DMC Cheetah batch in one process. `WarpBatch` pauses the garbage collector during the first call and
  the capture, and collects when a temporary array finds no free buffer.
- Classic control `render` of fewer than 8 rows with a per-environment appearance draws the frames. Before,
  it raised `ValueError`. The check for the source appearance compares with the defaults of the variation
  space.
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

- The DMC copies of MuJoCo Warp internals, `dmcontrol/simulation/mjwarp_runtime.py` and `mjwarp_pair.py`
  (1,715 lines): the dense pyramidal and elliptic Newton loops with ordered reductions, the constraint-row
  ordering, the plane-capsule contact-frame and capsule-capsule contact corrections, and the paired two-round
  graph. Their tests are removed with them.
- The MJX physics of `OGBCube` on CUDA (`cube_mjx.py`). MuJoCo Warp steps CUDA Cube batches.
- `mujoco-mjx` and `jax` from the `ogbench-manipulation-gpu` extra. No Cube or Scene code uses them.
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
