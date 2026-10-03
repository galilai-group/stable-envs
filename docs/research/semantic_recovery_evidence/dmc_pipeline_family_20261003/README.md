# DMC component and pipeline batch sweep

The two-device Humanoid B64 smoke independently passed and is preserved in
../dmc_pipeline_smoke_20261003/completed-v3. This sweep uses the same harness
against current frozen integration source for eleven accepted process routes:
acrobot, pendulum, cartpole, reacher, cheetah, hopper, walker, ball_in_cup,
finger, manipulator and humanoid. Each registration must pass its own CPU and
CUDA checks; Quadruped is excluded, not promoted through sibling evidence.

Each task requests one RTX 3090, eight physical CPUs, 64 GiB and four hours;
three array tasks may run concurrently. B64/256/1024, T1/8 and no/final/all-frame
selections retain all States. Software-GL CPU and NVIDIA CUDA output execute in
separate interpreters. Construction, restoration and warmup are outside timers.
Reports include task stepping without State capture plus IPC/transfer, State
export/IPC/transfer, saved-State rendering, and ordinary versus deferred online
scheduling. Six samples, mean/SD and exact complete-State/task/frame checks are
required per cell. Rates count real transitions and emitted images separately.

Installed renderer extras from the passed smoke are reused read-only; every
recorded package file hash is checked before each task. Cache/output paths are
per-task on scratch. No accepted environment or prior evidence is overwritten.
These remain scheduling comparisons, not upstream speedups. Same-device source
baselines and the complete 35-registration performance table remain open.
