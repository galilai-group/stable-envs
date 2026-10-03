# Navigation serial/parallel source comparison: prepared, not cluster-accepted

The pinned source is [stable-worldmodel 176d54532b3c0a251a4f509b74f2fdacc143a63d](https://github.com/galilai-group/stable-worldmodel/tree/176d54532b3c0a251a4f509b74f2fdacc143a63d).
The source manifest hashes registration, spaces/utilities and all three actual
class implementations. No upstream environment logic is copied into production
or modified for this benchmark. Namespace loading skips unrelated package
initializers; the independent existing oracle is used only for construction,
reset and untimed complete-State snapshot/restoration.

| Registration | Pinned class | Source stepping/rendering | Library execution |
| --- | --- | --- | --- |
| swm/TwoRoom-v1 | TwoRoomEnv, envs/two_room/env.py | Scalar CPU Torch; no device argument; CPU grids and NumPy output | Native batched Torch, CPU/CUDA |
| swm/Piecewise-v0 | PiecewiseEnv, envs/piecewise/piecewise_env.py | Scalar CPU Torch; no device argument; CPU grids and NumPy output | Native batched Torch, CPU/CUDA |
| swm/SimplePointMaze-v0 | SimplePointMazeEnv, envs/simple_point_maze.py | Scalar NumPy dynamics, Matplotlib Agg rendering | Native batched Torch, CPU/CUDA; source-compatible cached goal |

All three registrations directly select these classes and specify no registered
time limit. No auto-reset is added. The source APIs expose neither a batch axis
nor a GPU device selection. Moving individual tensors would constitute a source
port, not a supported source GPU baseline. CUDA source rows therefore mean CPU
simulation/rendering plus delivery to CUDA. This is explicit in every CSV row.

The benchmark runs source serially and in eight persistent Gymnasium spawn
workers with shared observation buffers and owned parent output copies. Each
worker owns a source shard; no source get_state is inserted into timed stepping.
Numeric task fields, selected images and valid masks cross shared buffers.
SimplePointMaze's 500x500 cached info.goal is retained at every step. State
snapshots/restoration for parity checks use the oracle outside timing.

Our ordinary native-batched public step additionally returns complete final
State. Intermediate State history is disabled in this source comparison; the
slice-only production API is unchanged. B16/64 and T1/8 cover no/final/all frame
selection. Parent Torch uses eight threads for **both** output devices, so source
CPU rendering is not artificially limited when delivering to CUDA. Each source
worker uses one thread. All strategies share the same eight-core allocation.

Both callers stop terminal lanes within each action chunk, pad invalid task
slots with zero and hold final States/images. Seed 17 plus lane offset is used
without editing sampled starting States. TwoRoom/Piecewise B2/T8 actually have
nine valid transitions, not sixteen. Steps/s counts valid transitions. FPS
counts selected emitted images, including held terminal images; it is not a
claim that each such image required a new rasterization. Full State/task outputs
and source serial/parallel images must match exactly. Library/source image MSE
must be <=0.001 for every selected frame. Shared-buffer output ownership is
checked after a different subsequent action sequence.

The final frozen local bundle independently passes all three B2/P2 CPU runs:
6 parity cases, 18 six-sample online timing cells and 3 six-sample component
cells per registration. Maximum normalized MSE: 1.124e-9 TwoRoom/Piecewise,
1.9094e-5 SimplePointMaze. Versions: Torch 2.14.0, NumPy 2.5.3, Gymnasium 1.3.0,
Matplotlib 3.10.9, TensorDict 0.14.2. These small local runs validate the harness;
they are not cluster performance acceptance or a new broad renderer corpus.

The strict cluster summarizer requires both devices, all B16/64 cases, eight
cores/workers, RTX 3090 identity, source/implementation provenance, exact gates,
all six samples and complete component coverage. Raw task-step-with-State,
get_state and saved-State rendering are measured separately on our ordinary
route. They are stage diagnostics, not independent additive throughput estimates
or source speedup claims. The source tables retain serial and parallel rates,
mean/SD and ratios against both, including the faster source strategy.

`run.sh` uses the existing pinned Oscar Matplotlib extras read-only. Verify the
233-file candidate manifest and six-file source manifest before starting. The
candidate reconstructs from base 4dd645b5 plus the three new Navigation benchmark
scripts committed with this evidence. No production changes. The job is **not
submitted** because Oscar SSH authentication is unavailable. `submission.json`
records that distinction; CPU/CUDA cluster results remain open. Earlier running
Cube/Scene and classic-source jobs are separate and must not be restarted merely
because their status cannot currently be observed.
