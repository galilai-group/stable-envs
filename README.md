<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/logo-dark.png">
  <img src="docs/assets/brand/logo.png" alt="stable-envs" width="340">
</picture>

**Famous research environments, rendered on the GPU as a pure function of a compact state, batched over
thousands of environments in one launch.**

[![License: MIT](https://img.shields.io/badge/license-MIT-black.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-black.svg)](pyproject.toml)

The consequence is that you stop storing image datasets: store the states, regenerate the frames
live, and choose the camera, the resolution and the appearance **afterwards**.

That last part is the point. A stored corpus fixes every one of those choices at collection
time — before the experiment that needs them exists. An ablation over 3 resolutions and 8
appearances is 24 datasets to generate, name, version and keep consistent; add an axis and it
multiplies again. A state corpus is written **once** and the cross product becomes an argument
to `render()`. On PushT, 29 appearance dimensions give **1,024 distinct looks in one 2.85 ms
batch** — 715× faster than looping the same renderer over the same styles — and the default
is the official appearance, bit for bit, so it costs nothing to ignore.

The storage win is universal (1,929× against raw pixels, 90× against JPEG). The *speed* win is
per-family and is **not** universal — against nvJPEG on the GPU it is a wash — see [docs/benchmarks.md](docs/benchmarks.md)
for which families beat which read path, including the ones that do not.

PushT is 5 floats per step against a 96×96×3 frame: 20 bytes against 27,648. Those 20 bytes come back as
a frame in 0.34 µs on one H200 — about three million frames a second against the official
renderer's few hundred, with the current run always in
[`examples/results/bench_gpu_vs_cpu.md`](examples/results/bench_gpu_vs_cpu.md) — and **100% of
the interior pixels are bit-identical** to it.

## Keep your training loop. Delete your image dataset.

Your loop already reads frames from somewhere — an image folder, a Lance table, a WebDataset shard.
That *somewhere* is what this replaces. The model, the optimiser, the sampler and the loop body do not
change.

<table>
<tr><th align="left">Before — 27 KB per sample, on disk</th><th align="left">After — 20 B per sample, in RAM</th></tr>
<tr><td valign="top">

```python
ds = ImageFolder("frames/", to_tensor)
loader = DataLoader(ds, batch_size=1024,
                    num_workers=8)

for frames, target in loader:
    frames = frames.cuda(non_blocking=True)
    loss = mse(net(frames), target)
    loss.backward(); opt.step()
```

</td><td valign="top">

```python
env = se.make("PushT", batch=1024, device="cuda")
ds = TensorDataset(states)
loader = DataLoader(ds, batch_size=1024)

for (s,) in loader:
    frames = env.set_state(s).render()   # <- the swap
    loss = mse(net(frames), env.get_state())
    loss.backward(); opt.step()
```

</td></tr></table>

`render()` returns `(B, H, W, 3)` uint8 **already on the device**. Nothing is copied from host, nothing
is decoded, and no worker process exists to configure.

**It is the same picture** — the official CPU renderer and the GPU renderer, fed the *same recorded
states*. One state stream, two renderers, not two rollouts that look alike:

https://github.com/galilai-group/stable-envs/raw/main/docs/assets/media/PushT/sidebyside.mp4

More per family in the [gallery](https://galilai-group.github.io/stable-envs/gallery/).

**What actually changes, stated precisely.** This replaces the *dataset*, not the simulator — worth
being exact about, because that is where a drop-in claim usually turns out to be false. Rendering is a
drop-in: `render` is a pure function of a `(B, state_dim)` stack, same states in, same pixels out.
Stepping is not: `StableEnv` deliberately has no universal `step`, because a contact solver cannot make
a bit-exactness claim the way a rasterizer can ([why](docs/concepts/physics.md)). So the shape that
works is **your existing simulator produces states, stable-envs turns states into pixels**. If your
pipeline only ever kept frames, there is a one-time step: re-run your rollouts recording the compact
state instead of images.

## Install

```bash
pip install -e ".[all]"           # 191 envs, one interpreter
pip install -e .                  # or just the core: numpy, torch, gymnasium
pip install -e ".[pusht]"         # gym_pusht, pymunk, pygame — to run the parity tests
pip install -e ".[hf]"            # datasets, for the HuggingFace corpus round-trip
```

Four packages in `[all]` need `--no-deps` or an override, because their declared pins are contradicted
by the combination this project actually runs; the recipe is on that extra in `pyproject.toml`.
Overcooked's five envs are the one family that needs its own interpreter.

A CUDA device is required: the product is a GPU renderer, and the test suite skips loudly rather than
going green without one.

```python
import stable_envs as se

env = se.make("PushT", batch=16384, size=96, device="cuda")

states = env.sample_states(16384)             # (16384, 5) float32, the env's own reset distribution
frames = env.render(states)                   # (16384, 96, 96, 3) uint8, on device
```

`env.describe()` returns the state layout, the reference env id and every appearance knob — enough to
consume an env whose source you have never read.

## Three ways to use it

### 1. Render inside your training step — the recommended one

Bring your own dataset and `DataLoader` **over states**, and render in the forward pass. stable-envs
supplies the renderer; slicing a `(N, state_dim)` float32 array is something torch and HuggingFace
already do well, so there is no library dataset class and no `collate_fn` in this path.

**Put the whole corpus in RAM and slice it.** This is the one place storing states instead of frames
should change how you write the loop. States are small enough that the entire dataset is resident —
PushT is 20 bytes a row, so 200,000 states is 4 MB — and once it is, **the loader has no I/O left to
hide**, which is the only reason a `DataLoader` is shaped the way it is. A stock loader calls
`__getitem__` once per sample and hands the results to `default_collate` to stack back together. When
each `__getitem__` is a disk seek and a JPEG decode, that is worth paying and workers are worth
spawning. When each one is a 20-byte view of an already-contiguous tensor, it is about **3.4 µs of
Python per row** spent taking an array apart and immediately reassembling it. Ask the sampler for the
batch instead, and index once:

```python
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

env = se.make("PushT", batch=4096, size=96, device="cuda")
ds = TensorDataset(torch.as_tensor(states))                  # the whole corpus, in RAM
sampler = BatchSampler(RandomSampler(ds), batch_size=4096, drop_last=False)
loader = DataLoader(ds, sampler=sampler, batch_size=None)    # torch's own collate, one slice

for (s,) in loader:
    frames = env.set_state(s).render()          # (4096, 96, 96, 3) uint8, on device
    loss = criterion(net(frames), target(env.get_state()))
```

Measured on PushT at 96 px, ms per batch, same states and same render throughout:

| batch | per-item + `default_collate` | `BatchSampler`, one slice | ceiling (no loader at all) |
|---|---|---|---|
| 256 | 3.426 | 2.666 | 2.524 |
| 1024 | 5.600 | 2.886 | 2.774 |
| 4096 | **17.705** | **3.677** | 3.369 |

At B≤256 the render dominates and a plain `DataLoader(ds, batch_size=256)` is fine. At 4096 the
per-item path is **4.8×** slower, and it is not the bytes: with the render removed, the fetch alone is
14.890 ms per-item against 0.794 ms for one batched index. Two corollaries that are easy to get
backwards — **`num_workers` is the wrong fix** (eight workers reach 6.397 ms, still 1.7× off, spending
processes and IPC to parallelise Python overhead that should not exist; there is no disk here to
overlap with), and **putting the states on the GPU is *slower***, 21.191 ms at B=4096 and the worst of
every option measured, because a CUDA `TensorDataset` makes 4,096 tiny device slices. Keep states on
the host and let `set_state` do one transfer.

`drop_last` is left at torch's default of `False`, because the short final batch of an epoch is
**rendered, not dropped**: `set_state` takes fewer rows than the env holds and draws exactly those.
Measured on one H200, that costs nothing — PushT at 96 px renders 4,095 states in 3.371 ms against
3.341 ms for 4,096. `env.get_state()` hands back those same rows, on device, so the target cannot drift
out of step with the frames.

`set_state` also checks what it is handed, because the reshape behind it has two failure modes that
produce a *valid* array: a batch of frames passed where states were meant, and a row that is the wrong
width. Both used to render a plausible picture of nothing. So do NaN and inf. Each is refused by name:

```
PushT.set_state: expected states shaped (batch, 5) and got (256, 96, 96, 3) uint8. That is a batch
of FRAMES, not states. set_state/render run in the other direction: they take the 5 number(s) a
frame is drawn FROM (agent_x, agent_y, block_x, block_y, block_angle) and produce the pixels —
`frames = env.set_state(states).render()`. ...
```

To keep the allocation out of the hot loop, render into a buffer you own:

```python
out = torch.empty((256, 96, 96, 3), dtype=torch.uint8, device="cuda")
env.render(batch_of_states, out=out)
```

`out` must be contiguous, and a non-contiguous one is refused rather than filled: the renderer writes
the frames linearly through a flat view of the destination, so a strided buffer would receive them
scattered across its neighbouring slices. That is what a time or camera axis in front of the frame axis
produces — `buf[:, t]` of a `(B, T, H, W, 3)` rollout buffer is strided — so for that layout, assign
(`buf[:, t] = env.render(states)`) or put the frame axis outermost and render into `buf[t]`.

One measured caveat, and it is about torch rather than about this package. A `DataLoader` indexes a
map-style dataset one row at a time, at roughly 3.4 µs of Python per row: invisible at batch 64, and
14.9 ms per batch at 4,096 — 4× the render it is feeding. If you train at large batches, hand the
dataset whole index lists instead, which needs no custom collate:

```python
from torch.utils.data import BatchSampler, RandomSampler
sampler = BatchSampler(RandomSampler(range(len(ds))), batch_size=4096, drop_last=False)
loader = DataLoader(ds, batch_size=None, sampler=sampler)      # ds[list] -> (B, state_dim)
```

| PushT, 96 px, one H200, states → frames | B=64 | B=256 | B=1,024 | B=4,096 |
|---|---|---|---|---|
| your own `DataLoader` over `TensorDataset` | 2.760 ms | 3.545 | 5.637 | 19.266 |
| the same with `num_workers=8` | 3.076 | 3.095 | 3.619 | 7.120 |
| the same with a `BatchSampler` | **2.601** | **2.666** | **2.886** | **3.677** |
| no loader at all (the ceiling) | 2.452 | 2.462 | 2.774 | 3.369 |

A real training loop built this way, with the render and the forward/backward timed apart:
[`examples/train_from_states.py`](examples/train_from_states.py); under DDP,
[`examples/ddp_train_from_states.py`](examples/ddp_train_from_states.py).

At batch 16,384 and 96px a frame costs 0.34 µs, of which the rasterizer is 0.20 µs; the rest is building
the scene in Python. Full breakdown: [Throughput](docs/benchmarks.md).

### 2. Re-render a stored dataset

A trajectory corpus already contains the compact state at every step; the frames stored beside it are
redundant with those numbers.

```python
from stable_envs import render_states, StateDataset, TrajectoryDataset

frames = render_states("PushT", states, size=96, chunk=8192)   # (..., 96, 96, 3), streamed
```

`render_states` preserves leading dimensions, so a `(n_traj, T, D)` corpus comes back as
`(n_traj, T, H, W, 3)`, and it chunks the input so peak memory is bounded by the chunk rather than by the
corpus — which is what lets a dataset larger than device memory pass through.

`StateDataset` is the migration shim: it stores states and yields **frames**, so a pipeline already
built around an image folder can point at states without its training step changing.

```python
ds = StateDataset("PushT", states, size=96)

ds.compression
# {'n': 100, 'state_bytes': 2000, 'style_bytes': 0, 'stored_bytes': 2000,
#  'frame_bytes': 2764800, 'frame_shape': (96, 96, 3), 'ratio': 1382.4}

loader = DataLoader(ds, batch_size=1024, shuffle=True,
                    collate_fn=se.collate_batched, num_workers=0)
```

That one argument is the whole cost of the shim, and it is not avoidable for free: `StateDataset`
renders in the batched `__getitems__` hook rather than per item — one launch for the whole batch is the
entire point — so it hands back an already-assembled `(B,H,W,3)` tensor that `default_collate` would try
to stack a second time. `num_workers=0` belongs with it because the frames are already on this
process's GPU. (Yielding per-item samples instead, so `default_collate` works, was measured: for
`TrajectoryDataset` at B=1,024 it is 8.741 ms/batch against 0.202, a 43× tax to undo a gather that had
already happened.)

`TrajectoryDataset` is the same idea one level up: `(n_traj, T, state_dim)` plus per-trajectory style,
yielding windows that never straddle an episode boundary, with a HuggingFace round-trip that stores
`(states, style)` together — because a corpus that keeps only the dynamic half re-renders under whatever
appearance happens to be set, which is a coincidence rather than reproduction.

```python
traj = TrajectoryDataset("PushT", states, window=4, stride=2)   # (n_traj, T, 5)
traj[[0, 1, 2]]["states"].shape        # (3, 4, 5) — the time axis is always present
traj.all_frames()                      # re-render the whole corpus, chunked
```

Worked end to end, with the save/load assertion that the reloaded corpus renders bit-identical frames:
[`examples/hf_roundtrip.py`](examples/hf_roundtrip.py).

### 3. Roll out trajectories

A convenience, not the route to a training set: most envs here ship a renderer and no simulator, and
the ones that do ship one are listed below. If you already have states — from an offline dataset, from
the official CPU env, from anywhere — use them; `env.sample` is for when you have none.

`env.sample` generates states under a policy and never touches disk. The policy interface mirrors
`stable_worldmodel.policy.BasePolicy` — `get_action(obs)` plus `set_env(env)` — with the one deliberate
change that `get_action` receives a batch and returns `(B, action_dim)`.

```python
traj = env.sample(steps=100, batch=256, policy=se.RandomPolicy())
# states  (256, 101, state_dim)
# actions (256, 100, action_dim)
# style   (256, style_dim)          — frames only when render=True, and off by default
```

This requires `env.step`, a batched GPU simulator. **Eighteen envs ship one** — seventeen of which roll
out on a machine with no optional reference packages installed, `Sokoban` being the one that needs
`gym_sokoban` to construct — and each states its fidelity against the official CPU env in its own terms,
because they are not the same kind of claim:

| envs | fidelity vs the official simulator | B=256 | B=16,384 | official CPU env |
|---|---|---|---|---|
| MiniGrid / BabyAI (6, top-down) | **exact** — `exact_frac` 1.0, max err 0 over 16 × 100 steps | 1.66–1.71 M | 48–79 M | 8,100–10,700 |
| stable_worldmodel nav + SimplePointMaze (5) | **exact** — `exact_frac` 1.0, max err 0 over 16 × 100 steps | 0.34–1.12 M | 21–72 M | not timed ¹ |
| classic control (5) | 1.1e-16 after one step, 4.4e-13 after 100 — a 1-ulp `sin`/`cos` difference | 0.62–8.5 M | 37–456 M | 31,000–488,000 |
| PushT (contact) | *median 0.68 px over 100 steps*, explicitly **not** bit-exact | 32,700 | 1.08 M | 2,363 |

env-steps/s on one H200, eager; capturing the whole step into one CUDA graph is worth a further 3.2–6.3×
at these batch sizes (measured, [`examples/rollout_with_policy.py`](examples/rollout_with_policy.py)).
The remaining envs have no `step`, and that is the deliberate split described below rather than an
oversight — a renderer can be bit-exact against the official reference and a reimplemented *contact
solver* cannot. `sample` says so precisely on those envs and names the alternative — supply states you
already have, or bring a simulator — rather than failing somewhere deep.

¹ Their reference is a `stable_worldmodel` CPU env stepped in a Python loop, whose rate is set by how
busy the node is rather than by the simulator: the official-renderer column elsewhere in this repo has
moved by more than 100× with load. Against a GPU column of 21–72 M env-steps/s the ratio is an order of
magnitude either way, so it is left unmeasured rather than quoted to three digits.

Our own column does not travel unconditionally either, and the split is measurable per row. A render
whose process CPU time is near 0 per second of wall clock is one kernel on a device the job owns, and
repeats across nodes to under a percent. The ray-cast families are not that: they run at 100–206% and
are **host-bound**, so those rows are a property of the node's processor and its load —
`DMCBallInCupCatch` measured 3,664 frames/s idle and 271–978 contended, a 13× spread. Every row of
[`examples/results/decode/ALL.md`](examples/results/decode/ALL.md) carries its own `cpu%` for that
reason, and the ones that sit at or above 100% say so.

The two claims are kept apart on purpose. The PushT **renderer** is bit-exact on interior pixels; the
PushT **simulator** tracks pymunk to a median 0.68 px and is not bit-exact, because contact is chaotic
and every replacement engine diverges. `stable_envs.testing.conformance`'s
`check_step_matches_reference` measures whatever an env declares as its reference, and reports a
first-step integrator control that no amount of divergence can hide inside — the kinematic agent
(9.5e-5 px) for PushT, one step of the closed form for classic control.
[`examples/rollout_with_policy.py`](examples/rollout_with_policy.py) runs the native path, asserts the
tolerance, and keeps a `ReferencePushT` that drives B copies of the official CPU env as the worked
example for envs with no `step`.

## Appearance is part of the state

Every env declares a `variation_space`: a tree of samplable Spaces covering colours, scales, positions
and toggles, mirroring `stable_worldmodel.spaces` so envs port between the two unchanged.

```python
env.variation_space.names()
# ['agent.color', 'agent.inner_color', 'agent.scale', 'background.color', 'block.color',
#  'block.outline_color', 'block.scale', 'goal.angle', 'goal.color', 'goal.position', 'goal.scale',
#  'rendering.render_goal', 'walls.color', 'walls.visible']

frames = env.render(states, variation={"block.color": [220, 40, 40]})

# one colour per environment — 4096 different appearances, still one kernel launch
import numpy as np
per_env = np.random.randint(0, 256, size=(4096, 3)).astype(np.float32)
frames = env.render(states, variation={"block.color": per_env})
```

Defaults are exactly the official appearance, so `scale = 1.0` and the stock palette reproduce the
reference renderer bit-for-bit, and a test asserts that identity. Domain randomisation is a tensor, not a
loop.

Style is stored as its own flat per-environment tensor because a dataset that keeps only the dynamic half
can regenerate a frame only under whatever appearance happens to be set at render time — a coincidence
rather than reproduction. `test_style_is_state` asserts that `(state, style)` determines the frame.

## The registry

Two claims, and they no longer have the same answer, so this page keeps them apart. **Storage** holds
for every family: summed over the registry the state that determines a frame is 37,228 B against
71,805,384 B of raw pixels, a factor of **1,929**, with a per-environment median of 4,332× — and the
resolution and the appearance are chosen at render time, which no frame corpus offers. No throughput
result moves that. **Regeneration cost** is per family. Against one core of JPEG decode almost
everything wins; against eight worker processes batch-decoding JPEG-in-Lance, which is the strongest
disk-based path measured here, **17 of 23 families** win and 6 do not — HumanoidBench, DM Control,
gymnasium's MuJoCo locomotion, Meta-World, RoboCasa and CarRacing. The per-family ratios are
regenerated from the measurement JSONs into
[`examples/paper/tables/family.tex`](examples/paper/tables/family.tex) by
[`examples/paper/make_tables.py`](examples/paper/make_tables.py).


**196 environments in 24 families** at the last census (2026-08-10, `scratch/census/union.json`), every
parity number measured against the official renderer. No single interpreter can import every family, so
that count is the union over **15** of them and the widest one sees **147** — `len(stable_envs.REGISTRY)`,
not this number, is the current answer on your machine, and `stable_envs.envs.import_errors()` says which
families a given machine is missing a reference package for. The registry is under active development and
runs ahead of the sweep the last two columns come from.

| family | envs | parity | GPU frames/s | vs official |
|---|---|---|---|---|
| Griddly | 7 | **interior 100%, all pixels 100%**, max abs diff 0 on all seven | 2,307,114–9,438,816 | 18,907–82,365× |
| Sokoban | 1 | **100% of all pixels**, max abs diff 0 | 9,571,978 | 37,785× |
| Overcooked | 5 | **interior 100%, all pixels 100%**, max abs diff 0 on all five ⁶ | 3,615,605–8,145,368 | 4,419–87,586× |
| MiniGrid + BabyAI | 7 (+165 via `register_all`) | **max abs diff 0** over 21,216 frames, all 172 ids | 711,991–4,479,313 | 1,256–10,200× |
| MiniHack | 1 | **100% of all pixels** | 244,456 | — |
| Jumanji | 8 | **100% of all pixels**, bit-identical on all eight | 409,277–1,286,366 | 6,455–145,393× |
| Procgen (`maze`, `heist`, `miner`, `chaser`) | 4 | **100% of all pixels**, max abs diff 0, 240 frames/game | 313,372–1,588,402 | 323–1,723× ⁵ |
| PushT | 1 | **100% of interior pixels** | 1,256,557 | 2,664× |
| OGBench mazes | 25 | **100% of interior pixels**, no tolerance relaxed | 48,978–206,740 | 110–630× |
| stable_worldmodel nav | 5 | **100% of interior pixels** on the reported draw; floor declared at 0.999995 ⁷ | 71,140–187,629 | 151–1,356× |
| Crafter | 1 | **100% of all pixels**, max abs diff 0 over 1,111 frames | 665,836 | — |
| Craftax Classic | 1 | **100% of all pixels**, max abs diff 0 | 569,934 | — |
| PGX (tic-tac-toe, connect four, othello, go 9×9) | 4 | **interior 1.000000** on all four; all pixels 0.915308–0.983681 | 78,322–109,484 | 1,010–3,931× |
| Classic control | 5 | **100.00000%**, 256/256 frames exact per env | 33,216–39,843 | 61–148× |
| D4RL maze2d / PointMaze | 4 | **100% of interior pixels**, 99.49–99.77% of all, no tolerance relaxed | 88,303–88,935 | 40–48× |
| DM Control | 16 (11 parity, 5 visual) | **interior 1.000000** where the metric is defined on all eleven, no tolerance relaxed ² | 3,813–140,391 | 11–421× |
| LunarLander | 1 | **100% of interior pixels**, 99.97% of all | 18,786 | 56× |
| highway-env | 6 | **100% of interior pixels**, 99.93–100.00% of all | 1,520–20,389 ¹ | — ¹ |
| Fetch | 12 | interior **0.9988–0.9990**, all pixels 0.922–0.937, mean 0.79–0.85/255 ¹⁰ | 6,008–6,164 | 15–19× |
| HumanoidBench | 17 | **visual tier** — SSIM 0.849–0.931, interior-exact 0.607–0.943 ⁸ | 9,330–16,835 | 183–547× |
| Gymnasium MuJoCo | 10 (4 parity, 6 visual) | **100% of interior pixels** on three; 99.999963% on InvertedPendulum ³ | 4,404–10,931 | 3–31× |
| Meta-World | 50 | interior-exact **median 0.999700**, min 0.980766, 48/50 above 0.99 ⁹ | 1,039–2,554 | 5–325× |
| RoboCasa | 4 | **visual tier** — SSIM 0.821–0.921, 4.8–15.8% of interior pixels exact | 5,122–11,687 | 11–60× |
| CarRacing | 1 | **100% of all pixels**, max abs diff 0 | 179 ⁴ | 1× ⁴ |


> **The `ours` column of that sweep predates two kernel changes and understates the 2-D families
> badly.** It was taken at 05:58 on 2026-08-11; the fused 2-D rasterizer and the fused ray-cast path
> landed after it. Re-measured at 256 px / batch 1,024 the same day, `CarRacing` goes from 179 to
> **21,943** frames/s and `Roundabout` from 1,520 to **88,471** — 123x and 58x — while the ray-cast
> families move by less at that resolution. The current per-family ratios, against the eight-worker
> JPEG-in-Lance bar rather than against one core, are regenerated from the storage sweep in
> [`examples/paper/tables/family.tex`](examples/paper/tables/family.tex). Re-run
> `examples/bench_vs_decode.py` per tag and `scratch/census/merge_bench.py` to refresh the table below.

The last two columns are one sweep —
[`examples/results/decode/ALL.md`](examples/results/decode/ALL.md), all 196 envs, each at its own native
frame size and its own `default_supersample`, at the best batch that sweep measured. That is not the same
thing as each family's peak: PushT's best cell there is batch 4,096, while [Throughput](docs/benchmarks.md)
records 2,914,271 frames/s for it at batch 16,384. A blank ratio means no reference package was installed in
the interpreter that owns that env, which is an absence rather than a zero. Read the ratio as an order of
magnitude in either direction: the official column is a single-threaded Python renderer sharing a node and
has moved by more than 100× with load, and the ray-cast families are host-bound on our side too (see the
footnote to the rollout table above).

¹ The wide highway-env range is resolution, not efficiency: Roundabout, Intersection and Racetrack
draw 600×600, four times the pixels of Highway and Merge. Per-env numbers are in
[the highway-env note](docs/research/highway.md).

² Sixteen cells, eleven at parity and five in the visual tier. On the eleven, `parity_interior_min` is
**not relaxed** — interior agreement measures exactly 1.000000 over 64 states on three independent
draws, wherever the metric is defined. DM Control's floor is a checkerboard under a spotlight gradient
and its sky is a speckled gradient, so a 3×3 flat neighbourhood — which is what the interior mask keeps
— barely exists: `interior_frac` is 0.000000 on acrobot, point_mass and both reachers and 0.0001–0.0052
on the rest, so on those four the parity test falls back to all-pixel agreement (90.6–99.8% of pixels
bit-identical, mean 0.02–0.68/255). 98.4–100.0% of every disagreeing pixel is inside the ray-cast mover
pass and the baked plate is 99.967–100.000% exact on the pixels it owns. One cell declares anything at
all: `DMCFingerTurnEasy` sets `parity_all_exact_min = 0.89`, which is the size of its movers rather than
the renderer — all-pixel exactness is interior exactness weighted by how much of the frame moves, and
finger-turn_easy's seven movers cover 11.7% of an 84 px frame against cartpole's 3.0%. Needs its own
venv pin (`mujoco==3.11` + `dm_control==1.0.44`) — see [the DM Control note](docs/research/dmcontrol.md).

³ All ten of gymnasium's MuJoCo envs are registered, in two different tiers. Reacher, Pusher,
InvertedPendulum and InvertedDoublePendulum are at parity. The six locomotion domains (HalfCheetah,
Hopper, Walker2d, Ant, Humanoid, Swimmer) are a **documented NO-GO for the parity tier** — they ship in
the visual tier instead, see below — and not for the usual reasons: their frame *is* a measured pure
function of `qpos` and their
models are mesh-free primitives. The camera rides the body, so there is no static plate (202–255/255
over 29–57% of pixels), and the floor is a minification-filtered checker: on five of the six, GL has
averaged the far half of it into one colour (luminance spread 0.0–6.8 about 93.7, against 74.9–90.1
spanning black to white in the near field). InvertedPendulum declares `parity_interior_min = 0.99999` for 5 disagreeing interior pixels
out of 13,401,944 over 256 states, every one inside a mover and off by exactly 1/255. This is also the
one family where the official renderer is competitive — these scenes are 3–20 primitives, so MuJoCo's
rasterizer costs triangles while an analytic caster costs rays — which is why its ratio column is the
narrowest in the table. See [the Gymnasium MuJoCo note](docs/research/mujoco_locomotion.md).

⁴ CarRacing does not beat its reference; it draws with it. The table's row is the sweep's; the paired
measurement, both sides on one node, is 175 f/s for the env as it stands at batch 1024 against 174.5 f/s
for the official renderer on one CPU core — a 1.01×
"speedup", and the only row here where the GPU renderer is not faster than the thing it reproduces.
The reason is the reference's draw order, not the kernel: `_render_road` calls `gfxdraw.aapolygon`
before `filled_polygon` for every quad and the anti-aliased fringe blends against whatever is already
on the canvas, so a frame is a chain of 81–118 alpha-blend steps that cannot be merged. Each step's
fill is `O(batch × 800,000)`, so the per-call cost is nearly flat in batch (2.7 s at 8, 5.8 s at
1024) and throughput is linear in it — 3.0 f/s at batch 8, 34.4 at 128, 112.4 at 512. Two measured
routes to more are open and neither is implemented: capturing the step chain as a CUDA graph gave
297 f/s at batch 128 on a synthetic pipeline (8.6× the eager 34.4 at that batch), and a fused span
fill would leave only the 0.173 s serial floor. Parity is settled; this is throughput work. See
[the CarRacing note](docs/research/carracing.md).

⁵ Procgen's ratio in the table is against one thread of its own C++ renderer, which is the reference
every other row uses. Measured instead against each game's official *peak* over `num` (8 threads) it has
a **crossover**: procgen's own renderer is fast on 12.3 kB frames, so this one only wins once the batch
amortises a chain of serial composites. On that comparison the GPU path
overtakes at **batch 128 for `maze`, 256 for `miner`, 2048 for `heist` and 2048 for `chaser`**, and
below that it is slower than the reference — by two orders of magnitude at batch 1. The full sweep is
in [the pixel-native note](docs/research/pixel_native.md). Quoted at batch 4096: 27.97× / 3.06× /
17.83× / 2.63×. `bigfish` was also built and is bit-exact and **not shipped**: 33 entity slots is 33
serial full-frame composites, it stays at **0.47× and never crosses over**, and its spawner cast
cannot be given a fixed-width state that is both lossless and free of dead dimensions.

⁶ This is JaxMARL's `OvercookedVisualizer`, not `overcooked_ai_py`'s `StateVisualizer`, and the two
draw different pictures — on the same `cramped_room` board, 450×525 against 300×375 with 48 distinct
colours against 33. No pixel comparison between them is meaningful, and this family reproduces the
JaxMARL renderer only. See [the Overcooked note](docs/research/overcooked.md).

⁷ `Piecewise`, `PiecewiseZones`, `TwoRoom`, `TwoRoomTarget` and `SimplePointMaze`. On the four
`stable_worldmodel` navigation envs, interior agreement rounds to 100.000% on the reported draw, and
over 200 seeded draws of 64 states it is 100% on most and 99.99991% on about one in eight — never more
than 2 pixels out of ~2.3 million interior pixels, never more than 1/255, always on one channel, from a
one-ulp `exp` difference. `parity_interior_min` is declared at 0.999995 rather than left at 1.0 for that
reason, because seeding the test to a clean draw would have hidden a real disagreement rather than
explained it. `SimplePointMaze` is at 100.000% of interior pixels with no tolerance relaxed, 99.9690%
of all pixels.

⁸ Built and measured, but **not registered by default**: `stable_envs/envs/humanoidbench.py` raises at
import unless `STABLE_ENVS_HUMANOIDBENCH=1`, because `mujoco_warp`'s batched render is not
bit-reproducible across a long conformance session — 2–3 of ~660 tests fail every run by 1–2/255, on a
different cell each time, and the registry's contract includes "the same state gives the same frame".
17 of HumanoidBench's 49 scenes are covered; the other 32 are an evidenced NO-GO. See
[the HumanoidBench note](docs/research/humanoidbench.md).

⁹ Meta-World's earlier NO-GO is **withdrawn**, and both gates it rested on are closed. All 50 v3 tasks
ship in the parity tier at `parity_interior_min = 0.978` and `parity_all_exact_min = 0.80`, state =
`qpos` plus the episode's live model slice — 0 to 6 named floats, which makes a *different* episode
restore bit-exactly on 50/50 against 0/50 for the do-nothing control. Interior-exact median 0.999700
over 64 states × 50 tasks, min 0.980766, 48/50 above 0.99; 9.2× the official renderer at batch 256 /
480 px and 21.1× at 224 px. See [the Meta-World note](docs/research/metaworld.md).

¹⁰ The Fetch numbers are the ones measured *after* a rounding fix, and the class's own `parity_note`
still quotes the values from before it. `mujoco_warp.render` regenerates its megakernel inside the
call, so a CUDA graph captured outside a `with gl_rounding()` block replays the truncating kernel for
the rest of the run — the difference against correct rounding was `{0, -1}` and nothing else, on
15.8% of Fetch interior channels, 21.2% of Meta-World's and 63.8% of HumanoidBench's. Capturing inside
the block moves Fetch's interior parity 0.9923 → 0.9988. The declared floor is still
`parity_interior_min = 0.99`; a floor is a contract and the measurement is what it has to clear.


```python
sorted(se.REGISTRY)                  # every env, by class name
print(se.format_registry())          # ...with its FIDELITY TIER, which the names alone do not carry
```

Envs are identified by class name; `stable_envs.envs.import_errors()` reports any env module that failed
to import, so a missing optional dependency costs one env rather than the package.

### The visual tier

Most rows above are **parity tier**: the frame *is* the official frame. **Thirty-two classes**, across
four families, are **visual tier** — a different render of the same scene, fast enough to generate
training frames live and measurably not the official pixels. Two of those families are entirely visual
tier and two are split, which is why they appear above with two counts.

**RoboCasa**, four classes: 7.0x the official rasterizer at batch 256 / 256 px, 17.9x at 128 px, mean
5.7-32.1/255, SSIM 0.92-0.82, 4.8-15.8% of interior pixels exact against the 1.0 every row above
declares. The residual is one named mechanism — `mujoco_warp` derives texture UVs for `PLANE` and
`MESH` only, so the 178-401 textured boxes in a kitchen are drawn one flat colour.

**Gymnasium's six MuJoCo locomotion domains** (HalfCheetah, Hopper, Walker2d, Ant, Humanoid,
Swimmer), rendered by nvdiffrast: 23-27x the official rasterizer at batch 256 / 240 px, mean
1.97-4.39/255, **SSIM 0.947-0.981**, 54-88% of interior pixels exact. Their residual is also one named
mechanism, and it is the one that keeps them out of the parity tier: MuJoCo's checker floor is
minified toward the horizon and a renderer that resolves one texel per pixel paints speckle where GL's
mipmap has averaged. nvdiffrast's prefiltered sampler takes the far tenth of that floor from 0.0001 to
0.66-0.81 of pixels bit-exact — not to 1.0. The four parity-tier envs of the same family are
unaffected and still claim the official frame; needs the `mujoco-locomotion-visual` extra.

**HumanoidBench**, seventeen classes (Unitree H1 and H1-with-hands), rendered by `mujoco_warp`: mean
2.6-8.4/255, within-8 0.795-0.958, **SSIM 0.845-0.931**, 60.7-94.3% of interior pixels exact, at 437x
HumanoidBench's own renderer at batch 64. What decides the tier here is that there is no static plate to
bake at all: every camera in the family is `mode="trackcom"`, so hiding every mover and rendering eight
states still leaves the "static" picture differing by 54/255 over 21.7% of its pixels. The residual is
the same minified checker floor, degenerate in this family — a 2 cm grid line every 2 m on otherwise
flat grey rather than a full-contrast checker — so drawn as-is it is 0.877 interior-exact rather than
the 0.22 the same mechanism costs Gymnasium locomotion, and the module draws the floor's dominant texel
instead, which is measurably closer on every metric at once. Registers only under
`STABLE_ENVS_HUMANOIDBENCH=1`.

**DM Control's five body-tracking domains** (`cheetah-run`, `walker-walk`, `hopper-hop`,
`humanoid-stand`, `swimmer-swimmer6`), also through `mujoco_warp`: mean 4.4-8.1/255, 70-91% of pixels
within 8, SSIM 0.66-0.96. Two mechanisms, measured per surface off MuJoCo's own segmentation buffer: the
speckled-gradient skybox, which is 48-89% of the frame on cheetah, hopper and walker and which
`mujoco_warp` samples with a single unfiltered tap, and the minified checker floor again. The eleven
fixed-camera cells of the same family are unaffected and still claim the official frame.

#### Learned render correctors — faster *and* closer, and a different kind of frame

A visual-tier cell may additionally ship a **learned render corrector**: render deliberately cheap
(112 px, textures at mip level 2) and hand the result to a 441 k-parameter convolutional network, 7.9 µs
a frame, fitted to predict the official 256 px frame. `RoboCasaCounterToSinkCorrected` reaches
**16,959 f/s at mean 2.48/255, within-8 0.935 and SSIM 0.970**, against 5,756 f/s at 6.41 / 0.833 /
0.907 for `RoboCasaCounterToSink` — 2.9× faster and 2.6× closer at once, and past the 15,937 f/s that
eight workers reach reading these frames off disk.

**It changes the decoder, not the contract.** A fixed checkpoint over a render that is already a pure
function of `(state, style)` is still a pure function of `(state, style)`, so `render` is still
`f(get_state())` and every test in `tests/gpu/test_reproducibility.py` applies unchanged. What changes
is what the picture is made of: most of the win is that the network *paints a marble texture its input
does not contain*, so that detail is synthesised — right statistics, wrong pixels, and a frozen
pretrained encoder can tell even where the pixel metrics improve 2.5×. These are separate registry
entries declaring `render_decoder = "learned"`, and **anyone who will be evaluated on official pixels
has to record which of the two produced their training frames, because it is not visible in the
picture.**

**It needs a batch.** The network runs at a fixed batch of 128 — that is what makes batch invariance
structural rather than lucky — so a caller asking for fewer frames pays for 128 anyway: 416 f/s at
batch 1 against the plain class's 828, crossing over between 16 and 64 and reaching 2.95× by 1024. A
per-step interactive loop should stay on the plain cell; a training loop that renders a batch is what
this is for.

A corrector is per (scene, camera): applied to a camera 40 cm away it is no better than a plain resize,
and on the wrist camera it is 2.6× *worse than doing nothing* — a failure the pixel mean does not catch
and SSIM and within-8 do. So each checkpoint is keyed to its cell and a mismatch raises rather than
falling back, and `examples/rendernet/train_corrector.py` refuses to write one that does not beat the
render it replaces on all three quantities, at the default appearance and under the env's declared
variation space. No weights are shipped and none are required:
[`examples/rendernet/`](examples/rendernet/) builds one in about forty minutes, and with no checkpoint
present nothing is registered and the plain cell is untouched.

That tier is **suitable for live training-data generation and is NOT a substitute for a stored dataset
when a policy will be evaluated on official pixels.** `format_registry()` marks it,
`describe()["fidelity_tier"]` reports it, and `test_render_parity_with_official_renderer` skips those
envs rather than being relaxed for them. See [Fidelity tiers](docs/concepts/fidelity-tiers.md), and
[`examples/robocasa_visual/`](examples/robocasa_visual/) for the side-by-side video that shows the
residual instead of describing it. The registry grows
between benchmark sweeps, so `len(stable_envs.REGISTRY)` is the current answer.
Per-env tables: [Throughput](docs/benchmarks.md). Side-by-side video against the official renderer, on
identical states: [Gallery](docs/gallery.md).

Every throughput figure in this section and the table above is at the supersample the class itself
declares, which is what `se.make(name)` hands you and the picture every parity number is measured on.
That is not a formality, and getting it wrong is a correction this repo has already had to make: **44
registered classes set `default_supersample = 2`** — all 25 OGBench mazes, 11 DM Control domains, 4 D4RL
mazes and 4 MuJoCo locomotion envs — while the published rows were being taken with `--ss 1`, so they
described a different picture, four times cheaper in rays. Measured in one process on an idle node,
`supersample=1` against the class's own value: `DMCBallInCupCatch` at 84 px batch 4,096 is 4,272 against
3,683 frames/s, `AntmazeLargeNavigate` at 224 px 70,598 against 55,947, `MujocoPusher` at 240 px 14,938
against 4,475. The ratio is not a constant, because these renders are host-bound and supersample 2
quadruples the per-chunk Python loop, so it grows with how contended the node is. The flag now defaults
to `native`; the A/B is in `make_env` in
[`examples/bench_gpu_vs_cpu.py`](examples/bench_gpu_vs_cpu.py).

Families that were *examined and rejected* are written up with the gate that rejected them rather than
omitted, and that list is maintained where the evidence is — the [research notes](docs/research/), and
[when does this pay off](docs/research/when_does_this_pay_off.md) for the rule they were tested against.
It is deliberately not duplicated here: a no-go that is later reopened by a better measurement (DM
Control was one, blocked by a `dm_control`/`mujoco` version skew rather than by the thesis) has to change
in one place, not two.

## Why you can trust the frames

### Pixel parity, split so it cannot hide anything

Interior pixels — those whose 3×3 neighbourhood in the reference is one flat colour — must be **100%
exact**. There is no coverage ambiguity there, so a disagreement is a wrong colour, a wrong shape or a
wrong place. All-pixel agreement is held to a much looser bar, because edges are two rasterizers'
anti-aliasing rules landing on different sides of a rounding and the reference is not ground truth there.

Folding the two together is the failure mode: a real geometry error is invisible in an aggregate. The
block outline being mitred instead of round costs 2.2% of interior pixels, which vanishes in an
all-pixel average and is exactly 0 versus 1.0 in the interior column.

An env may declare a lower interior floor only together with a `parity_note` naming the mechanism, which
the test prints. That keeps a genuine limitation attached to the one env that has it instead of lowering
the bar for everything.

An env whose *renderer* provably cannot be the official one does not get a looser floor at all: it
declares the [visual tier](docs/concepts/fidelity-tiers.md), sets `parity_interior_min = None`, is
skipped by this test, and is asserted by a separate one against its own measured mean-error / within-8 /
SSIM thresholds. The strong claim stays strong and the new claim is separately checkable — which is the
whole reason there are two tests rather than one threshold spanning both.

### A pretrained policy cannot tell our frames from the official ones

The strongest test available: take an official pretrained world model, run its own evaluation unmodified,
and swap only the pixels it sees. LeWorldModel's `quentinll/lewm-pusht` checkpoint, LeWM's own `eval.py`,
the official corpus, identical episode indices, start steps and CEM seed in both arms, 6 seeds × 50
episodes:

| renderer | success rate | paired test vs official |
|---|---|---|
| official (pygame) | **88.0%** (264/300) | — |
| stable-envs, every pixel | **87.7%** (263/300) | McNemar **p = 1.000**, Δ −0.3 pts |
| blank frames (control) | 4.0% | p < 0.001, Δ −92 pts |

The control arm is the point. An evaluation that cannot detect a broken renderer cannot certify a working
one; this one detects blank frames at p < 0.001 and still cannot distinguish our frames from the official
ones. Swapping only the live observations the agent acts on — about 1,490 renders per run — leaves the
success count identical.

A result we did not go looking for: **the stored corpus is a degraded copy of the renderer's output.**
Its frames carry 1,582 distinct colours where live pygame produces 146 and a flat-shaded raster produces
7 — the entire palette. Every edge has a compression ringing band, and no single-stage resize filter
(9 tried) or downscale/upscale round trip (8 tried) reproduces it. Regenerating from states is therefore
not a lossy substitute for those 46 GB of stored pixels; it is strictly closer to the reference than
reading them off disk.

Full protocol, per-seed numbers and the surface-by-surface breakdown:
[`examples/lewm_pusht/`](examples/lewm_pusht/).

### The suite checks the thesis, not just the renderer

The conformance suite is parameterised over the registry, so a new env is covered without editing a test.

    sbatch scripts/run_gpu_tests.sbatch      # or: pytest tests/ -v -rs   on a GPU node

| check | what it rules out |
|---|---|
| render parity vs the official renderer | interior pixels must be 100% exact |
| state round-trip | `set_state(get_state())` is the identity |
| every state dim reaches the render | a dead dimension means the state is not minimal, or the renderer ignores it |
| batch invariance | frame *i* of a batch equals the frame rendered alone |
| backend agreement | the fused kernel matches the readable torch reference |
| variation coverage | a knob in the space but not in `scene()` advertises support that does not exist |
| style is state | `(state, style)` determines the frame |
| **state sufficiency** | two *reference* envs driven to the same state must render the same frame |

The last one is not decoration: it runs against the official env alone, so it can be answered for a
candidate env in an afternoon, before any porting work. It caught two real bugs in how PushT is driven,
both of which affect anyone restoring PushT from a stored state — `gym_pusht._set_state` sets position
before angle, and Chipmunk rotates about the body's centre of gravity, so the position you read back is
not the one you wrote (up to 70 px of error, depending on the block's *previous* angle); and a manually
moved body keeps drawing at its old pose until `space.reindex_shapes_for_body`. Neither is a rendering
bug and neither would have been found by comparing pixels.

What each check rules out, with the measured numbers: [Testing and conformance](docs/testing.md).

## When does this pay off?

A list of successes is not a result. The useful thing is a rule that says in advance where the approach
helps and where it cannot:

> It pays when the frame is a lookup into a decoder that is a **constant of the family**, indexed by a
> small per-step state. It fails when the decoder is itself content — regenerated per episode
> (RoboCasa) or written per title (Atari).

Four gates, each measurable before writing a renderer, in cost order:

| gate | test | threshold | cost |
|---|---|---|---|
| G3 decoder stability | reset twice, compare the compiled scene / atlas / primitive set | identical | ~20 s, most decisive |
| G1 purity | force two internal states to the same candidate state, render, diff | 0 pixels | an afternoon |
| G2 compression | render-relevant *slice* over frame bytes, amortized per episode | < 1 | minutes |
| G4 economics | regeneration vs decode, and the render's share of a step | > 1 | an hour |

Twenty-four families and 196 environments have been measured against that rule, and it produces both the
wins and the losses. A selection, with the full table in the note:

| verdict | families | decided by |
|---|---|---|
| **GO** | PushT, MiniGrid + BabyAI, classic control, MiniHack/NLE, OGBench mazes | all four gates clear |
| **GO** | DM Control, fixed-camera domains (frame = `f(qpos)` on all 17 probed, 0 dead dims) | all four gates clear |
| **GO** | D4RL maze2d / PointMaze (4 f32 vs a 224² frame = 1/9,408; `nmesh = 0`) | all four gates clear |
| **GO, conditional** | Craftax (store the render slice, not `EnvState`) | G2 |
| **GO, conditional** | Crafter (same game, same failure: its world arrays are 1.67× the frame, the render window is 1/38) | G2 |
| **GO** | Overcooked-AI — shipped, 5 JaxMARL layouts, 100% exact (1,980 frames rebuilt from a position-independent atlas, 0 mismatches); Griddly — shipped, 7 games, 100% exact (0 collisions over 5,352 cross-history repeats; official renderer is *slower* than decode) | all four gates clear |
| **GO** | Procgen `maze`/`heist`/`miner`/`chaser` (bit-exact; the earlier "conditional" was an effort estimate, and section 2 of the pixel-native note records why it was wrong) | all four gates clear |
| **NO-GO** | Procgen `bigfish` (renders bit-exactly, 240 frames, max abs diff 0 — but 24 fish at once against a pool whose slots 7+ are never occupied) | **G5**, the fixed-cast gate it produced |
| **NO-GO** | MinAtar (clears G1/G2/G3; the observation *is* the state and the frame is 300 bytes nobody trains on) | G4 |
| **NO-GO** | Octax (CHIP-8 in JAX: a genuine pure function of state, and the framebuffer *is* the state — 1.015× the frame) | G2, degenerately |
| **NO-GO** | PixelBrax (`nmesh = 0` and 98.8% of the step is the render, but the renderer that exists replaces 1.43 cores) | G4 |
| **GO, conditional** | KAGE-Bench (5.68–6.69× as shipped — `bg_image` is a whole frame inside the state — and 0.051× with the window packed *and* the background stored as an index) | G2 |
| **GO at the parity tier, loses G4** | Meta-World, all 50 v3 tasks — the earlier NO-GO on G1 and G3 is withdrawn: a reset writes placement into the *model*, but restoring 14 model arrays makes a different episode bit-exact on 50/50 against 0/50 for the control. It is 1.2–3.8 cores against one core of decode and 0.20–0.49× against eight workers, so it ships with the ratio printed | G1+G3 withdrawn; G4 lost |
| **NO-GO at the parity tier**, shipped at the visual tier | RoboCasa (exact rendering replaces 0.37 of a CPU core; the fast renderer replaces 5.1 and is 5.9–23.1% interior-exact) | G4 on the loop |
| **NO-GO** | Atari/ALE (identical 128 B of RAM gives different frames on 13.75% of pairs) | all four |
| **GO, on a wider state** | LunarLander — `render()` mutates what it draws, and the fix is to put the exhaust in the state rather than to exclude the env: bounded at 6 drawn particles, deterministic, and a different env driven to the same 38 numbers renders 0/255 | G1 re-asked |
| **NO-GO** | DM Control on the **2-D** path (skybox, shadowmap, perspective) | G3 |

The direction of the surprises is what makes this predictive rather than a rationalisation: **visual
complexity is not the axis — decoder stability is.** OGBench's articulated humanoid reduces to 19
analytic primitives with `nmesh = 0`; NetHack's 1.27 MB frame is a single gather. Atari, the visually
simplest family in the set, fails all four gates. RoboCasa fails not for photo-realism but because its
scene graph is regenerated every episode.

One derived figure puts wins and losses on one axis — CPU cores of JPEG decoding that one GPU replaces,
each family against one core decoding *its own* frames from memory, which is the case most favourable to
a stored dataset. Across the 24: Griddly **1,789–3,648**, Overcooked **1,241–1,861**, MiniGrid / BabyAI
**330–1,226**, PushT **122.5**, OGBench **11.9–93.6**, Fetch **4.3–6.9**, HumanoidBench **2.3–6.8**,
Gymnasium MuJoCo **1.6–4.2**, Meta-World **1.2–3.8**, RoboCasa **1.7–3.1** at the visual tier, CarRacing
**0.0**. Every family's row is in
[`examples/results/decode/ALL.md`](examples/results/decode/ALL.md). RoboCasa is the instructive one, and
it is measured twice: the renderer that is *exact* replaces **0.37** of a core, and the one that is fast
replaces 5.1 at 5.9–23.1% of interior pixels exact. One core is also the generous bar — against eight
workers batch-decoding JPEG-in-Lance a family divides by roughly the worker count, and several of these
rows cross below 1.

Full derivation, per-gate measurements and the recorded gaps:
[when does this pay off](docs/research/when_does_this_pay_off.md).

## How it works

An env implements one method that carries the whole claim:

```python
def scene(self, states, variation) -> Scene2D:   # (B, state_dim) -> a batch of 2-D scenes
```

`Scene2D` is a small IR — convex polygons, discs, thick segments, composited back-to-front with flat
colours — and one fused Triton kernel rasterizes any of it. An env that can produce a `Scene2D` from a
state has *proved* the frame is a function of the state, because there is nowhere else for information to
come from: no env instance, no simulator handle, no hidden buffer.

Everything is per-environment tensors, so a batch of 16,384 can each carry a different block colour,
agent radius or background and still render in one launch.

Not every env goes through the IR. The tile-atlas families (MiniGrid, BabyAI, MiniHack, Craftax) render
by gathering from a precomputed atlas, which is exact by construction rather than by matching a
rasterizer — and it is those families that reach byte-exact parity rather than interior-exact. The
OGBench mazes ray-cast analytic primitives. Those envs override `render` and the suite skips the checks
that need a `Scene2D`, applying the rest.

- [The Scene2D IR](docs/concepts/scene-ir.md) — the primitive vocabulary, and the three details in it
  that are load-bearing.
- [The fused rasterizer](docs/concepts/rasterizer.md) — why one kernel, what it costs, and the 73–85× the
  readable torch reference costs instead.
- [Variation spaces](docs/concepts/variation-spaces.md) — the knobs, and which ones measurably vary per
  environment today.
- [Adding an environment](docs/adding-an-env.md) — the worked PushT port. Start here.

### Physics is separate, on purpose

Rendering and simulation are separable, and keeping them apart is what makes the repo useful. A renderer
can be bit-exact against the reference. A contact solver cannot — contact is chaotic, so any
reimplementation diverges. Conflating them lets an "our env matches" claim rest on the half that does
not. The common case needs no simulator at all: you are re-rendering states you already have.
[Why physics is separate](docs/concepts/physics.md).

### The 3-D path

The 2-D IR does not reach MuJoCo-based families, and none of Brax / MJX / mujoco_warp ships a batched
renderer as a headline feature, so this was the open risk. Measured at 224px on one H200, MJWarp
(`mujoco-warp` 3.11) renders antmaze at 53,549 frames/s and humanoidmaze at 60,132, against MuJoCo EGL's
185 and 178 — and EGL gets *slower* with threads, so there is no cheap CPU-side answer.

Fidelity splits by material, not by complexity. On flat-material scenes MJWarp reaches 98.9% (antmaze) and
92.5% (humanoidmaze) of interior pixels bit-identical, MAE ≤ 0.3/255, the residual being a 1px band where
MuJoCo multisamples and MJWarp casts one ray. On heavily textured assets it does not: a RoboCasa kitchen
reaches 6-23% of interior pixels exact, and the earlier "7.6%, flat saturated colour" reading was a
separate bug — MJWarp never reads alpha, and RoboCasa leaves 154 fully transparent marker geoms in the
rendered group. MJWarp is therefore adopted for flat-material scenes and explicitly not shipped for
textured ones. Backend comparison, and three traps that each yield a confident
wrong answer: [Choosing a 3-D backend](docs/research/backend3d.md).

## Documentation

```bash
python -m venv .venv-docs && . .venv-docs/bin/activate
pip install mkdocs mkdocs-material "mkdocstrings[python]"
mkdocs build --strict          # or: mkdocs serve
```

The API reference is generated from source docstrings by mkdocstrings, which reads them statically, so
building the site needs neither torch nor a GPU.

| | |
|---|---|
| [Home](docs/index.md) | the claim and the measured tables |
| [Concepts](docs/concepts/) | the IR, the rasterizer, variation spaces, why physics is separate |
| [Adding an environment](docs/adding-an-env.md) | the worked PushT port |
| [Testing](docs/testing.md) | what each check rules out |
| [Throughput](docs/benchmarks.md) · [Gallery](docs/gallery.md) | the numbers, and the video |
| [Research notes](docs/research/) | per-family feasibility studies, including the no-gos |
| [Examples](examples/) | training from states, rollouts, domain randomisation, the HF round-trip, the LeWM evaluation |

## License

MIT. See [LICENSE](LICENSE).
