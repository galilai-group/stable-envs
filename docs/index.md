![](assets/brand/mark.svg){ width="76" }

# stable-envs

**Famous research environments, rendered on the GPU as a pure function of a compact state, batched over
thousands of environments in one launch.**

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
per-family and is **not** universal — against nvJPEG on the GPU it is a wash — see [Throughput](benchmarks.md)
for which families beat which read path, including the ones that do not.

PushT is 5 floats per step against a 96×96×3 frame: 20 bytes against 27,648. Those 20 bytes come back as
a frame in 0.34 µs on one H200, and 100% of the interior pixels are bit-identical to the official
`gym_pusht` renderer.

```python
import stable_envs as se

env = se.make("PushT", batch=16384, size=96, device="cuda")
states = env.sample_states(16384)            # (16384, 5) float32, the env's own reset distribution
frames = env.render(states)                  # (16384, 96, 96, 3) uint8, on device
```

## Keep your training loop. Delete your image dataset.

Your loop already reads frames from somewhere — an image folder, a Lance table, a WebDataset shard.
That *somewhere* is what this replaces. The model, the optimiser, the sampler and the loop body do not
change.

=== "Before"

    ```python
    ds = ImageFolder("frames/", transform=to_tensor)     # 27 KB per sample, on disk
    loader = DataLoader(ds, batch_size=1024, num_workers=8)

    for frames, target in loader:
        frames = frames.cuda(non_blocking=True)
        loss = mse(net(frames), target)
        loss.backward(); opt.step(); opt.zero_grad()
    ```

=== "After"

    ```python
    env = se.make("PushT", batch=1024, size=96, device="cuda")   # +2 lines
    ds = TensorDataset(states)                                   # 20 B per sample, in RAM
    loader = DataLoader(ds, batch_size=1024)                     # no workers needed

    for (s,) in loader:
        frames = env.set_state(s).render()                       # <- the swap
        loss = mse(net(frames), env.get_state())
        loss.backward(); opt.step(); opt.zero_grad()
    ```

`render()` returns `(B, H, W, 3)` `uint8` **already on the device**. Nothing is copied from host,
nothing is decoded, and no worker process exists to configure.

### It is the same picture

The official CPU renderer and the GPU renderer, fed the **same recorded states**. Not two rollouts that
look alike — one state stream, two renderers:

<video controls muted loop playsinline width="100%" poster="assets/media/PushT/grid.png">
  <source src="assets/media/PushT/sidebyside.mp4" type="video/mp4">
</video>

More per family in the [Gallery](gallery.md); the per-env numbers behind the claim are in
[Fidelity tiers](concepts/fidelity-tiers.md).

### What actually changes, stated precisely

This replaces the **dataset**, not the simulator — worth being exact about, because that is where a
drop-in claim usually turns out to be false.

<div class="grid cards" markdown>

- **Rendering is a drop-in.**

    `render` is a pure function of a `(B, state_dim)` stack. Same states in, same pixels out — for most
    families byte-identical to the official renderer.

- **Stepping is not.**

    `StableEnv` deliberately has no universal `step`, and the rest say what to use instead. A contact
    solver cannot make a bit-exactness claim the way a rasterizer can:
    [Why physics is separate](concepts/physics.md).

</div>

So the shape that works is: **your existing simulator produces states, stable-envs turns states into
pixels.** If today you run a simulator once and save PNGs, you keep the first half and throw away the
second.

!!! warning "You need states, not just frames"

    If your pipeline only ever kept frames, there is a one-time step: re-run your rollouts recording
    `env.get_state()` — or whatever your simulator's compact state is — instead of images. That corpus
    is what you store from then on.

## The measured claim

All figures on one NVIDIA H200, torch 2.11.0+cu130, triton 3.6.0 — and re-measured unchanged on torch 2.13.0+cu130 / triton 3.7.1 (geomean 1.003x over 12 cells, `scratch/census/triton_ab.sbatch`), each env at its own
`default_supersample` — 1 for PushT, and 2 for the 44 classes whose reference is MSAA'd. Full tables and
method: [Throughput](benchmarks.md).

| | PushT @ 96px | PushT @ 224px |
|---|---|---|
| official CPU renderer | 491 frames/s | 438 frames/s |
| this repo, batch 16,384 | 2,914,271 frames/s | 839,360 frames/s |
| speedup | **5,931×** | **1,915×** |
| interior-pixel parity vs the official renderer | **100.00%** | 99.9985% |
| state vs raw frame | 20 B vs 27,648 B | 20 B vs 150,528 B |

Those five rows are a transcription of one run of `examples/bench_gpu_vs_cpu.py`. The file that script
writes,
[`examples/results/bench_gpu_vs_cpu.md`](https://github.com/galilai-group/stable-envs/blob/main/examples/results/bench_gpu_vs_cpu.md),
is the source, and it is regenerated more often than this page is edited — when the two disagree,
believe the generated file. It also carries the supersample each row resolved to and the `cpu%` of the
timed loop, which is what says whether a row travels to another node.

Three caveats stated up front, because all three are visible in the tables. At batch 1 the GPU path is
the same order as the reference (558 against 491 frames/s), because 91% of that call is building the
scene in Python and there is nothing to amortise it over. At the headline batch, 38% of the 96px wall
time is still that scene build rather than rasterizing — the rasterizer alone runs at 4.89 M frames/s.
And neither column travels unconditionally: the same official renderers measured up to 11× slower on a
busier host, and the GPU column repeats across nodes to under a percent only where the device is the
limit. That 0.4% was verified on the fused 2-D rasterizer and does not extend to the ray-cast families,
which spend 100–206% of the timed loop on CPU — `DMCBallInCupCatch` measured 3,664 frames/s idle against
271–978 contended, a 13× spread. Read the speedup as an order of magnitude.

## Quickstart

```bash
pip install -e .                  # numpy, torch, gymnasium
pip install -e ".[pusht]"         # gym_pusht + pymunk + pygame, for the parity tests
pip install -e ".[hf]"            # datasets, for the HuggingFace corpus round-trip
```

```python
import numpy as np
import stable_envs as se

env = se.make("PushT", batch=4096, size=96, device="cuda")

states = env.sample_states(4096)             # (4096, 5) float32, the env's own reset distribution
frames = env.render(states)                  # (4096, 96, 96, 3) uint8, on device

# render into a pre-allocated buffer to keep the allocation out of a training loop
import torch
out = torch.empty((4096, 96, 96, 3), dtype=torch.uint8, device="cuda")
env.render(states, out=out)

# appearance knobs; defaults are exactly the official appearance
env.variation_space.names()
frames = env.render(states, variation={"block.color": [220, 40, 40]})

# per-environment colours: one value per env, still one kernel launch
per_env = np.random.randint(0, 256, size=(4096, 3)).astype(np.float32)
frames = env.render(states, variation={"block.color": per_env})
```

`env.describe()` returns the state layout, the reference id and the variation leaf names — enough to
consume an env you have never read the source of. `stable_envs.envs.import_errors()` reports any env
module that did not import, so a missing optional dependency loses one env rather than the package.

### Put the whole corpus in RAM and slice it

This is the one place where storing states instead of frames should change how you write the loop, so
it is worth saying why rather than just showing the call.

**States are small enough that the entire corpus fits in RAM.** PushT is 20 bytes a row against a
27,648-byte frame; 200,000 states is 4 MB. Once the data is resident and tiny, **the loader has no I/O
left to hide** — and hiding I/O is the only reason a `DataLoader` is shaped the way it is.

A stock loader calls `__getitem__` once per sample and hands the results to `default_collate`, which
stacks them back together. When each `__getitem__` is a disk seek and a JPEG decode, that per-sample
cost is worth paying and workers are worth spawning. When each one is a 20-byte view of a tensor that
was already contiguous, you are paying roughly **3.4 µs of Python per row** to take an array apart and
immediately reassemble it. So ask the sampler for the batch instead, and index once:

```python
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

ds = TensorDataset(torch.as_tensor(states))                  # the whole corpus, in RAM
sampler = BatchSampler(RandomSampler(ds), batch_size=4096, drop_last=False)
loader = DataLoader(ds, sampler=sampler, batch_size=None)    # torch's own collate, one slice

for (s,) in loader:
    frames = env.set_state(s).render()                       # (4096, 96, 96, 3) uint8, on device
    loss = criterion(net(frames), target(env.get_state()))
```

Measured on PushT at 96 px, ms per batch, same states and same render throughout:

| batch | per-item + `default_collate` | `BatchSampler`, one slice | ceiling (no loader at all) |
|---|---|---|---|
| 256 | 3.426 | 2.666 | 2.524 |
| 1024 | 5.600 | 2.886 | 2.774 |
| 4096 | **17.705** | **3.677** | 3.369 |

At B≤256 the render dominates and the choice barely matters. At 4096 the per-item path is **4.8×**
slower, and it is not the bytes: with the render removed, the fetch alone is 14.890 ms per-item against
0.794 ms for one batched index.

Two corollaries that are easy to get backwards:

* **`num_workers` is the wrong fix.** Eight workers take 17.705 ms to 6.397 — still 1.7× off the
  one-slice path, and it spends processes and IPC parallelising Python overhead that should not exist.
  There is no disk in this path to overlap with.
* **Putting the states on the GPU is *slower*.** 21.191 ms at B=4096, the worst of every option
  measured, because a CUDA `TensorDataset` makes 4,096 tiny device slices. Keep states on the host and
  let `set_state` do one transfer.

`drop_last` is left at torch's default of `False`: the short final batch of an epoch is rendered rather
than dropped, and `set_state` draws exactly those rows — 3.371 ms for 4,095 PushT states against 3.341
for 4,096. `set_state` also refuses, by name, the three inputs that used to render a plausible picture
of nothing: frames passed where states were meant, a row of the wrong width, and NaN or inf.

The library's own datasets exist for the cases that are genuinely about rendering or about episode
structure:

```python
from stable_envs import render_states, StateDataset, TrajectoryDataset

frames = render_states("PushT", states, size=96, chunk=8192)   # (..., 96, 96, 3), streamed

ds = StateDataset("PushT", states, size=96)      # stores states, YIELDS FRAMES: an image-folder shim

ds.compression
# {'n': 100, 'state_bytes': 2000, 'style_bytes': 0, 'stored_bytes': 2000,
#  'frame_bytes': 2764800, 'frame_shape': (96, 96, 3), 'ratio': 1382.4}

loader = DataLoader(ds, batch_size=1024, shuffle=True,
                    collate_fn=se.collate_batched, num_workers=0)
```

`render_states` chunks the input so peak memory is bounded by the chunk rather than by the corpus, which
is what lets a dataset larger than device memory pass through. `StateDataset` is the drop-in for an
image folder — it hands back pixels, so an existing training step does not change — and it needs
`collate_fn=stable_envs.collate_batched`, because it renders in the batched `__getitems__` hook rather
than per item (one launch for the whole batch is the entire point) and so returns an already-assembled
`(B,H,W,3)` tensor, which `default_collate` treats as a *list* of samples and fails on with
`stack(): argument 'tensors' must be tuple of Tensors, not Tensor`. `num_workers=0` belongs with it
because the frames are already on this process's GPU.

`TrajectoryDataset` is the same idea one level up: `(n_traj, T, state_dim)` plus per-trajectory style,
yielding windows that never straddle an episode boundary, with a HuggingFace round-trip that stores
`(states, style)` together — because a corpus that keeps only the dynamic half re-renders under whatever
appearance happens to be set, which is a coincidence rather than reproduction.

```python
traj = TrajectoryDataset("PushT", states, window=4, stride=2)   # (n_traj, T, 5)
traj[[0, 1, 2]]["states"].shape        # (3, 4, 5) — the time axis is always present
traj.all_frames()                      # re-render the whole corpus, chunked
```

## How it works

An env implements one method that carries the whole claim:

```python
def scene(self, states, variation) -> Scene2D:   # (B, state_dim) -> a batch of 2-D scenes
```

`Scene2D` is a small IR — convex polygons, discs, thick segments, composited back-to-front with flat
colours — and one fused Triton kernel rasterizes any of it. An env that can produce a `Scene2D` from a
state has *proved* the frame is a function of the state, because there is nowhere else for information to
come from: no env instance, no simulator handle, no hidden buffer.

Everything is per-environment tensors, so a batch of 16,384 can each carry a different block colour and
still render in one launch. Domain randomisation is a tensor, not a loop.

Physics is deliberately not part of the contract. A renderer can be bit-exact against the reference; a
contact solver cannot, and conflating them lets an "our env matches" claim rest on the half that does
not. See [Why physics is separate](concepts/physics.md).

## Where to go next

<div class="grid cards" markdown>

- **[The Scene2D IR](concepts/scene-ir.md)** — the primitive vocabulary every env emits, and the three
  details in it that are load-bearing.
- **[The fused rasterizer](concepts/rasterizer.md)** — why one kernel, what it costs, and the 73–85× the
  readable torch reference costs instead.
- **[Variation spaces](concepts/variation-spaces.md)** — the appearance knobs, and a measured table of
  which ones actually vary per environment today.
- **[Adding an environment](adding-an-env.md)** — the worked PushT example. Start here if you are
  porting something.
- **[Testing and conformance](testing.md)** — what each check rules out, with the measured numbers.
- **[Gallery](gallery.md)** — side-by-side video against the official renderer, on identical states.

</div>

## The registry

**196 environments in 24 families** at the last census (2026-08-11), every parity number measured
against the official renderer. No single interpreter can import every family, so that count is the union
over 15 of them and the widest single one holds 147 — `scratch/census/union.json` is the source. The
families below are the ones whose parity has been written up, and the registry runs ahead of the table:
`len(stable_envs.REGISTRY)` is the current answer, and `stable_envs.envs.import_errors()` explains any
shortfall.

| family | envs | parity |
|---|---|---|
| Classic control | 5 | **100.00000%**, 256/256 frames exact per env |
| MiniGrid + BabyAI | 7 (+165 via `register_all`) | **max abs diff 0** over 21,216 frames, all 172 ids |
| MiniHack | 1 | **100% of all pixels** |
| Craftax Classic | 1 | **100% of all pixels**, max abs diff 0 |
| PushT | 1 | **100% of interior pixels** |
| highway-env | 6 | **100% of interior pixels**, 99.93–100.00% of all ¹ |
| Sokoban | 1 | **100% of all pixels**, max abs diff 0 |
| LunarLander | 1 | **100% of interior pixels**, 99.97% of all |
| OGBench mazes | 25 | **100% of interior pixels**, no tolerance relaxed |

The throughput columns that used to sit here have been removed rather than updated. They were a second
copy of numbers that live in [Throughput](benchmarks.md) and in
[`examples/results/decode/ALL.md`](https://github.com/galilai-group/stable-envs/blob/main/examples/results/decode/ALL.md),
and a second copy is a copy that goes stale on its own schedule — every one of them was measured at a
forced `supersample = 1` and before two kernel changes. One table, regenerated from the JSONs, is the
arrangement this repo can keep honest.

¹ The wide highway-env range is resolution, not efficiency: Roundabout, Intersection and Racetrack
draw 600×600, four times the pixels of Highway and Merge. Per-env numbers are in
[the highway-env note](research/highway.md).

They reach exactness by three different routes, and the route is what the throughput tracks. The
tile-atlas families compose the reference's own cached tile images rather than redrawing them, so they
are byte-exact by construction; the `Scene2D` path has to match a rasterizer, and matches it everywhere
the answer is unambiguous; the OGBench mazes ray-cast analytic primitives against a baked static plate.
The OGBench row above is at the shipped defaults (224px, supersample 2), which is what `se.make` returns
and what its parity is measured on. Every OGBench figure published before 2026-08-11 was taken at
supersample 1 instead — a different picture, four times cheaper in rays — so those cells were faster than
the environment a user gets. The row is re-measured; the correction is in
[Throughput](benchmarks.md).

The registry grows between doc builds, so `sorted(stable_envs.REGISTRY)` is the current answer, and it
also shrinks on a machine that is missing a reference package — DM Control needs its own pin
(`mujoco==3.11` + `dm_control==1.0.44`), and without it those 16 envs are absent rather than broken.
`stable_envs.envs.import_errors()` names any family in that state.

### Four families are not in that table, on purpose

Every row above is **parity tier**: the frame is the official frame. Thirty-two classes are **visual
tier** — a different render of the same scene, fast enough to generate training frames live and
measurably not the official pixels. RoboCasa's four (7.0× the official renderer at batch 256, 256 px;
mean 5.7–32.1/255, SSIM 0.92–0.82, 4.8–15.8% of interior pixels exact against the 1.0 everything above
declares), gymnasium's six MuJoCo locomotion domains through nvdiffrast (23–27× at batch 256,
240 px; mean 1.97–4.39/255, SSIM 0.947–0.981, 54–88% interior exact), DM Control's five body-tracking
domains through `mujoco_warp` (225–364× at batch 1024; mean 4.35–8.07/255, SSIM 0.661–0.964, 70.0–90.5%
of pixels within 8/255), and HumanoidBench's seventeen, also through `mujoco_warp` (437× at batch 64,
256 px; mean 2.6–8.4/255, SSIM 0.845–0.931, 60.7–94.3% interior exact). Each attributes its residual to
named mechanisms rather than leaving it as noise: textured primitives drawn flat for the first, MuJoCo's
minification-filtered checker floor for the second, a `trackcom` camera that moves the static half of
the scene with the state for the third — so the baked plate the other eleven DM Control cells rely on
does not exist — and a floor grid that `mujoco_warp` cannot mipmap for the fourth.
`print(stable_envs.format_registry())` marks them, `describe()["fidelity_tier"]` reports it,
and the parity test skips them rather than being relaxed for them. What that licenses and what it does
not is [Fidelity tiers](concepts/fidelity-tiers.md); the residual is shown beside the official frame in
[`examples/robocasa_visual/`](https://github.com/galilai-group/stable-envs/tree/main/examples/robocasa_visual).

Feasibility notes, including the families that were examined and rejected, are in the research section;
the current verdict list lives there rather than being copied here, so that a no-go reopened by a better
measurement changes in one place. A family where the thesis does not hold is a result worth writing down,
not a gap to hide: [when does this pay off](research/when_does_this_pay_off.md).

## Building these docs

```bash
python -m venv .venv-docs && . .venv-docs/bin/activate
pip install mkdocs mkdocs-material "mkdocstrings[python]"
mkdocs build --strict     # or: mkdocs serve
```

The API reference is generated from the source docstrings by mkdocstrings, which reads them statically —
building the site does not require torch, triton or a GPU.
