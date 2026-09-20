# Learned render correctors

A visual-tier cell renders the same scene as the official renderer with a different rasterizer, and
declares how far apart the two are. A **render corrector** buys back most of that distance and a large
part of the throughput at the same time: render deliberately cheap — 112 px, textures at mip level 2 —
and hand the result to a 441 k-parameter convolutional network fitted to predict the official 256 px
frame.

`RoboCasaCounterToSink`, 64 states × 4 draws at 256 px, worst draw, end to end including inference on
one H200:

| | f/s | mean \|diff\| | within 8/255 | SSIM |
|---|---:|---:|---:|---:|
| `RoboCasaCounterToSink` — MJWarp at 256 px | 5,756 | 6.41 | 0.833 | 0.907 |
| `RoboCasaCounterToSinkCorrected` | **16,959** | **2.48** | **0.935** | **0.970** |

2.9x faster and 2.6x closer, and past the 15,937 f/s that eight worker processes reach reading these
frames off disk — which is the number that decides whether regenerating beats storing.

## Build one

```
python examples/rendernet/train_corrector.py RoboCasaCounterToSink
```

About forty minutes on one GPU, most of it the official renderer producing targets. It writes
`~/.cache/stable-envs/rendernet/RoboCasaCounterToSinkCorrected.pt` (override the location with
`STABLE_ENVS_RENDERNET_CACHE`), and `RoboCasaCounterToSinkCorrected` appears in the registry on the
next import. **The repository ships code, not weights** — the same rule that makes it ship no compiled
RoboCasa kitchen and no MiniGrid atlas.

Useful flags: `--no-seal` measures and reports without writing; `--report FILE` dumps every draw;
`--epochs`, `--n-train`, `--net` for the sweep. `--from-prototype PATH` adopts an already-fitted
state dict and still measures and still gates it.

## It can refuse, and that is the point

The script will not write a checkpoint unless the corrected path beats the plain class on **mean
|diff|, within-8 and SSIM**, on the same state draws, **both** at the default appearance and under the
env's own declared variation space. Three separate comparisons and never an aggregate, because the way
a corrector fails is that it lowers a frame's average error while making more of its pixels wrong:
applied to a kitchen it was not fitted on, it paints marble-like speckle onto plain cabinet fronts, the
mean ties (33.19 against 32.85 for a plain resize) and SSIM falls (0.763 against 0.792).

The same three comparisons run in the conformance suite
(`test_a_learned_decoder_beats_the_render_it_replaces`), so a render path that drifts after the weights
were sealed is caught in the interpreter it drifted in.

## A corrector is per (scene, camera)

Zero-shot, the `RoboCasaCounterToSink` corrector on cells it was not fitted for: 5.94/255 on a camera
40 cm away against 6.14 for a plain resize, and 11.35 on the wrist camera against 4.39 — **worse than
doing nothing, 2.6×**. So the checkpoint is keyed to `(env class, family, scene, camera, MJCF hash)`
and the key is verified at load; a mismatch raises `CellMismatch` rather than falling back. Copying the
file under another cell's name does not work, because the key is inside it.

Handing the network the renderer's own depth and per-geom identity does not fix this and was measured:
worth 0.14/255 in distribution, costs 3.6 µs a frame, and makes the cross-kitchen SSIM *worse*. The
full study, including that negative result and the frontier sweep behind the architecture choice, is
`docs/research/rendernet.md`.

## What it changes, and what it does not

The frame remains a pure deterministic function of `(state, style)`: the checkpoint is fixed, the
render feeding it is already such a function, so `render` is still `f(get_state())`, still
batch-invariant, still reproducible bit-for-bit by anyone holding the weights. Every test in
`tests/gpu/test_reproducibility.py` applies unchanged.

What changes is the **decoder**. Most of the win is that the network paints a marble texture its input
does not contain — `mujoco_warp` draws RoboCasa's textured `BOX` primitives as one flat colour, so the
detail came out of the weights, not out of the frame. It has the right statistics and the wrong pixels,
and a frozen pretrained encoder can tell even where the pixel metrics improve by 2.5×. These classes
declare `render_decoder = "learned"`, and anyone who will be evaluated on official pixels needs to
record which of the two produced their training frames, because it is not visible in the picture.

See [`docs/concepts/fidelity-tiers.md`](../../docs/concepts/fidelity-tiers.md) for the contract and
`stable_envs/render/rendernet.py` for the loader and the identity lock.
