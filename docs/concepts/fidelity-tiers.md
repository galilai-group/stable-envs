# Fidelity tiers: parity and visual

Every environment in this repo declares one of two tiers, and the tier is part of the contract rather
than a caveat in a README. `describe()` reports it, `stable_envs.format_registry()` marks it, the class
docstring says it, and the conformance suite applies a different assertion to each.

```python
import stable_envs as snv
print(snv.info("PushT")["fidelity_tier"])                     # 'parity'
print(snv.info("RoboCasaCounterToSink")["fidelity_tier"])     # 'visual' (after contrib.load(), see below)
```

RoboCasa, used as the example on this page, is a renderer-only port kept in the repository's `contrib/`
directory and is not part of the released package; `import contrib; contrib.load()` from a checkout
registers it.

There is a **second, independent field**, and a consumer of the visual tier has to read both:

```python
snv.info("RoboCasaCounterToSink")["render_decoder"]            # 'renderer'
snv.info("RoboCasaCounterToSinkCorrected")["render_decoder"]   # 'learned'
```

`render_decoder` says *what drew the frame*: a rasterizer, or a rasterizer followed by a fixed neural
checkpoint trained to predict the official frame. [The learned decoder](#the-learned-decoder) below is
the whole of it, and it is not a footnote — the difference between "MuJoCo's renderer approximated"
and "a learned approximation of MuJoCo's renderer" is invisible in the picture and visible only here.

## What the two tiers claim

| | **parity** | **visual** |
|---|---|---|
| the frame is | **the official frame** | **a different render of the same scene** |
| geometry, camera, materials, state | the official ones | the official ones |
| rasterizer | ours, verified against the official one | a different one, much faster |
| declared as | `parity_interior_min` (1.0 unless a mechanism is stated) | `visual_fidelity`: mean abs error, % within 8/255, SSIM |
| asserted by | `test_render_parity_with_official_renderer` | `test_visual_tier_matches_its_declared_fidelity` |
| a dataset regenerated from states is | interchangeable with a stored one | **not** interchangeable with a stored one |

A parity env answers "if I had stored these frames, would they be these frames?" with *yes, byte for
byte, on every interior pixel*. That is what licenses the repo's headline claim — stop storing image
datasets — and it is a claim about identity, not about resemblance.

A visual env answers a different question. The scene is the same scene, the state is the same state, the
camera is in the same place, and the picture is drawn by a rasterizer that disagrees with the official
one in ways the env measures and names. It exists because the parity bar is the wrong bar for one real
use, and refusing that use would not make the bar stronger.

## What the visual tier does and does not license

!!! success "Suitable for live training-data generation"
    Store states, regenerate frames in the forward pass, randomise camera and lighting per environment
    inside one launch. The pixels are self-consistent — the same state gives the same frame, the frame
    is a function of the state, and the appearance knobs are per-environment tensors — which is what a
    training loop needs.

!!! danger "NOT a substitute for a stored dataset when a policy will be evaluated on official pixels"
    The train and eval pixel distributions differ, by the amount in `visual_fidelity`, and **nothing in
    this repo measures what that difference costs a policy.** If the evaluation runs on official
    frames and a stored dataset exists, use the stored dataset. This is the distinction the tier exists
    to make; it is not a hedge.

Two smaller consequences follow from the same fact and are worth stating separately, because each has
been assumed away before:

* **Do not compare a number measured here against a published number measured on official pixels.** A
  benchmark result is a claim about a pixel distribution as much as about a policy.
* **Do not read a visual-tier residual as a bug report.** It is declared, measured, and attributed to a
  named mechanism in `visual_note`. A residual that grows past the declared thresholds *is* a bug, and
  that is exactly what the test asserts.

## Why the parity test was not simply relaxed

The obvious alternative is one test with a per-env threshold, spanning "every interior pixel is
bit-identical" and "the picture looks right". It was not done, for a reason this repo has watched play
out: the floor becomes the contract. A single threshold that some env sits at 0.05 on is a threshold
everyone can move, and the next env that cannot reach 1.0 moves it again — while the strong claim, which
is the whole warrant for the project, quietly stops being tested for anyone.

So the two claims are kept mechanically apart:

* `test_render_parity_with_official_renderer` is **unchanged** and **skips** visual-tier envs. Its
  floor is still `INTERIOR_EXACT_MIN = 1.0`, still relaxable only per-env with a stated `parity_note`.
* `test_visual_tier_matches_its_declared_fidelity` runs only for visual-tier envs and asserts three
  quantities that fail for different reasons: `mean_abs` catches a global shift, `within8_frac` catches
  a region going wrong while the mean stays small, and `ssim` catches structure — a surface losing its
  texture moves SSIM and can leave both of the others where they were.
* A visual env sets `parity_interior_min = None`. `register()` refuses it otherwise, so the parity gate
  cannot be applied to it, cannot be silently loosened for it, and cannot be passed by it.

The visual test deliberately does **not** assert anything about interior-pixel exactness. The env
reports that number, because it is the honest measure of what the tier gave up, and turning a number
reported for honesty into a target is how it stops being reported.

## Declaring the visual tier

`register()` enforces all of this at import time, so a half-declared tier is an error rather than a
review comment:

```python
@register
class MyEnv(Env):
    """... VISUAL TIER — these are not the official pixels. <what the residual is> ..."""

    fidelity_tier = "visual"
    parity_interior_min = None          # required: the parity gate does not apply
    visual_fidelity = {
        # MEASURED
        "mean_abs": 6.55, "within8_frac": 0.830, "ssim": 0.909,
        # THRESHOLDS the suite asserts, clear of the measurements
        "mean_abs_max": 8.5, "within8_min": 0.78, "ssim_min": 0.87,
        "measured_at": "256 px, 64 states from sample_states, H200, ...",
    }
    visual_note = "<the mechanism, and what the tier does not license>"
```

The measurements and the thresholds are both required and are deliberately different numbers: a
threshold equal to the measurement flakes on the next state draw, and a measurement inferred from a
threshold is not a measurement. Set the thresholds from the *spread over independent state draws*, not
from one; `examples/robocasa_visual/calibrate_tier.py` is the worked example.

The docstring check is mechanical on purpose. `describe()` and the registry listing carry the mark, but
the first thing a person reads is `help(cls)`, and a marker that depends on remembering to write it is
the kind of rule this repo makes impossible instead of asking for.

## The learned decoder

The only learned decoder in the repository, RoboCasa's (`contrib/render/rendernet.py`), is in
`contrib/` and is not part of the released package.

A visual-tier cell may additionally ship a **learned render corrector**: a small convolutional network
(441 k parameters, 7.9 µs a frame) that maps a deliberately cheap render — 112 px, textures at mip
level 2 — to the 256 px frame MuJoCo's own renderer would have drawn. Such a class declares
`render_decoder = "learned"`, and it is a **separate registry entry** from the plain cell:

```python
snv.make("RoboCasaCounterToSink")           # MJWarp at 256 px.        render_decoder 'renderer'
snv.make("RoboCasaCounterToSinkCorrected")  # MJWarp at 112 px + a CNN. render_decoder 'learned'
```

It is faster **and** closer to the official frame, at the same time. On `RoboCasaCounterToSink`, 64
states × 4 draws at 256 px, worst draw, end to end including inference:

| | f/s | mean \|diff\| | within 8/255 | SSIM |
|---|---:|---:|---:|---:|
| `RoboCasaCounterToSink` — MJWarp at 256 px | 5,756 | 6.41 | 0.833 | 0.907 |
| `RoboCasaCounterToSinkCorrected` | **16,959** | **2.48** | **0.935** | **0.970** |

**It needs a batch, and this is the one thing to check before switching.** The network runs at a fixed
batch of 128 (see [below](#reproducibility-and-the-one-thing-to-record)), so a caller who asks for
fewer frames pays for 128 anyway. At the env's own surface on one H200:

| batch | 1 | 16 | 64 | 256 | 1024 |
|---|---:|---:|---:|---:|---:|
| `RoboCasaCounterToSink` f/s | 828 | 4,174 | 5,338 | 5,697 | 5,756 |
| `RoboCasaCounterToSinkCorrected` f/s | 416 | 5,137 | 11,889 | 16,365 | 16,959 |
| ratio | **0.50×** | 1.23× | 2.23× | 2.87× | 2.95× |

It crosses over between 16 and 64 and beats the 15,937 f/s disk read from 256 up. **Single-frame
rendering is half the speed of the plain class**, so a per-step interactive loop is the case where the
plain cell is still the right answer; a training loop that renders a batch is the case this is for.

### What it does not change

**The contract.** The corrector is a fixed checkpoint applied to a render that is already a pure
function of `(state, style)`, so `render` is still `f(get_state())`, still batch-invariant, still
deterministic, still reproducible bit-for-bit by anyone holding the weights. Every test in
`tests/gpu/test_reproducibility.py` applies to these classes unchanged and unrelaxed.

### What it does change

**What the picture is made of.** Most of the win is that the network *paints a marble texture its
input does not contain*: `mujoco_warp` draws RoboCasa's textured `BOX` primitives as one flat colour,
so the detail the corrector supplies came out of its weights, not out of the frame. It has the right
statistics and the wrong pixels — and a frozen pretrained encoder can tell the difference even where
the pixel metrics improve by 2.5×. In DINOv2 and ResNet-18 embeddings the corrected frame sits
1.2–1.7× *further* from the official one than the plain render does, while being 2.5× closer in pixels.
Nothing here licenses "a model would not notice".

!!! danger "Say which one your dataset came from"
    Someone evaluating a policy on official RoboCasa pixels needs to know whether their training
    frames came from *MuJoCo's renderer approximated* or from *a learned approximation of MuJoCo's
    renderer*. **That is not visible in the picture, only in the declaration.** Record
    `describe()["render_decoder"]` alongside the states, and do not describe these frames as MuJoCo's
    renderer at reduced fidelity.

### A corrector is per (scene, camera), and a mismatched one is worse than nothing

This is the property everything else is built around, because it fails silently. Zero-shot, the
`RoboCasaCounterToSink` corrector applied to cells it was not fitted on:

| applied to | corrector | bilinear resize | the render it replaces |
|---|---:|---:|---:|
| same kitchen, camera moved 40 cm | 5.94 | 6.14 | 4.75 |
| same kitchen, wrist camera | 11.35 | **4.39** | 3.94 |
| a different kitchen | 33.19 (SSIM 0.763) | 32.85 (SSIM 0.792) | 32.20 (SSIM 0.815) |

On the different kitchen it paints marble-like speckle onto plain cabinet fronts that carry no texture
in the official frame. Read the last row twice: **the mean ties and SSIM and within-8 are worse.** A
corrector can lower a frame's average error while making more of its pixels wrong, which is why the
gates below are three separate comparisons and never an aggregate.

So it is a lock rather than a warning. Every checkpoint is keyed to `(env class, family, scene,
camera, MJCF hash)`; `load` verifies that key against the env and **raises `CellMismatch`** on any
disagreement. There is no fallback to a neighbour's weights, no "close enough", and copying the file
under another name does not work — the key is inside it.

### How a class comes to exist

**No checkpoint, no class.** The repository ships code, not weights, exactly as it ships no compiled
RoboCasa kitchen and no MiniGrid atlas. A corrector is built on the machine that wants it:

```
python examples/rendernet/train_corrector.py RoboCasaCounterToSink
```

which generates its own pairs from the env and its official reference renderer, fits the network,
measures the corrected path **and the plain class on the same draws**, and seals a checkpoint into
`~/.cache/stable-envs/rendernet/` — but only if the corrector beats the plain render on mean |diff|,
within-8 **and** SSIM, both at the default appearance and under the env's own declared variation
space. A corrector that is not better than the renderer it replaces cannot be produced by the
supported path. With no checkpoint present, nothing is registered and the plain cell is untouched: the
absence of weights removes an env, it never changes one.

The declaration travels **inside** the weights file, because `visual_fidelity` here describes a
(renderer, checkpoint) pair rather than a class. A new sealed checkpoint dropped into the cache
directory registers a new class carrying its own measured numbers, with no code change.

### The second declaration, and why a renderer does not need one

A learned class declares `variation_fidelity` as well as `visual_fidelity`: the same three quantities,
measured with a per-state appearance drawn from `variation_space` instead of at the default.
`register()` refuses a learned decoder without it.

The asymmetry is measured. A rasterizer that agrees with the official one at one camera agrees at
another — same arithmetic, pointed somewhere else — so one declaration at the default appearance
describes the whole space. A network does not generalise that way: a corrector fitted at one camera
pose scores **3.25/255 there and 9.06 under the same env's declared ±8° of pan**, which is worse than
not correcting at all (8.91) *and* worse than the render it replaces (7.93). A default-appearance
declaration would have said 3.25 while every consumer using the variation knobs — the reason this
family exists — got the worst of the three. Training on the declared variation costs nothing and
improves the default case too (2.65 against 3.25), so the shipped correctors are fitted that way and
`test_a_learned_decoder_holds_up_under_its_declared_variation` re-measures it.

### Reproducibility, and the one thing to record

The network runs in fp16, `channels_last`, compiled with
`torch.compile(mode="max-autotune-no-cudagraphs")` at a **fixed batch of 128** and replayed from a CUDA
graph. Two of those are correctness rather than speed:

* **The fixed batch.** Inductor's autotuner benchmarks per shape, and two convolution tile
  configurations sum a 3×3×C window in different orders; fp16 carries 11 bits of mantissa and the last
  act of the network is to round to 1/255. The same checkpoint at a network batch of 64 and at 128
  produces *different pixels*. Every call is therefore padded to exactly 128 rows and the padding
  discarded, which makes batch invariance and batch-size independence hold by construction — the same
  argument `humanoidbench.render` makes for pinning MJWarp's world count.
* **Inductor is allowed here and nowhere else.**
  [Writing a bit-exact Triton kernel](bit-exact-triton.md) keeps the rasterizer and ray-casters off
  Inductor because it fuses floating-point operations, and those are parity-tier paths where a 1-ulp
  change disqualifies the claim. This path is a network's approximation of another renderer: fp fusion
  here is arithmetic, not a defect, and the fidelity it produces is re-measured rather than assumed.
  Measured cost of the fusion: ~11% of pixels move by 1/255 against the eager path, for 22.1 µs →
  7.9 µs a frame.

`STABLE_ENVS_RENDERNET_COMPILE` and `STABLE_ENVS_RENDERNET_BATCH` change the arithmetic, so **a dataset
generated through a learned decoder should record them** along with the states — they are the only two
settings in this repository that do. The autotuner's choice was measured to be stable: three
cold-cache runs on two H200 nodes plus a warm run all produced the same frame digest.

## What is in the tier today

The Gymnasium MuJoCo locomotion, RoboCasa and HumanoidBench cells described here are renderer-only
ports kept in `contrib/`; they are not part of the released package.

**Gymnasium's six MuJoCo locomotion domains** — HalfCheetah, Hopper, Walker2d, Ant, Humanoid, Swimmer
— rendered by `nvdiffrast` (`stable_envs/render/nvdiff.py`). Mean 1.97–4.39/255, 83.9–94.5% of pixels
within 8/255, SSIM 0.947–0.981, at 23–27× the official rasterizer at batch 256. The residual is one
named mechanism and it is the reason these six are not in the parity tier while the other four of the
same family are: MuJoCo's `MatPlane` is a checker seen from ~3 m, so most of it is minified and what
GL draws out there is a mipmapped average — luminance spread 0.0–6.8 about a single grey where the
near field spans black to white. A renderer that resolves one texel per pixel paints speckle instead,
which is what caps `mujoco_warp` at 0.0017–0.0069% of far-field floor pixels bit-exact. nvdiffrast
samples a prefiltered pyramid and reaches 66–81% of them; the same pipeline with the pyramid switched
off lands back at 0.01%, which is the control that says the two were failing for one reason.

**DM Control's five body-tracking domains** — `DMCCheetahRun`, `DMCWalkerWalk`, `DMCHopperHop`,
`DMCHumanoidWalk`, `DMCSwimmerSwimmer6` — rendered by `mujoco_warp`. Mean 4.35–8.07/255, 70.0–90.5%
within 8/255, SSIM 0.661–0.964, at 225–364× the official renderer at batch 1024. Camera 0 on these is
`mode="trackcom"`: it rides the body's centre of mass, so the static half of the scene moves with the
state and the plate that makes the other eleven DM Control cells parity does not exist for them. Two
named mechanisms, and the first is the one that separates this family from the Gymnasium six above —
DM Control's skybox is a gradient with `mark="random"` specks and MJWarp samples it with one
unfiltered tap, which is 48–89% of the frame on three of the five and is what holds SSIM down; the
second is the same minified checker floor. nvdiffrast closes the floor and is **not** a drop-in here:
these scenes' model lights are `trackcom` too, and `nvdiff.Lights` carries one position for the whole
batch. A sixth domain (`fish`) was measured and does not ship.

`RoboCasa` — four classes, one compiled kitchen each, rendered by `mujoco_warp`'s batch raytracer. The
residual is one named mechanism: MJWarp derives texture UVs for `PLANE` and `MESH` only, so a textured
`BOX` or `CYLINDER` samples texel (0, 0) and the whole surface comes out one flat colour, and a RoboCasa
kitchen has 178–401 such geoms. A marble counter is drawn as one grey.

**HumanoidBench's seventeen H1 and H1-hands cells** — rendered by `mujoco_warp`. Mean 2.62–8.40/255,
79.5–95.8% within 8/255, SSIM 0.849–0.931, with per-cell `ssim_min` declared between 0.80 and 0.89
(`contrib/humanoidbench.py`, `FIDELITY`). Every camera is `trackcom`, so there is no static
plate here either, and the same minified-checker mechanism as the Gymnasium six applies. These are
registered only under `STABLE_ENVS_HUMANOIDBENCH=1`, because `mujoco_warp`'s batched render is not
bit-reproducible across a long suite session — the cause is now known and is **not** in this repo:
`mujoco_warp` builds one BVH over `nworld x ngeom` primitives, so identical geometry in different
world slots gets differently shaped subtrees and closest-hit ties break by visit order. Filling all 64
worlds with the same `qpos` and comparing against slot 0 isolates it: `h1-balance_simple` differs on 2
pixels at 1/255, in slots 21 and 42, and eight sibling cells differ on none. The upstream fix it
needs is a deterministic tie-break at equal ray parameter.

That made **thirty-two classes** in the tier while the package registered all of them: six MuJoCo
locomotion, five DM Control, seventeen HumanoidBench and four RoboCasa, against 164 in the parity tier.
Of these, only the DM Control cells remain in the released package. A cell with a sealed corrector adds
one more — `RoboCasaCounterToSinkCorrected` — and the count is machine-dependent for the reason
[above](#how-a-class-comes-to-exist): no weights, no class. `stable_envs.format_registry()` prints the
current answer and marks the learned ones separately.

[`examples/robocasa_visual/`](https://github.com/galilai-group/stable-envs/tree/main/examples/robocasa_visual)
shows it side by side with the official frame and the difference amplified 4×, on the best and the worst
of the four kitchens measured, so the quality can be judged rather than taken on the numbers.
RoboCasa remains a **NO-GO for the parity tier**, and this tier does not overturn that.
