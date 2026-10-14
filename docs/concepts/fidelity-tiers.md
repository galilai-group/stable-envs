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
from one: measure several seeded draws and keep each threshold clear of the worst one.

The docstring check is mechanical on purpose. `describe()` and the registry listing carry the mark, but
the first thing a person reads is `help(cls)`, and a marker that depends on remembering to write it is
the kind of rule this repo makes impossible instead of asking for.

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

**DM Control's six body-tracking cells** — `DMCCheetahRun`, `DMCHopperHop`, `DMCHumanoidWalk`,
`DMCQuadrupedWalk`, `DMCSwimmerSwimmer6`, `DMCWalkerWalk` — rendered by `mujoco_warp`. Their
`visual_fidelity` declarations (`VISUAL_FIDELITY` in each cell's module under
`stable_envs/envs/dmcontrol/`) measure against `dm_control`'s `physics.render` at camera 0 on mujoco
3.12.0: mean 0.89–2.53/255, 95.3–98.6% of pixels within 8/255, SSIM 0.984–0.994. Camera 0 on these is
`mode="trackcom"` (Quadruped's measured camera is 2, also trackcom): it rides the body's centre of
mass, so the static half of the scene moves with the state and the plate that makes the other twelve
DM Control cells parity does not exist for them. The share of the frame that is skybox is 0.0 for
Humanoid and Swimmer, 0.148 for Quadruped, 0.478 for Walker, 0.831 for Cheetah and 0.893 for Hopper;
the floor share is 0.959, 0.981, 0.806, 0.448, 0.127 and 0.082 in the same order. nvdiffrast is
**not** a drop-in here: these scenes' model lights are `trackcom` too, and `nvdiff.Lights` carries
one position for the whole batch.

| cell | mean abs /255 | within 8/255 | SSIM |
|---|---:|---:|---:|
| `DMCCheetahRun` | 1.29 | 0.964 | 0.990 |
| `DMCHopperHop` | 0.89 | 0.976 | 0.990 |
| `DMCHumanoidWalk` | 2.23 | 0.967 | 0.984 |
| `DMCQuadrupedWalk` | 1.16 | 0.986 | 0.994 |
| `DMCSwimmerSwimmer6` | 2.53 | 0.963 | 0.994 |
| `DMCWalkerWalk` | 1.75 | 0.953 | 0.991 |

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

The released package holds eight visual-tier classes among the 75 that register without PyFlyt: the
six DM Control cells above, `OGBCube` and `OGBScene`; the other 67 are parity tier. The `contrib/`
ports add six MuJoCo locomotion, seventeen HumanoidBench and four RoboCasa classes.
`stable_envs.format_registry()` prints the current answer.

RoboCasa remains a **NO-GO for the parity tier**, and this tier does not overturn that.
