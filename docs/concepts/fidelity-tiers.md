# Fidelity tiers

Every class declares one of two tiers in `fidelity_tier`. The set is `stable_envs.env.FIDELITY_TIERS`:
`("parity", "visual")`. The default is `parity`.

| | parity | visual |
|---|---|---|
| the frame is | the official frame | a different render of the same scene |
| geometry, camera, materials, state | official | official |
| declared with | `parity_interior_min` (default 1.0) | `visual_fidelity`, `visual_note` |
| checked by | `test_render_parity_with_official_renderer` | `test_visual_tier_matches_its_declared_fidelity` |
| use for | anything, including evaluation on official pixels | training data, but not evaluation on official pixels |

## Reading the tier

```python
import stable_envs as snv
from stable_envs.registry import format_registry

snv.info("PushT")["fidelity_tier"]        # 'parity'; no construction needed
print(format_registry())                  # every class, visual ones marked
env.describe()                            # on a constructed env
```

`env.describe()` returns `fidelity_tier` and `parity_interior_min`. For a visual-tier class it also
returns:

- `visual_fidelity`: the measured values and the thresholds the tests assert.
- `visual_note`: what the residual is and where it comes from.
- `not_for`: what the frames must not be used for.
- `reference_versions`, if declared: the upstream versions the measurement used.

`format_registry()` prints the current count of parity and visual classes.

## Parity checks

`test_render_parity_with_official_renderer` renders 64 states from the env's reset distribution with
both renderers and asserts:

| check | default | meaning |
|---|---|---|
| interior pixels exact | `parity_interior_min = 1.0` | pixels whose 3×3 neighbourhood in the reference is one flat colour must match exactly |
| all pixels exact | `ALL_EXACT_MIN = 0.90` | edge pixels may differ because the two rasterizers anti-alias differently |
| mean absolute difference | `MEAN_ABS_MAX = 4.0` / 255 | catches a small global colour shift |

A class can lower `parity_interior_min` or set `parity_all_exact_min`. It must then give a
`parity_note` that names the cause. `test_relaxed_parity_floors_are_all_explained` checks this for
every registered class.

## Visual checks

`test_visual_tier_matches_its_declared_fidelity` runs only for visual-tier classes. It asserts three
numbers against the class's own `visual_fidelity` thresholds:

| assertion | catches |
|---|---|
| `mean_abs <= mean_abs_max` | a global shift: a light, a material, a colour rule |
| `within8_frac >= within8_min` | one region going wrong while the mean stays small |
| `ssim >= ssim_min` | lost structure, such as a surface losing its texture |

The test does not assert interior-pixel exactness for the visual tier.

## Declaring the visual tier

`register()` checks the declaration at import time. A visual-tier class must have:

- `fidelity_tier = "visual"` and `parity_interior_min = None`
- `visual_fidelity` with `mean_abs`, `within8_frac`, `ssim`, `mean_abs_max`, `within8_min`, `ssim_min`
  and `measured_at`
- a non-empty `visual_note`
- the words `VISUAL TIER` in the class docstring

A parity class must not carry `visual_fidelity`.

```python
@register
class MyEnv(Env):
    """... VISUAL TIER: these are not the official pixels. <what the residual is> ..."""

    fidelity_tier = "visual"
    parity_interior_min = None
    visual_fidelity = {
        "mean_abs": 1.29, "within8_frac": 0.964, "ssim": 0.990,          # measured
        "mean_abs_max": 1.7, "within8_min": 0.92, "ssim_min": 0.95,      # asserted
        "measured_at": "84 px, 64 states x 4 draws, <GPU>, <versions>",
    }
    visual_note = "<the cause of the residual>"
```

Measure several independent state draws. Set each threshold clear of the worst draw.

## Visual-tier classes

The released visual-tier classes include the DM Control body-tracking cells, `OGBCube`, `OGBScene`
and `PyFlytRocketLanding`. The DM Control cells use a camera that follows the body, so they render the
full scene with MJWarp on CUDA or CPU3D on CPU. The table gives their declared values.
The measurements compare against `dm_control`'s `physics.render` on an RTX 3090 with MuJoCo 3.12.0, at
84 px (Quadruped at 224 px).

| class | mean abs /255 | within 8/255 | SSIM |
|---|---:|---:|---:|
| `DMCCheetahRun` | 1.29 | 0.964 | 0.990 |
| `DMCHopperHop` | 0.89 | 0.976 | 0.990 |
| `DMCHumanoidWalk` | 2.23 | 0.967 | 0.984 |
| `DMCQuadrupedWalk` | 1.16 | 0.986 | 0.994 |
| `DMCSwimmerSwimmer6` | 2.53 | 0.963 | 0.994 |
| `DMCWalkerWalk` | 1.75 | 0.953 | 0.991 |
