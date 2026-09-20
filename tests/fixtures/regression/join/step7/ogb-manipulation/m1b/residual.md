# M1b — the authorised residual pass: what is left after the spot correction

Job **6397621**, `gpu2106`, RTX 3090, 8 real states, 224 px, `front_pixels`, `gl_rounding`. Same
recipe as M1: every term neutralised **in the MjModel**, both renderers driven from that same model,
host render-noise floor measured (**0** on every row), verdict flags carried. The spot correction is
applied under every row except the first.

| term | interior | static | mover | mover n | **composited** | interior mean abs | signed |
|---|---:|---:|---:|---:|---:|---:|---|
| no correction (M3 baseline) | 0.0099 | 0.0029 | 0.5072 | 836 | 0.9236 | 8.888 | −8: 15.3 % |
| **spot fix only** | **0.3473** | 0.3430 | 0.7568 | 584 | **0.9341** | 2.648 | 0: 54.5 % |
| spot fix + headlight off | 0.3480 | 0.3218 | 0.8955 | 6881 | 0.9506 | 2.496 | 0: 55.1 % |
| spot fix + spot light off | 0.4606 | 0.4599 | 0.3481 | 1396 | 0.9347 | 1.537 | 0: 62.0 % |
| **spot fix + global light off** | **0.7016** | 0.7005 | 0.6702 | 752 | 0.9397 | 1.811 | 0: 76.4 % |
| spot fix + ambient = 0 | 0.2971 | 0.2937 | 0.3372 | 1124 | 0.9313 | 3.014 | 0: 52.4 % |

## The ordered residual

1. **The `global` DIRECTIONAL light is now the largest term.** Neutralising it takes interior from
   0.347 to **0.7016**, +0.355 — bigger than everything else remaining put together. It is the
   `castshadow = 1` light, and M1 already found that turning *shadows* off makes agreement worse, so
   the disagreement is in the light itself rather than in its shadow.
2. **The spot still carries a residue after its angular term is fixed**: removing it entirely adds a
   further +0.113 (0.347 → 0.461). MJWarp takes `attenuation = 1/(1,d,d²)·(1,0,0) = 1`, i.e. no
   distance falloff; MuJoCo evidently does something else with the same `light_attenuation`.
3. **The headlight is a mover-pixel term, not a floor term.** Interior barely moves (0.3473 →
   0.3480) while mover exactness goes 0.757 → 0.896. That is consistent with a headlight pointing
   along the camera axis: it lands on the things facing the camera.
4. **Ambient is not a term** — zeroing it makes agreement *worse* (0.297), so MJWarp already agrees
   with MuJoCo about it.

## The number that decides the tier

**`composited_all_exact_frac` sits at 0.923–0.951 on every row**, including the uncorrected one.
That is the figure a shipped cell on the `fetch.py` pattern would actually produce: the plate is
baked through MuJoCo itself and is exact *by construction* (phase 0 measured it bit-static over 6
states), so only the mover pixels are MJWarp's to get right, and they are 10–19 % of the frame. The
whole-frame interior figure of 0.347 is 99.2 % a statement about a floor the cell does not ask MJWarp
to draw.

**This does not make it a parity cell.** Fetch declares 0.9988 *interior*; this scene's mover pixels
are 0.30–0.47 all-exact even after the correction, and the interior metric over the composited frame
would still be dominated by terms 1 and 2 above wherever a mover meets the floor. The evidence
supports **`visual`**, with the residual declared term by term as above.

## A measurement caveat that must travel with these numbers

The mover columns are **not comparable across rows**. Each row calls `reset()` before rendering, and
`CubeEnv.reset` resamples its default variations from an RNG the gym seed does not control (phase 0,
G1d), so the cube's placement — and therefore the mover pixel count — changes between rows: `n`
swings from 584 to 6881. The two `restore` rows, which are the same configuration measured twice,
come back **identical** (interior 0.3463, mover 0.3829, n=1225), so the probe is reproducible for a
fixed configuration; it is the re-reset between configurations that moves the mover mask. The
interior and static columns are stable because the floor does not move. Treat the mover column as
indicative and the interior/static/composited columns as measured.

**Per the coordinator's bound, this was the one further attribution pass. Parity is not chased
beyond it.**
