# D4RL maze2d / PointMaze

Verdict: **go.** Four numbers a frame — the ball's `(x, y)` and the goal's `(x, y)` — against a
150,528-byte 224×224 frame, **9,408×**. Measured against `gymnasium_robotics` 1.4.2's
`PointMaze_*-v3`: **99.49–99.77 % of all pixels exact, mean |diff| 0.013–0.023/255, interior
agreement 1.000000** on all four scenes, at **48,648 frames/s (batch 4,096) against MuJoCo's own
2,293 f/s — 21×**.

D4RL's own `maze2d-*-v1` is not installable here (`d4rl` needs `mujoco_py`, which needs the MuJoCo 2.1
binaries; `import d4rl` → `ModuleNotFoundError`), so the maintained port is the reference. It is the
same environment and the same picture.

Shipped in `stable_envs/envs/maze2d.py` (four registered scenes). Probes:
`scratch/maze2d/{probe,probe2,probe3,proto,parity,order,xproc}.py`.

## The four gates

| gate | test | result |
|---|---|---|
| **G1 purity** | same `(qpos, goal)` from a different history; `qvel` × 100; a fresh env instance | **0 pixels** on all three, on all four scenes |
| **G2 compression** | render slice vs frame | **16 B vs 150,528 B = 1/9,408** |
| **G3 shared decoder** | two resets, two instances: `(nq, ngeom, nmesh, nsite)` | **identical**, and `nmesh = 0` in all four models — the whole scene is a plane, 19–63 boxes and 2 spheres |
| **G4 economics** | GPU regeneration vs the official renderer and vs decode | **48,648 f/s** vs 2,293 f/s official (**21×**) and ~6,762 f/s/core of 224² JPEG decode — **≈ 7 cores replaced** |

## Why this is not just OGBench's pointmaze again

Same body, three differences, and each one cost something:

1. **The goal is drawn.** OGBench renders no goal marker, so its state is `qpos` alone; here a
   translucent red sphere moves per episode and is therefore state.
2. **The camera is oblique** (elevation −45, not straight down), so the frame contains the model's
   `builtin="gradient"` skybox with `mark="random"` speckles and a 20×20 checkerboard receding to the
   horizon. That is exactly the scene content that makes DM Control a NO-GO for a hand-written 2-D
   path — and it costs nothing here, because it is static and the plate is baked through MuJoCo.
3. **The reference is `gymnasium`'s viewer, not `mujoco.Renderer`.** That turned out to be where the
   two real defects live.

## The transparency law, measured

The obvious model — source-over, `out = a·C + (1−a)·bg` with `a = 0.7` — is **wrong by a mean of
18.7/255 and exact on 0 % of target pixels**. Sweeping the site's alpha and fitting shows why:

| model | max residual over an alpha sweep |
|---|---|
| linear in `a` | **31.47/255** |
| quadratic in `a` | **0.84/255** |

The pixel is quadratic because MuJoCo blends a transparent geom over the background **twice** with the
same front-face colour, so

```
out = (2a − a²)·C_front + (1 − a)²·background
```

Checked on individual pixels: front `(240,0,0)` over background `(49,73,98)` at `a = 0.7` predicts
`(222.8, 6.6, 8.8)` and the reference draws `(222, 7, 9)`. The two weights sum to 1, which is what
makes the renderer two ordinary opaque casts plus a blend that is the identity wherever the target is
absent.

## Two defects in the reference

Neither is in this repo's code; both are invisible until you have a second implementation to compare
against, and both would silently corrupt anyone's stored PointMaze dataset.

**1. The free camera is a function of construction history, not of the state.**
`gymnasium.envs.mujoco.mujoco_rendering.BaseRender.__init__` sets

```python
for i in range(3):
    self.cam.lookat[i] = np.median(self.data.geom_xpos[:, i])
```

and the viewer is built lazily on the **first render call**, from whatever `data` holds at that
instant. `geom_xpos` includes `particle_geom`, which moves with the ball. Measured: two freshly built
`PointMaze_Medium-v3` instances driven to the same state disagree on **37.9 % of pixels, max 212/255**
— their `lookat` came out `(−0.25, 0.25, 0.2)` and `(0.25, 0.25, 0.2)`. Moving one of 40 geoms shifts
a median only when it crosses it, which is why UMaze, Open and Large agree at 0/255 and Medium does
not: **the defect is invisible on three quarters of the family.** Both the plate bake and the parity
reference now pin the viewer's construction at `qpos = 0`.

**2. The viewer never re-binds its GL context.** `make_context_current()` is called once, in
`__init__`; `render()` goes straight to `mjr_render` against whatever EGL context is current. MuJoCo's
own `mujoco.Renderer` — which `ogbench` uses — binds on every call, so a process holding both hands
the gymnasium viewer someone else's context. Symptom: the four PointMaze envs **pass** the
cross-process reproducibility gate on their own and **fail** it as soon as the run also contains the
OGBench mazes, because the pytest process and the render-worker process bind contexts in a different
order. One `make_context_current()` per bake fixes it.

A third, smaller one: **editing `model.tex_rgb` after a GL context exists does nothing.** The context
uploads texture data when it is built and never re-reads it, so the `decolor_floor` ablation was inert
and `check_variations_do_something` said so. Clearing the floor geom's `matid` instead is read by
`mjv_updateScene` every frame and takes effect immediately.

## Where the residual lives

| | share of frame | exact | within 1/255 | mean |
|---|---|---|---|---|
| static (baked plate) | 99.4–99.7 % | **99.997 %** | — | 0.0003–0.0007/255 |
| the two spheres | 0.25–0.57 % | 8.7–11.6 % | 42.7–46.7 % | 3.8–5.0/255 |

Every disagreement is inside or adjacent to a 12-pixel sphere, which is the same residual `ogbench.md`
records for its agent and has the same two causes: MuJoCo's 4× multisampling at a silhouette, and an
analytic shader standing in for the fixed-function pipeline. A grid search over
(ambient, diffuse, specular, exponent) around the model's declared headlight moved within-1 agreement
from 0.410 to 0.468 and whole-frame exactness from 0.995326 to 0.995455 — less than the multisampling
costs — so the **declared** constants are used rather than fitted ones.

`parity_interior_min` stays at **1.0**: the interior mask keeps only pixels whose 3×3 neighbourhood in
the reference is one flat colour, and a shaded 12-pixel sphere has none.

## Performance, and what it cost to get

| form | f/s at B = 4,096 |
|---|---|
| two untiled casts, whole-frame float64 blend | 3,548 |
| two tiled casts, whole-frame float64 blend | 7,985 |
| two tiled casts, masked blend, **batch-wide** occlusion fallback | 7,985 |
| two tiled casts, masked blend, **per-row** occlusion fallback | **48,648** |

The last row is the whole story: with 4,096 independent `(ball, goal)` pairs, roughly 0.5 % of rows
have the ball in front of the target — and taking the *whole batch* down the untiled path for them
costs a factor of six. Both the blend and the fallback are per-row now.
