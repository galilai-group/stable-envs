# Adding an environment

This is the page that matters. Everything else in the repo exists so that this procedure is short and so
that finishing it produces evidence rather than a claim.

PushT is the worked example throughout, because it is the env the contract was derived from and because
every one of its constants was *measured* against the reference rather than read off the source. The full
implementation is [`stable_envs/envs/pusht.py`](api/envs.md); this page is the reasoning behind it.

## Step 0: decide whether the env belongs here

One question: **is the frame a pure function of a compact state vector?**

Not "can I approximate it" — can you write down a fixed-width vector of numbers such that the picture is
determined by them and nothing else? If the answer needs an emulator's internal RAM, a 24 GB asset
library, or a mesh that changes shape per episode, the answer is no, and *saying so is a result*. The
[RoboCasa note](research/robocasa.md) is a worked no-go: the frames there are exactly reproducible from a
state, but the state is a 385 KB MJCF plus 24 GB of assets and changes shape every episode, which is not
the claim this repo makes.

There is a test for the yes case that does not involve writing any of our code, and it is worth running
first — see [state sufficiency](testing.md#state-sufficiency). If two instances of the *official* env
driven to the same state render different pixels, no GPU renderer can fix that, and you have learned it
in an afternoon rather than after a port.

## Step 1: declare the contract

Drop a module into `stable_envs/envs/` — that is the whole registration step. The package auto-discovers
its modules, so there is no shared import list to edit and no second place to forget. A module that fails
to import (a missing optional dependency) loses exactly that env, and `snv.import_errors()`
says why; the conformance suite prints it, so "my env vanished" is answerable in one line instead of
looking like a registration bug.

```python
from stable_envs.env import Env
from stable_envs.registry import register

@register
class MyEnv(Env):
    name = "my-env"                 # dataset name, hyphenated-lowercase
    state_dim = 5
    state_names = ("agent_x", "agent_y", "block_x", "block_y", "block_angle")
    default_size = 96
    reference_id = "gym_pusht/PushT-v0"
```

Construct with `num_envs=`. `snv.MyEnv(num_envs=8, device="cpu")` after registration.
The public base class is `Env`.

`state_names` is not decoration. The conformance suite asserts `len(state_names) == state_dim` with the
reasoning that a state vector whose fields are unnamed cannot be checked or reused by anyone, and the
dead-dimension test prints these names when a dimension turns out not to reach the renderer.

`register` refuses duplicate **class names** rather than silently overwriting, and the class name is the
lookup key — there is no parallel alias table to keep in sync. That is also why the block above says
`MyEnv` and not `PushT`: pasting the shipped env's own declaration a second time raises
`ValueError: env name collision`, which is the intended behaviour and not a bug to work around.

Two more class attributes exist for the case where exactness is genuinely unreachable:

```python
parity_interior_min = 1.0    # required fraction of interior pixels that must be bit-identical
parity_note = ""             # why it is below 1.0, and what would close it. Required if it is.
```

The default is 1.0 and that is the expectation. Relaxing it without a `parity_note` is a test failure,
because a tolerance with no stated mechanism is indistinguishable from a bug being tolerated. The
motivating case is MuJoCo's Gouraud shading of tessellated spheres, which no per-pixel analytic shader
reproduces exactly — a declared, explained limit visible in the report, rather than a shared threshold
loosened for everyone.

### If the renderer provably cannot be the official one

`parity_interior_min` is for an env whose renderer is *ours* and nearly exact. It is **not** the tool for
an env whose only fast renderer is a different rasterizer that will never agree pixel for pixel. That
env declares the [visual tier](concepts/fidelity-tiers.md) instead:

```python
fidelity_tier = "visual"
parity_interior_min = None          # required: the parity gate does not apply to this env
visual_fidelity = {...}             # measured mean_abs / within8_frac / ssim, plus the thresholds
visual_note = "..."                 # the mechanism, and what the tier does not license
```

`register()` enforces every part of that, including the words `VISUAL TIER` in the class docstring, so a
half-declared tier is an import error rather than a review comment. Read the tier page before reaching
for it: it makes a weaker claim, it is asserted by a different test, and it disqualifies the env from the
use the strong claim licenses — training on regenerated frames and evaluating on official pixels.

## Step 2: find the reference's real constants

This is where the time goes, and it is the step that cannot be skipped or guessed. PushT's docstring
records five constants that all read plausibly wrong from the source:

| what | the plausible reading | what is actually true | how it was found |
|---|---|---|---|
| y axis | `512 - y`, per the long flip docstring in the draw override | no flip at all: `positive_y_is_up` is False, so `to_pygame` is `(int(x), int(y))` | the plausible version puts the agent at row 348 where the reference draws it at 162 |
| block transform | `pos + R(θ) @ (v - cog)`, since the T sets `cog = (0, 45)` | `pos + R(θ) @ v_local` | checked against `pymunk.Body.local_to_world` over 6 random poses: 5.7e-14 for the right one, 44.6 px for the tempting one |
| block fill colour | `(142, 163, 183)` from truncating `light_color`'s 1.2× scale | `(143, 163, 184)` — `SpaceDebugColor.as_int()` **rounds** | off-by-one on every interior pixel of the block |
| wall stroke | half-width 2.0, from `draw_fat_segment(radius=2)` | half-width 2.5 — a width-4 line *plus*, because `r > 2`, an orthogonal quad extended by `radius` and round caps | pixel count: 2.0 gives 8,012 wall pixels against the reference's 10,009; 2.5 matches |
| draw order | either | the block paints **over** the agent | `pymunk.debug_draw` walks `space.shapes`, so the order is an undocumented dict-iteration detail; recovered by rendering both orders against the reference |

The draw-order entry carries its own lesson. The first search ran over random rollout poses, where the
agent and the block almost never overlap, and came back a meaningless near-tie (0.9127 against 0.9110).
The answer only appeared once poses with actual overlap were added. **A parity search over states where
the thing you are resolving does not occur will confidently return noise.**

## Step 3: write `make_variation_space`

Every knob that changes the picture without changing the state.

```python
def make_variation_space(self) -> Dict:
    return Dict({
        "agent": Dict({
            "color": RGBBox(init_value=AGENT_C),
            "inner_color": RGBBox(init_value=AGENT_IN),
            "scale": Box(low=0.25, high=3.0, shape=(1,), init_value=[1.0]),
        }),
        ...
        "rendering": Dict({"render_goal": Discrete(2, init_value=1)}),
    })
```

Two rules:

**Defaults must be exactly the reference appearance.** `scale = 1.0` is a multiplier on the reference
geometry, not an absolute size, so "matches the official renderer" is an identity a test can assert
rather than a number nobody re-checks.

**Do not declare a knob you have not wired into `scene()`.** `check_variations_do_something` moves every
declared leaf to the far end of its range and fails if the frame does not change. A knob in the space but
not in the scene advertises support that does not exist, and this is the check that stops it shipping.

## Step 4: write `scene()` — the load-bearing method

```python
def scene(self, states, variation=None, style=None) -> Scene2D:
    s = torch.as_tensor(states, device=self.device, dtype=torch.float32).reshape(-1, self.state_dim)
    B = s.shape[0]
    sty = self.resolve_style(style, variation, batch=B)     # every appearance knob, per environment
    sb = SceneBuilder(B, device=self.device, vertices=4)    # ... , style=sty) — see 4. below

    # draw order is the reference's: goal, then walls, then agent, then block on top
    gverts = self._t_verts(gpos, gang, gscale)
    for q in range(2):
        sb.add_poly(gverts[:, q], gcol, enabled=show_goal)
    for a, b in WALLS:
        sb.add_segment(a, b, WALL_HALF_W, wcol, enabled=show_walls)
    sb.mark_static()                       # goal + walls depend on variation only, so bake them

    sb.add_disc(s[:, 0:2], AGENT_RADIUS * ascale, agent_color,
                inner_radius=AGENT_INNER_RADIUS * ascale, inner_color=agent_inner)
    bverts = self._t_verts(s[:, 2:4], s[:, 4], bscale)
    for q in range(2):
        sb.add_poly(bverts[:, q], bfill, outline_color=bline, outline_w=OUTLINE_HALF_W * bscale)

    return sb.build(background=bg, world=(0.0, 0.0, SCREEN, SCREEN), flip_y=False)
```

Five things to get right:

1. **Add primitives in the reference's draw order**, back to front. Index order is the z-buffer.
2. **Keep the env's own units.** PushT hands the builder a 512-unit board and declares
   `world=(0, 0, 512, 512)`. Nothing converts to pixels; the rasterizer does that once, correctly, for
   everybody.
3. **Call `mark_static()` after the state-independent primitives.** Six of PushT's nine primitives are
   static, and baking them is worth
   [1.2–2.3× end to end](concepts/scene-ir.md#baking-the-static-prefix) once the batch is large enough
   to earn it, and measurably nothing below that, because the rasterizer will not attempt it below a
   measured floor. You do not have to prove the prefix is batch-uniform either: the rasterizer measures
   it and quietly draws it per frame when it is not, so a prefix you were wrong about costs speed and
   never pixels.
4. **Consider passing the resolved style to the builder** (`SceneBuilder(B, ..., style=sty)`). It is
   optional, the two envs that ship a prefix today do not do it, and it is worth 0.21–0.29 ms per render
   whenever a caller randomises the appearance per environment: the builder can then rule the bake out
   from `StyleTensor.shared` for free instead of the rasterizer measuring it every call.
5. **Use `enabled=` for optional primitives, never a shorter array.** Shape must not depend on state.

The reason `scene()` is where the claim lives: an env that can produce a `Scene2D` from a state has
*proved* the frame is a function of the state, because there is nowhere else for information to come
from. No env instance, no simulator handle, no hidden buffer is in scope.

### When `scene()` is the wrong tool

It is optional. An env whose frame is not drawn from vector primitives overrides `render` directly and
leaves `scene` unimplemented; the conformance suite skips the checks that need a `Scene2D` and applies
all the rest.

The case that motivated this is the tile family. A MiniGrid frame is a grid of tiles, and upstream
already caches each tile image under a key that is a function of the cell contents alone — so the
appearance set is finite and can be precomputed once into an atlas. Rendering is then a gather with no
rasterization at all, exact by construction rather than by matching a rasterizer pixel for pixel. Pushing
that through a polygon IR would re-derive images the source already caches, and would trade an exact
answer for an approximate one. See [`stable_envs.render.tilemap`](api/tilemap.md).

The test to apply: if the reference draws your env from shapes, emit a `Scene2D`. If it composes cached
images, gather them.

## Step 5: write `sample_states`

```python
def sample_states(self, batch, rng=None):
    """gym_pusht's reset distribution: agent U[50,450]^2, block U[100,400]^2, angle U[-pi,pi]."""
    rng = rng or self.rng
    return np.stack([...], axis=1).astype(np.float32)
```

This must be the env's *own* reset distribution, not a convenient box. Every conformance number is
measured on these states, and a parity test on states the env never reaches proves very little — that is
the same failure as the draw-order search above, generalised.

## Step 6: bridge to the official renderer

Three static methods, and they are what turn "looks right" into a number:

```python
@staticmethod
def reference_env(size=96): ...          # construct the official env
@staticmethod
def reference_set_state(u, st): ...      # place it at a state
@staticmethod
def reference_render(env, states): ...   # official frames for those states
```

`reference_set_state` is where PushT's two hardest bugs live, and both affect anyone restoring PushT from
a stored state, not just this repo:

**Set the angle first, then the position.** Chipmunk stores the body transform about its centre of
gravity, and the T's is `(0, 45)`. Setting `position` and then `angle` rotates the body about a point
that is not where you just placed it, so the position you read back is not the one you wrote: target
`(256, 256)` comes back as `(284.99, 266.58)` after a previous angle of 0.0, `(241.63, 233.62)` after 1.3,
and `(325.91, 202.86)` after −2.0. Up to ~70 px of error, *depending on the block's previous angle*, so
the same requested state lands somewhere different depending on history. Reversing the order round-trips
to 2.8e-14 regardless. `gym_pusht._set_state` uses the broken order deliberately — its own comment says
the angle "should theoretically be set first" but does the opposite "for compatibility with legacy data".

**Reindex both bodies.** Chipmunk caches each shape's world transform and refreshes it on a space step or
an explicit `space.reindex_shapes_for_body`. A manually moved body keeps *drawing* at its old pose while
`body.position` already reports the new one. Reindexing only the block left the agent painted at a fixed
`(72.4, 56.5)` for every state — the reference frames silently stopped being a function of the agent's
coordinates, which presents as a bug in *our* renderer.

**Assert the round-trip inside `reference_render`.** If the reference env does not actually end up at the
requested state, every parity number computed from it compares two different scenes:

```python
got = np.array([*u.agent.position, *u.block.position, u.block.angle], np.float64)
if np.abs(got - np.asarray(st, np.float64)).max() > 1e-6:
    raise AssertionError("reference env did not reach the requested state ...")
```

Without that assertion the setter-order bug produces a plausible-looking parity number instead of a
failure, and it looks like a rendering discrepancy.

## Step 7: run the suite and read the numbers

Nothing needs editing — the suite is parameterised over the registry, so the new env is covered the
moment it is registered (`snv.list_envs()` / `snv.spec("MyEnv")`):

```bash
sbatch scripts/run_gpu_tests.sbatch     # or, on a GPU node:
pytest tests/ -v -rs
```

The bar is [interior pixels 100% exact](testing.md). If you are below it, the diff is a real
geometry or colour error and the [checklist of what each check rules out](testing.md) is the place to
start. Do not relax the threshold; that is the one number that makes the repo's claim worth anything.

The examples pick the env up automatically too:

```bash
python examples/bench_gpu_vs_cpu.py --out examples/results   # GPU vs the official CPU renderer
python examples/gallery.py --out docs/assets/media           # grid, parity panel, side-by-side video
```

Envs without a `reference_env` still get GPU throughput and a grid; the comparison artifacts are skipped
rather than faked.

## Checklist

- [ ] the frame is a function of a fixed-width state, and the official env agrees (state sufficiency)
- [ ] the module is in `stable_envs/envs/` and the class carries `@register`
- [ ] `state_dim`, `state_names`, `default_size`, `reference_id` declared
- [ ] every reference constant *measured*, with the measurement recorded in the docstring
- [ ] `make_variation_space` defaults reproduce the reference exactly
- [ ] `scene()` in reference draw order, env-native units, `mark_static()` placed correctly — or
      `render()` overridden, if the reference composes cached images rather than drawing shapes
- [ ] `sample_states` is the env's own reset distribution
- [ ] `reference_env` / `reference_render`, with the state round-trip asserted
- [ ] `pytest tests/` green on a GPU node, interior parity 1.0 — or `parity_interior_min` relaxed *with*
      a `parity_note` naming the mechanism, or the [visual tier](concepts/fidelity-tiers.md) declared in
      full if the renderer provably cannot be the official one
