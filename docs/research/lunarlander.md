# LunarLander-v3

Verdict: **go — but on a wider contract than "render the observation".** The classic-control note
excluded this env for a specific reason: `render()` runs `for obj in self.particles: obj.ttl -= 0.15`
and then `self._clean_particles(False)`, so rendering twice gives two different images and the frame is
not a function of anything the env exposes as state. That is an argument for *widening* the state, not
for excluding the env, and the three questions that decide which now have numbers attached.

Package: `gymnasium` 1.2.3 with `Box2D` 2.3.10, `pygame-ce` 2.5.7.
Module: `stable_envs/envs/lunarlander.py`. Probes: `scratch/lunarlander/`.

## 1. The three deciding measurements

**A. Is the mutation inside `render()` deterministic?** Yes. Restore an env to a recorded state and
render it twice: **max absolute pixel difference 0**. Render twice *without* restoring: **63/255**.
The second number is the mutation, and it is exactly why the particles have to be in the state.

**B. Is the particle list bounded?** Yes, in the only regime that produces frames at all. A discrete
LunarLander creates at most one particle per `step`, and only when `render_mode` is set. The discrete
env fixes `m_power = s_power = 1.0`, so every particle starts at `ttl = 1.0` and loses 0.15 per render;
the seventh decrement takes it below zero.

| policy | particles drawn, peak | particles alive if `render()` is never called |
|---|---|---|
| main engine every step | **6** | 118 over 118 steps |
| main and side alternating | **6** | 126 over 126 steps |
| uniform random | **6** | 77 over 102 steps |

The right-hand column is the honest caveat: nothing except `render` prunes the list, so a training loop
that steps without rendering accumulates particles without limit. That regime has no frames in it, so
it has no state-to-frame claim to make; `reference_get_state` raises rather than truncating if a state
ever needs more than six.

**C. Does a fixed-width state determine the frame?** Yes, and this is the measurement that decides the
verdict. A **different env instance**, with a different terrain and twenty steps of a different history
behind it, driven to the same 38 numbers, renders **max absolute pixel difference 0** against the
original. The frame is a pure function of (terrain, lander, legs, particles).

So the contract is "replay the simulator's render state", the same one MiniGrid's top-down classes and
PushT already use. The 8-D observation is a lossy function of it: it carries neither the terrain, nor
the leg joint angles, nor the exhaust.

## 2. Results

| quantity | value |
|---|---|
| state | 11 terrain heights + 3 poses x 3 + 6 particles x 3 = **38** numbers |
| compression | 720,000-byte frame against 38 float32, **4,737x** |
| **interior-pixel parity** | **100.00000%**, 0 differing interior pixels over 32 states |
| all-pixel parity | **99.9691%**, mean 0.0218/255, 74.2 differing px a frame |
| GPU throughput, one H200 | **3,148 frames/s** at B = 256, **11,047 f/s** at B = 1,024 |
| official CPU renderer | **402 frames/s** |

The all-pixel residual is 74 pixels a frame and it decomposes exactly:

| cause | px/frame |
|---|---|
| `pygame.draw.aalines`, the outline this port does not reimplement | 34.5 |
| `pygame.draw.polygon` vs `gfxdraw.filled_polygon` at scanline ends (the terrain edge) | 30.6 |
| the rest, all boundary | ~9 |

Every one of them is within a pixel or two of a shape edge, which is why interior parity is untouched.

## 3. Three rasterizers in one frame, and only one of them is exact

```
pygame.draw.polygon  +  gfxdraw.aapolygon  +  pygame.draw.aalines     lander, legs
pygame.draw.polygon  +  gfxdraw.aapolygon                             terrain, flags
pygame.draw.circle   x2                                               particles
pygame.draw.line(width=1)                                             flagpoles
```

`stable_envs/render/gfxdraw.py` reproduces the SDL_gfx path exactly, because SDL_gfx takes Sint16
vertices and is integer arithmetic all the way down. `pygame.draw` is a different renderer:

**`pygame.draw.polygon` is not `gfxdraw.filled_polygon`.** Measured over 300 random convex polygons
they differ on **3.63% of lit pixels**, and it is not the vertex handling — both truncate, verified by
shifting an axis-aligned quad: 0.4 px and 0.9 px move nothing, 1.0 px moves one column. The
disagreements are all at the ends of scanlines, and the two rules were read off row by row on a
polygon where they disagree without a candidate reproducing both ends. What is used here is the exact
SDL_gfx scanline rule, so the residual is the difference between the two, ~30 px a frame, at the
terrain's own edge. Closing it needs `draw_fillpoly`'s rule recovered, which is open.

**`pygame.draw.aalines` is a sub-pixel path unrelated to gfxdraw's Wu line** and is not reimplemented.
It paints the lander's and legs' outline in `color2 = (77,77,128)`, measured at 324 px per body on a
test quad, of which 67 lie inside a 1-px erosion of the fill and 8 inside a 2-px erosion — so the whole
of it is excluded from the interior mask, which is why it costs nothing there and 34.5 px a frame in
the all-pixel number.

**`pygame.draw.circle` is a 12-pixel stencil here, not a circle.** Every particle has
`radius = 2/SCALE`; `radius * SCALE` is 2.0000001 and truncates to 2, and the centre truncates too
(60.0, 60.4, 60.5 and 60.9 all give the same columns). So the shape is a constant, read off pygame
once. It is not symmetric — it spans `dy` and `dx` from −2 to +1:

```
(-2,-1) (-2,0)
(-1,-2) (-1,-1) (-1,0) (-1,1)
( 0,-2) ( 0,-1) ( 0,0) ( 0,1)
( 1,-1) ( 1,0)
```

## 4. Two things the source gets you wrong

**The landing flags are drawn once per FIXTURE, so the number of live particles changes their colour.**
The `for x in [helipad_x1, helipad_x2]` loop is nested *inside* `for f in obj.fixtures`, so the flags
are painted `len(particles) + 3` times per frame. The opaque parts of a flag are idempotent; its
`gfxdraw.aapolygon` fringe is not, because a Wu blend at alpha `(255*w)>>8` never reaches the source
colour and converges toward it with repetition. Measured on the flag alone, against one repetition:

| repetitions | 2 | 3 | 7 |
|---|---|---|---|
| differing px | 40 | 40 | 40 |
| max difference | 62/255 | 98/255 | 152/255 |

So the draw order is reproduced exactly — particle, flag, flag, particle, flag, flag, …, lander, flag,
flag, leg, flag, flag — and a batch holding different numbers of particles gates the missing ones off
rather than issuing a different number of draw calls.

**The state's `ttl` is the value the particle is DRAWN with, not the value the simulator holds.** This
is what makes the state minimal rather than merely sufficient. `render` decrements every `ttl` and
drops the dead prefix before it draws anything, so the env's list is one particle longer than the
frame's — and always by the same slot, the oldest, because every `ttl` starts at exactly 1.0. Storing
the list verbatim leaves that slot's x and y unable to move any pixel, which the conformance suite
reported as a dead dimension. `reference_get_state` therefore applies the decrement and the prune, and
`reference_set_state` adds the decrement back so the reference's own `render` lands on the stored
values.

## 5. Throughput: this renderer is launch-bound, and it was 30x worse before anyone looked

The first working version rendered at **262 frames/s at B = 256 — slower than the pygame renderer it
replaces.** Profiling by disabling one stage at a time (`scratch/lunarlander/profile.py`) put all of it
in one place:

| stage disabled | frames/s at B=256 | ms/batch |
|---|---|---|
| nothing (as first written) | 262 | 979 |
| the anti-aliased edges | 15,040 | 17 |
| `clip_line` stubbed out | 853 | 300 |
| fills only | 20,680 | 12 |

A Wu edge costs a six-iteration Cohen-Sutherland clip — about 150 tiny kernel launches whatever its
length — so the price of the anti-aliasing pass is the NUMBER of calls, at roughly **8.6 ms a call, at
any batch size**. Three changes, none of which touched a pixel:

1. **the flags' fringe is a constant**, so it is recorded once at construction and replayed as a
   scatter (they are drawn up to ten times a frame): 513 ms of the 979;
2. **every edge of every polygon is issued in one call** instead of one call per edge — the terrain's
   40 edges together rather than four at a time;
3. **each terrain quad's fill pays for its own 61-px band** rather than for the frame, since its left
   and right edges are vertical and at known columns. Without that, 31 polygons each read and rewrote
   the whole `(B, 400, 600, 3)` buffer: 9 GB a batch.

Result: **3,148 f/s at B = 256 and 11,047 f/s at B = 1,024**, with the parity numbers **bit-identical**
before and after — 2,376 differing pixels over 32 states either way.

Change 2 gives up the ordering of writes where two edges land on the same pixel, i.e. at shared
vertices, because a batched scatter reads every destination before any of them is written. That it
costs nothing is measured, not assumed, and it is not luck: `aapolygon` draws every edge in ONE colour
and the fill underneath is that same colour, so the two candidate values at a shared vertex are a blend
of a colour over itself.

## 6. Not ported

**Physics.** Box2D's contact solver is chaotic in exactly the way `core/env.py` describes, and the legs
are two revolute joints with motors and angle limits; a reimplementation would diverge and the number
to quote would be a distribution rather than an equality. States come from the official env.

**`LunarLanderContinuous`.** It differs only in `m_power`/`s_power` being continuous in [0.5, 1.0] and
in being able to fire both engines in one step — which changes the particle bound (up to two per step)
and the `ttl` alphabet. It is the same renderer with a different `N_PARTICLES`, and it is not
registered because that bound has not been measured.

**Resizing.** Every primitive is hardcoded in pixels — a 34 px lander, a 2 px exhaust particle, a 50 px
flagpole — so there is no reference frame at any other resolution and `render(size=)` refuses.

## 7. The dependency is hard, and the module says so at import

`render` is pure torch and imports nothing that can fail, but the env is still unusable without Box2D
and pygame: `sample_states` rolls the official env out, because the terrain, the leg joint angles and
the exhaust have no closed form. An env that renders but cannot say what to render is not an env, so
the dependency is declared as a hard one — the module probes the failing path at import (`reset` and
one `render`, milliseconds) and raises `ImportError`, which `stable_envs/envs/__init__.py` records and
`import_errors()` names. The env is then cleanly **absent** rather than present and failing inside
every fixture. `import Box2D` alone would not do: gymnasium raises `DependencyNotInstalled` from inside
`envs/box2d/lunar_lander.py`, and pygame is a separate requirement that only `render` reaches.
