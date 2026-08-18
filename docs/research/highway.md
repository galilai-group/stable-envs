# highway-env

Verdict: **go.** A highway-env frame is a pure function of four numbers per vehicle —
`x`, `y`, `heading`, `crashed` — because the camera is locked to the ego, so the ego's own
coordinates fix the window and the lane markings follow from the static road network plus
that window. **All six target ids are now registered** — `Highway`, `Merge`, `Roundabout`,
`Intersection`, `Racetrack` and `Parking` — and all six reach **100% interior-pixel parity**
with the official pygame renderer. The last two needed three things the first four did not:
four tyre sprites per ego, which is the pivot rule applied twice; the ego's steering angle in
the state; and a per-env answer to what colour a vehicle is, which is not one answer for the
family (section 2a).

Package: `highway-env` 1.12.1, `pygame-ce` 2.5.7.
Module: `stable_envs/envs/highway.py`. Probes: `scratch/highway/`.
`scratch/highway/calib_all.py` is the parity table for all six plus the two calibration
questions the new pair raises; `scratch/highway/diag.py <Env>` classifies every differing
pixel by the colour the reference has there, which names the primitive rather than the file.

Reproduce, on a GPU node: `sbatch scratch/highway/run_style.sbatch` runs the parity
measurement for the first four envs, the conformance suite and the throughput table;
`run_all.sbatch` additionally sweeps the primitive-count ceiling described in section 4.
`scratch/highway/calib.py <Env> <n>` is the parity number on its own, and it prints where
the disagreeing pixels are and which reference colour they should have had, so a miss is
attributable to a primitive rather than to "rendering".

## 1. Results

Parity is measured over 64 states drawn from each env's own spawn distribution, against the
official renderer driven to the same state; throughput is one H200 at `ss=1`, batch 16384.
The GPU column was measured twice, on different nodes, and the two runs agree to within
0.4% (21,342/21,300, 27,618/27,537, 1,716/1,717, 2,473/2,475 frames/s). The
official-renderer column is single-core pygame on a shared node and is **not** reproducible
that way: the same measurement gave Merge 422 f/s on an idle node and 3 f/s while three
other jobs were resident. Treat it as an order of magnitude, not a benchmark.

| env | state | primitives | interior exact | all-pixel exact | mean abs | GPU frames/s | official |
|---|---|---|---|---|---|---|---|
| Highway (600x150) | 48 floats | 134 | **100%** | 99.933% | 0.076/255 | **21,342** | 335 f/s |
| Merge (600x150) | 20 floats | 130 | **100%** | 99.993% | 0.009/255 | **27,618** | 422 f/s |
| Roundabout (600x600) | 20 floats | 604 | **100%** | 99.949% | 0.078/255 | **1,716** | 90 f/s |
| Intersection (600x600) | 40 floats | 372 | **100%** | 99.990% | 0.011/255 | **2,473** | 117 f/s |
| Racetrack (600x600) | 9 floats | 356 | **100%** | 99.980% | 0.029/255 | **2,585** | 87 f/s |
| Parking (600x300) | 7 floats | 242 | **100%** | 99.997% | 0.002/255 | **7,147** | 126 f/s |

A 600x150x3 frame is 270,000 bytes against 48 floats, a **1,400x** shrink; the roundabout's
600x600 frame is 1,080,000 bytes against 20 floats, **13,500x**; and parking-v0's is
540,000 bytes against **seven**, a **19,300x** shrink — the largest in the family, because
its whole cast is one vehicle, one goal and four walls whose poses are constants.

The parity columns for the first four are re-measured over 32 states and the
Intersection row is **better than it was** (99.960% -> 99.990%): the float32 rotation band in
section 2a was found while porting parking-v0 and it was costing intersection-v0 too, since
its ego approaches at exactly `+-pi/2`. The Racetrack and Parking GPU numbers are at batch
4,096 rather than 16,384.

Everything that is not exact is a boundary pixel, and the causes are known and
quantified in section 3. With every heading set to zero (the common case on a straight
highway, since the reference snaps any heading under 2 degrees to 0) the agreement is
**100% of all pixels**, not just interior ones.

## 2. Six things the source gets you wrong

Each of these cost real debugging time and none is visible by eye.

**`SDL_VIDEODRIVER=dummy` makes highway-env render nothing.** `EnvViewer.__init__` sets
`self.enabled = False` when it sees that driver, and `display()` returns immediately, so
`render()` hands back a surface nothing was ever drawn on. Measured: 1 unique colour and
mean 0.0 under `dummy`, against 7 unique colours and mean 103.8 under `offscreen`. This is
the standard headless recipe, and it silently turns any parity measurement into a
comparison against a black image. `reference_env` forces `offscreen` while the viewer is
constructed and then asserts `viewer.enabled`.

**The vehicle is not drawn as a rotated rectangle.** `VehicleGraphics.display` rasterises
axis-aligned rectangles into a 38x38 per-pixel-alpha sprite, calls
`pygame.transform.rotate` — nearest neighbour, fixed point — and blits the result at a
float origin that `Rect` truncates to a whole pixel. It is nevertheless *exactly* a
polygon: the rotation reads `src[floor(A·d + t)]`, and `floor(u) == n` is a half-open box
test in the rotated frame, so the pixels showing one sprite rectangle form a rotated
rectangle. Confirmed by searching every offset that reproduces the reference on 97
headings: an analytic rectangle matches **100% of pixels** at the right pivot.

**The pivot is not the vehicle's screen position.** Measured over 181 headings and sprite
sizes S in {11, 14, 21, 38}, with `Wr = int(S*(|cos a| + |sin a|))` the rotated surface
width:

| case | offset |
|---|---|
| angle a multiple of 90 degrees | `(-0.5*(S mod 2), -0.5*(S mod 2))` |
| `Wr` odd | `(-0.5, -0.5)` |
| `Wr` even | `(-1.0, -0.5)` |

The parity that matters is `Wr`'s, not `S`'s and not `Wr - S`'s: S=11 with Wr=13 gives
-0.5 while S=38 with Wr=46 gives -1.0, and both have an even `Wr - S`. Getting this wrong
puts every car up to a pixel off. The x and y rules differ because `blit_rotate` builds
the two coordinates from `min_box` and `max_box`, which are not symmetric.

## 2a. Four more, found while porting racetrack-v0 and parking-v0

**"A multiple of 90 degrees" is not an exact test, and taking it as one costs a pixel.**
`pygame.transform.rotate` takes its axis-aligned fast path on a *band* around each quarter
turn, and the band is exactly half a **float32** ulp wide. Bisected on 490x490, 294x294 and
38x38 sprites, a rotation stops being bit-identical to the quarter turn at:

| anchor | 90 | 180 | 270 |
|---|---|---|---|
| threshold (degrees) | 3.815e-06 | 7.629e-06 | 1.526e-05 |
| i.e. | 2⁻¹⁸ | 2⁻¹⁷ | 2⁻¹⁶ |
| half an ulp of | 90 | 180 | 270 | 

So the angle is rounded to float32 and *then* tested. A heading of `pi/2` that has passed
through a float32 state comes back as 90.0000025 degrees — inside the band — and an
exact-equality test sends it down the `Wr` branch instead. Worth 23 px a frame on
parking-v0's 14x14 goal landmark alone, and it was also costing intersection-v0, whose ego
approaches at exactly `+-pi/2`: fixing it moved that env from 99.960% to 99.990% all-pixel.
The mirror image of the same trap: a class constant of `np.pi/2` narrowed to float32 *by our
own code* misses the band the reference hits. parking-v0's four walls are built at exactly
`np.pi/2`, and narrowing them was worth **1,140 px a frame**, all of it on their 1 px
outline. `scratch/highway/probe_rot_tolerance.py`.

**The ego is not green, and what colour it is is a property of the env rather than of the
family.** `VehicleGraphics.get_color` reads `vehicle.color` if it is set, and only then falls
through `crashed` -> `LinearVehicle` -> `IDMVehicle` -> `MDPVehicle` -> `DEFAULT_COLOR`. So:

* racetrack-v0's ego is a plain `Vehicle` with no pinned colour, which falls all the way to
  `DEFAULT_COLOR` — **yellow**, the colour an obstacle is elsewhere in the family. Assuming
  the family's green cost **1,717 interior pixels a frame** before the diff named it;
* parking-v0 pins `vehicle.color = EGO_COLOR`, so its ego is green — **and its `crashed` flag
  can never change a pixel**, because the pinned colour is read first. `crashed` is therefore
  not a state dimension there and `crashed.color` is not a knob; carrying either would be a
  dead dimension and an inert knob. `reference_set_state` asserts the colour is still pinned,
  so a package that stopped pinning it fails loudly rather than drawing a green car where the
  reference draws a red one.

**Road objects are not culled; vehicles are.** `VehicleGraphics.display` returns early on
`surface.is_visible(position)` and `RoadObjectGraphics.display` has no such test. Measured on
parking-v0 — whose walls are 70 m and 42 m long, i.e. 490 px and 294 px — culling them the way
vehicles are culled costs 7 px a frame. Small, and the reason is worth recording: a wall
leaves the frame along its **7 px thickness** and the visibility margin is 50 px, so it is
already invisible by the time its centre is culled. The rule is followed because it is the
reference's, not because the difference is large.

**The transcribed road is checked against the package, not trusted.** racetrack-v0 has 18
lanes with radii and phases written in degrees and parking-v0 has 28 generated from a loop;
a transcription that size has a defect rate, and a wrong radius is not a crash but a lane
marking a few pixels off, which looks exactly like a rasterizer bug. `_check_lanes` re-derives
every lane's geometry from the official `road.network` every time a reference env is built —
and caught a `deg(165)` that should have been `deg(170)` on the first run. It compares as a
multiset rather than in order, because `RoadNetwork.graph` groups every `a->b` lane before
every `b->c` one while `MergeEnv._make_road` writes them section by section, and every marking
is drawn in one colour so the sequence cannot matter.

**The ego's screen x is 179, not 180.** `move_display_window_to` sets
`origin = ego - centering*size/scaling`, and `pos2pix` then computes
`int((ego - origin) * scaling)`, reconstructing `0.3*600` through two subtractions. In
float64 that lands at 179.99999999999997 and truncates to 179. The camera quantisation is
therefore done in float64 here; float32 rounds the other way and shifts the entire frame
by a pixel.

**The black outline is drawn after the headlights.** A headlight that reaches the body's
front edge is painted over by the 1 px outline. Expressing this as a filled black
rectangle *under* an inset body only reproduces it if the headlight rectangles are first
clipped to the inset body; without the clip the headlights bleed 6 px over the outline per
vehicle.

And two found while porting rather than while calibrating:

**A capsule is not a Bresenham line.** `pygame.draw.line` at width 1 keeps one pixel per
column of the major axis, i.e. it thresholds the distance along the *minor* axis at 0.5,
while a capsule thresholds the *perpendicular* distance. Scaling the capsule half-width by
`|major| / length` converts one into the other. Over 400 random segments: **12.48%** of lit
pixels wrong with a flat 0.5 half-width, **1.79%** with the scaled one, and 283 of 400 lines
pixel-exact instead of 106. highway-v0's markings are horizontal, where the factor is 1,
so the error only appeared once curved roads were added.

**An even-width line is not centred on its endpoints.** intersection-v0 runs at
`scaling = 5.5*1.3`, so `pix(STRIPE_WIDTH) = 2`. Every symmetric candidate — capsule,
scaled capsule, square-capped quad — came out ~50% wrong, which is what an unshifted band
looks like when the truth is shifted by a whole pixel: two pixels cannot straddle a pixel
centre symmetrically. Read off directly, a width-2 horizontal line at y=30 lights rows
`[30, 31]`, width 3 lights `[29, 30, 31]`, width 4 lights `[29..32]`, and reversing the
endpoints moves none of them, so the shift is absolute rather than direction-dependent.
The rule is therefore: threshold the minor-axis distance at `w/2` about a centre offset by
`+0.5` when `w` is even. Measured: **0.25%** of lit pixels wrong on axis-aligned segments
and 1.35% on arbitrary ones, against ~50% for anything symmetric.

## 3. What is still not exact, and why

**Rotated vehicles, ~0.03% of pixels.** pygame's rotation is fixed point:
`isin = (int)(sin*65536)`, and destination pixels are chosen by an integer shift. At
boundary ties that lands on the other side of the edge from an exact analytic test. This
is not a calibration error: searching every sub-pixel offset on a 0.04 px grid, 133 of 145
headings are already pixel-exact under the rule above, and for the remaining 12 the best
achievable by *any* translation removes only 12 of 22 wrong pixels. Setting every heading
to zero gives 100% of all pixels exact, which localises the residual to the rotation.

**Curved lane markings, ~0.05% of pixels in the roundabout and the intersection.** The
reference draws arcs as chains of short `pygame.draw.line` calls at arbitrary angles, and
the rules above still miss ~1.8% of lit pixels at width 1 and ~1.4% at width 2. The
markings are 1-2 px wide, so none of them is an interior pixel and interior parity is
unaffected.

**One bug found this way.** Before the fix, the roundabout was missing **66% of its white
pixels**. A circular lane's `local_coordinates` projects the camera onto the whole circle,
so for a 14.66 m arc the projection routinely lands tens of metres outside `[0, length]`;
a stripe window centred on the camera then clipped to a degenerate segment and the arc
vanished. The window is now clamped to cover the lane whenever the lane is short enough to
fit, and only follows the camera when it is not.

## 4. The primitive-count wall (removed — kept as history)

This section describes a wall that no longer exists. It is kept because it is the kind of
constraint that comes back silently the moment anyone reintroduces loop unrolling, and
because nothing else in the repo records what it cost.

`_scene_kernel` used to walk the primitive list with `tl.static_range`, which makes the
primitive count a compile-time constant and unrolls the kernel. Measured on an H200 with a
synthetic scene of N quads at 600x150:

| N | 64 | 128 | 200 | 256 | 300 | 320 | 340 | 350 |
|---|---|---|---|---|---|---|---|---|
| compile + first call | 7.7s | 13.6s | 21.0s | 27.4s | 31.3s | 33.1s | 35.6s | **ptxas aborts (SIGABRT)** |

roundabout-v0 needs 604 primitives and intersection-v0 needs 372, so both were
unrenderable, and roundabout aborted the *process* rather than raising. The abort was not
state-dependent and was not an unchecked index: the same fixed, in-distribution state
rendered fine on the torch backend and aborted on the fused one; an all-zero state aborted
identically; and a synthetic scene of 350 identical quads containing no env code at all
aborted the same way. It happened on the first Triton render, before any pixel was touched.

`raster2d: runtime primitive loop` replaced `tl.static_range` with a runtime `range()` and
made `N`/`PSTART` runtime arguments. Both envs render and are registered.

**The unroll was not even a win at this scale.** It costs 7.6% on PushT, which has 9
primitives. Measured on the same states and the same hardware before and after, at 134
primitives it was costing a factor of four:

| env | primitives | unrolled | runtime loop |
|---|---|---|---|
| Highway | 134 | 5,347 f/s | **21,342 f/s** |
| Merge | 130 | 7,182 f/s | **27,618 f/s** |

So the ceiling was not the only cost of unrolling — past a few dozen primitives the kernel
stops fitting whatever it needs to fit and the throughput collapses well before ptxas
refuses to compile at all. A scene-complexity knob that silently degrades 4x and then
aborts is worth 7.6% on the smallest env in the repo only if nobody measures the others.

What has *not* changed is that the kernel evaluates every primitive at every pixel: a 3x3 px
headlight still costs what a full-screen polygon costs, which is why the 600x600 envs are an
order of magnitude slower per frame than the 600x150 ones despite the same output bandwidth.
A bounding-box reject would be the next real win.

## 5. The state, and why `crashed` is in it

Per vehicle, in the reference's own draw order (`road.vehicles`):

| field | why |
|---|---|
| `x`, `y` | position; for the ego it also fixes the camera and hence the stripe phase |
| `heading` | the sprite rotation, snapped to 0 below 2 degrees by the reference |
| `crashed` | `VehicleGraphics.get_color` paints a crashed vehicle red |

`crashed` is not decoration. A state without it does not determine the frame in exactly
the situation these environments exist to study, and the conformance suite would not
notice, because a rollout that never crashes never exercises it. `sample_states` therefore
puts a crashed vehicle next to the ego in half of every batch.

`reference_set_state` rebuilds `road.vehicles` outright rather than repositioning whatever
happens to be there. Several of these envs decide their cast at random —
`IntersectionEnv._spawn_vehicle` draws against a probability and `_clear_vehicles` deletes
vehicles that leave — so the number of vehicles after `reset` is not a property of the env
id. The ego object is kept, because it has to stay the class the action type produced or
it renders in the wrong colour and `env.vehicle`, which fixes the camera, stops pointing
at it.

## 6. The tyres, and what is still not ported

**racetrack-v0 and parking-v0 were in this section, and the reason they were is now the
interesting part.** Their ego is a plain `Vehicle`, and `VehicleGraphics` draws tyres only
for `Vehicle`/`BicycleVehicle` — an exact `type()` check, so an `IDMVehicle` never gets them
— as four sprites of `pix(1) x pix(0.3)`, i.e. 5x1 px at scaling 5.5 and 7x2 at scaling 7.
Each is its own `blit_rotate` INSIDE the vehicle surface, at the steering angle, and the
vehicle surface is then rotated by the heading and blitted again. So the pivot rule of
section 2 applies twice:

```
local centre  = tire_position + pivot(S_tire, steering) + R(steering) @ tyre-rect offset
screen centre = vehicle centre + R(heading) @ (local centre - S_vehicle/2)
screen angle  = heading + steering
```

`tire_angles` is `[0, 0, steering, steering]`, so only the FRONT pair turns; the rear pair is
blitted at angle 0, which takes both `pygame.transform.rotate`'s identity path and the
quarter-turn branch of the pivot. The heading is snapped to zero below 2 degrees and the
steering is not — the snap happens once, on `v.heading`, after the tyres are already drawn.

This cannot be exact by construction, and the reason is worth stating rather than hiding: the
outer rotation resamples the vehicle surface with `src[floor(A·d + t)]`, so a rectangle of
vehicle-surface PIXELS maps to a rotated rectangle tested at the *floored* coordinate rather
than at the pixel itself. On a body 27 px long that is invisible; on a tyre `pix(0.3) = 1` px
thick it is up to a pixel of the tyre. Measured, against the reference, with and without them:

| env | with tyres | without |
|---|---|---|
| Racetrack | 70.5 px/frame differ | 85.8 |
| Parking | **6.0** | 46.8 |

So they are worth drawing — on parking-v0 they are seven eighths of what would otherwise be
wrong — and what remains is 6 px a frame. `ego_steering` earns its state dimension by the same
kind of measurement: setting it to zero moves **7.2 px a frame** in the OFFICIAL renderer, up
to 195/255.

**`show_trajectories`** (the ghost history of past poses) and the lidar overlay are still not
ported, and neither is on by default. `ObservationGraphics.display` draws only for
`LidarObservation`, and no registered env here uses one.

intersection-v0 *was* in this section for two reasons and neither survived. The primitive
count (measured at 372, not the ~350 estimated) stopped mattering when the wall went, and
the width-2 marking path turned out to be calibratable — see section 2. It is registered
and exact on the interior. Two details worth keeping: its ego is appended to
`road.vehicles` **last**, so it paints on top of the traffic, which at a junction is the
difference between the ego being visible and being buried; and its left-turn lanes carry
`line_types = [NONE, NONE]`, so they contribute no primitives at all.
