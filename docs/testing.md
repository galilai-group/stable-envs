# Testing and conformance

The claim "you never need to store frames again" is only worth making if the regenerated frame is the
frame you would have stored. That is an empirical question with a number attached, not a design
intention, and the suite is where the number comes from.

Every check below corresponds to a specific way the claim can be false. The suite is parameterised over
the registry (`snv.list_envs()`), so an env added tomorrow is covered without editing a test.

Markers: `unit`, `integration`, `gpu`, `regression`, `slow`. The extra markers
`reference`, `smoke` and `selftest` stay. Directory markers are applied automatically, so a
file that forgot `pytestmark` is still selected by its folder.

```bash
pytest -m unit                                 # CPU-only family/unit contracts
pytest -m integration                          # multi-component (safety net, SWM interop)
pytest -m "regression and not gpu"             # frozen corpora, digests, support matrix
pytest -m "regression and gpu"                 # GPU regression (needs CUDA)
pytest                                         # full suite
sbatch scripts/run_gpu_tests.sbatch            # full suite, crash-isolated
sbatch scripts/run_smoke_tests.sbatch          # one env per family              ~3.4 min
sbatch scripts/run_smoke_tests.sbatch --fast   # one env per render backend      ~0.9 min
sbatch scripts/run_unit_tests.sbatch           # CPU contracts + CPU regression
```

`tests/unit/` renders nothing: the style batching rule, the dataset indexing and episode
boundaries, the policy and rollout API, the corrector's refusals, the benchmark scripts' guards,
and the repository's own citations, docs nav and launcher agreement. `tests/regression/` is
the frozen regression oracle (corpora, digests, declarations, provenance). Green on
`-m unit` says the contracts hold and says nothing about the frames;
`scripts/run_gpu_tests.sbatch` is still the release gate.

The full suite's wall time tracks the registry, so it is quoted with the size it was measured at rather
than as a property of the suite: **45.5 min for 2,580 passed / 648 skipped / 0 failed over a registry of
64**, 3,228 collected, one segment, 133 EGL contexts created and 0 released, on one H200
(gb-west-a1-gpu-021). `scripts/run_unit_tests.sbatch` is the CPU-only subset of that at 2.9 min.

The full suite and `-m gpu` require a CUDA device by design and skip loudly rather than passing
without one. The product is a GPU renderer; a suite that goes green on CPU-only CI proves nothing
about the frames. `-m unit` is the CPU-only contract layer and does not need CUDA.

## Running it so that it cannot take itself down

The suite has aborted its own interpreter four times in a week while being green, and a suite that can
do that is defective whatever its exit code says: `SIGABRT` out of MuJoCo, EGL or the CUDA driver is not
an exception, so pytest writes no report for the test that died and never runs the ones after it.

The sbatch scripts therefore go through `stable_envs.testing.supervisor` rather than calling pytest
directly. It writes one breadcrumb per test, and when the interpreter dies it names the test that killed
it, records it as **CRASHED**, and resumes the run past it — a crash costs one test instead of the run.
On a clean run it costs one extra process and nothing else. (`pytest --forked` is the obvious
alternative and cannot be used: torch refuses to initialise CUDA in a forked child even when the parent
never touched it, so every GPU test errors. Measured, not assumed; see the module docstring.)

Three more things run alongside it, each answering an incident rather than a hypothesis:

* **A resource guard.** EGL context creations and releases are counted per test. Releasing a context
  corrupts rendering in every other live one *silently*, and it has surfaced three times as pixel
  differences in an unrelated env twenty minutes later. It is now an immediate failure named
  `EGL_CONTEXT_RELEASED` on the test that did it, with the traceback of the release site. Creating
  contexts in proportion to the batch is `EGL_CONTEXTS_SCALE_WITH_BATCH`. A healthy full run creates one
  per plate-baking env and releases **0** — 133 created, 0 released on the run above — and that line is
  printed at the end of every run.
* **A node gate**, before the first test: what does SLURM think of this host, and does the *official*
  renderer return the same pixels twice? One of the four "incidents" was a node the cluster had
  condemned 41 minutes before the job started (`NHC: auto-reboot`), and the suite was correctly
  reporting that the official renderer was broken — but nothing in the output named the host, so it
  cost hours. An unfit node now ends the run with exit code **78**, the hostname and the `sinfo` reason.
* **A per-test timeout** (900 s, ~16× the slowest measured test) plus a watchdog in the supervisor, which
  kills a segment that has gone quiet. The timeout is the fast path; nothing in-process can interrupt a
  thread blocked inside a driver call, which is what a hung EGL or CUDA operation is.

`tests/integration/test_safety_net.py` tests all of that by crashing real processes, and
`tests/gpu/test_crash_isolation.py` does it on the GPU with live EGL contexts (opt-in:
`SE_SELFTEST_CRASH=1`, since those tests abort the interpreter on purpose).

Test order is not load-bearing: with `pytest-randomly` installed, `-p randomly --randomly-seed=N`
reorders collection and gives the same results as the default order. It is off by default (`-p
no:randomly` in `addopts`) because shuffling the full run would rebuild module-scoped env fixtures —
a MuJoCo plate bake apiece.

## The checks

### Render parity against the official renderer

`test_render_parity_with_official_renderer` renders 64 states from the env's own reset distribution
through both our kernel and the official env, and reports agreement split two ways.

**Interior pixels must be 100% exact.** An interior pixel is one whose 3×3 neighbourhood in the reference
is a single flat colour, so there is no coverage ambiguity there: a disagreement means a wrong colour, a
wrong shape, or a wrong place. The default floor is `INTERIOR_EXACT_MIN = 1.0`.

An env may declare a lower floor in `parity_interior_min`, but only together with a `parity_note` saying
what the mechanism is — the test asserts the note exists and prints it in the report. That is the
difference between a limitation and a loosened threshold: the note keeps the exception attached to the
one env that needs it, visible in the output, instead of lowering the bar for everything. The case it
exists for is MuJoCo's Gouraud shading of tessellated geometry, which no per-pixel analytic shader
reproduces exactly.

**All-pixel agreement is held to a much looser bar** (`ALL_EXACT_MIN = 0.90`, `MEAN_ABS_MAX = 4.0`). Edge
pixels are the two rasterizers' anti-aliasing rules landing on different sides of a rounding. gym_pusht
draws at 512 and `cv2.resize`s down, so its edges are its own resampling filter rather than true
coverage; demanding exactness there would force us to reimplement pygame's coverage rules instead of
rendering the scene correctly.

Folding the two together is the thing to avoid. A real geometry error is small in aggregate — the block
outline being mitred instead of round costs 2.2% of interior pixels, which is invisible in an all-pixel
average but is exactly 0 versus 1.0 in the interior column.

Measured, PushT vs `gym_pusht/PushT-v0`, 64 sampled states:

| | interior exact | all exact | mean abs. diff | interior share of frame |
|---|---|---|---|---|
| 96px, ss=1 | **100.00%** | 90.70% | 2.47 / 255 | 78.8% |
| 96px, ss=2 | **100.00%** | 93.79% | 1.35 / 255 | 78.8% |
| 224px, ss=1 | 99.9985% | 95.60% | 1.19 / 255 | 86.4% |
| 224px, ss=2 | 99.9985% | 97.04% | 0.57 / 255 | 86.4% |

The 224px row is 41 disagreeing pixels out of 2.75 M interior pixels across 64 frames. The likely
explanation is the mask rather than the render: interior-ness is defined by a flat 3×3 neighbourhood *in
the reference*, and the reference's 512→224 downsample can leave a patch flat while the pixel itself is
a blend, which classifies a few edge pixels as interior. The default resolution, which is where the
parity claim is stated, is exact.

Visually, on frames from a real `gym_pusht` rollout:

![CPU official, GPU, and the absolute difference](assets/media/PushT/parity.png)

The red in the third column is the anti-aliased outline of the T and the agent, and nothing else.

**This test is not relaxed for the visual tier.** An env that declares `fidelity_tier = "visual"` is
*skipped* here and asserted separately, below. The floor above stays `INTERIOR_EXACT_MIN = 1.0` for
everyone it applies to. See [Fidelity tiers](concepts/fidelity-tiers.md) for why the two claims are kept
mechanically apart rather than merged behind one per-env threshold.

### Visual-tier fidelity against the official renderer

`test_visual_tier_matches_its_declared_fidelity` runs for visual-tier envs *only* — the ones whose frame
is a different render of the same scene rather than the official frame — and asserts the numbers that
env declared in `visual_fidelity`. Three quantities, because they fail for different reasons:

| assertion | catches |
|---|---|
| `mean_abs <= mean_abs_max` | a global shift: a light, a material, a colour-resolution rule |
| `within8_frac >= within8_min` | a REGION going wrong while the mean stays small |
| `ssim >= ssim_min` | structure — a surface losing its texture moves SSIM and can leave both others put |

Measured, `RoboCasaCounterToSink` vs MuJoCo's own raster of the same compiled kitchen, 64 states × 4
independent draws at 256 px:

| env | mean abs | within 8/255 | SSIM | interior exact |
|---|---|---|---|---|
| `RoboCasaCounterToSink` | 6.58 | 82.9% | 0.907 | 12.9% |
| `RoboCasaCounterToSinkRight` | 5.79 | 84.3% | 0.921 | 7.9% |
| `RoboCasaCounterToSinkWrist` | 9.89 | 82.1% | 0.857 | 15.8% |
| `RoboCasaCounterToSinkL45S33` | 32.10 | 36.7% | 0.821 | 4.8% |

The interior-exact column is reported and **not asserted**. It is the honest measure of what the tier
gave up — every parity env in the registry declares 1.0 there — and asserting a low floor on it would
turn a number reported for honesty into a target.

A visual env must set `parity_interior_min = None`; `register()` refuses it otherwise, so the parity gate
cannot be applied to it, loosened for it, or passed by it. `test_fidelity_tier_is_fully_declared` runs for
every env and re-asserts that enforcement, so an under-declared tier fails loudly rather than falling
between the two tests.

### A learned decoder: two more assertions, and neither is an aggregate

A visual-tier class may draw its frame through a fixed neural checkpoint
([the learned decoder](concepts/fidelity-tiers.md#the-learned-decoder)). Such a class declares
`render_decoder = "learned"` and is asserted by everything above *plus* two tests that exist only
because a network is in the path:

| test | asserts |
|---|---|
| `test_a_learned_decoder_beats_the_render_it_replaces` | on the same states, against this machine's official renderer, the learned frame is better than the plain class's on **mean_abs, within8_frac AND ssim** — separately |
| `test_a_learned_decoder_holds_up_under_its_declared_variation` | the same three, re-measured with a per-state appearance drawn from `variation_space`, against the class's `variation_fidelity` |

The three comparisons are separate and never combined, and that is the whole design. A corrector
applied off its training distribution **lowers a frame's average error while making more of its pixels
wrong**: on a kitchen it was not fitted for, the mean ties (33.19 against 32.85 for a plain resize) and
SSIM falls (0.763 against 0.792) because it paints marble-like speckle onto plain cabinet fronts. A
single aggregate would have passed that.

The second test exists because for a *renderer* it would be redundant and for a *network* it is the
whole risk: a rasterizer that agrees with the official one at one camera agrees at another, while a
corrector fitted at one camera pose scores 3.25/255 there and 9.06 under the env's own declared ±8° of
pan — worse than not correcting at all. `register()` refuses a learned decoder that does not declare
`variation_fidelity`, so the measurement cannot be skipped by omission.

`tests/gpu/test_rendernet.py` covers what is left: that the delivered frame is exactly the cheap render
put through the corrector, that the corrector was actually applied (a checkpoint that failed to load
still produces a perfectly reasonable picture), that a caller's batch size cannot reach the autotuner's
decision, and that a checkpoint keyed to one cell refuses every other. `tests/unit/test_rendernet_identity.py`
asserts the refusals on synthetic checkpoints, with no GPU and no cache.

### State round-trip

`set_state(get_state())` must be the identity to float32. Measured for PushT: max absolute error
**0.0**.

Rules out: a state representation that quietly normalises, clips or reorders, which makes every stored
state a slightly different state when it comes back.

### Every state dimension reaches the render

Each dimension is overwritten with the value that dimension takes in another sampled state, and the frame
must change; a dimension is declared dead only if several donors and then a set of additive nudges all
leave the frame bit-identical. Measured for PushT: **no dead dimensions**.

Rules out two different failures at once. A dimension the renderer ignores means either the state is not
minimal — you are storing something that does not matter — or the renderer is wrong and is drawing
something at a fixed pose. Both are silent, and the second is exactly what the missing
`reindex_shapes_for_body` produced in the reference.

Substitution rather than a fixed nudge, because the probe has to land somewhere the env actually reaches.
The check used to add `+25.0`, which is invisible to any integer dimension whose renderer wraps it modulo
a period dividing 25: on a 25×25 MiniGrid it reported `agent_x` and `agent_y` dead while they were
demonstrably live, and on a 22×22 grid of the same family it reported nothing. The verdict depended on
the grid width — a property of the probe, not of the env. A donor value respects integer alphabets,
bounded ranges and wrap-around without knowing anything about them.

### Batch invariance

Frame *i* of a batch must equal the frame rendered on its own. Measured for PushT: max difference
**0/255**.

Rules out: the entire performance story being a lie. Every throughput number in this repo assumes
batching is semantically free, so a batch-dependent render would mean the fast path and the correct path
are different paths.

### Backend agreement

The fused Triton kernel must agree with the readable torch reference implementation. Measured for PushT:
**100.00%** of pixels exact, max difference **0**.

Rules out: a fast kernel nobody can check. The torch path is short enough to read as a definition of what
the picture should be; keeping it and diffing against it is what makes the kernel trustworthy rather than
merely tested against itself.

### Variation coverage

Every declared variation leaf is moved to the far end of its range; the frame must change. Measured for
PushT: **no inert variations** out of 14 leaves.

Rules out: a knob wired into the space but not into `scene()` — advertised support that does not exist.

It does *not* rule out a knob that changes the picture for the whole batch when it was supposed to change
it per environment; see [what actually varies per environment](research/variation_collapses.md#what-actually-varies-per-environment)
for the measured breakdown of that case.

### Determinism

The same states must render identically twice. Measured for PushT: identical.

Rules out: any dependence on allocation order, cache state, or a stale static-prefix bake.

### State sufficiency

This one is not about our renderer at all, and it is the most important check in the suite.

Two instances of the **official** env are driven to the same states through different histories (one in
order, one reversed) and their frames must match exactly. If the compact state really determines the
frame, they will. If they do not, the state does not determine the frame for that env, and no GPU
renderer can fix that — the thesis is false there and the correct response is to not add the env.

It is also the check that pays for itself before any porting work happens: it runs against the reference
alone, so it can be answered for a candidate env in an afternoon.

For PushT it caught two real bugs in how the reference is *driven*, both of which affect anyone restoring
PushT from a stored state:

- `gym_pusht._set_state` sets `position` then `angle`. Chipmunk rotates about the body's centre of
  gravity, `(0, 45)` for the T, so the position read back is not the one written — up to **70 px** of
  error, and the error depends on the block's *previous* angle. Reversing the order round-trips to
  2.8e-14.
- A manually moved body keeps drawing at its old pose until `space.reindex_shapes_for_body`, so the
  reference frames silently stopped being a function of the agent's coordinates.

Neither is a rendering bug and neither would have been found by comparing pixels.

## Thresholds, and why they are where they are

| constant | value | reasoning |
|---|---|---|
| `INTERIOR_EXACT_MIN` | 1.0 | interior pixels have no anti-aliasing ambiguity; anything below 1.0 is a real error |
| `ALL_EXACT_MIN` | 0.90 | edges are two rasterizers' coverage rules disagreeing; the reference is not ground truth there |
| `MEAN_ABS_MAX` | 4.0 / 255 | catches a systematic colour shift that leaves the interior fraction intact only if the shift is small |
| `Env.parity_interior_min` | 1.0 | per-env override of the interior floor; requires `parity_note` |
| `Env.visual_fidelity` | — | the visual tier's own thresholds, set from the SPREAD over independent state draws; required in full, or `register()` refuses the class |

If a new env cannot reach interior 1.0, the right move is to find the constant that is wrong — see the
[measured-constants table](adding-an-env.md#step-2-find-the-references-real-constants) for the five that
were wrong in PushT — not to lower the bar. The [visual tier](concepts/fidelity-tiers.md) is not an
escape from that: it is for an env whose *renderer* provably cannot be the official one, it makes a
weaker and separately-stated claim, and it disqualifies the env from the use the strong claim licenses.

## Using the checks outside pytest

They are plain functions returning dicts of numbers, so they can drive a report or a benchmark:

```python
from stable_envs.testing import conformance as C

C.check_state_roundtrip(env)        # {'max_abs_err': 0.0}
C.check_state_dims_matter(env)      # {'dead_dims': []}
C.check_backend_agreement(env)      # {'exact_frac': 1.0, 'max_abs': 0, ...}
C.pixel_agreement(ours, official)   # all-pixel and interior-pixel agreement
```

API details: [`stable_envs.testing`](api/testing.md).
