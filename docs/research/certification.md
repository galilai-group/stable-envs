# Certification: what "done" means for a stable-envs renderer

!!! info "This page is frozen policy, not a measurement"
    Every numeric threshold below was chosen from the **existing committed benchmark corpus**, before
    any fresh RTX 3090 or Apple-M5 measurement was taken, precisely so that a later result cannot
    move the bar it is being judged against. The derivations are shown so the choices can be argued
    with; the numbers themselves are fixed. The machine-readable copy is
    [`scratch/certification.json`](https://github.com/galilai-group/stable-envs/blob/main/scratch/certification.json),
    and `tests/unit/test_certification.py` fails if the two drift apart.

    **No benchmark was run to produce this page.** It reads
    `examples/results/bench_gpu_vs_cpu.json` (133 env×size rows, NVIDIA H200),
    `examples/results/decode/ALL.json` (196 rows), `examples/results/dino/dino_agreement.json`
    (192 envs), `examples/lewm_pusht/results/` (300 paired episodes) and
    [`scratch/swm_matrix.json`](https://github.com/galilai-group/stable-envs/blob/main/scratch/swm_matrix.json).

---

## 1. Project objective

The claim this project is trying to earn the right to make is:

> A frame is a pure function of a compact state, and regenerating it is so much cheaper than storing
> or officially re-rendering it that you should stop doing either — **and nothing downstream can tell.**

Two halves, and they fail differently. The first half is an economics claim and is settled by
throughput. The second is a semantic claim and is settled by **task success under a renderer swap**,
not by PSNR, not by pixel identity, and not by feature cosine.

The governing product criteria are unchanged: rendering should be very fast; CPU is a first-class
target rather than a fallback; GPU is a first-class target especially at large batch; small visual
approximations are acceptable *if downstream behaviour is preserved*; physics stays exact where
practical and is explicitly labelled where it is not; no learned renderer is required.

The template for the semantic half already exists, on PushT:

| arm | successes | rate | paired vs official |
|---|---|---|---|
| official (`swm/PushT-v1`, pygame) | 264/300 | 88.0 % | — |
| stable-envs, every pixel swapped | 263/300 | 87.7 % | 12/11 discordant, **p = 1.000** |
| blank frames (control) | 2/50 | 4.0 % | 46/0 discordant, **p < 0.001** |

The control row is what makes the null mean anything: the evaluation *can* move by 84 points, and it
does not move when the renderer changes. Every downstream gate in this document is built to that
shape.

---

## 2. Why Step 1's axes stay independent

Step 1 spent its whole effort separating things that are routinely conflated. Collapsing them again
here would throw that away. The following is a **coherent** state, not a contradiction, and the
scheme below must be able to represent it without an asterisk:

```text
renderer exists                  = yes
registered default render        = covered
pixel fidelity tier              = visual, not parity
CPU speed                        = unmeasured
GPU speed                        = fast
physics                          = reference_only
VariationMap                     = missing
downstream renderer validation   = ready
standalone task semantics        = different
```

That is `swm/HumanoidDMControl-v0` as of Step 1. The renderer draws its states correctly — the
registered default (`humanoid-walk`) and the stable-envs env (`humanoid-stand`) compile to a
byte-identical render model after `reset()` — and only the *reward* differs, which reaches no pixel.
A single `partial` label would have destroyed that finding.

Six separations are therefore load-bearing, and each gets its own gate below:

| separation | why it must not collapse |
|---|---|
| renderer coverage ≠ task semantics | swm keeps producing the states; stable-envs supplies pixels only. A different reward never reaches the renderer. |
| render fidelity ≠ physics fidelity | `reference_only` physics is a fully certifiable configuration. Renderer completion never waits on a fast `step()`. |
| CPU support ≠ CPU speed | `device="cpu"` returning a frame is not a throughput claim. Across the **34 matched renderer rows** the CPU-support axis reads `yes` on 18, `unknown` on 12 and `unknown_dependency_absent_here` on 4 — and `se_cpu_fps = needs_measurement` on all 34. The other 8 rows have no renderer to run (6 renderer gaps, 2 symbolic ids), so CPU support is **n/a** there rather than absent or unknown. |
| GPU throughput ≠ provenance | The committed corpus is H200/VastAI. It is context for Sami's RTX 3090, never certification of it. |
| variation compatibility ≠ baseline usability | 30 matched ids have no `VariationMap`. None of them is blocked from downstream validation, which uses the registered default appearance. |
| pixel/DINO agreement ≠ task success | Feature cosine is a tripwire. Only a paired task gate licenses a semantic-equivalence claim. |

And one more, which Step 1 added by measurement: **absent ≠ unmeasured**. A probe that could not
import a dependency records `unknown_dependency_absent_here`, never `no`.

---

## 3. Certification axes

Nine independent axes. Each carries its own status from
`{not_applicable, unknown, pending, pass, fail, partial}` plus, where relevant, a tier.

| axis | gate | statuses beyond pass/fail |
|---|---|---|
| A · renderer correctness | §4 | `partial` (documented observation subset) |
| B · renderer fidelity tier | §4.3 | `parity` \| `visual` |
| C · CPU performance | §5 | `CPU_UNMEASURED` … `CPU_EXCEPTIONAL` |
| D · GPU performance | §6 | `GPU_UNMEASURED` … `GPU_EXCEPTIONAL` |
| E · physics fidelity | §7 | the Step 1 taxonomy |
| F · SWM render compatibility | §8 | — |
| G · variation space | §9 | `VARIATIONS_NONE` … `VARIATIONS_COMPLETE` |
| H · downstream semantics | §11 | `DOWNSTREAM_DIRECT` \| inherited \| none |
| I · registered-default coverage | Step 1, §12.1 | `full` \| `partial` (documented subset) \| `absent`. **Gates completion**: only `full` can reach `ENVIRONMENT_DONE`. |

**Nothing derives one axis from another.** A `fail` on C does not touch D; an `unknown` on G does not
touch A or H. The *derived* states of §12 are conjunctions over several axes at once — that is what
makes them derived — but no axis ever changes another axis's value.

---

## 4. Gate A — renderer correctness

> Does the renderer faithfully encode the visual scene implied by the SWM/reference state?

This gate is **entirely** the existing conformance suite. No parallel checker is defined here.
`stable_envs/testing/conformance.py` supplies the checks; `tests/gpu/test_conformance.py` and
`tests/gpu/test_reproducibility.py` run them parameterised over `REGISTRY`;
`stable_envs/testing/gate.py` declares which envs one interpreter must be able to reach and why
anything is excluded.

### 4.1 Check classification

**Mandatory** — every certified env, both tiers, no exceptions:

| check | what a failure means |
|---|---|
| `check_state_roundtrip` | `set_state`/`get_state` is lossy; states cannot be stored |
| `check_state_dims_matter` | a state dimension reaches no pixel — the frame is not a function of the state |
| `check_batch_invariance` | frame *i* depends on its neighbours in the batch |
| `check_batch_size_invariance` | the same state renders differently at a different batch size |
| `check_order_independence` | permuting the batch changes per-state output |
| `check_determinism` | two renders of the same states differ |
| `check_render_is_pure` | rendering mutates state, or reads something outside `(state, style, size)` |
| `check_chunked_render_matches` | a loader's chunking changes the picture |
| `check_input_form_invariance` | numpy vs torch vs list gives different pixels |
| `check_set_state_copies` | the caller's array aliases internal storage |
| `check_style_is_state` | style is not carried per-row and cannot be batched |
| `check_default_style_is_official` | `se.make(name)` does not give the official appearance |
| `check_per_env_variation` | one env's variation leaks into another's frame |
| the fidelity declaration is complete | `check_fidelity_declaration` — the tier is asserted, not declared |

**Family-specific** — mandatory *where the env declares the capability*, `not_applicable` otherwise:

| check | applies when |
|---|---|
| `test_render_parity_with_official_renderer` | `fidelity_tier == "parity"` and a reference renderer is installable. Explicitly **not** relaxed for the visual tier — a visual env is skipped here and asserted against its own declared numbers instead. |
| `check_visual_fidelity` / `check_visual_fidelity_under_variation` | `fidelity_tier == "visual"` |
| `check_backend_agreement` | the class declares more than one backend (`raycast3d` + `mjwarp`, `raycast3d` + `nvdiff`) |
| `check_colors_apply_exactly` | the env exposes colour variations |
| `check_variations_do_something` | `variation_space` is non-empty |
| `check_step_matches_reference` | `step_reference_checkable = True` — see §7 |
| `check_resolution_consistency` | the env can render at more than one size |

**Diagnostic only** — informative, never a gate:

- `check_reference_renderer_is_stable` — a property of the *reference*, not of us. A reference that
  is itself unstable invalidates a parity number rather than failing our renderer.
- `test_state_is_sufficient` — the thesis stated about the reference env. Failing it is a finding
  about the upstream env, not a defect here.
- `pixel_agreement` / `ssim` reported outside a declared floor.

**Not applicable** — envs with `render_decoder == "learned"` skip the parity checks and take
`test_a_learned_decoder_beats_the_render_it_replaces` instead. No env in the SWM matrix is in this
class; the branch exists for RoboCasa's RenderNet.

### 4.2 Additional requirement from Step 1

**The configuration being certified must be the registered SWM default.** Step 1 established this by
measurement (the render-model signature over geoms, sites, materials, camera and lights, taken
*after* `reset()`), and it caught real surprises: `ReacherDMControl` defaults to `hard`, not `easy`;
`CartpoleDMControl` to `swingup_sparse`. A certification run must record the signature it certified.

Gate A is `partial` — not `fail` — where the renderer covers a **documented** proper subset of the
default observation, with the exclusions declared in the class and measured on what is drawn.
`swm/CraftaxClassicPixels-v1` is the only such row (world layer certified; inventory strip, night
noise field and the sleeping desaturation declared out).

`partial` is a passing state for this gate and it caps the ladder: see §12.2 — a `partial` env can be
certified per backend at any performance tier, and can never be `ENVIRONMENT_DONE`.

### 4.3 Fidelity tiers

Both tiers are certifiable. Pixel identity is **not** universally required.

**`parity`.** The renderer intends to reproduce the official frame. Evidence: `interior_exact_frac ≥
parity_interior_min` (every env in the repo declares `1.0`; any relaxation must be explained, which
`test_relaxed_parity_floors_are_all_explained` already enforces). "Interior" excludes edge
anti-aliasing, where two independent rasterisers are entitled to disagree and neither is ground truth.

**`visual`.** The renderer intentionally simplifies appearance. Evidence:

- the declared `visual_fidelity` floors (mean abs, SSIM) hold, including *under* the declared variation;
- every task-critical object and geometry is present — this is `check_state_dims_matter`, not an eyeball;
- determinism and batch-consistency are held to the identical mandatory bar;
- the representation tripwire of §10 passes;
- **broad semantic-equivalence claims require project-level downstream evidence (§11).**

A `visual` renderer is never held to a parity threshold, and a `visual` tier is never by itself a
reason to withhold certification.

---

## 5. Gate C — CPU performance

### 5.1 What is measured

```text
official / reference renderer on CPU     vs     stable-envs renderer on CPU
```

at the same env state, output resolution, resolved supersample and **thread budget**. Renderer-only
and full `step + render` are timed separately and reported in separate tables.

Define, on the same host:

```text
S_cpu    = stable-envs CPU frames/s  /  official CPU renderer frames/s
cores_cpu = stable-envs CPU frames/s  /  one-core JPEG-q90 decode frames/s   (of this env's own frame)
```

### 5.2 Required views

Batch: **1, 16, 64, 256, 1024**, and 4096 where memory-safe.
Threads: **1** and **all physical cores**, wherever the backend can be pinned. A number taken with
unrestricted machine-wide parallelism is not a CPU certification; it is a node measurement, and it is
recorded as such along with the load average.

### 5.3 Frozen tiers

| tier | condition on `S_cpu` |
|---|---|
| `CPU_UNMEASURED` | no measurement exists |
| `CPU_REGRESSION` | `S_cpu < 1` |
| `CPU_PARITY` | `1 ≤ S_cpu < 2` |
| `CPU_FASTER` | `2 ≤ S_cpu < 5` |
| `CPU_FAST` | `S_cpu ≥ 5` **and** `cores_cpu ≥ 1` |
| `CPU_EXCEPTIONAL` | `S_cpu ≥ 20` **and** `cores_cpu ≥ 10` |

`S_cpu` is the value at the **1-thread** budget. The all-cores number is reported alongside and never
substituted for it, because scaling a Python renderer across a node is a different achievement from
making it fast.

### 5.4 Where those numbers come from

There is **no stable-envs CPU throughput anywhere in the committed corpus.** Every `cpu` column in
`bench_gpu_vs_cpu.json` and `decode/ALL.json` is the *official* renderer — Step 1 recorded this on
all **34 matched renderer rows** as `se_cpu_fps = needs_measurement`; the 6 renderer gaps and 2
symbolic ids are **n/a**, since they carry no stable-envs env to time. So the numerator distribution
does not exist and cannot be used to calibrate. The thresholds are therefore anchored to a
distribution that *does* exist, using the repo's own existing economics criterion.

`docs/research/when_does_this_pay_off.md` §G4 sets the bar for whether regeneration is worth doing at
all: **regeneration must be faster than decoding a stored frame.** Transplant that from GPU to CPU
and it becomes a statement about `S_cpu`: for a given env, the CPU renderer must beat one core of
JPEG-q90 decode, i.e.

```text
S_cpu  ≥  (one-core decode fps) / (official CPU renderer fps)   ==   "G4-on-CPU break-even"
```

That ratio *is* measurable from the committed corpus — both terms are in `decode/ALL.json`. Over the
187 rows that carry both:

| | min | p10 | p25 | **median** | p75 | p90 | max |
|---|---|---|---|---|---|---|---|
| all families | 1.11 | 2.27 | 3.62 | **7.03** | 21.2 | 76.4 | 999 |
| 3-D scene class | 1.11 | 2.05 | 3.54 | **5.04** | 10.3 | 45.2 | 102 |
| 2-D raster/tile class | 2.28 | 3.08 | 4.72 | **10.35** | 46.4 | 335 | 999 |

Every tier boundary is a percentile of that distribution, rounded to a round number:

- `CPU_PARITY` floor **1** — slower than the renderer you replaced is a regression by definition.
- `CPU_FASTER` floor **2** ≈ p10 (2.27): at 2× the easiest tenth of the corpus has cleared G4.
- `CPU_FAST` floor **5** ≈ the 3-D-class median (5.04): at 5×, a *typical* 3-D env has reached the
  point where regenerating on one core is as cheap as reading. The `cores_cpu ≥ 1` conjunct makes
  that literal rather than approximate, per env.
- `CPU_EXCEPTIONAL` floor **20** ≈ p75 (21.2): three quarters of the corpus has cleared G4.

### 5.5 When the reference renderer does not exist

`S_cpu` is a ratio against the official renderer, and some envs have no installable one in any
interpreter (Craftax, MiniHack, Merge). The ratio is then undefined and must not be fabricated. Since
§12.2 makes `CPU_FAST` mandatory for completion, those envs need the same escape §6.8 gives the GPU
side — otherwise "no upstream package" would silently mean "can never be finished".

They certify on `cores_cpu` alone, with the qualifier `NO_REFERENCE`:

| tier | condition |
|---|---|
| `CPU_FASTER` | `cores_cpu ≥ 1` |
| `CPU_FAST` | `cores_cpu ≥ 3` |
| `CPU_EXCEPTIONAL` | `cores_cpu ≥ 10` |

The floors are the primary ladder's own spacing applied from the G4 bar: `CPU_FAST` is 2.5× the
`CPU_FASTER` floor and `CPU_EXCEPTIONAL` is 4× `CPU_FAST` in §5.3, so 1 → 2.5 → 10, rounded to
1 / 3 / 10. They are deliberately far below the GPU's `NO_REFERENCE` ladder of 1 / 10 / 100, because
on GPU a whole device competes with one core of decode, while here one core competes with one core.

As on the GPU side, a `NO_REFERENCE` row can never contribute to a parity claim or to §11.

### 5.6 The rule that must not be softened

An environment is **not** CPU-certified fast because `device="cpu"` returned a frame. Gate A support
and Gate C speed are different claims with different evidence, and Step 1 kept them apart on every
row. `render_executes = yes` maps to no CPU tier at all; the tier stays `CPU_UNMEASURED`.

---

## 6. Gate D — GPU performance

### 6.1 Two questions, two tables, never merged

**Renderer microbenchmark.** `batched states already in device memory → renderer → frames`. No
`DataLoader`, no `default_collate`, no host transfer inside the timed loop. This is the number the
tiers below are computed from.

**End-to-end input pipeline.** `stored render states → sampler/DataLoader → transfer → renderer →
frames`. This is where `default_collate` vs batched indexing vs the no-loader ceiling belong, and it
is the only table in which loader overhead may appear. Existing repo measurements put that overhead
in view around B = 4096, which is one of the reasons B = 4096 is mandatory.

A row from one table is never quoted as if it came from the other.

### 6.2 Required batch sweep

**Mandatory: B = 1, 16, 64, 256, 1024, 4096.** Optional where memory-safe and informative: 8192,
16384. **B = 4096 is the headline point** and may be omitted only with a recorded reason.

Feasibility is already established: across the 129 measured env×size rows that reach B = 4096, peak
device memory is **10.7 GB worst case** (MiniHack; next is Pendulum at 10.4 GB) and **no row exceeds
24 GB** — so every currently-measured env fits B = 4096 on a 24 GB RTX 3090.

B = 1 is mandatory for a different reason: it is where launch overhead is visible, and the existing
corpus shows the GPU path at batch 1 is *slower than the official CPU renderer* for most envs
(median speedup **0.6×**, p90 0.9×). No env is ever certified on a B = 1 number.

### 6.3 Two derived quantities

On the same host, at the best **certified** batch (i.e. from the mandatory list, not from 65536):

```text
R      = stable-envs GPU frames/s  /  official CPU renderer frames/s
cores  = stable-envs GPU frames/s  /  one-core JPEG-q90 decode frames/s
```

`cores` is the repo's existing G4 economics metric, already the headline column of
`examples/results/decode/ALL.md`.

### 6.4 Backend classes

The measured distribution is strongly bimodal by render backend, so one global threshold would be
wrong. Two classes, assigned deterministically from the `backends` field that
`scratch/census/census.py` already records per class:

| class | backends | families | n rows |
|---|---|---|---|
| **S** — 3-D scene | `mjwarp`, `raycast3d`, `nvdiff` | DM Control, Fetch, OGBench, D4RL PointMaze, MuJoCo locomotion, Meta-World, HumanoidBench, RoboCasa | 138 |
| **R** — 2-D raster/tile | `scene`, `raster2d`, `tilemap`, `gfxdraw`, embedded atlas | PushT, classic control, highway-env, MiniGrid/BabyAI, Crafter, Craftax, Griddly, Overcooked, Sokoban, Jumanji, Procgen, PGX, LunarLander, CarRacing, MiniHack, swm nav | 58 |

Measured on H200, `R` = GPU at best batch ÷ official CPU renderer:

| class | min | p05 | p25 | median | p75 | max |
|---|---|---|---|---|---|---|
| S | 5.5 | 12.1 | 13.0 | 25.2 | 212 | 596 |
| R | 55.5 | 76.0 | 398 | 3,129 | 40,035 | 132,228 |

### 6.5 Transferring an H200 corpus to an RTX 3090

This is the one place where the existing evidence cannot be used as-is, and saying so is part of the
specification. `R` is a **cross-device ratio**: moving from an H200 to an RTX 3090 divides the
numerator without touching the denominator, so the historical ratios are not portable.

The fused rasterizer is documented bandwidth-bound on its own output, so the scaling factor used
here is the memory-bandwidth ratio:

```text
H200 HBM3e   ≈ 4.8 TB/s
RTX 3090 GDDR6X ≈ 936 GB/s
                        ->  5.13x
```

Predicted RTX 3090 distribution (`R_H200 / 5.13`):

| class | p25 | median | p75 |
|---|---|---|---|
| S | 2.54 | 4.91 | 41.4 |
| R | 77.6 | 610 | 7,804 |

This is a **prediction, and it is pre-registered.** It is deliberately pessimistic for host-bound
rows (§6.7), whose throughput is set by the node's processor and will not fall by 5×. If Step 3
measures a systematically different factor, that is a finding to report — not a reason to move the
thresholds.

### 6.6 Frozen tiers

| tier | class S (3-D scene) | class R (2-D raster/tile) |
|---|---|---|
| `GPU_UNMEASURED` | no measurement | no measurement |
| `GPU_REGRESSION` | `R < 1` | `R < 1` |
| `GPU_PARITY` | `1 ≤ R < 2` | `1 ≤ R < 10` |
| `GPU_FASTER` | `2 ≤ R < 4` | `10 ≤ R < 500` |
| `GPU_FAST` | `R ≥ 4` **and** `cores ≥ 1` | `R ≥ 500` **and** `cores ≥ 1` |
| `GPU_EXCEPTIONAL` | `R ≥ 40` **and** `cores ≥ 10` | `R ≥ 5000` **and** `cores ≥ 10` |

Anchors, all from the predicted-3090 table in §6.5: the `GPU_FASTER` floor is the class p25, the
`GPU_FAST` floor is the class median, the `GPU_EXCEPTIONAL` floor is the class p75 — each rounded
down to a round number. The `cores` conjuncts are the repo's own G4 bar (`cores ≥ 1`) and one order
of magnitude past it.

An env that clears the ratio bar but fails `cores` is capped at `GPU_FASTER` and carries the
qualifier `BELOW_G4`. That is not a formality: `DMCBallInCupCatch` measures **0.59 cores** on the
H200 — the only row in the 196-env corpus below the G4 bar — while its ratio is 12.5×. It is faster
than the official renderer and still loses to reading the frame off disk. The scheme has to be able
to say both.

### 6.7 Mandatory qualifiers, independent of tier

| qualifier | condition | consequence |
|---|---|---|
| `BATCH_SCALES` | `fps(4096) ≥ 20 × fps(1)` | **required for any tier above `GPU_PARITY`.** Bar set below the observed minimum across 129 rows (22.8×) so it fails only a renderer that is not actually batching — a per-frame loop scores ≈1×. |
| `NO_LARGE_BATCH_REGRESSION` | `fps(4096) ≥ 0.9 × fps(256)` | required. Catches memory thrash at large batch. 1 of 129 rows sits between 0.9 and 1.0 (Roundabout, 0.93). |
| `HOST_BOUND` | `cpu_frac ≥ 1.5` at the certified batch | the row is a measurement of **this node's processor under this node's load**, not of the device. It must be reported with thread count and load average, and may not be quoted as a device capability or compared across nodes. |

`HOST_BOUND` is not hypothetical. `DMCReacherHard` measures `cpu_frac = 2.95` — two to three cores
saturated inside a per-chunk Python loop with the device waiting — and the same configuration
elsewhere in this repo has produced a 13× spread across nodes on a column that once claimed 0.4 %
reproducibility.

### 6.8 When the reference renderer does not exist

Some envs have no installable official renderer in any interpreter (Craftax, MiniHack, Merge). `R` is
undefined and must not be fabricated. Those envs certify on `cores` alone, with the qualifier
`NO_REFERENCE`:

| tier | condition |
|---|---|
| `GPU_FASTER` | `cores ≥ 1` |
| `GPU_FAST` | `cores ≥ 10` |
| `GPU_EXCEPTIONAL` | `cores ≥ 100` |

`NO_REFERENCE` rows can never contribute to a parity claim, and never to §11.

---

## 7. Gate E — physics fidelity

Independent of everything above. **Fast rendering is the project; fast physics is optional.** No
renderer's certification waits on a `step()`.

The Step 1 taxonomy, unchanged, with the evidence each class requires. Tolerances are read from
`check_step_matches_reference`, over its default **16 episodes × 100 steps**.

| class | requirement | frozen bound | measured today |
|---|---|---|---|
| `n/a` | no dynamics in scope | — | 8 rows |
| `reference_only` | the official simulator remains the source of state | none — **fully certifiable** | 24 rows |
| `exact_fast_step` | bit-identical transition | `exact_frac == 1.0` and `max_err == 0.0` | MiniGrid, Sokoban, SimplePointMaze, swm_nav: `exact_frac 1.0`, max deviation 0.0 |
| `near_exact_fast_step` | same published closed form, same arithmetic; differs only where transcendentals do | `first_step_max_err ≤ 1e-12` **and** `max_err ≤ 1e-9` | classic control after 100 steps: 3.3e-16 (MountainCar) … 4.4e-13 (Acrobot); after one step ≤ 2.2e-16 |
| `approximate_fast_step` | a genuinely different solver | see below | PushT |

The `near_exact` bounds sit roughly 2,000× above the worst value the family actually produces
(4.4e-13) and far below anything physically meaningful (1e-9 rad on an O(1) rad angle), so they fail
on an algorithmic change and not on a CUDA-vs-glibc ulp.

`approximate_fast_step` requires all four of:

1. **explicitly selectable** — never the silent default (§13);
2. a **kinematic control group** that agrees to float precision, because a term with no contacts
   acting on it has no excuse to drift: `control_max_err < 1e-3` (PushT measures 9.5e-05);
3. a **median**, not a maximum, on the chaotic gate group — contact is chaotic and two solvers that
   disagree by an ulp at first touch disagree by centimetres later: PushT's frozen bound is
   `block_median_err < 2.0` px at 100 steps (measured 0.68, against a mean of 20.7);
4. the gate group must actually have **moved** in ≥ 75 % of episodes — an episode where the gate
   never moved agrees trivially and is evidence of nothing;

and, before it may become a recommended default, a **task-level validation** under §11.

A `reference_only` env is certified with the badge `PHYSICS_REFERENCE_ONLY`, which is a statement of
scope, not a deficiency.

---

## 8. Gate F — stable-worldmodel render compatibility

Compatibility means the fast renderer can be dropped into the intended SWM workflow **without
changing the semantics of state generation**. The adapters exist and are generic over `REGISTRY`;
this gate is about evidence, not new code. Reuse `stable_envs/compat/worldmodel.py` and
`examples/swm_compat/`; do not write another adapter.

Requirements, all mandatory:

1. the registered/default SWM state is extractable through the declared hook, and stable-envs
   consumes it unchanged;
2. the returned frame has the shape, layout and dtype `swm`'s wrapper stack expects
   (`AddPixelsWrapper` → `EverythingToInfoWrapper` → `EnsureInfoKeysWrapper` → `ResizeGoalWrapper`);
3. **every `info` key the official env produces is produced by the adapter**, with the same shapes —
   the measured bar is `facade_missing == []` and `batched_missing == []`, which
   `examples/swm_compat/results/swm_workflow.json` records for `swm/TwoRoom-v1` over 8 envs;
4. batched operation works where the family supports it (`attach_batched_pool`);
5. **only `render()` is swapped.** The official env keeps its dynamics, so reward and task semantics
   are untouched by construction. This is why a differing standalone task name (§2) cannot block
   this gate;
6. no accidental per-frame reconstruction of expensive reference render state — the RoboCasa lesson:
   per-scene setup was 10.04 s against a 0.028 s loop, and a single end-to-end number hid which term
   was losing;
7. variation compatibility is recorded **separately** (§9) and never gates this axis.

Only an arm that holds everything but the renderer fixed produces a meaningful pixel comparison;
arms that differ in simulator visit different states and comparing their frames compares different
episodes.

---

## 9. Gate G — variation space

Its own axis, its own states. **A missing `VariationMap` never blocks Gate A, D, F or H**, because
the downstream gate uses the registered default appearance. 30 of the 34 matched Step 1 rows are
`VARIATIONS_UNMAPPED` and 31 are simultaneously `ready_for_downstream_validation`; that combination
must stay representable.

| state | meaning |
|---|---|
| `VARIATIONS_NONE` | the SWM wrapper exposes no `variation_space` (4 Step 1 rows) |
| `VARIATIONS_UNMAPPED` | leaves exist; no `VariationMap` entry (30 rows) |
| `VARIATIONS_PARTIAL` | some leaves classified; the remainder explicitly listed |
| `VARIATIONS_COMPLETE` | every leaf classified and the checks below pass (4 rows) |

`VARIATIONS_COMPLETE` requires:

- every SWM variation leaf classified as `identical` / `renamed` / `scaled` / `swm_only`;
- stable-envs-only leaves listed explicitly;
- `check_variations_do_something` and `check_colors_apply_exactly` pass on every visual leaf — a
  variation that changes no pixel is not a mapping, it is a typo;
- `check_per_env_variation` passes, so one env's variation cannot leak into another's frame.

### 9.1 The high-risk class: coupled variations

Where one leaf changes **both** visual geometry and physics/collision geometry, the renderer must not
show one geometry while the reference simulator integrates another. This is the only failure mode in
this document that produces frames that are individually plausible and collectively a lie, and no
pixel check can catch it — the frame is a correct rendering of the wrong world.

Coupled leaves are therefore flagged `COUPLED` at classification time and require, in addition:

- the leaf is applied to the reference simulator and to the renderer **from the same value**, in the
  same call, with no independent default on either side;
- a state-sufficiency check *after* the variation is applied, not only at the default.

Acrobot's `physics.link_length_*` and Fetch's object scales are in this class.

---

## 10. Representation-level tripwire

Not a gate on its own and never a substitute for §11 — a regression detector, cheap enough to run on
every optimization pass.

Using `examples/dino_agreement.py`, with the blank-frame arm as the scale:

```text
A = dino_cos(ours, official) / dino_cos(blank, official)
```

| tier | bound on `A` | corpus today |
|---|---|---|
| `parity` | `A ≤ 0.10` | n = 159, median 0.014, p90 0.033, **max 0.087** |
| `visual` | `A ≤ 0.25` | n = 33, median 0.104, p90 0.160, **max 0.224** |

Both bars pass on the entire current corpus by construction — that is the point. They are frozen so
that a later visual approximation which quietly moves a family toward the blank-frame end is caught
by a cheap check rather than by an expensive downstream one. Normalising by the blank arm rather than
by an absolute cosine keeps the bar meaningful across resolutions and families.

For context, a JPEG-q95 round-trip — an artifact every dataset in this space already tolerates —
scores a median cosine of 0.0099 against the official frame, and 94 of 192 envs are already *closer*
to the official frame than that.

---

## 11. Gate H — downstream semantic validation

> Does replacing the official renderer with stable-envs materially change world-model / control success?

### 11.1 Design, frozen

Paired, everything held fixed but the renderer:

```text
same checkpoint · same episodes · same seeds · same start steps · same goal offsets
same planner seed · same reference physics · official renderer  vs  stable-envs renderer
```

Plus a **mandatory visually-destructive control arm** (blank frames, or an equivalent that destroys
the image while leaving the pipeline intact).

### 11.2 Acceptance rule, frozen before any run

This is an **equivalence** test. A non-significant p-value is not evidence of equivalence and is not
accepted as one.

Let `b` = episodes won only by the official arm, `c` = won only by ours, `n` = paired episodes,
`Δ = (c − b)/n` in percentage points.

| requirement | bound |
|---|---|
| paired episodes | `n ≥ 300` (≥ 6 seeds × ≥ 50 episodes) |
| **equivalence** | the two-sided 95 % CI on `Δ`, `Δ ± 1.96·√(b+c)/n`, lies **entirely within ±5 points** |
| reported alongside | McNemar exact two-sided *p*, Wilson 95 % CI on each arm's rate, `n`, seeds, `b`, `c` |
| **control arm** | control `Δ ≤ −20` points **and** McNemar exact `p < 0.01` |
| power guard | if discordance `(b+c)/n > 0.19`, the ±5 CI cannot close at `n = 300`; extend to `n ≥ 3.8416 · (b+c)/n / 0.0025` before judging |

If the control arm does not degrade, the outcome is **`INCONCLUSIVE`**, never `PASS`. An evaluation
that cannot tell a blank frame from a real one has not tested the renderer.

### 11.3 Why ±5 points

Two independent reasons, both from committed data:

1. **It is well inside the noise the evaluation already carries.** Seed-to-seed spread of the
   *official* PushT arm alone is 80–96 % — a 16-point range. A renderer-swap effect the study is
   asked to exclude should be small relative to that, and 5 points is under a third of it.
2. **The test has the resolution to fail.** At `n = 300` and the observed discordance of 23/300, the
   95 % half-width is **3.13 points**, comfortably inside ±5, so a genuine 5-point degradation would
   be detected rather than absorbed. The margin is not so wide that it passes automatically.

Applied to the existing PushT result: `b = 12, c = 11, n = 300` → `Δ = −0.33`, 95 % CI
**[−3.47, +2.80]** ⊂ ±5 → equivalence **PASS**; control `Δ = −84.0`, `p = 2.8e-14` → control **PASS**.
The rule certifies the evidence the project already has, which is the minimum test of a rule that
was written afterwards.

---

## 12. Per-backend, environment-level and project-level "done"

Three different claims, and conflating them is how "CPU is a first-class target" quietly becomes
"CPU is a fallback we did not get to". They are separate states with separate names.

### 12.1 Per-backend: `BACKEND_CERTIFIED(device)` and `BACKEND_CERTIFIED_PARTIAL(device)`

A **single device path** reaches a certified state when all of the following hold *for that device*:

```text
renderer exists
every mandatory Gate A check passes, and every applicable family-specific one
Gate F SWM render integration passes
performance on THIS device has been measured over the full required sweep
this device's tier is GPU_FAST / CPU_FAST or above
BATCH_SCALES and NO_LARGE_BATCH_REGRESSION hold          (GPU only)
```

**Which** certified state depends on one further fact, and only on that fact:

| registered-default render coverage | state |
|---|---|
| **full** — the renderer draws the whole registered SWM default observation | `BACKEND_CERTIFIED(device)` |
| **`partial`** — a *documented and measured* proper subset (§4.2) | `BACKEND_CERTIFIED_PARTIAL(device)` |

Both states are **parameterised by device** and written that way in every report:
`BACKEND_CERTIFIED(gpu)`, `BACKEND_CERTIFIED_PARTIAL(cpu)`. Certifying one device says nothing about
the other.

`BACKEND_CERTIFIED_PARTIAL` is a real, useful state and not a euphemism for failure. A partial
renderer is fully benchmarkable, fully conformance-tested, usable in the SWM workflow, and may carry
any performance tier including `GPU_EXCEPTIONAL`. What it may not do is claim the env is finished
while part of the picture the registered default produces is still not drawn.

Gates A and F are shared, so they are proved once and count for both devices. Everything downstream
of `PERFORMANCE_MEASURED` is per-device, because that is where the two paths genuinely differ.

### 12.2 Environment-level: `ENVIRONMENT_DONE`

An environment is **done** when the registered default is fully covered **and both** device paths are
certified:

```text
registered-default render coverage == full
BACKEND_CERTIFIED(cpu)  AND  BACKEND_CERTIFIED(gpu)
   i.e.  CPU tier >= CPU_FAST   AND   GPU tier >= GPU_FAST
```

Two exemptions that a "done" label invites, and neither exists:

**No advertised-target exemption.** An env does not become done by declining to advertise CPU; CPU is
a first-class target, so an env with `CPU_REGRESSION` is not finished, it is half-finished, and the
state model has to say so. The only relief available is the honest one — the `NO_REFERENCE` fallback
of §5.5 / §6.8 when no official renderer exists to form a ratio against, which changes the *metric*,
not the bar.

**No partial-coverage exemption.** This is the invariant, and it is not negotiable by any combination
of the other badges:

> **Gate A = `partial` can never produce `ENVIRONMENT_DONE`.**

`BACKEND_CERTIFIED_PARTIAL(cpu) + BACKEND_CERTIFIED_PARTIAL(gpu)` does **not** compose into
`ENVIRONMENT_DONE`; it composes into "this env is fast on both devices and still does not draw the
whole frame". Concretely, `swm/CraftaxClassicPixels-v1` renders the world layer of the 63×63
observation and declares three exclusions — the two-row **inventory strip**, **night** frames, and
the **sleeping-player** desaturation. Every one of those is part of what the registered default
actually shows an agent. No throughput number retires them.

`PERFORMANCE_MEASURED` remains a separate rung, and a *failing* tier still satisfies it. Measuring a
regression is progress; it is simply not completion. An env therefore sits in one of four legible
places rather than one ambiguous one:

| state | meaning |
|---|---|
| `PERFORMANCE_MEASURED` | both sweeps run, tiers assigned, at least one below its bar |
| `BACKEND_CERTIFIED_PARTIAL(gpu)` / `…(cpu)` | that device path is finished, on a documented subset of the frame |
| `BACKEND_CERTIFIED(gpu)` / `…(cpu)` | that device path is finished, on the whole registered default |
| `ENVIRONMENT_DONE` | full coverage, and both devices certified |

Environment-level completion **does not** require a dedicated DINO-WM campaign for that env. 34 such
studies would be expensive and would add little over a representative design.

### 12.2.1 What full coverage adds up to

The invariant exists because the project has a countable coverage target, and a `partial` row that
could reach `ENVIRONMENT_DONE` would let that count be reached without the work being done:

```text
33  registered defaults fully covered today
 1  CraftaxClassicPixels, once the inventory strip, night and sleeping-player are drawn
 6  true renderer gaps, once implemented
--
40  of 40 pixel-producing swm ids
```

The 2 symbolic ids are outside that denominator entirely and terminate at
`NOT_APPLICABLE_SYMBOLIC` — they produce no frame, so there is nothing for a renderer to cover.
40/40 is therefore the whole of the target, and every row in it must be `full`, never `partial`.

### 12.3 Project-level: semantically certified

The stronger claim —

> stable-envs preserves downstream world-model / control utility

— is licensed only when the representative suite of §13 passes under the §11 rule. It is a claim
about the *project*, and individual envs inherit it as `DOWNSTREAM_INHERITED`, which is explicitly
weaker than `DOWNSTREAM_DIRECT` and is never reported as if it were the same evidence.

---

## 13. Representative downstream gates

**Adopted: PushT + Reacher**, with one conditional third arm.

### PushT — `swm/PushT-v1`

2-D `scene` backend (class R), `parity` tier, `approximate_fast_step` physics available as a separate
mode, contact-heavy control, `VariationMap` complete. It already carries the only
`downstream_policy_or_worldmodel` evidence in the matrix, and its evaluation is *demonstrated*
pixel-sensitive by the blank-frame control. It is the positive template, and re-running it under this
spec costs nothing new.

### Reacher — `swm/ReacherDMControl-v0`

The complement on every axis that matters: class S (`mjwarp` + `raycast3d`), 3-D MuJoCo scene,
`reference_only` physics, `parity` tier, visually grounded control.

Three Step 1 findings make it a better choice than it looks:

- the registered default is **`reacher-hard`, not `reacher-easy`** — resolved by measurement against
  the suite. A study run against `easy` would be testing the wrong env;
- the wrapper's 3 tasks (`easy`, `hard`, `qpos_match`) collapse to **2 render-distinct
  configurations**, and stable-envs covers both — so the pair can be used as a robustness check
  across a render-changing parameter;
- it measures `cpu_frac = 2.95`, i.e. **`HOST_BOUND`** — which makes it the right env to keep the
  performance methodology honest, not a reason to avoid it.

### Conditional third arm — a `visual`-tier env

Neither PushT nor Reacher is `visual` tier, so the pair cannot license a semantic claim about visual
renderers. **If** any `visual`-tier renderer is to be advertised as downstream-certified, one
`visual` arm must pass first; the recommended env is `swm/WalkerDMControl-v0` (`DMCWalkerWalk`,
class S, `A = 0.115` against the blank control). Until then, `visual`-tier envs may be
`ENVIRONMENT_DONE` but do not inherit the project-level semantic claim.

---

## 14. The certification ladder

Sequential rungs — each requires the one above. The ladder **forks by device** at
`PERFORMANCE_MEASURED` and rejoins only at the top:

```text
RENDERER_EXISTS
    ↓  every mandatory Gate A check
CONFORMANCE_VALIDATED
    ↓  Gate F
SWM_RENDER_COMPATIBLE
    ↓  both required sweeps run, both tiers assigned (a failing tier still counts)
PERFORMANCE_MEASURED
    ├─↓  CPU tier >= CPU_FAST                      ├─↓  GPU tier >= GPU_FAST
    │                                              │    + BATCH_SCALES
    │                                              │    + NO_LARGE_BATCH_REGRESSION
BACKEND_CERTIFIED*(cpu)                       BACKEND_CERTIFIED*(gpu)
    └──────────────────────┬───────────────────────┘
                           ↓  both, AND registered-default coverage == full
                    ENVIRONMENT_DONE

  * = BACKEND_CERTIFIED(device)          if coverage is full
      BACKEND_CERTIFIED_PARTIAL(device)  if coverage is a documented subset -- terminal,
                                         no path from here to ENVIRONMENT_DONE
```

Independent badges, attached in any combination:

```text
CPU_REGRESSION | CPU_PARITY | CPU_FASTER | CPU_FAST | CPU_EXCEPTIONAL | CPU_UNMEASURED
GPU_REGRESSION | GPU_PARITY | GPU_FASTER | GPU_FAST | GPU_EXCEPTIONAL | GPU_UNMEASURED
HOST_BOUND | BELOW_G4 | NO_REFERENCE | COVERAGE_FULL | COVERAGE_PARTIAL
VARIATIONS_NONE | VARIATIONS_UNMAPPED | VARIATIONS_PARTIAL | VARIATIONS_COMPLETE
PHYSICS_REFERENCE_ONLY | PHYSICS_EXACT | PHYSICS_NEAR_EXACT | PHYSICS_APPROX_VALIDATED
DOWNSTREAM_DIRECT | DOWNSTREAM_INHERITED
TIER_PARITY | TIER_VISUAL
```

There is deliberately **no single overall score.** This is a legal state and must render without
comment:

```text
BACKEND_CERTIFIED(gpu) · GPU_FAST · CPU_REGRESSION · VARIATIONS_UNMAPPED · PHYSICS_REFERENCE_ONLY
```

That env has a finished GPU path, an unfinished CPU path, no variation mapping and no fast physics —
and it is **not** `ENVIRONMENT_DONE`. Every one of those five facts is separately true, separately
actionable, and none of them is inferred from another. What the state model must never do is let the
`GPU_FAST` badge stand in for the missing `CPU_FAST` one.

Envs that cannot enter the ladder carry a terminal state instead: `RENDERER_ABSENT` (the 6 Step 1
renderer gaps) or `NOT_APPLICABLE_SYMBOLIC` (the 2 symbolic ids).

---

## 15. Re-certification after optimization

Aggressive simplification is expected. These rules say what must be re-run, and they are the reason
§10's tripwire exists.

### 15.1 Pure implementation optimization

Kernel fusion, memory layout, removing Python loops, launch reduction, vectorization — anything where
the rendered output is unchanged.

```text
required:  full Gate A conformance rerun  +  full performance rerun (both devices, full batch sweep)
not required: downstream rerun, PROVIDED the outputs are bit-identical or within the
              existing declared fidelity contract, and §10 is unmoved
```

The cheap proof that outputs are unchanged is the **render-model signature** Step 1 already computes
(sha256 over geom/site types, sizes, colours, materials, body and camera transforms, lights, taken
after `reset()`). If it is unchanged and conformance is green, the picture did not move.

### 15.2 Visual approximation change

Simplified geometry, cheaper shadows, approximate tendons, reduced material complexity, lower-cost
anti-aliasing, camera approximation.

```text
required:  Gate A conformance + fidelity rerun (declared floors re-asserted, and re-declared if lowered)
           §10 representation tripwire
           the §11 representative gate, IF the change touches a certified representative family
```

A change that lowers a declared `parity_interior_min` or a `visual_fidelity` floor is a **tier
change**, not an optimization, and re-enters at §4.3.

### 15.3 Physics approximation change

```text
required:  §7 physics conformance for the NEW class
           task-level validation under §11
```

Certification is **never inherited across a materially different solver.** Moving an env from
`near_exact_fast_step` to `approximate_fast_step` voids `PHYSICS_NEAR_EXACT` immediately, even if
every pixel test still passes.

### 15.4 Configuration changes that silently void Gate A

Changing `default_size`, `default_supersample`, the default style, or the registered SWM default the
env is certified against invalidates conformance and every performance number, because the picture
being timed is no longer the picture that was certified. Step 1 already found that forcing a
supersample instead of resolving it makes 44 classes report a cheaper picture than they ship.

---

## 16. Optional speed / fidelity modes — contract only

Specified now, **implemented later**, so that the contract is not written to match whatever was built.

```text
render_quality = "reference" | "balanced" | "fast"
```

| mode | contract |
|---|---|
| `reference` | the highest-fidelity stable-envs path available for the env. The default. Must meet its declared tier's evidence in §4.3. |
| `balanced` | optional intermediate. Must declare its own fidelity floors and meet §10. |
| `fast` | may simplify appearance aggressively. **May only be advertised as certified for an env or family that has passed the §11 downstream gate under `fast`** — not under `reference`. |

```text
physics_mode = "reference" | "fast"
```

`reference` is the default and users are **never** silently switched to approximate physics. A `fast`
physics mode carries its §7 class in its name in every report, and an env whose `fast` physics is
`approximate_fast_step` cannot present it as the default until §7's task-level validation passes.

Mode is part of a certification's identity: `GPU_FAST` earned under `render_quality="fast"` is
recorded as `GPU_FAST@fast` and does not transfer to `reference`.

---

## 17. Sanity check against real Step 1 rows

Applied on paper, no experiments run. GPU tiers use the H200 measurement with the §6.5 factor
applied, marked *(predicted)* — Step 3 replaces them with measurements.

### `swm/PushT-v1` — the already-validated row

```text
class R · parity · approximate_fast_step · VariationMap complete
R = 2,481 (H200)  ->  484 (predicted 3090)      cores = 153.5 -> 29.9
```

| axis | state |
|---|---|
| ladder | today `SWM_RENDER_COMPATIBLE`. `BACKEND_CERTIFIED(gpu)` needs the predicted `GPU_FASTER` to come in at or above 500; `ENVIRONMENT_DONE` needs that **and** `CPU_FAST`, which is unmeasured |
| GPU | `GPU_FASTER` (predicted) — 484 is just under the class-R `GPU_FAST` floor of 500 |
| CPU | `CPU_UNMEASURED` |
| physics | `PHYSICS_APPROX_VALIDATED` — control 9.5e-05, block median 0.68 px, both inside §7 |
| variations | `VARIATIONS_COMPLETE` |
| downstream | `DOWNSTREAM_DIRECT` — passes §11.2 today |

The flagship env landing one notch below `GPU_FAST` on the prediction is exactly what a pre-registered
threshold is for. Step 3 will settle it, and the bar does not move either way.

### `swm/ReacherDMControl-v0` — parity + reference physics

```text
class S · parity · reference_only · registered default = reacher-hard · 2 of 2 render-distinct configs
R = 137.5 (H200), cpu_frac 2.95  ->  HOST_BOUND, no device scaling applied
cores = 4.68
```

| axis | state |
|---|---|
| GPU | ratio clears the class-S `GPU_EXCEPTIONAL` floor of 40, but `cores = 4.68 < 10` → **`GPU_FAST`**, with `HOST_BOUND` |
| CPU | `CPU_UNMEASURED` |
| physics | `PHYSICS_REFERENCE_ONLY` — full certification, no deficiency |
| variations | `VARIATIONS_UNMAPPED` (9 leaves) — does not block anything |
| downstream | representative arm; `DOWNSTREAM_DIRECT` pending |

### `swm/BallInCupDMControl-v0` — the known-problematic backend

```text
class S · parity · reference_only · task semantics match
R = 12.5 (H200), cpu_frac 1.20     cores = 0.59  <-- the only row in 196 below G4
```

| axis | state |
|---|---|
| GPU | ratio would reach `GPU_FAST`; `cores < 1` caps it at **`GPU_FASTER` + `BELOW_G4`** |
| everything else | Gate A passes, `PHYSICS_REFERENCE_ONLY`, `VARIATIONS_UNMAPPED` |

`BELOW_G4` is the whole point: this renderer is 12.5× the official one *and* it is still cheaper to
read the frame off disk. The scheme states both without contradiction, and flags the env as the first
place optimization should go.

### `swm/HumanoidDMControl-v0` — visual tier, task semantics differ

```text
class S · VISUAL · reference_only
registered default = humanoid-walk;  stable-envs = humanoid-stand
render model after reset() is BYTE-IDENTICAL; only the reward differs, and it reaches no pixel
R = 359 (H200) -> 70 (predicted)     cores = 10.6 -> 2.07
```

| axis | state |
|---|---|
| Gate A / coverage | `registered_default_render_covered = True`, kind `same_scene_reward_differs` |
| standalone task semantics | **`False`** — a separate axis, recorded, and it blocks nothing here |
| GPU | ratio ≥ 40, but predicted `cores = 2.07 < 10` → **`GPU_FAST`** |
| tier | `TIER_VISUAL` → §10 bound is 0.25; measured `A = 0.095` → pass |
| downstream | `ready`, but `visual` tier means it cannot inherit a project-level semantic claim until the §13 conditional arm passes |
| CPU | `render_executes = unknown_dependency_absent_here` → `CPU_UNMEASURED`, **not** `CPU_REGRESSION` |

That last line is the absent-vs-unmeasured rule doing its job.

### `swm/CraftaxClassicPixels-v1` — partial observation coverage, no reference

```text
class R · parity on what it draws · no installable official renderer
R = undefined  ->  NO_REFERENCE     cores = 102.6 -> 20.0 (predicted)
```

| axis | state |
|---|---|
| Gate A | **`partial`** — world layer certified at 100 % of all pixels exact, mean abs 0.000, max 0, over 64 states at both tile sizes; inventory strip, night noise field and sleeping desaturation declared out in the class |
| GPU | `NO_REFERENCE` path: `cores ≥ 10` → **`GPU_FAST`**, qualifier `NO_REFERENCE` |
| CPU | `CPU_UNMEASURED`; it will certify on the §5.5 `NO_REFERENCE` ladder too, since `S_cpu` is as undefined as `R` |
| variations | `VARIATIONS_NONE` — the wrapper exposes no variation space |
| downstream | `ready_on_documented_subset`; may never contribute to a parity claim |
| completion | best reachable state is `BACKEND_CERTIFIED_PARTIAL(cpu)` + `…(gpu)`. **Not** `ENVIRONMENT_DONE`, at any throughput, until the inventory strip, night frames and the sleeping-player desaturation are drawn |

Two mechanisms are exercised here at once, and neither is a fudge: `partial` records a *declared and
measured* subset, and `NO_REFERENCE` prevents inventing a ratio against a renderer that cannot be
installed. The third thing this row tests is the §12.2 invariant — it is the one env in the matrix
that could otherwise have been counted as finished while three declared exclusions were still
outstanding, and it is the 34th of the 40 pixel-producing ids in §12.2.1.

### `swm/OGBMaze-v0` — partial render-parameter coverage

```text
class S · parity · reference_only · one swm id standing for a 15-cell family
registered default (pointmaze-large) covered by PointmazeLargeNavigate / PointmazeLargeStitch
R = 539 (H200) -> 105 (predicted)     cores = 59.4 -> 11.6
```

| axis | state |
|---|---|
| registered default | **covered** — this is the axis certification runs on |
| render parameter space | **`partial`** — 4 of 15 render-distinct configurations unrendered (`antmaze-arena`, `humanoidmaze-arena`, `humanoidmaze-teleport`, `pointmaze-arena`) |
| GPU | `R ≥ 40` and `cores ≥ 10` → **`GPU_EXCEPTIONAL`** |
| downstream | `ready` — the gate uses the registered default, which is covered |

The two coverage axes disagree and both are true. Certification follows the registered default;
`partial` parameter coverage is a published limitation, not a blocker.

### `swm/QuadrupedDMControl-v0` — a true renderer gap

```text
no stable-envs env matched.  gap_reason = existing_research_promising
```

| axis | state |
|---|---|
| ladder | **`RENDERER_ABSENT`** — cannot enter |
| every performance/fidelity axis | `not_applicable`, never `fail` |
| variation space | `present`, 10 leaves — recorded, because it is what the eventual mapping must cover |
| downstream | `blocked_renderer_gap` |

A gap is an absence of evidence and is never scored as a failure. The row still carries the facts
that make the work estimable: `quadruped-walk` passes state sufficiency 23/23 and compiles to
`nmesh = 0`, so the mesh blocker that rules out `dog` does not bind, and it is the only id probed at
`camera_id = 2`.

### What the seven rows prove about the scheme

| requirement | shown by |
|---|---|
| already downstream-validated | PushT |
| parity + reference physics | Reacher, BallInCup |
| known slow/problematic backend | BallInCup (`BELOW_G4`), Reacher (`HOST_BOUND`) |
| visual-tier renderer | Humanoid |
| partial observation coverage | CraftaxClassicPixels |
| partial render-parameter coverage | OGBMaze |
| renderer missing | Quadruped |
| a coherent multi-badge state | Humanoid: covered + task-semantics-differ + visual + unmapped + ready |

---

## 18. What later work must produce

Not part of this specification's rules, but the direct consequence of them. **This is a list of
required evidence, not a project plan** — the sequencing and the step numbers are the coordinator's,
and nothing here should be read as assigning work to a numbered step that has not been briefed.

The one exception is fresh benchmarking, which the Step 2 brief already names as Step 3:

- **Step 3 (benchmarking)** — the CPU and GPU sweeps of §5.2 / §6.2 on the M5 and Oscar's RTX 3090,
  in the two separate tables of §6.1, with `cpu_frac`, thread count and load average on every row;
  and a deterministic `benchmark JSON → certification label` mapping that reads only the constants in
  `scratch/certification.json`.

Unscheduled, in no implied order:

- the §11 downstream gate on PushT and Reacher, with `n`, seeds and the control arm pre-registered
  before the run;
- `VariationMap` work scored on §9, coupled leaves (§9.1) first, since those are the only ones that
  can produce a correct rendering of the wrong world;
- the §13 conditional `visual`-tier arm, if and only if a `visual` renderer is to be advertised as
  downstream-certified;
- closing the six §17 renderer gaps, which re-enter this ladder at `RENDERER_EXISTS`.

Open questions recorded rather than resolved:

1. **The 5.13× device factor is a prediction.** It is derived from memory bandwidth on the stated
   grounds that the fused rasterizer is output-bandwidth-bound, and it is knowingly pessimistic for
   `HOST_BOUND` rows. Step 3's first job is to report the factor it actually measures.
2. **No stable-envs CPU throughput exists anywhere.** §5's tiers are anchored to the G4-on-CPU
   break-even distribution, which is real, but the numerator distribution is genuinely unknown. If
   Step 3 finds the CPU path is uniformly far above or below the ladder, that is a finding about the
   renderer, and the ladder stays put.
3. **`swm/PushT-Discrete-v1`** shares PushT's renderer and its downstream evidence. Whether it
   inherits `DOWNSTREAM_DIRECT` or only `DOWNSTREAM_INHERITED` is deliberately left open until §11 is
   re-run; this spec's default is the weaker one.
