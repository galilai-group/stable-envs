# `check_colors_apply_exactly` on lit renderers — decision packet for Sami

**Step 8, Workstream E (final certification harness and provenance).**
Baseline `43ac95238f7472137619f3e69e7636e11dd088b1`.
**No code change was made to the check.** This packet exists because the honest answer changes a
frozen Step-2 requirement, and section 4 of the Step-8 plan says that is your call, not the
coordinator's.

The Step-7 escalation this closes is recorded verbatim at `docs/research/step7_coverage.md:278` and
in `examples/results/step7/dashboard.yaml` (`escalations`, 2026-09-15, from W6): *"declare not
applicable for lit RGB leaves with the law cited, or extend the shared check to accept the
law-transformed colour (an extension of a Gate-A check = Sami's call) — open (Sami did not rule)."*

---

## 1. What the check does today

`stable_envs/testing/conformance.py:693-715`, unchanged since Step 2:

```python
def check_colors_apply_exactly(env) -> dict:
    layout = {p: (o, w) for p, o, w in env.variation_space.flat_layout()}
    st = env.sample_states(env.B)
    wrong = []
    for path, (off, w) in layout.items():
        node = env.variation_space
        for part in path.split("."):
            node = node[part]
        if w != 3 or not hasattr(node, "high") or int(np.max(node.high)) != 255:
            continue  # only RGB leaves have a checkable answer
        for target in ([17, 33, 200], [200, 17, 33]):  # two, so a channel swap cannot pass both
            sty = np.tile(env.default_style(), (env.B, 1))
            sty[:, off : off + w] = np.asarray(target, np.float32)
            f = env.render(st, style=sty).cpu().numpy()
            if not (np.abs(f.astype(np.int16) - np.asarray(target, np.int16)).sum(-1) == 0).any():
                wrong.append(f"{path}->{target}")
    return {"colors_not_applied": wrong}
```

Precisely:

| property | value |
|---|---|
| subject | every width-3 variation leaf whose `high` maxes at 255 (`RGBBox`) |
| what varies | style only, from `default_style()` with that leaf's 3 slots overwritten |
| targets | `[17, 33, 200]` and `[200, 17, 33]` — a permutation pair, so a channel swap fails one |
| assertion | **existential and bit-exact**: at least ONE pixel in the batch equals the triple in all three channels |
| tolerance | **zero**. No epsilon, no interior mask, no count threshold, no per-row requirement |
| failure expression | the knob's path and the target. **No "how close did it get" number is captured** |
| skip branch | none — an env with no RGB leaf returns `[]`, a vacuous pass |

The docstring states the intent exactly, and the intent is not in dispute:

> *"`check_variations_do_something` only asks whether a knob moves pixels; a knob wired to the wrong
> primitive, or scaled, or with its channels swapped, passes that and is still wrong. This asks
> whether the requested colour is literally present."*

**Consumers.** One test, `tests/gpu/test_conformance.py:253` (`tests/` was frozen in Step 7). One
judge, `scratch/step4/certify.py:164`, bound string *"an RGB knob puts that colour in the frame"*.
`examples/fidelity_step4.py:295` records it `not_applicable` only when the env declares an empty
variation space.

**There is no opt-out mechanism.** Grepping `stable_envs/envs/*.py` for a per-check applicability key
finds exactly one precedent, and it is for a different check: `resolution_consistency_applicability`
on `craftax.py:865` and `craftax_full.py:835`.

---

## 2. What Step 2 froze

Colour exactness is named, by function name, in exactly one frozen place —
`scratch/certification.json:337-352`, `"variations"`:

```json
"complete_requires": [
  "every SWM leaf classified as identical / renamed / scaled / swm_only",
  "stable-envs-only leaves listed explicitly",
  "check_variations_do_something and check_colors_apply_exactly pass on every visual leaf",
  "check_per_env_variation passes"
]
```

Prose twin, `docs/research/certification.md:529-533`. Two facts matter and they pull in opposite
directions:

1. **The frozen text names the check with no tolerance, no law-awareness and no escape hatch.** Any
   change to what the check asserts touches a frozen Step-2 requirement.
2. **But that requirement is `VARIATIONS_COMPLETE`, not Gate A.** `"blocks_other_gates": false`, and
   `docs/research/certification.md:516` says outright: *"A missing `VariationMap` never blocks Gate
   A, D, F or H."*

In the Gate-A taxonomy the check sits in the **family-specific** table
(`docs/research/certification.md:149`), under *"mandatory where the env declares the capability,
`not_applicable` otherwise"* (`:143`). It is **not** on the 14-row mandatory list.

### A live inconsistency worth fixing whichever way you rule

The Step-7 Gate-A harness records family-specific checks without judging them
(`scratch/step7/closure/gate_a_new_cells.py:13-20`). So the Rocket row reads `"result": "PASS"`,
`"failing_checks": []` while carrying ten `colors_not_applied` entries, and its note asserts the
failure is *"neither a Gate-A check"*
(`examples/results/step7/closure2/coverage_matrix.json:2747`). That phrase contradicts
`certification.md:143,149`, where a family-specific check **is** part of Gate A once applicable.
Today the cell is simultaneously recorded as passing Gate A and as failing a Gate-A check. That is a
reporting defect independent of the invariant question, and §6 proposes the fix.

---

## 3. Official renderer behaviour under its lighting law

### 3.1 Rocket — bullet3 TinyRenderer, a **multiplicative** law

Measured and confirmed empirically by W6
(`examples/results/step7/rocket-landing/findings.md:1209-1217`), from
`TinyRenderer.cpp`'s `Shader::fragment`:

```text
out_i = int( 255 · c_i · ( A + shadow·( D·max(0, n·l) + S·max(0, r_z)^spec ) ) · L_i )
```

clamped at 255, **truncated not rounded**, defaults `A = 0.6`, `D = 0.35`, `S = 0.05`,
`spec = 2`, `L = (1,1,1)`, shadow off, light direction the normalised world-fixed `(-50, 30, 100)`.
No gamma. **The specular multiplies the material colour** — unlike MuJoCo's additive white.

**I re-derived the ceiling independently rather than quoting it.** Maximising
`f(n) = 0.6 + 0.35·max(0, n·l) + 0.05·max(0, r_z)²` over a 2001 × 4001 grid of unit normals, with
`r = 2(n·l)n − l` and `l` the normalised `(-50, 30, 100)`:

```text
max f = 0.993567   at n ≈ (-0.3257, 0.1954, 0.9251),  n·l = 0.9904,  r_z = 0.9686
```

This reproduces W6's figure to six decimal places. Note the maximum is **not** at `n = l`, where
`f = 0.987313` — it is slightly off-axis, where the diffuse and specular terms trade off. The
consequence for the check's two probe targets:

| requested channel | brightest the reference can emit | reachable? |
|---|---|---|
| 200 | `int(200 · 0.993567)` = **198** | no |
| 33 | **32** | no |
| 17 | **16** | no |

**`f < 1` for every normal, at every pixel, for any correct implementation of this reference.**
The exact RGB triple can appear only where the whole bracket equals 1, and it never does.

Corroborating empirical constants from the same findings (`:1218-1220`): ambient-only body pixels
equal `int(255·c)` exactly; the darkest default body pixel equals `int(0.6·255·c)`; the brightest is
`0.991·int(255·c)`.

**The official PyFlyt frame fails this check too.** Phase 0 measured it directly: the brightest body
pixel is `0.991 · int(255·c)` (`findings.md:251-257`). A check that the reference renderer itself
cannot pass is not measuring our renderer.

**And the wiring is independently known to be correct.** `review_phase1.md:885-894` Appendix 1: the
cell's colour table matches `getVisualShapeData` to **1.19e-08** on every geom. The five knobs are
wired to the right primitives, unscaled, unswapped — which is the entire property this check exists
to establish.

### 3.2 MuJoCo — an **additive** law that clamps at 1

The contrast is the reason the other lit cells pass. MuJoCo's specular term adds white rather than
multiplying the material colour, so the bracket reaches and clamps at 1.0 and `int(255·c·1) = int(255·c)`
appears wherever a surface is lit to saturation (`review_phase1.md:891-892`;
`findings.md:1211-1212`).

Worth stating plainly: **no separate analytic MuJoCo derivation artifact exists.** The support here
is empirical, and it is broad — every MuJoCo-backed cell in the corpus records
`colors_not_applied: []`. If you want the law written down as carefully as TinyRenderer's before
ruling, that is a small, bounded measurement and I did not do it.

---

## 4. stable-envs behaviour — the whole corpus

From the most recent committed artifacts, `examples/results/step7/closure2/`:

| outcome | cells |
|---|---|
| **FAIL** | **`PFRocketLanding` only** — 10 entries (`pad.color`, `rocket.body_color`, `rocket.booster_color`, `rocket.fin_color`, `rocket.leg_color`, each × both targets), `closure2/gate_a/cuda_rocket.json` |
| PASS | `CraftaxClassic`, `CraftaxFull`, `DMCManipulatorBringBall`, `DMCQuadrupedWalk`, 14 DMC, 4 DMC-visual, 12 Fetch, 11 OGBench maze, 6 raster/classic-control, PushT, 5 SWM — 47 configs `status: pass` |
| NOT_APPLICABLE | `OGBCube`, `OGBScene` — *"the env declares an empty variation space"* |

Historical baseline: `docs/research/step4_fidelity_baseline.md:195` — **49 pass, 0 fail**.
**PFRocketLanding is the first and only failure this check has ever recorded.**

### The corpus already routes around the check, nine times, in prose

This is the strongest single piece of evidence, and it is easy to miss. Nine env modules explain *in
their own docstrings* that they deliberately declare **no RGB leaf** because their output cannot
contain a literal requested colour:

| env | file:line | mechanism |
|---|---|---|
| `CarRacing` | `carracing.py:62-66` | a 10.4× box downscale: a flat region of value `v` comes out at `v−2` |
| `Crafter` | `crafter.py:445-451` | `daylight == 1` at one step in 300, so a world colour is essentially never painted literally |
| `Procgen` | `procgen.py:899-901` | knobs are gains, not RGB: *"an RGB leaf here would fail `check_colors_apply_exactly` for the right reason"* |
| `PGX` | `pgx.py:568-575` | the check *"correctly refused"* the outline stroke on three games |
| `Jumanji` | `jumanji.py:805-808` | palette is `Box (V,3)`, not `RGBBox`, so the check skips it |
| `SimplePointMaze` | `simple_point_maze.py:57-62` | `show_goal=False` registered because a 128/255-composited disc can never satisfy it |
| `Sokoban` | `sokoban.py:304-310` | ten RGB leaves precisely *because* `room_to_rgb` composites nothing |
| `Overcooked` | `overcooked.py:428-436` | per-colour leaves because a 3×3 downsample of same-label subpixels lands exactly |
| `swm_nav` | `swm_nav.py:34-39` | passes only because Gaussian normalisation makes the argmax pixel exactly `c` |

Read together these say something specific: **the check's real subject is not "is the knob wired
correctly" but "is this renderer's output path colour-preserving"** — and nine authors have already
had to answer that by choosing a declaration shape rather than by passing or failing. The Rocket is
the first cell where the honest answer was to declare the leaf anyway, because the knob *is* an RGB
knob in the reference's own API; it is the transport that is lossy.

The one written proposal on record, `docs/research/swm_coverage.md:249-258`, reached the same place
from the compositing side: *"this is the one place where a shared check, rather than the criterion or
our renderer, is what stands in the way"*, and proposed a class-level `translucent_color_paths`.

---

## 5. The three candidate invariants

### (a) Literal RGB — the status quo

> *the requested triple appears verbatim in at least one pixel.*

**Keeps:** maximum strength where it holds; zero implementation risk; no frozen text changes.
**Costs:** it is **unsatisfiable by any correct implementation** of a multiplicative-law reference.
It fails the official frames too. It cannot distinguish "wrong wiring" from "lit renderer", which is
the one distinction it was written to make. Under it, `PFRocketLanding` can never clear
`VARIATIONS_COMPLETE`, and the only ways to make it "pass" are to delete five correctly-wired,
semantically-required SWM leaves or to unlight the renderer — both of which make the library less
truthful. That is the *"weaken fidelity to make a cell pass"* failure mode inverted: here the check
pushes toward deleting a true declaration.

### (b) Law-transformed RGB

> *the requested triple appears verbatim, OR the renderer's response to the knob matches the
> OFFICIAL renderer's response to the same knob.*

**Keeps:** the full discriminating power. A wrong primitive, a scale error and a channel swap all
still fail, because the official frame's response differs from ours in all three cases. It is
strictly stronger than (a) for lit cells, since it compares against a measured reference rather than
a constant.
**Costs:** it needs the official renderer available at check time, which a CPU-only or
reference-free interpreter will not have — so it needs a `blocked_environment` path, exactly like the
parity checks already have (`stable_envs/testing/provenance.block_on_reference_mismatch`). It is a
genuine extension of a Gate-A check. It is more code in a policy-frozen area.
A cheaper variant — *assert `int(255·c·f_max)` is reached, with `f_max` read from a class-level
declared shading law* — avoids needing the reference but makes the harness trust a declared constant,
which is a weaker provenance than a measurement.

### (c) Not applicable for lit RGB leaves, with the law cited

> *a leaf whose transport applies a documented lighting law that cannot reach unit gain is recorded
> `not_applicable` with the law and its measured ceiling, and `check_variations_do_something` plus the
> under-variation fidelity numbers carry the wiring claim instead.*

**Keeps:** it is the existing precedent, in two forms — the nine envs above that route around the
check by declaration, and `resolution_consistency_applicability`, which is *the same shape of
problem* (a relation the official renderer does not itself satisfy) and which Sami already ruled on,
on 2026-09-15, in favour of declaring inapplicability with the oracle cited. It needs no reference at
check time. It is small.
**Costs:** it is strictly weaker than (b). `check_variations_do_something` alone would pass a
channel swap. It needs a real applicability declaration with a *measured* ceiling, not a free-text
opt-out, or it becomes a way to silence the check.

---

## 6. Recommendation

**Recommended: (c) now, with (b) as the stated successor — and the two must land together with the
Gate-A reporting fix.**

Reasoning, in the order that decided it:

1. **The premise, not the renderer, is what is wrong.** The check assumes flat shading. On a
   multiplicative-law reference the exact RGB is mathematically out of reach — proved here to six
   decimals, and true of the official frames. Keeping (a) would record a permanent FAIL whose cause
   is the reference's shading model. Step 8's decision taxonomy has a bucket for exactly that, and it
   is not `RELEASE_BLOCKING`.
2. **The precedent is yours and it is recent.** On 2026-09-15 you ruled on CraftaxClassic's
   resolution consistency: *"If the official is itself not resolution-consistent under night … mark
   `check_resolution_consistency` not applicable for the night band with that result as
   justification … No weakened or invented bound."* The colour case is the same shape — a relation
   the reference does not satisfy — with the same evidence standard available. (c) applies your own
   rule; (b) goes further than it.
3. **The wiring claim does not go uncovered.** The Rocket's knobs are already established correct by
   measurement against `getVisualShapeData` to 1.19e-08 on every geom, and by
   `check_variations_do_something` and the under-variation fidelity numbers. What (c) gives up is a
   *redundant* assertion on this cell, not the only one.
4. **(b) is better and should be built — but not under time pressure and not as a side effect.** It
   is a genuine extension of a Gate-A check in policy-frozen `tests/gpu/`, it needs a reference-
   availability path, and it deserves its own review. Recommending (c) *and* naming (b) as the
   successor is the honest split: ship the truthful record now, build the stronger invariant
   deliberately.

**Three conditions, without which (c) is a loophole:**

- The applicability declaration must carry the **measured ceiling** (`max_shading_factor: 0.993567`)
  and the artifact that measured it — the same discipline `DOWNSAMPLE_EXACT_FRAC` is held to. A
  free-text opt-out is not acceptable.
- The check must **record how close it got** (`best_abs_diff` per leaf), so a wiring regression that
  moves a knob off its primitive still shows as a number moving even while the leaf is
  `not_applicable`. Today the check captures nothing but the knob's name, which is why this question
  took a bespoke probe to answer.
- **Fix the Gate-A reporting inconsistency in the same change** (§2): a family-specific check on an
  applicable cell either is part of Gate A or is not, and right now the Rocket row says both.

**What is NOT recommended, explicitly:** relaxing the check globally, adding a tolerance to it,
deleting the Rocket's five RGB leaves, or leaving the current contradictory record standing.

---

## 7. Decision requested

| # | question | recommendation |
|---|---|---|
| 1 | Adopt **(c)** — declared, measurement-backed inapplicability for lit RGB leaves — as the Step-8 answer? | **yes** |
| 2 | Commission **(b)** — the law-aware/reference-compared form — as the successor invariant, on its own review? | **yes**, not in Step 8 |
| 3 | The frozen `VARIATIONS_COMPLETE` clause in `scratch/certification.json` names this check with no applicability escape. Adding one edits frozen Step-2 text. Approve that edit? | **yes**, scoped to applicability only — no tolerance, no threshold |
| 4 | Fix the Gate-A record so a family-specific check on an applicable cell is judged consistently, and re-state the Rocket row? | **yes** |
| 5 | Want the MuJoCo additive law written down analytically (currently empirical only) before ruling? | your call; ~1 bounded measurement |

Until you rule, the state is unchanged and truthful: `PFRocketLanding` records
`check_colors_apply_exactly` as **FAIL with the mechanism attached**, and this workstream changed no
line of the check.
