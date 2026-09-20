# Step 7 W4 `dmc-quadruped` — independent review, Phase 1

Reviewer: independent (did not write the code). Read-only against
`/Users/samib/stable-envs-step7-dmc-quadruped` @ `9188c2c5c40ad73f4ea41a05ca014ba0ab77d2a0`
(branch `step7/dmc-quadruped`), baseline `76dc5116f432e61dd47fba8f9e879cf9212c8e46`.
Environment: `/Users/samib/se6-venv/bin/python` (torch 2.14 CPU, mujoco 3.12.0, dm_control 1.0.45),
`MUJOCO_GL=glfw`, Apple M5. Oscar not attempted (denied). Nothing in the worktree was modified;
a detached baseline worktree was created under the scratchpad and removed at the end.

## VERDICT: APPROVE-WITH-CHANGES

Every substantive claim I could test locally reproduced, several bit-for-bit. The ellipsoid
math is correct against an independent closed form; the regression claim is real; Gate A
reproduces exactly at HEAD; the SWM transcription is correct against the actual SWM source.
The required changes are three small documentation/hygiene defects and one real (minor) resource
leak. None of them touches a number.

---

## 1. Ellipsoid math — VERIFIED CORRECT

`stable_envs/render/raycast3d.py::_ellipsoid` (lines 304–341).

Checked against an **independent** reference (`numpy.roots` on the float64 quadratic in the geom
frame, nearest positive real root, gradient normal) — deliberately not the author's
`1/size^2`-weighted rearrangement.

```
scratchpad/ell_check2.py  (6000 rotated anisotropic scenes: aimed-at-surface, origin-INSIDE,
                           pointing-AWAY; float64)
aimed fuzz (float64): hits=3540 inside_cases=1978 hit/miss mismatches=0
                      rel_t_max=1.734e-13  n_max=3.910e-11
```

```
scratchpad/ell_check.py   (2000 random rotated fp32 scenes through the vectorised path)
unit-normal max dev on hits: 4.77e-07        (1034 hits)
surface residual max on hits (fp32): 2.24e-03
outside rays with n.d >= 0 (should be 0): 0 of 1031
non-unit d: hit-set equal True, max rel |t*k - t| = 6.1e-05
bound violations (|p-pos| > max(size) + 1e-5): 0 of 1034
```

- **Rotated frame**: `_mtv`/`_mv` are used exactly as `_capsule` does; the 90°-about-z permutation
  case and the random-quaternion fuzz both land on the written-down answer.
- **Unit-scale reduction**: confirmed algebraically and numerically identical to the reference
  quadratic.
- **Nearest positive root**: `t = (-b-sq)/a; t = where(t>0, t, (-b+sq)/a)` — same spelling and same
  convention as `_sphere`. 3540 fuzz hits, zero root-selection disagreements.
- **Inside-origin**: `c < 0 ⇒ sq > |b| ⇒ t_near < 0` always, so the far root is taken. Verified
  directly (`t=0.2566928232003637` vs reference `0.25669282320036374`, on-surface residual 2.2e-16)
  and over 1978 inside cases in the fuzz.
- **Normal**: gradient `p_l / size²`, normalised, rotated by `_mv`. Unit to 4.8e-7 in fp32; outward
  on every one of 1031 outside-origin hits; outward (i.e. *not* flipped toward the ray) at the exit
  point for inside-origin rays — matching `_sphere`, which also does not flip.
- **Degenerate semi-axis**: `hit` requires all three `size > 0`, so a zero axis is a miss and the
  `clamp(min=1e-12)` never divides by zero. No NaN (test + my own run).
- **Tangent**: `disc > 0` strictly, so exact/near tangents miss — identical convention to `_sphere`.
  Measured: offset `+1e-6` from tangent → MISS, `-1e-6` → hit at t=1.99905.

### Triton twin — textually equivalent; NOT EXECUTED (no Triton locally)

Read side by side with the torch path. Frame change (`olx0 = r00*ax + r10*ay + r20*az`) and normal
rotation (`nx = r00*nlx + r01*nly + r02*nlz`) match the CAPSULE/BOX branches' conventions exactly;
`div_rn`/`sqrt_rn` are used everywhere torch uses IEEE `/` and `sqrt`; the association order of
`a`, `b`, `c`, `disc`, the gradient and its norm is identical to the torch spelling; the miss
sentinel and the `min(t, 1e6)` clamp before the surface point match. The branch is inserted as
`elif kind == 4:` between CAPSULE/CYLINDER and BOX, so the `else: # PLANE` fallback and every
existing kind are untouched; it sets exactly the `t, nx, ny, nz` the kernel returns.
**I could not run `tests/gpu/test_fast_paths.py::test_the_ellipsoid_twins_are_bit_identical_on_both_casts`**
— bit-identity of the twin remains UNVERIFIED and must run on Oscar before certification. The
author says the same in findings §P4.

### Bounds — conservative, and other kinds unchanged

- `Primitives.bound_radius` ELLIPSOID branch = `float(s[i].max())`. The ellipsoid is contained in
  the sphere of radius max(a,b,c) about the geom origin. Measured: 0 violations in 1034 hits.
- `_bound_radius(ELLIPSOID)` = elementwise `max`, same value.
- `_BOUNDED_KINDS` is **unchanged** (`{CAPSULE, CYLINDER, BOX}`) — ELLIPSOID is therefore always
  tested, which is conservative (slower, never wrong). `_kind_is_bounded` and `COVER_COMPACTION`
  behaviour for every other kind is byte-for-byte unchanged (no diff hunk touches them).
  Consequence: the torch `_bound_radius` ELLIPSOID branch is presently unreachable from the casts;
  it is pinned by the new test, so it is defensive rather than dead-and-untested. The author
  correctly records that admission is a measurement (`r5/r6_kind_cost`), pending on Oscar.

---

## 2. Regression / byte identity — REPRODUCED (14/14)

Independent harness (my own, not the author's `replay_step4.py`): rendered the first 16 frozen
Step-4 states of each DMC config on CPU in a pristine `76dc511` detached worktree and in the branch
worktree, sha256 of the raw uint8 buffers.

```
byte-identical: 14/14  (16 frames each, CPU, 84 px)
DMCAcrobotSwingup DMCBallInCupCatch DMCCartpoleBalance DMCCartpoleSwingup DMCCheetahRun
DMCFingerSpin DMCFingerTurnEasy DMCFingerTurnHard DMCHopperHop DMCHumanoidStand
DMCPendulumSwingup DMCReacherEasy DMCReacherHard DMCWalkerWalk
```
Zero differing pixels anywhere; the author's `phase1/replay/cpu_branch_vs_baseline_identity.json`
(64 frames, 451,584 px/config, `all_identical: true`) is corroborated.

**Not reproducible locally:** the 11 OGBench maze cells (`ogbench` is not installed in
`/Users/samib/se6-venv` — `ModuleNotFoundError`, both trees). `stable_envs/envs/ogbench.py` is the
only other `raycast3d` consumer. I accept the author's argument by inspection: the raycast3d diff
adds only (a) a new `ELLIPSOID = 4` constant, (b) a new `elif kind == ELLIPSOID` arm in
`intersect()` placed after all existing arms, (c) a new `elif kind == 4` arm in the Triton kernel
placed before the PLANE `else`, (d) a new `elif` in `Primitives.bound_radius` / `_bound_radius`,
(e) one extra code in the `_kind_tensor` error allow-list. No existing kind's code path changes.
Still, CPU identity on the maze cells is an Oscar item (`certify_cpu.sbatch` step 1) and should not
be waived.

---

## 3. Test suites — NO NEW FAILURES; the baseline failure is identical

```
branch   tests/unit -q --deselect tests/unit/test_examples.py
         -> 1 failed, 734 passed, 6 skipped, 111 deselected
baseline tests/unit -q --deselect tests/unit/test_examples.py
         -> 1 failed, 713 passed, 6 skipped, 111 deselected
```
The single failure is `tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves`
on **both** trees, with the **same 9 unresolvable paths** (`scratch/carracing/logs/`,
`scratch/robocasa/out/corpus`, `scratch/sboughan/…`, `scratch/step6/gfxdraw/cost_floor.{py,cpp`).
The branch only adds `findings.md` as an extra *citation site* for three of the already-broken
paths; it introduces no new broken path. The author's "734 passed" claim is exact.

Targeted subsets (`-k "raycast3d or cpu3d or dmcontrol or declarations"`):
branch 238 passed / 1 skipped, baseline 217 passed / 1 skipped, 0 failures either side.

---

## 4. `dmcontrol.py`

### Hunk confinement — CONFIRMED
`git diff -U0 76dc5116..HEAD -- stable_envs/envs/dmcontrol.py` hunk headers:
`@@ -279`, `-2241`, `-2250`, `-2476`, `-2499`, `-2562`, `-2578`, `-2623`, `-2629`.
Baseline region map: `_DMCModel` 863–1209, `DMControlSuite` **1212–2064**, MJWarp TrackCom 2065–,
`DMControlTrackCom` 2169–, module tail from 2476. **No hunk falls inside 863–2064**, so W3's
`DMControlSuite` region and the shared `_DMCModel` body are untouched. The two hunks inside
`_make_classes` (2623, 2629) are inside the *trackcom* loop body only. Textual merge contention with
W3 is limited to the shared `VISUAL_FIDELITY` dict literal and `_make_classes`, both additive.

### `_used_materials` — the test assertion is REAL and I verified it independently
Ran `cls._used_materials()` on both trees and also `_DMCModel._used_materials.__func__(cls)`:

| cell | baseline effective | branch effective | changed |
|---|---|---|---|
| DMCCheetahRun | `grid, self` | `grid, self` | no |
| DMCHopperHop | `grid, self` | `grid, self` | no |
| DMCWalkerWalk | `grid, self` | `grid, self` | no |
| DMCHumanoidStand | `grid, self` | `grid, self` | no |
| DMCSwimmerSwimmer6 | `grid, target, self, effector, eye` | identical | no |
| DMCQuadrupedWalk | (did not exist) | `grid, self, effector` (base census also had `eye`) | n/a |

So the override is a genuine no-op on all five pre-existing trackcom cells; only quadruped loses
`eye`. I confirmed `eye` has **zero geom wearers** and is worn by `pupil_r`/`pupil_l` (group 0,
alpha 1) alone — so the restriction is justified, and the base census would indeed have declared a
knob `check_variations_do_something` could not honour.

### Provisional VISUAL_FIDELITY entry — adequately marked
`_QUADRUPED_AT` says, verbatim, "CPU backend cpu3d with torch 2.14.0 on an Apple M5 … PROVISIONAL
until the MJWarp draws"; `measured_with = {"torch": "2.14.0"}`; `reference_versions = _AT_VERSIONS
= {"mujoco": "3.11.0"}` (the family convention; dm_control 1.0.44 is named in the prose string, as
on the siblings). Threshold margins sit inside the family's existing spread:

| cell | mean ceiling | within8 floor | ssim floor |
|---|---|---|---|
| DMCSwimmerSwimmer6 | 1.391× | 0.9269× | 0.9533× |
| **DMCQuadrupedWalk** | **1.374×** | **0.9313×** | **0.9404×** |
| DMCCheetah/Hopper/Walker/Humanoid | 1.297–1.310× | 0.9286–0.9524× | 0.9322–0.9447× |

The 1.374× ceiling is looser than the stated "~1.3×" convention but is bracketed by swimmer's
1.391×, so it is within precedent. Not a required change.

### SWM default — VERIFIED against the real SWM source
`stable_worldmodel` is **not installed** in `/Users/samib/se6-venv`, but the source checkout is at
`/Users/samib/stable-worldmodel` (`176d545`). Read
`stable_worldmodel/envs/dmcontrol/{quadruped.py,dmcontrol.py}`:

- `self.camera_id = 2 if domain == 'quadruped' else 0` — **camera 2 confirmed**.
- `def render(self, width=224, height=224, camera_id=None)` — **224 px confirmed**.
- `_WALK_SPEED = 0.5`, `_DEFAULT_TIME_LIMIT = 20`, `make_model(floor_size=20*0.5)`,
  `Move(desired_speed=0.5)`, `control.Environment(..., time_limit=20, control_timestep=0.02)`,
  `action_scale.Wrapper(min=-1, max=1)`, the `b'file="./common/'` fixup, `model_dir=suite_dir`,
  `assets=quadruped.common.ASSETS` — all transcribed exactly by `_swm_quadruped_env`.
- `modify_mjcf_model` at `init_value`: grid rgb1 `[.1 .2 .3]` / rgb2 `[.2 .3 .4]`; `material self`
  rgba `[.6 .3 .3 1]`; floor friction[0] = 1.0 with the `[1, .005, .0001]` None-fallback; `torso`
  density 1000; `foot_back_left` density 1000; `light` diffuse `0.7*ones(3)` — **all six match the
  transcription line for line**, including the friction fallback.
- `swm/QuadrupedDMControl-v0` → `QuadrupedDMControlWrapper` in `envs/__init__.py`: confirmed.

Model census I ran on `_swm_quadruped_env(0)`: `ngeom=20` = 12 capsule + 4 sphere + 2 cylinder +
1 plane + **1 ellipsoid (`torso`)**; **no HFIELD geom** (`nhfield=1` but no geom of type 1 — the
terrain belongs to `quadruped-escape`, not `walk`); `nmesh=0`; camera 2 is named `y`, `mode=2`
(trackcom), fovy 45. The Step-7 plan's "1 HFIELD" line for this id is therefore moot, and the
author is right to have shipped ellipsoid only.

---

## 5. Gate A — REPRODUCED BIT-FOR-BIT AT HEAD

I regenerated the corpus from scratch in the scratchpad and re-ran
`scratch/step7/dmc-quadruped/gateA_local.py` at HEAD `9188c2c`:

- Candidate frozen corpus is **reproducible**: my regenerated `DMCQuadrupedWalk.npz` is
  `np.array_equal` to the committed `phase1/corpus/states/DMCQuadrupedWalk.npz` ((64, 23)).
- **14/14 mandatory checks `measured`** with zero-diff evidence: batch-invariance, batch-size
  (1/7/64), chunked (1/7/64/4096), default-style, determinism (`identical: true`),
  fidelity-declaration (`agrees: true`), input-form, order/duplicate, per-env-variation (no
  collapse), render-is-pure (all 0), set_state-copies, `state_dims_matter` (no dead dim of 23),
  state-roundtrip (0.0), style-is-state. Identical to the author's artifact.
- `check_visual_fidelity` on the 3.12 arm, **my run vs the author's committed
  `phase1/gateA/cpu_local_mj3120.json` — every field identical to the last digit**:

```
mean_abs 3.446892874581473   within8_frac 0.9003040547273597   ssim 0.954998479126862
exact_frac 0.1508636474609375  interior_frac 0.028498838705785356
interior_exact_frac 0.19565169456020814  interior_mean_abs 3.838757690017861
max_abs 245  n_states 64  interior_px 90702  interior_bad_px 72956
```
All three inside the declared ceilings (4.7 / 0.84 / 0.90) even on the undeclared 3.12 arm.
`check_backend_agreement` not_applicable (no Triton), `check_visual_fidelity_under_variation`
not_applicable, `render_parity_with_official_renderer` not_applicable (visual tier).
Wall time 205 s.

**The 3.11.0 + dm_control 1.0.44 numbers (3.42 / 0.902 / 0.957) are UNVERIFIED** — that arm does
not exist locally. They are however within 0.03/255, 0.002 and 0.002 of the 3.12 numbers I did
reproduce, which is a consistency argument, not a verification.

Provenance nit: the committed Gate A artifact records `git_commit: 68b411c` with `git_dirty: true`
— i.e. it was produced from the working tree *before* the dmcontrol commit that creates the cell
existed. The numbers are genuine (I reproduced them at `9188c2c`), but the stamp is misleading.

---

## 6. Hygiene — PASS

```
git log --format=%B 76dc5116..HEAD | grep -ci -E "co-authored-by|generated with"   -> 0
git log --format='%an <%ae>' 76dc5116..HEAD | sort -u -> sami-bg <samibghanem@gmail.com>
git diff --stat 76dc5116..HEAD -- examples/results/step4 examples/results/step5 \
                                  examples/results/step6                          -> (empty)
```
Minor: 11 `.npz` artifacts under `examples/results/step7/dmc-quadruped/{phase0,phase1}/` are
**untracked** (not gitignored), which is why every generated provenance stamp says
`git_dirty: true`. None of them is cited by name in `findings.md`, so no citation dangles.

---

## 7. Policy — PASS

- **No silent CUDA fallback.** `DMControlTrackCom.__init__` dispatches `cuda → _context`,
  `cpu → _cpu_scene`, anything else raises. The new refusal sits *inside* the `cpu` arm and raises;
  it never reaches the GPU path.
- **Swimmer refusal is by name only.** `CPU_UNCERTIFIED = {"DMCSwimmerSwimmer6": …}` keyed on
  `type(self).__name__`. I enumerated the 26 ids SWM registers
  (`stable_worldmodel/envs/__init__.py`): there is **no swimmer id**. No SWM id is touched. The GPU
  path is unaffected.
- **No per-environment hack in shared renderer code.** `cpu3d.py`'s entire diff is
  `MOVER_KINDS = (SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX)` plus the constant; grepping
  `cpu3d.py`/`raycast3d.py` for env names finds only doc prose. `cpu3d` is kind-generic apart from
  the `MOVER_KINDS` membership test (line 284). The ellipsoid is a genuine shared primitive.

---

## REQUIRED CHANGES (APPROVE-WITH-CHANGES)

1. **Fix the temp-directory leak in `_swm_quadruped_env`** (`stable_envs/envs/dmcontrol.py`,
   `td = tempfile.mkdtemp(prefix="se_swm_quadruped_")`). The directory is never removed. SWM itself
   avoids this by holding a `tempfile.TemporaryDirectory()` on the wrapper. `_swm_quadruped_env` is
   called on **every** `sample_states()` and `reference_env()` call, not just once, so a long-lived
   process grows without bound. Demonstrated:
   ```
   dirs matching /var/folders/.../se_swm_quadruped_*:  54 already present (author's own runs)
   after 3 more calls: 57      each dir 17,436 bytes (quadruped.xml)
   ```
   Smallest fix: build the XML into a `tempfile.TemporaryDirectory()` kept alive only as long as
   `Physics.from_xml_path` needs it and `shutil.rmtree` it afterwards (the compiled `Physics` does
   not need the file after compilation), or attach the `TemporaryDirectory` to the returned env the
   way SWM does.

2. **Declare the undrawn-site residual in the shipped `visual_note`.** `QUADRUPED_NOTE` enumerates
   floor-reflection, far checker, skybox, body and ellipsoid torso but is silent on the two `pupil`
   sites (group 0, alpha 1) that `physics.render` draws and neither backend does. `findings.md`
   documents it (0.01 % of frame, mean 148/255 on those pixels, 0.5 % of the residual); the
   *declaration the env carries* does not. I measured it directly at the registered default:
   ```
   camera 2, 224 px, reset state, sites-on vs sites-off in mujoco 3.12:
   52 / 50,176 px differ, max 186/255, mean 0.0725/255
   ```
   The family's convention is that the note names every residual. Add a clause (and note that
   `site torso` — a visible red group-0 site — is in the same bucket).

3. **Fix the now-false comment in `DMControlTrackCom._cpu_scene`'s `except ValueError` handler**
   (`stable_envs/envs/dmcontrol.py` ≈ line 2311): it still says "swimmer is the case: its bodies are
   mjGEOM_ELLIPSOID, which no analytic CPU intersector in raycast3d covers". That intersector now
   exists and swimmer is refused earlier, by `CPU_UNCERTIFIED`. The handler itself is still correct
   and useful (MESH/SDF/HFIELD would hit it) — only the prose is stale.

## RECOMMENDED (not blocking)

4. `_SWMQuadrupedModel._bundle` is a ~25-line verbatim copy of `_DMCModel._bundle` with one line
   changed. The author already flags the loader-hook cleanup as post-W3 work, but nothing will
   *detect* a future divergence. Add a cheap guard — e.g. a unit test comparing the two functions'
   `co_code`-independent behaviour on a shared invariant, or at minimum a test that the mixin's
   pristine dict has the same key set as the base's — until the hook lands.
5. Regenerate `phase1/gateA/cpu_local_mj3120.json` at HEAD (`9188c2c`) so the artifact's
   `git_commit` matches the commit that defines the cell, and commit-or-delete the 11 untracked
   `.npz` so `git_dirty` stops being `true` in every stamp.
6. Do not let the Oscar items lapse before certification: the Triton twin's bit-identity test
   (`test_the_ellipsoid_twins_are_bit_identical_on_both_casts`) has **never been executed**, the
   11 OGBench maze cells' CPU identity is unverified, and the VISUAL_FIDELITY entry is explicitly
   provisional CPU-measured where the family convention is three MJWarp draws under the declared
   3.11 pair.

## Commands run (abridged)

```
git log --format=%B 76dc5116..HEAD | grep -ci -E "co-authored-by|generated with"
git diff -U0 76dc5116..HEAD -- stable_envs/envs/dmcontrol.py | grep '^@@'
git worktree add <scratch>/base-76dc511 76dc5116f432e61dd47fba8f9e879cf9212c8e46 --detach
MUJOCO_GL=glfw se6-venv/bin/python -m pytest tests/unit -q --deselect tests/unit/test_examples.py -p no:randomly   # both trees
MUJOCO_GL=glfw se6-venv/bin/python -m pytest tests/unit -q -k "raycast3d or cpu3d or dmcontrol or declarations"    # both trees
MUJOCO_GL=glfw se6-venv/bin/python <scratch>/dump.py <tree> <out> <14 DMC configs>   # byte-identity, both trees
MUJOCO_GL=glfw se6-venv/bin/python <scratch>/ell_check.py ; <scratch>/ell_check2.py  # ellipsoid vs numpy.roots
MUJOCO_GL=glfw se6-venv/bin/python <scratch>/mats.py <tree>                          # _used_materials, both trees
MUJOCO_GL=glfw se6-venv/bin/python <scratch>/eyes2.py                                # site census + sites-on/off render
MUJOCO_GL=glfw OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 se6-venv/bin/python \
  scratch/step7/dmc-quadruped/gateA_local.py --config DMCQuadrupedWalk --device cpu \
  --states-dir <scratch>/gateA/states --out <scratch>/gateA/out.json
git worktree remove <scratch>/base-76dc511 --force && git worktree prune
```
