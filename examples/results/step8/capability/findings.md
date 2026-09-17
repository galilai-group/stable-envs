# Step 8 — shared capability: per-value texel substitution on the baked plate

**Verdict: `PATCH` for DM Control, `FINDING` + escalation for Fetch, `NO-OP` for OGB.**

* **Baseline** `2b4b5c3` (`step8/integration`) · **branch** `step8/capability-plate-texels`
* **Scope ruling applied**: the ONE shared capability approved in principle — mechanism class (ii) of
  [`per_world_model_arrays.md`](per_world_model_arrays.md) §4. Class (i), (iii), (iv) and (v) are out
  of scope for this ticket and are untouched.
* **Local venvs**: `/Users/samib/se6-venv` (CPU suite), `/Users/samib/se8-vm-venv` (live swm tiers),
  `/Users/samib/se8-fetch-venv` (Fetch probe). `MUJOCO_GL=glfw` throughout.

---

## 1. Step 1 — the Fetch skybox probe, run FIRST as the packet required

Artifact: [`fetch_skybox_probe.json`](fetch_skybox_probe.json) · source
`scratch/step8/capability/fetch_skybox_probe.py`.

**Verdict: `CONFIRMED_WITH_A_NEW_BLOCKER`.** The packet's inference is right; the reason recorded at
`fetch.py:589-592` was wrong; and a *different* blocker, which nobody had named, stops the leaf.

*Method note.* `FetchEnv` construction is blocked locally by the numpy 2.5.3 + mujoco 3.12 upstream
defect (`gymnasium_robotics/utils/mujoco_utils.py:142` asserts on the compiled joint type), so the
probe compiles the four scenes' MJCFs directly and renders them through `mujoco.Renderer` at the
family's `DEFAULT_CAMERA_CONFIG` free camera at 480 px — the same raster `_bake_plate` takes. That
is the official renderer path for the question being asked; it does not exercise our compositor,
and it does not need to, because the compositor's use of the plate is not in doubt.

| what was measured | result |
|---|---|
| Is the skybox in the baked plate? | **YES.** Substituting its texels moves **171,620 / 230,400 px (74.5 %)** on push, pick_and_place and reach, **143,275 (62.2 %)** on slide |
| Substitution vs a full recompile at the same colour | ≤ **1/255** on ≤ 5,903 px (≤ 2.6 %); **0 px, 0 tex bytes** at the two probe values whose channels quantise exactly |
| Default restore after every probe value | **byte-identical**, all four scenes |

The ≤1/255 residual is the **recompile** arm differing from swm, not the substitution: swm's own
`background.color` (`stable_worldmodel/envs/gymnasium_robotics/fetch.py:400-430`) **does not
recompile either** — it writes `(bg * 255).astype(np.uint8)` flat over the skybox texels and calls
`mjr_uploadTexture`, the same truncation `_plate_texels.quantise` implements. Our substitution is
byte-identical to swm's operation by construction; MuJoCo's gradient compiler rounds where both of
them truncate.

### The new blocker, and why it is not a capability gap

swm's operation **flattens** the skybox to one colour. The Fetch skybox is a **16-colour gradient**
(measured: 16 unique texel values over 6,144 texels, rgb1 `.44 .85 .56` → rgb2 `.46 .87 .58`).
Therefore:

* no value of a **flat-fill** leaf reproduces our registered default appearance, and
* no **two-colour masked** substitution — the shape that works for DM Control's checker — reproduces
  swm's flat sky.

So the leaf cannot be offered without moving this family's **default frame**, which is a frozen pixel
oracle on the Step-4 and Step-7 corpora. That is a **re-freeze** decision (plan §19) and a change to
a frozen contract (plan §4 escalation list), not an implementation decision. **Escalated, not taken.**
`background.color` stays `partial`; its residual, and `FetchBase.make_variation_space`'s own
docstring, now carry the measured reason instead of the refuted one. No Fetch behaviour changed.

*(Three shapes that were considered and rejected, so the coordinator does not have to re-derive them:
a sentinel "pristine" default is discontinuous at the default value and Gate A samples through it; a
(2,3) rgb1/rgb2 leaf cannot reconstruct a 16-level gradient without reimplementing MuJoCo's builtin;
a multiplicative tint is continuous and identity-at-default but invents a law swm does not perform.)*

---

## 2. Step 2 — what was implemented

### `stable_envs/envs/_plate_texels.py` (new, shared)

The OGB plate-texture edit lifted out of `ogb_manip.py`: `quantise` (MuJoCo **truncates**
`int(255*v)`), `texel_slots` (masks taken once from the **pristine** texture, with an optional
explicit-mask form and a check that the masks agree with the texture), `substitute` (write →
`make_current` → `mjr_uploadTexture`, unconditionally, on the renderer's **own** GL context) and
`texel_key` (the plate-cache key is the **quantised bytes**, and `None` at the registered default).

### DM Control parity tier — `floor.color`, 8 cells / 7 swm ids

`DMControlSuite` gains `floor.color`, shape `(2,3)` — swm's own shape and order, rgb1/rgb2 of the
builtin `grid` checker — with the default **read from `dm_control`'s `common/materials.xml`**, the
same file swm's `init_value` transcribes, so "our default IS swm's default" holds by construction.

Two things this had to get right that the packet did not anticipate:

1. **`_bundle`'s one-`Renderer`-per-(domain, size) policy is unchanged.** Its comment says only a
   texture edit would need a second GL context; the escape it does not consider is an in-place texel
   write on the **existing** model and context, which is what this does. No context is created,
   closed or duplicated.
2. **`markrgb` equals `rgb2` in the suite's own MJCF.** The 1,196 edge-mark texels are therefore
   byte-identical to the 44,402 rgb2 texels in the compiled texture, and swm's recompile moves the
   second set while leaving the first at its literal `.2 .3 .4`. An equality mask — the OGB shape —
   would move **both**, and the marks are the floor's grid lines. `_grid_masks` separates them
   positionally with a four-line probe checker and **asserts** the separation by rebuilding the real
   texture from it and requiring MuJoCo's own bytes.

### OGB — deliberately not refactored (`NO-OP`)

The brief allowed refactoring `ogb_manip.py` onto the shared helper "only if the diff is small and
byte-identical". It is not: OGB's `_apply_floor` also maintains `rig.floor`, its masks are taken by
equality (correct **there** — OGB's `markrgb` is 204,204,204 and distinct), and its `_floor_key`
folds into `_PLATE_CACHE`'s tuple differently. The refactor is 3 call sites of behaviour change for
0 behaviour gain on a family whose evidence is already merged and whose GPU evidence I cannot re-run
this session. **`STEP9_CLEANUP`.** The duplication is one function and is now documented on both
sides.

### Fetch — no code change beyond correcting a refuted comment. See §1.

---

## 3. Step 3 — VariationMaps and declaration deltas

| family | leaf | before | after | ids |
|---|---|---|---|---|
| DM Control parity | `floor.color` | `unsupported` | **`mapped`** (`identical`) | **7** (Acrobot, BallInCup, Cartpole, Finger, Manipulator, Pendulum, Reacher) |
| DM Control trackcom | `floor.color` | `unsupported` | `unsupported`, **narrower reason** | 5 (Cheetah, Walker, Hopper, Humanoid, Quadruped) |
| Fetch | `background.color` | `partial` | `partial`, **corrected reason** | 12 |
| OGB | `floor.color` | `mapped` | `mapped` (unchanged) | 2 |

`identical`, not `renamed` or `scaled`: same dotted path, same shape, same order, same units, same
default. Counts, re-derived from the **live** swm space by
`scratch/step8/variations-dmc-control/build_classification.py`: **mapped 46 → 53, unsupported 94 →
87**, total 161 unchanged.

**Declaration delta** — [`declaration_delta.json`](declaration_delta.json), before = integration
`2b4b5c3`, after = this branch, cpu/84 px:

| cells | `style_dim` | `default_style_digest` | default frame |
|---|---|---|---|
| 12 `DMControlSuite` cells | **+6** (11→17, 14→20, 18→24, 22→28) | **moves** (the row is 6 floats wider) | **byte-identical** |
| 5 `DMControlTrackCom` cells | unchanged | unchanged | byte-identical |
| OGB, Fetch | unchanged | unchanged | unchanged (no code change) |

The default-style digest moving while the default *frame* does not is the expected shape and is the
reason both are recorded: a wider row is a declaration move that Gate A must re-baseline, not a
pixel move.

---

## 4. Evidence

### Byte-equality against a full recompile — the claim the whole capability rests on

`scratch/step8/capability/bake_vs_recompile.py`, two processes (dm_control's asset table and this
module's caches are process-global), **12 parity cells × 3 probe colours × 8 states at 84 px**:

| probe | rgb1 / rgb2 | `tex_data` identical | frames identical |
|---|---|---|---|
| 0 | `.9 .1 .1` / `.1 .1 .9` | **12/12** | **12/12** |
| 1 | `0 0 0` / `1 1 1` | **12/12** | **12/12** |
| 2 | `.37 .61 .24` / `.05 .95 .5` | **12/12** | **12/12** |

Artifacts `bake_vs_recompile_probe{0,1,2}.json`, verdict **IDENTICAL** on all three. The recompile
arm patches `dm_control.suite.common.ASSETS['./common/materials.xml']`, so it is a genuine recompile
of the suite at that `floor.color` — the same thing swm's `mark_dirty` path does.

### Default identity vs the integration baseline

All 12 parity cells, `sample_states(4, rng=0)` at 84 px on cpu: **byte-identical** to the same render
from `/Users/samib/stable-envs-step8-integration` at `2b4b5c3`. Recorded per cell in
`declaration_delta.json`.

### Tests

| suite | venv | result |
|---|---|---|
| `tests/unit/test_plate_texels.py` (new, 23 tests) | `se6-venv` | **23 passed** |
| `tests/unit/test_variation_maps_dmc_control.py` | `se8-vm-venv` (live swm) | **147 passed** (138 before + 9 new) |
| `tests/unit/test_variation_maps_dmc_control.py` | `se6-venv` | 19 passed, 128 skipped |
| `tests/unit/test_variation_maps_fetch.py`, `test_variation_map_partial.py` | `se6-venv` | **passed** |
| `tests/unit` (whole suite) | `se6-venv` | **1,234 passed**, 195 skipped, 2 deselected |

The 2 deselected are the two failures A already recorded as **pre-existing at the baseline**
(`test_examples.py::…robocasa_visual/render_mjwarp.py`, `test_repo_contracts.py::
test_every_cited_scratch_path_resolves`); both were re-confirmed failing on the integration tree.

`tests/gpu` **not run** — see §5.

### Regeneration

`leaf_classification.json`, `support_matrix.{json,md}`, `performance/tiers.json` and
`step8_performance.md` are regenerated with their own generators and committed. Note the existing
convention these generators carry: each embeds `generated_at_head_sha`, so a regeneration is
byte-identical only when run at the HEAD the artifact records — one commit behind the commit that
contains it. That is how every merged Step-8 copy of these files already reads (the baseline's matrix
records `0ee3335` while HEAD was `2b4b5c3`); it is not something this patch introduced and there is
no fixpoint to reach.

### GPU evidence — **BLOCKED**

Not run, and not runnable this session. Oscar SSH returned `Permission denied
(keyboard-interactive)` on **26 attempts over ~2 h** (two retry loops, 21:57–23:57 EDT), which is
past the brief's 30-minute retry budget; and allocation `6443480` ended at ~22:45 EDT, so the single
`3090-gcondo` sbatch the brief authorises could no longer have started. **No job was submitted and
no job ID exists.** The sbatch is written and ready at
`scratch/step8/capability/gpu_evidence.sbatch` for whoever holds the next allocation.

Consequently the merge gate's items 4, 5 and 7 are **NOT satisfied** for this patch and it must not
merge on this evidence. What is outstanding, and only what is outstanding:

1. default-frame digest identity vs `$SCR/se8/baseline` on the frozen Step-7/Step-4 corpora, the
   **8 affected parity cells**;
2. isolation — `floor.color` moves only plate pixels;
3. custom → default sequences (leak);
4. targeted **Gate A on cuda** with the new `style_dim` (a declaration re-baseline, per §3);
5. paired throughput at B=1 and B=4096, which is the term to watch: this family's plate cache is
   `_SCENE_CACHE_MAX = 96` and a high-cardinality `floor.color` batch bakes one scene per distinct
   **quantised** colour. The key is on the bytes precisely so that near-identical floats share a
   bake, but a genuinely random floor over a large batch is a re-bake storm and the sweep must
   include one.

**One representative per domain is NOT acceptable for this family** and I state it because the brief
offered it: `PLATE_PATHS` here is *every* leaf and the plate is baked per (domain, task, camera,
size, appearance), so the eight cells do not share a plate path — they share the *code*, not the
bake. All eight need their own default-identity row.

---

## 5. Merge gate (plan §18)

| # | item | status |
|---|---|---|
| 1 | explicit workstream scope | **YES** — capability class (ii) only; §2 names what was refused |
| 2 | before/after diagnosis | **YES** — §1 probe, §4 byte-equality, §3 declaration delta |
| 3 | targeted unit tests | **YES** — 23 new helper tests + 9 new map tests, §4 |
| 4 | affected Gate-A / Gate-F checks | **NO — BLOCKED**, §4. Gate A must re-run on cuda with the new `style_dim` |
| 5 | affected frozen-corpus replay | **NO — BLOCKED**, §4 |
| 6 | pixel-movement classification | **YES** — *no default pixel moves*, 12 cells, cpu-measured; a non-default `floor.color` moves plate pixels only, which is item 4's isolation check |
| 7 | performance regression check | **NO — BLOCKED**. Hot-path code changed (one `mjr_uploadTexture` per plate-cache **miss**, never per render); the cache-thrash sweep is the real question |
| 8 | independent reviewer verdict | **NOT YET** — coordinator to assign |
| 9 | semantic follow-up decision | **NONE WARRANTED**: no default frame moves, so no downstream number can move. A caller who *uses* the new leaf is doing domain randomisation, which is what the leaf is for |
| 10 | dashboard update | **coordinator** — this workstream has no dashboard row yet |

## 6. Buckets (plan §3)

| item | bucket |
|---|---|
| DM Control parity `floor.color` | **RELEASE_BLOCKING**, resolved — a registered appearance leaf that moves 6,408–53,246 official px was `unsupported` on 7 ids |
| Fetch `background.color` skybox half | **RELEASE_BLOCKING → escalated.** The mechanism exists; offering the leaf moves a frozen default. Sami's call: re-freeze Fetch's default appearance, or accept `partial` permanently |
| `fetch.py:589-592`'s stated reason | **UPSTREAM_FINDING (ours)** — a refuted reason in shipped documentation; corrected in place |
| DM Control trackcom `floor.color` (5 ids) | **DOCUMENTED_LIMITATION** — no plate, so the capability does not apply |
| OGB refactor onto the shared helper | **STEP9_CLEANUP** |
| `_SCENE_CACHE_MAX` / `_PLATE_CACHE_MAX` under a high-cardinality style batch | **DEFERRED_OPTIMIZATION**, but its *measurement* is merge-gate item 7 and is outstanding |

## 7. Open items for the coordinator

1. **Submit `scratch/step8/capability/gpu_evidence.sbatch`** on the next allocation. Until it
   returns, this patch is not mergeable.
2. **Decide the Fetch escalation** (§1). It is a re-freeze question, not an engineering one.
3. `VariationMap` has no **one-to-many** mapping mechanism. Even with the skybox implemented, swm's
   single 3-wide `background.color` would reach two of our leaves and could not be declared
   `mapped`. `worldmodel.py` is a shared file this ticket may not edit; flagging rather than doing.
4. The `per_world_model_arrays.md` packet says class (ii) unlocks "21 leaves across 21 swm ids in 3
   families". **The measured figure is 9 leaves across 9 ids in 2 families** — OGB's 2 (already
   merged) plus DM Control's 7. The 12 Fetch ids are blocked by §1's default problem, which the
   packet's cost estimate ("M — one texture, one key entry") did not see.

## GPU evidence (added by the coordinator once Oscar accepted SSH; tested tree `64a38e7`)

All artifacts under `examples/results/step8/capability/gpu/` (Oscar paths rewritten to `$SCR`).
Every comparison is within-job on one card; the four jobs ran on the `gpu` partition
(`norm-gpu` QOS) because the 3090 group quota was held by unrelated jobs.

| merge-gate item | evidence | verdict |
|---|---|---|
| default frames unchanged | job 6459781, gpu3002 (L40S): `declaration_{before,after}.json`, 17 DMC cells, `default_frame_digest` identical in every cell; `style_dim` 11→17 / 14→20 / 18→24 / 22→28 on the 12 parity cells, unchanged on the 5 trackcom cells | identical |
| bake vs full recompile on cuda | same job: `bake_vs_recompile_gpu_probe{0,1,2}.json` | IDENTICAL ×3 |
| 5 frozen-corpus replay | job 6460099, gpu1504: `digest_replay_{before,after}.json`, 8 cells (7 parity + Manipulator), digests identical before/after | identical |
| 4 Gate A on cuda with the new style_dim | job 6460346, gpu1506 (A40): `gate_a_cuda_dmc_parity.json` via `examples/fidelity_step4.py` (the driver that certified these cells in Steps 4–7), 7 configs; job 6460099: `gate_a_cuda_dmc_manipulator.json` via the closure2 driver on the Step-7 corpus. Judged locally with the frozen `scratch/step4/certify.py` judge: **8/8 PASS, 14/14 mandatory, no family failure, nothing blocked** | PASS |
| 7 paired throughput | job 6460099: `throughput_{before,after}.json` (8 cells, B=1 and B=4096, default / shared floor / 96 distinct floors). Default-style B=4096 ratios 0.92–1.15 on seven cells; Manipulator read 0.72 on that mixed node. Re-measured in job 6460478, gpu1501 (RTX A6000, quiet), three ALTERNATING rounds: Manipulator 22236/22545/22567 → 21833/21677/21697 fps (spread 1.01, ratio **0.96**); Cartpole 44983/45577/45788 → 42074/41748/42218 (spread 1.02, ratio **0.92**). The 0.72 was a shared-node artefact; the real default-path cost is 4–8 % at B=4096 (the per-render floor-colour read of the style tensor), inside the 1.25× paired tolerance and far from any tier boundary (`GPU_FAST` needs R ≥ 4). `DEFERRED_OPTIMIZATION`. The 96-distinct-colour batch runs at ~1,200 fps (Manipulator 127) because every distinct quantised colour bakes one plate: a documented cost of USING the leaf at high cardinality, not of the default path | no regression |

The three EGL `Renderer.__del__` tracebacks after each driver's "wrote" line are the known
interpreter-teardown noise (Step-7 NODE-2, retracted), not measurement failures.
