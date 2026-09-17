# Step 8 — reference versions and provenance

!!! abstract "What this page settles"

    Every final fidelity claim has to name a reproducible, appropriate reference version and
    explain the drift around it. This page does that for the four DM Control trackcom cells, for
    the `mujoco` / `mujoco-warp` 3.13 question, and for the Antmaze bake.

    Nothing here rendered a new pixel on the trackcom cells, moved a threshold, edited a frozen
    corpus, or changed a dependency pin. The Step-4 judge `scratch/step4/certify.py` is untouched.

| | |
|---|---|
| workstream | Step 8 **F** — reference versions and provenance |
| baseline | `43ac95238f7472137619f3e69e7636e11dd088b1` |
| artifacts | `examples/results/step8/provenance/` |
| generators | `scratch/step8/provenance/` |
| tests | `tests/unit/test_step8_provenance.py` |
| cluster | SLURM allocation `6443480`, `gpu2264`, NVIDIA GeForce RTX 3090, driver 575.57.08 |

---

## 1. Cheetah, Hopper, Walker, Humanoid — one sentence

Their fidelity numbers were declared against **mujoco 3.11.0**, the Step-7 sweep measured them
**twice** — once against 3.11.0 and once against 3.12.0 — and the support matrix was reading the
3.12.0 arm because of an alphabetical tiebreak. Against the reference their declaration names, all
four **PASS Gate A** with no failing and no blocked check.

### What each arm measured

Both arms ran in **one SLURM job** (`6443480`) on **one card** (`gpu2264`, RTX 3090, driver
575.57.08) at the Step-7 integration HEAD `fec4f6ee`, on the same frozen corpora, with the same
`check_seed` 4041. The only difference is the interpreter.

| | declared arm | drifted arm |
|---|---|---|
| shard | `examples/results/step7/closure2/replay/dmcvis.json` | `examples/results/step7/closure2/replay/dmc.json` |
| shard tag | `adhoc` | `dmc` |
| `mujoco` | **3.11.0** | 3.12.0 |
| `dm_control` | 1.0.44 | 1.0.45 |
| `mujoco_warp` | 3.11.0 | 3.12.0 |
| satisfies `reference_versions` | **yes** | no |

`reference_versions` is a field, not prose: `_AT_VERSIONS = {"mujoco": "3.11.0"}` in
`stable_envs/envs/dmcontrol.py`, published by `describe()` and read by the certifier through
`stable_envs.core.env.reference_versions_of`. It names the **reference side only**. Our own
renderer's versions live in `measured_with`, which is recorded and never compared — a bump of our
renderer that moves a number is a regression to see, not a reason to refuse to judge.

### Gate A, both arms, frozen judge

`scratch/step4/certify.py:verdict`, unmodified, mandatory **and** family_specific:

| cell | declared 3.11 arm | drifted 3.12 arm |
|---|---|---|
| `DMCCheetahRun` | **PASS** | FAIL (`check_resolution_consistency`), visual fidelity `blocked_environment` |
| `DMCHopperHop` | **PASS** | FAIL (`check_resolution_consistency`), visual fidelity `blocked_environment` |
| `DMCWalkerWalk` | **PASS** | FAIL (`check_resolution_consistency`), visual fidelity `blocked_environment` |
| `DMCHumanoidStand` | **PASS** | BLOCKED (visual fidelity `blocked_environment`) |

The numbers, against the declaration's own ceilings:

| cell | declared | ceiling / floors | **3.11 arm** | 3.12 arm |
|---|---|---|---|---|
| `DMCCheetahRun` | 5.01 / 0.882 / 0.661 | ≤6.5 / ≥0.84 / ≥0.62 | **4.992 / 0.8820 / 0.6613** | 6.542 / 0.7961 / 0.5816 |
| `DMCHopperHop` | 4.35 / 0.905 / 0.684 | ≤5.7 / ≥0.86 / ≥0.64 | **4.373 / 0.9042 / 0.6843** | 6.578 / 0.7912 / 0.5440 |
| `DMCWalkerWalk` | 8.07 / 0.700 / 0.826 | ≤10.5 / ≥0.65 / ≥0.77 | **7.862 / 0.7137 / 0.8245** | 16.008 / 0.5524 / 0.5334 |
| `DMCHumanoidStand` | 6.26 / 0.735 / 0.868 | ≤8.2 / ≥0.69 / ≥0.82 | **6.271 / 0.7317 / 0.8672** | 6.299 / 0.7302 / 0.8658 |

(mean \|diff\| /255 · within-8 fraction · SSIM, 64 states at 84 px, `check_seed` 4041.)

Every 3.11 number lands inside the declaration's own recorded three-draw spread. This is not a new
result: Step 6's `texture-fidelity` workstream measured the same thing on two other nodes
(`examples/results/step6/texture-fidelity/findings.md`, jobs 6343431 / 6343699 / 6346226) and the
closure2 sweep reproduces it at the integration HEAD.

### Why the matrix was reading the other one

`scratch/step4/certify.py:merge` keeps, of two rows that measured the same number of checks, the
one whose **shard tag sorts last**:

```python
def _richness(row):
    n = sum(1 for g in ("mandatory", "family_specific")
            for v in (row.get(g) or {}).values() if v.get("status") == "measured")
    return (row.get("status") == "measured", n, row.get("shard") or "")
```

Both arms measure 18 checks. `"adhoc" < "dmc"`, so the declared-reference arm lost — to a tiebreak
that knows nothing about reference versions. `examples/results/step7/closure2/replay/certify/certification.json`
is **not edited**; it stays the record of what was judged when.

The fix is in the join, not in the judge and not in the renderer:
`scratch/step8/provenance/reference_arm.py` judges **both** arms with the frozen judge and selects
the authoritative one with `stable_envs.testing.provenance.reference_version_mismatch(declared,
recorded) == []` — the same function the certifier uses. If no arm satisfied the declaration the
record would say `UNMEASURED` and print the exact gap; if two did it would refuse to choose.

!!! note "No new declaration was needed"

    The brief allowed an additive declaration on the four classes if the harness required one. It
    does not. `reference_versions` already exists, `describe()` already publishes it, and the judge
    already reads it — which is precisely why the 3.12 arm reads `blocked_environment` rather than
    `FAIL` on `check_visual_fidelity`. The only thing missing was a join that honoured it. Adding a
    second declaration naming an artifact path would have put a build detail in the env class.

### The 3.12 finding is kept, in the same row

The four matrix rows now read `PASS` **and** carry
`DOCUMENTED_LIMITATION reference/upstream drift`, generated from the same record. Both facts, one
row. The mechanism, from Step 6's per-surface localisation off MuJoCo's segmentation buffer:

- MuJoCo **3.12.0** changelog item 15 (commit `cc7fb98c`) gives built-in geometries explicit texture
  coordinates and anchors a **finite plane's** texture to its bottom-left corner instead of its
  centre. `mujoco_warp` still implements the centred `render_gl3.c settexture` mapping.
- So the **reference's** floor checker is phase-shifted and ours is not. The reference's floor moves
  14.8 / 16.9 / 25.2 / 0.8 /255 between the two versions on cheetah / hopper / walker / humanoid;
  its skybox moves 0.01–0.06; our frames move 0.0002 (99.98 % of pixels exact).
- Humanoid is the control: its floor is square (100×100), so the re-anchoring shifts it by whole
  checker periods and nothing visible happens. The other three have floors 0.8–1.0 m half-width on
  the camera-facing axis, a non-integer number of periods.
- Compiled texture bytes, the material table, the suite XMLs and `dm_control`'s render path are
  **byte-identical** across 3.11.0 and 3.12.0. One upstream change explains all of it.

**Classification: `DOCUMENTED_LIMITATION`** (with an `UPSTREAM_FINDING` inside it — the divergence
between libmujoco's plane mapping and `mujoco_warp`'s is upstream's, and `mujoco_warp` is the one
that has not followed). Certifying these cells needs an interpreter pinned to `mujoco==3.11.0` /
`dm-control==1.0.44` / `mujoco-warp==3.11.0`; `pyproject.toml`'s `dmcontrol` extra is a **floor**
(`mujoco>=3.11`), not a pin, which is how a 3.12 interpreter came to judge them in the first place.
Turning that floor into a pin is a `pyproject` change and is **not made here**.

### `check_resolution_consistency` is a different finding

It never calls the reference, so the version drift above cannot be its cause. Yet it separates the
arms at the Step-7 HEAD — cheetah 0.533854 (3.11) against 0.522233 (3.12) on the same 8-state
protocol draw, on the same card, in the same job. Recorded, not explained:

| cell | frozen bound | floor (−0.005) | 3.11 arm | 3.12 arm |
|---|---|---|---|---|
| `DMCCheetahRun` | 0.533850 | 0.528850 | **0.533854** | 0.522233 |
| `DMCHopperHop` | 0.550220 | 0.545220 | **0.550223** | 0.536547 |
| `DMCWalkerWalk` | 0.628950 | 0.623950 | **0.628951** | 0.612581 |
| `DMCHumanoidStand` | 0.603650 | 0.598650 | **0.603653** | 0.603901 |

The 3.11 arm reproduces the bound to **≤4e-6** on all four. The bound was measured on an H200 under
**mujoco 3.9** — a pre-3.12 interpreter — so "the arm that reproduces it is the pre-3.12 one" is
consistent, but the carrier is not identified here: Step 6 measured this number as **identical to
four decimals** across its own 3.11 and 3.12 arms (`findings.md` S6), and this sweep does not
reproduce that. Two stable values exist across four campaigns (0.5304 / 0.5339 in Step 6; 0.5208 /
0.5222 in Step 4 and in closure2's 3.12 arm), and §2 below adds a third data point: the value under
**mujoco_warp 3.13** is 0.522233, the closure2-3.12 value exactly.

**This is an open item, classified `DEFERRED_OPTIMIZATION` → really `STEP9_CLEANUP`**: it does not
block the claim, because the arm the declaration names reproduces the bound. It does mean the
number is interpreter-sensitive in a way nobody has attributed, and §4 lists it.

---

## 2. `mujoco` / `mujoco-warp` 3.13 — recommendation: **do not bump**

Full record: `examples/results/step8/provenance/mjwarp313_verdict.json`, measurements under
`examples/results/step8/provenance/mjwarp313/`.

`$SCR/venvs/se8-mjwarp313` is `se7-ogb`'s pin set with **only** the MuJoCo pair moved to 3.13.0
(`scratch/step8/provenance/build_mjwarp313.sh`); both arms ran on `gpu2264` under allocation
`6443480`.

### Condition 1 — does it fix the defect? **Yes.**

`mujoco_warp` 3.13.0's `sample_texture` MESH branch:

```python
texcoord_offset = mesh_texcoord_offsets[mesh_id]
if texcoord_offset >= 0:
  # Some meshes may have no texcoord. The corresponding elements for these meshes in
  # mjm.mesh_texcoordadr (passed here as mesh_texcoord_offsets) are marked as -1, in
  # which case uv stays at its initialized value of (0.0, 0.0).
```

That is exactly the condition Step-7 W5 diagnosed
(`examples/results/step7/ogb-manipulation/findings.md` §11): OGBScene's two button geoms wear mesh
`stopbuttonrim` with `texcoordadr = -1, texcoordnum = 0`, and ≤3.12 indexed from a negative offset
and died with an illegal memory access. Confirmed on hardware in both interpreters:
`sample_texture_guards_missing_texcoords` is `false` under 3.12.0 and `true` under 3.13.0.

`texture_gaps` still reports the two geoms under 3.13, and that is correct: the helper describes the
**model** (a textured mesh with no UVs), not the renderer's tolerance of it. The model did not
change.

### Condition 2 — does the existing evidence still hold? **No.**

| `DMCCheetahRun` | 3.12 arm | 3.13 arm |
|---|---|---|
| `check_visual_fidelity` mean / within-8 / SSIM | 6.508 / 0.7969 / 0.5836 | 6.542 / 0.7961 / 0.5816 |
| `check_resolution_consistency` `exact_frac` | **0.533854** | **0.522233** |

The frozen floor is 0.528850. **A 3.13 bump turns a passing Gate-A check into a FAIL** on a cell the
texcoord defect never touched.

### Condition 3 — does it move unrelated renderers? **Yes.**

Frozen-corpus digests, same corpora, same card:

| cell | reference arm | 3.13 | identical |
|---|---|---|---|
| `OGBCube` | `d8676966e1370329…` (3.12, closure2 `cuda_ogbmanip.json`) | `dd84fe7965b3cfcb…` | **no** |
| `OGBScene` | `1ffbd6f7aa6c18c2…` (3.12, closure2 `cuda_ogbmanip.json`) | `2b12a93d84e437fe…` | **no** |
| `DMCCheetahRun` | `d314a7eab0123acb…` (3.12, this job) | `1fc511eaec271090…` | **no** |
| `FetchReach` | `c8855f3a7bb213ba…` (**3.11**, `se-s3b-fetch311`) | cannot build | — |

And the source says why. Beyond the texcoord guard, 3.13's `render.py` also:

- replaces a vertex normal more than ~37° off its face with the face normal (matching
  `mjr_uploadMesh`) — a **shading** change on every mesh;
- widens BVH traversal by `RAY_TOL_REL` / `RAY_TOL_ABS` and tie-breaks coincident hits to the higher
  geom index "like MuJoCo's GL depth test" — a **visibility** change on every scene with touching
  surfaces;
- turns the no-light fallback into a `shadow_light_fraction` parameter — a **lighting** change;
- adds an `_aa_resolve` accumulate-and-average kernel — a **sampling** change.

Each is plausibly an upstream correctness improvement. None of them is the texcoord fix, and every
one can move a pixel in a cell with no textured mesh at all.

!!! warning "`FetchReach` is a separate, older wall"

    `gymnasium-robotics`' `set_joint_qpos` assertion fires under `mujoco >= 3.12`, so the Fetch
    **reference** env cannot be constructed on 3.12 **or** 3.13 — measured again here, both arms,
    `AssertionError` before the renderer is reached. `se-s3b-fetch311` (mujoco 3.11.0) is the only
    interpreter where Fetch builds. This predates 3.13 (`docs/research/swm_matrix.md`) and is an
    `UPSTREAM_FINDING`; it is recorded here because a bump would make it the shipped configuration.

### And the bump buys nothing that is currently costing anything

OGBScene renders **today**, because the cell bakes the unsamplable material away
(`bake_unsamplable_textures` on `btn_metal`, the two button geoms) at a **declared** residual of
0.139 % of pixels with `interior_exact` 1.0. Dropping that bake under 3.13 would itself change
pixels and need a re-declaration.

**Recommendation: no bump.** What would change the answer: a release carrying the guard and nothing
else in the render path, or a deliberate re-declaration campaign that re-measures every mjwarp cell
under the new pair and re-freezes with the drift documented. Both are Step-9 scope. **No
`pyproject.toml` pin is changed by this workstream.**

---

## 3. Antmaze / bake nondeterminism — a provenance limitation of the oracle

Classified from the committed Step-6 and Step-7 artifacts
(`examples/results/step6/raycast3d/findings.md` §4 and `RAYCAST-2c / 2c'`,
`docs/research/step6_optimization.md`, `examples/results/step7/closure/findings.md`,
`docs/research/step7_coverage.md`). **No re-run was needed and none was made.**

### The three facts

1. **`AntmazeTeleportExplore` does not reproduce its frozen Step-4 `digest_ours` at the baseline
   either.** Measured on three jobs and two nodes, on both legs of every before/after:
   `483968227d628c59` against a frozen `06a2e039e1fdf3ad`. The other 20 live configs reproduce
   exactly. The digest mismatch is therefore **not** a property of any Step-6/7 change.
2. **The cause is the bake's history dependence, and it is deterministic, not random.** The plate
   `OGBenchMaze._bake` produces is a function of how many `update_scene`/`render` cycles the
   retained per-texture-state `mujoco.Renderer` has seen. Eight-bake digest sequences are
   byte-identical between two processes on two different code trees and across two nodes
   (`AntmazeGiantNavigate` `ABABABAB`, `HumanoidmazeMediumNavigate` `AAAAAAAA`). Driven as shipped,
   `ant/teleport` yields **four distinct plates in eight bakes**; a fresh `mujoco.Renderer` per bake
   yields **one** on every probed scene, which identifies the retained renderer as the carrier.
3. **Step 4's plate is none of them.** Step 4 ran the full check suite before taking the oracle, so
   its plate was a non-first bake; `AntmazeGiantNavigate` alternates between exactly two and its
   frozen digest **is** one of them, which is why that config replays green. Teleport's is a third
   image the probe did not reach.

### The classification

| item | bucket | why |
|---|---|---|
| `AntmazeTeleportExplore` frozen `digest_ours` mismatch | **`DOCUMENTED_LIMITATION`** | an **oracle hole**, not a renderer failure: the frozen digest records a plate that is a function of the measuring session's call history, so no correct implementation reproduces it. Gate A on this cell is **PASS** today (Step 6 recorded `FAIL → PASS`); the digest is not a Gate-A check. |
| the bake's history dependence itself | **`UPSTREAM_FINDING`** (with containment already merged) | the state is carried inside `mujoco.Renderer` across `update_scene`/`render`. Contained in Step 6 by replacing `_SCENE_CACHE`'s wholesale `clear()` with LRU eviction, which removes cache pressure as the trigger; a cold bake is still history-dependent. |
| `check_reference_renderer_is_stable` firing 12/12 on `PointmazeTeleportNavigate` | **`DOCUMENTED_LIMITATION`** | the **reference** rasterizer is not bit-reproducible for that scene on that host, deterministically and scene-specifically, reproducing Step 4's own record. When the reference is unstable that is a limit on what can be certified, not a stable-envs defect. |
| making the bake a pure function of `(scene, style)` | **`STEP9_CLEANUP`** | a fresh `mujoco.Renderer` per bake fixes it and **preserves every plate currently in use**, but costs one EGL context per bake against a measured budget of ~96 live contexts, and `check_style_is_state` alone bakes 64. The usable shape is a reset-without-new-context or a pool sized to the budget; neither is measured. |
| re-freezing `AntmazeTeleportExplore` | **`DEFERRED`** — explicitly **not** now | re-freezing a digest that is still a function of call history just moves the problem. Step 19's re-freeze policy is not satisfied: the corrected implementation does not exist yet. |

**The headline: if the reference is unstable, that is a provenance limitation, not a stable-envs
failure.** `AntmazeTeleportExplore` is not one of the 42 SWM ids, so no support-matrix row changes.

---

## 4. Open items

1. **`check_resolution_consistency`'s interpreter sensitivity** (§1). Two stable values across four
   campaigns and a third measurement under 3.13 matching one of them. The declared arm reproduces
   the frozen bound to 4e-6, so nothing is blocked; the carrier is unidentified. `STEP9_CLEANUP`.
2. **`pyproject.toml`'s `dmcontrol` extra is a floor, not a pin** (`mujoco>=3.11`). Turning it into
   a pin for certification interpreters is the clean close of §1 and is Sami's call — it is a
   dependency-contract change. Not made here.
3. **`stable_worldmodel` is recorded as `"present"`, without a version**, by the interpreter that
   measured the OGB cells. `reference_version_mismatch` treats "present" as unknown, so an OGB
   declaration naming an swm version could never be confirmed. Recorded, not fixed.
4. **The upstream report to file**: `mujoco_warp` shipped the UV-less-mesh guard in the same release
   as four independent render-path changes, which is what makes the fix unadoptable for a project
   holding frozen pixel oracles.
