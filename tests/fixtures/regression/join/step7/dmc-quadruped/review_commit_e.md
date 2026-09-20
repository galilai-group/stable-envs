# Step 7 W4 `dmc-quadruped` — independent re-check of commit (e) `ad842a8`

Reviewer: independent. Read-only against `/Users/samib/stable-envs-step7-dmc-quadruped`
@ `ad842a8` (branch `step7/dmc-quadruped`), parent `fa4ac534d5446613dac448e6ba4badedc5c8ef1a`.
Environment: `/Users/samib/se6-venv/bin/python` (torch 2.14 CPU, mujoco 3.12.0, dm_control 1.0.45),
`MUJOCO_GL=glfw`, Apple M5. No Oscar, no SLURM, no ssh. Nothing in the worktree was modified; the
`fa4ac53` comparison tree was materialised with `git archive | tar -x` into the scratchpad (no
`git worktree add`, so the branch repo's `.git` was not written to).

## VERDICT: APPROVE-WITH-CHANGES

Every number that lands in the shipped declaration reproduces exactly from the committed artifacts.
The thresholds follow the family's margin discipline and are strictly tighter than the provisional
entry they replace. No render-path file changed and I proved byte-identity on 6 cells. The changes
are two defects — one hygiene regression with a false claim attached, one latent test trap — plus
four nits. None of them touches a number.

---

## 1. Scope of the diff — PASS

`git diff fa4ac53..ad842a8 --stat`: 49 files, +10486/-71.

Under `stable_envs/` and `tests/` — **exactly two files**:

| file | change |
|---|---|
| `stable_envs/envs/dmcontrol.py` | +17/-16, entirely inside the `_QUADRUPED_AT` / `_QUADRUPED_AT_MEASURED_WITH` / `VISUAL_FIDELITY["DMCQuadrupedWalk"]` block |
| `tests/unit/test_raycast3d_primitive_bounds.py` | +13/-2, the Step-7 aggregate branch in `test_the_bounded_kinds_are_the_ones_the_committed_MEASUREMENT_chose` |

`git diff fa4ac53..ad842a8 --stat -- stable_envs/render/` is **empty**. Nothing outside the
declaration values, the bounds test, and documentation/artifacts.

Everything else is `examples/results/step7/dmc-quadruped/**` (artifacts + `findings.md` §P6) and
`scratch/` (sbatch scripts; `r5_kind_cost.py` / `r6_kind_cost_agg.py` gaining the ELLIPSOID row;
`measure_cell.py` gaining a numpy fallback for `scipy.ndimage.binary_dilation`).

Observation (not a defect): the `measure_cell.py` `_dilate` fallback is harness robustness rather
than part of (e). It is scratch-only, numerically equivalent to scipy's default 3x3-cross structure
at the same `iterations`, and produced no number committed in this commit.

## 2. Declaration vs the committed artifacts — PASS (one rounding nit)

Source of truth: `phase1/gpu/visual_calib_311/visual_DMCQuadrupedWalk.json` (3 draws x 64 states,
224 px) and `phase1/gpu/localize_311/localize.json` for the surface shares. Recomputed from `rows`:

| declared | value | artifact | ✓ |
|---|---|---|---|
| `mean_abs` | 3.52 | mean of rows 3.52312 (`mean_abs` 3.523) | ✓ |
| `draw_spread.mean_abs` | [3.520, 3.527] | min 3.519873 / max 3.526581 (`mean_abs_worst` 3.527) | ✓ |
| `within8_frac` | 0.897 | mean 0.897282 (`within8_frac` 0.8973) | ✓ |
| `draw_spread.within8_frac` | [0.8964, 0.8987] | min 0.896395 (`within8_worst` 0.8964) / max 0.898710 | ✓ |
| `ssim` | 0.955 | mean 0.954969 (`ssim` 0.955) | ✓ |
| `draw_spread.ssim` | [0.9536, 0.9557] | min 0.953625 (`ssim_worst` 0.9536) / max 0.955652 | ✓ |
| `all_exact_frac` | 0.171 | `exact_frac` 0.1709 | ✓ |
| `interior_frac` | 0.0284 | 0.028417 | ✓ |
| `skybox_share` | 0.148 | `localize_311` check `region_share.skybox` 0.147810 | ✓ |
| `floor_share` | 0.806 | `localize_311` check `region_share.floor` 0.806381 | ✓ |
| `interior_exact_frac` | 0.205 | draws [0.203653, 0.207905, 0.205292] — **mean 0.205617 → 0.206**; 0.205 is the MEDIAN | nit |

**NIT 1.** `interior_exact_frac` 0.205 is the median of the three draws, not the rounded mean
(0.206), while every other headline field in the entry is the rounded mean. It is explicitly "not a
threshold anywhere", so nothing depends on it — but the entry should be internally consistent, or
say it's the median.

### Thresholds — provenance discipline is intact; not fitted to the number

Ratios of threshold to the declared measurement (and to the worst draw, in brackets):

| cell | mean ceiling | within-8 floor | SSIM floor |
|---|---|---|---|
| DMCCheetahRun | 1.297 | 0.9524 | 0.9380 |
| DMCWalkerWalk | 1.301 | 0.9286 | 0.9322 |
| DMCHopperHop | 1.310 | 0.9503 | 0.9357 |
| DMCHumanoidStand | 1.310 | 0.9388 | 0.9447 |
| (DMCSwimmerSwimmer6, different `_AT`) | 1.391 | 0.9270 | 0.9533 |
| **DMCQuadrupedWalk (e)** | **1.307 [1.304]** | **0.9476 [0.9483]** | **0.9424 [0.9438]** |
| DMCQuadrupedWalk (provisional, retired) | 1.374 | 0.9313 | 0.9404 |

All three sit inside the four trackcom cells' band on all three axes — tighter on the ceiling than
the provisional entry was (which the phase-1 review flagged as looser than the "~1.3x" convention).
The thresholds moved **only in the strict direction**: `mean_abs_max` 4.7 → 4.6, `within8_min`
0.84 → 0.85, `ssim_min` 0.90 unchanged. They are round two-significant-figure numbers, not digits
of the measurement, so they are not tuned to it. No weakening anywhere.

### Version fields and the provisional marker

- `reference_versions` = `_AT_VERSIONS` = `{"mujoco": "3.11.0"}` — the same object the four trackcom
  cells use. ✓
- `measured_with` = `{"mujoco_warp": "3.11.0"}` (was `{"torch": "2.14.0"}`). ✓
- The `PROVISIONAL until the MJWarp draws` / `CPU backend cpu3d ... Apple M5` text is **gone**, and
  the new `_QUADRUPED_AT` names card, driver, `MUJOCO_GL=egl`, mujoco 3.11.0 + mujoco-warp 3.11.0,
  camera 2, 224 px, 3 draws, and the harness invocation. ✓
- The new string deliberately does **not** name a dm_control version — which is the family rule
  stated in the comment above `_AT_VERSIONS` ("neither string names a dm_control version, and a
  version nobody measured under is not provenance"). The provisional string *did* name it; the new
  one is now consistent with `_AT` / `_SWIMMER_AT`. dm_control 1.0.44 is recorded in the artifact
  provenance (`localize_311/localize.json` → `packages.dm_control: "1.0.44"`). Correct as shipped.

The harness is the family's own `scratch/dmcontrol/visual_calib.py` (unchanged by this commit), and
it measures through `se.make(..., device="cuda")` + `C.reference_env` / `cls.reference_render` — the
shipped path, three seeds (100/101/102), worst draw driving the ceiling. ✓

**NIT 2.** `_QUADRUPED_AT` credits "jobs 6394721 / 6394987, gpu2104 / gpu2107", but every artifact the
declaration is derived from is 6394987 on gpu2107 (6394721's outputs were overwritten; §P6 says so).
Co-attributing gpu2104 in the declaration itself is over-broad.

**NIT 3.** `visual_calib_311/visual_DMCQuadrupedWalk.json` — the single artifact the declaration is
read off — carries **no host/job/commit stamp of its own**. Provenance for it is inferred from the
co-located `provenance.txt` and from `gpu_declare2.sbatch` ordering. Every sibling artifact in the
same directory (`gateA_cuda_*`, `localize_*/localize.json`, the paired-sweep files) stamps
host + slurm job + `git_commit` in-band.

## 3. Identity artifacts — PASS

| artifact | n | result |
|---|---|---|
| `phase1/gpu/cuda_identity_raycast21.json` | 21 | `all_identical: true`; every config `max_abs_diff 0, px_differing 0, frames_differing 0` |
| `phase1/gpu/cuda_identity_trackcom4.json` | 4 (Cheetah/Hopper/Humanoid/Walker) | same |
| `phase1/cpu_oscar/cpu_identity25_node1913.json` | 25 (incl. all 11 maze cells) | same |

21 + 4 = 25 cuda configs, 0 differing; 25/25 CPU, 0 differing. The 11 OGBench maze cells the phase-1
review could not reach locally are covered by the CPU run. ✓

Provenance is per-directory rather than per-artifact: `phase1/gpu/provenance.txt` records
`job 6394721 node gpu2104` and `job2 6394987 node gpu2107`, `GIT_COMMIT fa4ac53…`,
`MERGED_COMMIT 0a029bd`; `phase1/cpu_oscar/provenance_node1913.txt` records `job 6394722
node node1913`, Xeon 8268, same two SHAs. `GIT_COMMIT` is `fa4ac53`, which is the correct tree
(verified below to equal `0a029bd` in code).

**NIT 3 (cont.)** the three identity JSONs themselves carry only `before_dir`/`after_dir`/`configs`
— no SHA, host, or job.

**"Kept results are gpu2107's" — consistent.** `gateA_cuda_mj3110.json` and `gateA_cuda_mj3120.json`
stamp `host: gpu2107.oscar.ccv.brown.edu`, `git_commit: fa4ac53…`, `git_dirty: null`;
`localize_311/localize.json` stamps `host gpu2107`, `slurm_job_id 6394987`. `gpu_declare2.sbatch`
(job name `se7-dmc-quadruped-gpu2`, `--exclude=gpu2104`) is the script whose steps 4–6 write
`visual_calib_311`, `gateA_cuda_*` and `localize_*`, so the declaration artifact is 6394987's by
construction. ✓  (Phase-1 recommendation 5 — "regenerate at a commit that defines the cell" — is
satisfied: the stamps now read `fa4ac53` with `git_dirty: null`, not `68b411c` + dirty.)

## 4. Kind cost / `_BOUNDED_KINDS` — main claim PASS; one latent defect, one labelling gap

- `stable_envs/render/raycast3d.py:665` is still `_BOUNDED_KINDS = frozenset({CAPSULE, CYLINDER, BOX})`
  — ELLIPSOID absent, and the whole `stable_envs/render/` tree is untouched by (e). ✓
- The committed aggregate is at `kind_cost_ellipsoid_node1913_agg.json`; the test reads
  `kind_cost_ellipsoid.json`. That file does not exist, so the new branch is inert today and the
  unit suite is green. ✓
- **Cannot silently admit.** I simulated the test's merge logic against the committed aggregate:
  - dropping in a file with `predicate.ELLIPSOID = true` → `set(_BOUNDED_KINDS) == measured` fails
    (`code=[BOX,CAPSULE,CYLINDER]` vs `measured=[BOX,CAPSULE,CYLINDER,ELLIPSOID]`) **and** the rule
    loop fails (`min(worst)=0.695` is not > 1.0). The code constant must move with it and the numbers
    must support it. ✓
- The load-contaminated runs are labelled — **in `findings.md` §P6 only** ("ran three times on
  node1913 at load ~34 … the node was too loaded to reproduce the committed predicate"), with the
  corroborating in-band tell that BOX collapses to 0.949 in run 2 against the frozen table's
  1.11–1.13. The aborted paired sweeps, by contrast, are labelled **in-band and correctly**:
  `cpu_1thread_paired_node1913_6394722.json` and `…_node2330_6395512.json` both carry
  `certifiable: false`, `envs: []`, `load_average_at_start` 34.52 / 32.40 vs the 12.0 bound, and a
  `not_certifiable_because` sentence. No tier is derived. ✓

### DEFECT A (latent, test-only) — the Step-7 branch is unsatisfiable for the kind it was added for

`test_the_bounded_kinds_are_the_ones_the_committed_MEASUREMENT_chose` keeps the Step-6
strict-separation assertion for unbounded kinds: `assert max(best) < 1.0`. ELLIPSOID's measured
best-case ratios are **1.427 / 1.213 / 1.371** and its worst-case **0.885 / 0.695 / 0.899** — it
straddles 1. Simulated both resolutions with the committed aggregate placed at the read path:

```
as-measured (ELLIPSOID false): FAIL — ELLIPSOID unbounded but best 1.427
hypothetical  (ELLIPSOID true): FAIL — set mismatch, and ELLIPSOID bounded but worst 0.695
```

So committing the honest aggregate under the canonical name turns the suite red **with no correct
resolution**, unless a quiet node happens to lift ELLIPSOID's worst ratio above 1 at every width in
every run. The strict-separation form was only safe for the Step-6 kinds because they had a 2.65x
gap (`separation.gap_x`). Consequently §P6's claim that "the test now knows how to read a Step-7
aggregate at that path when a quiet-node measurement lands" is **not true as written**.

Fix (test-only, small): the complement of the rule `bounded iff worst > 1 in every run` is
`min(worst) <= 1.0`, not `max(best) < 1.0`. Either weaken the unbounded assertion to that form for
the Step-7 row, or exempt a straddling kind explicitly and say why — and correct the §P6 sentence.

### Recommendation A2 — mark the contamination in-band

`kind_cost_ellipsoid_node1913_run{1,2,3}.json` and `…_agg.json` record host / job / threads / torch /
numpy but **no load average and no contamination flag**. Give them the same treatment the paired
sweeps got (`load_average_at_start` + a `not_authoritative_because` sentence), so the aggregate
cannot be promoted to the read path by someone who only reads the JSON.

## 5. Tests and byte identity vs the merged integration tree — PASS

```
tests/unit/test_raycast3d_primitive_bounds.py tests/unit/test_declarations.py
  -> 118 passed, 1 skipped
tests/unit -q --deselect tests/unit/test_examples.py
  -> 1 failed, 735 passed, 6 skipped, 111 deselected
```
The one failure is `test_repo_contracts.py::test_every_cited_scratch_path_resolves` — already red
before this commit, but made worse by it; see DEFECT B.

Code equivalence:
```
(in stable-envs-step7-integration) git diff 0a029bd fa4ac53 --stat -- stable_envs/ tests/   -> empty
(same repo)                        git diff 0a029bd ad842a8 --stat -- stable_envs/render/
                                        stable_envs/envs/dmcontrol.py stable_envs/core/
                                   -> stable_envs/envs/dmcontrol.py | 33 +++-- (17 insertions, 16 deletions)
```
i.e. `0a029bd` equals `fa4ac53` in code, and (e)'s only code delta against it is the declaration
block. No render path.

**Render digest, empirically** — CPU, first 8 frozen Step-4 states per config, sha256 of the raw
`uint8` buffer, `ad842a8` vs an extracted `fa4ac53` tree, `MUJOCO_GL=glfw`, `OMP_NUM_THREADS=1`:

```
DMCCheetahRun      ac2ff890631cf03f…  identical
DMCWalkerWalk      41266c21a1dc33e8…  identical
DMCHopperHop       694f0329ebcaf465…  identical
DMCHumanoidStand   29687a52a02fd740…  identical
DMCReacherEasy     cb03a4772ce6cde2…  identical
DMCFingerSpin      ba5c3b2d35fe6d1c…  identical
```
6/6 byte-identical — the four trackcom cells plus two raycast3d cells. (e) touches no pixel.

### Gate A artifacts — note, not a defect

The four committed Gate A runs were produced at `fa4ac53`, i.e. **against the provisional
declaration**: their `check_fidelity_declaration.evidence.visual_fidelity` embeds the retired entry
(3.42 / 0.902 / 0.957, thresholds 4.7 / 0.84 / 0.90) with `agrees: true`. All 14 mandatory checks are
`measured` on both devices under both arms, and the measured `check_visual_fidelity` observations all
clear the **new** thresholds with margin:

| arm | mean (< 4.6) | within-8 (> 0.85) | SSIM (> 0.90) |
|---|---|---|---|
| cuda 3.11 | 3.5089 | 0.9001 | 0.9557 |
| cpu 3.11 | 3.3752 | 0.9035 | 0.9586 |
| cuda 3.12 | 3.5491 | 0.8970 | 0.9538 |
| cpu 3.12 | 3.4167 | 0.9006 | 0.9564 |

So nothing is wrong — but Gate A has not been re-run against the entry the repo now ships, and an
auditor opening the artifact reads the superseded declaration. Worth a line in §P6, or a re-run at
`ad842a8` when the cluster is next available.

## 6. Hygiene — PASS except one regression

```
git log --format=%B fa4ac53..ad842a8 | grep -ciE "co-authored-by|generated with|claude"  -> 0
git log --format='%an <%ae>' fa4ac53..ad842a8 | sort -u -> sami-bg <samibghanem@gmail.com>
git diff --stat fa4ac53..ad842a8 -- examples/results/step4 examples/results/step5 \
                                    examples/results/step6                        -> (empty)
```
Committed bytes: ~1.35 MB total. Largest two are the 380 KB / 379 KB `localize_311/sheets/*.png`
contact sheets; the rest are JSON/MD. The two 8 MB `localize_*/frames/` `.npz` stacks are correctly
left untracked. No multi-MB frame stack committed. ✓

### DEFECT B — dangling cited-scratch paths go 9 → 21, and §P6 misattributes the failure

I ran `tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves` on both trees:

```
fa4ac53 (git archive into the scratchpad):  9 cited scratch path(s) do not resolve
ad842a8 (the branch worktree):             21 cited scratch path(s) do not resolve
```

The 12 new ones are all `scratch/sboughan/venvs/se-step5/**` and `scratch/sboughan/venvs/se-vis311/**`
paths, and they come from four Oscar stdout captures this commit adds:

```
examples/results/step7/dmc-quadruped/phase1/gpu/egl_probe_311_gpu2107.txt
examples/results/step7/dmc-quadruped/phase1/gpu/egl_probe_312_gpu2107.txt
examples/results/step7/dmc-quadruped/phase1/gpu/mujoco_warp_source_se-step5.txt
examples/results/step7/dmc-quadruped/phase1/gpu/mujoco_warp_source_se-vis311.txt
```

`test_repo_contracts._files()` scans `.txt` under `examples/`, and `_CITE = scratch/[A-Za-z0-9_./{}*,\-]+`
matches the tail of the absolute Oscar paths (`/oscar/scratch/sboughan/venvs/…`) inside those logs,
so each one registers as a repo citation that cannot be opened. It does not create a *new* failing
test — the test was already red on 9 Step-6 citations — but it more than doubles the contract's
outstanding debt in a commit whose stated job is provenance.

It also makes a claim in the commit false. §P6 says the unit failure is "the pre-existing
`test_every_cited_scratch_path_resolves` (Step-6 citations)". That was true of the Oscar run
(`unit_tests_node1913.txt`, executed at `fa4ac53`), but in the tree this commit produces, **12 of
the 21 are this commit's own Step-7 citations**.

Fix (documentation/hygiene, no number affected): scrub or relabel the absolute venv paths in those
four captures (e.g. rewrite `/oscar/scratch/sboughan/venvs/` to a non-matching form), or exclude
`examples/results/**` stdout captures from the citation scan; and correct the §P6 sentence.

### NIT 4 — §P6's untracked-artifact inventory is incomplete

§P6: "the two 8 MB `localize_*/frames/*.npz` stacks are left untracked; everything else is committed".
`git status` in the worktree also shows `phase1/gpu/localize_312/sheets/` (764 KB) untracked — the
asymmetry with `localize_311/sheets/`, which *is* committed, is unexplained — plus two uncommitted
scripts `scratch/step7/dmc-quadruped/sweep_quiet_{cpu,gpu}.sbatch`.

---

## REQUIRED CHANGES

1. **DEFECT B** — remove the 12 new dangling `scratch/sboughan/venvs/...` citations introduced by the
   four committed Oscar `.txt` captures (scrub the absolute paths, or exclude those captures from
   the scanner), and correct §P6's "(Step-6 citations)" attribution.
2. **DEFECT A** — fix the bounds test's Step-7 branch so a straddling ELLIPSOID can be recorded
   truthfully: replace `max(best) < 1.0` with `min(worst) <= 1.0` for the row decided by the
   worst-ratio rule (or exempt it explicitly), and correct §P6's "the test now knows how to read a
   Step-7 aggregate" sentence.

## RECOMMENDED (not blocking)

3. Give `kind_cost_ellipsoid_*.json` the paired-sweep treatment: `load_average_at_start` and a
   `not_authoritative_because` sentence in-band, so the aggregate cannot be promoted by someone who
   reads only the JSON.
4. Make `interior_exact_frac` 0.205 consistent with the other fields (rounded mean 0.206) or say it
   is the median.
5. Stamp `visual_calib_*/visual_<env>.json` with host / slurm job / git commit the way every sibling
   artifact in the directory is stamped; likewise the three identity JSONs.
6. Narrow `_QUADRUPED_AT`'s job/node credit to 6394987 / gpu2107, the job whose outputs were kept.
7. Note in §P6 that the committed Gate A artifacts predate the shipped declaration (their
   `check_fidelity_declaration` evidence shows the retired provisional entry), and that all four
   measured observations nevertheless clear 4.6 / 0.85 / 0.90; re-run at `ad842a8` when convenient.
8. Complete §P6's untracked inventory (`localize_312/sheets/`, the two `sweep_quiet_*.sbatch`).

## Commands run (abridged)

```
git diff fa4ac53..ad842a8 --stat ; -- stable_envs/ tests/ ; -- stable_envs/render/
git diff fa4ac53..ad842a8 -- examples/results/step4 examples/results/step5 examples/results/step6
git log --format=%B fa4ac53..ad842a8 | grep -ciE "co-authored-by|generated with|claude"
git log --format='%an <%ae>' fa4ac53..ad842a8 | sort -u
(in stable-envs-step7-integration) git diff 0a029bd fa4ac53 --stat -- stable_envs/ tests/
MUJOCO_GL=glfw se6-venv/bin/python -m pytest tests/unit -q -p no:randomly --deselect tests/unit/test_examples.py
MUJOCO_GL=glfw se6-venv/bin/python -m pytest tests/unit/test_raycast3d_primitive_bounds.py tests/unit/test_declarations.py -q
git archive fa4ac53 | tar -x -C <scratch>/base_fa4ac53   # comparison tree, no git worktree add
MUJOCO_GL=glfw se6-venv/bin/python -m pytest tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves -q   # both trees
MUJOCO_GL=glfw OMP_NUM_THREADS=1 se6-venv/bin/python <scratch>/digest.py <tree> <6 configs>       # both trees
se6-venv/bin/python <inline> : recompute declaration fields from visual_calib_311 rows
se6-venv/bin/python <inline> : simulate the bounds test with kind_cost_ellipsoid_*_agg.json at the read path
```

---

# Final verification before merge — tip `31bd3e9`

Re-check of `ad842a8` + `656f9db` (e2) + merge `b5c56ea` of `step7/integration` `5adc501` +
`31bd3e9` (INTEG-1 loader hook). Same environment; worktree read-only and unmodified (only the three
pre-existing untracked `localize_*` dirs remain — the two `sweep_quiet_*.sbatch` are now committed).
Comparison trees made with `git archive | tar -x`.

## VERDICT: APPROVE

Both defects are fixed at the level I asked for, the INTEG-1 hook is minimal and provably
pixel-neutral, and hygiene is clean over the whole range. One item to hand back to **integration,
not W4** (see the note under item 2).

### (1) DEFECT A — FIXED

`tests/unit/test_raycast3d_primitive_bounds.py` now reads:

```python
assert min(worst) <= 1.0, (
    f"{kind} is unbounded but its worst ratio is > 1 in every run ({min(worst):.2f})")
```

— the honest complement of `bounded iff worst > 1 in every run`, with a comment naming the straddle
that made the old `max(best) < 1.0` unsatisfiable. The aggregate is now committed **at the read
path**, `phase1/cpu_oscar/kind_cost_ellipsoid.json`, so the branch is live rather than inert. I
replayed the merge logic against the committed files:

```
read path exists: True
ELLIPSOID predicate: False -> measured set: ['BOX', 'CAPSULE', 'CYLINDER']   == _BOUNDED_KINDS ✓
  SPHERE    bounded=False min(worst)=0.272 clause_ok=True
  BOX       bounded=True  min(worst)=1.109 clause_ok=True      (Step-6 frozen row, not the loaded 0.949)
  CYLINDER  bounded=True  min(worst)=1.229 clause_ok=True
  CAPSULE   bounded=True  min(worst)=1.468 clause_ok=True
  ELLIPSOID bounded=False min(worst)=0.695 clause_ok=True      -> green UNBOUNDED verdict
```
`pytest …::test_the_bounded_kinds_are_the_ones_the_committed_MEASUREMENT_chose` passes. Only the
ELLIPSOID rows are merged from the Step-7 file, so the contaminated BOX row cannot displace the
frozen table. Admission is still two-sided: `predicate true` would require `min(worst) > 1.0` **and**
a matching `_BOUNDED_KINDS` edit. `ELLIPSOID` remains absent from `_BOUNDED_KINDS`.

**Recommendation A2 also taken**: `kind_cost_ellipsoid.json` now carries an in-band `load_note`
(node1913 job 6394722, the paired sweep's load 34.5, BOX's run-2 collapse to 0.95 against the frozen
1.11–1.13) plus a `provenance` block (git commit, merged commit, host, job).

### (2) DEFECT B — FIXED at W4's tip; the tip's 18 are inherited from integration

Computed with the test's own `CITATIONS`/`_expand` on extracted trees:

| commit | broken cited scratch paths |
|---|---|
| `76dc5116` (baseline) | 9 |
| `656f9db` (W4's e2 tip, pre-merge) | **9 — set is byte-identical to the baseline's** |
| `5adc501` (step7/integration) | 19 |
| `b5c56ea` / `31bd3e9` (tip) | 18 |

```
diff <(sort cits_656f9db) <(sort cits_76dc5116)      -> IDENTICAL
comm -23 <(sort cits_31bd3e9) <(sort cits_5adc501)   -> (empty)   # the tip adds nothing over integration
comm -13 <(sort cits_31bd3e9) <(sort cits_5adc501)   -> scratch/step7/dmc-quadruped/sweep_quiet_{cpu,gpu}.sbatch
```
So W4 introduces **zero** dangling citations, and the merge actually *repairs* one integration entry
(the dashboard cited W4's then-uncommitted `sweep_quiet_*.sbatch`, which e2 commits). The author's
"citation set = baseline 9" is exactly true of the W4 tree.

**Hand-back to integration (not a W4 defect).** The 9 tip-only entries all come from documents that
live on `step7/integration`:
```
scratch/sboughan/venvs/ , .../se-step5/** , .../se-vis311/**   <- examples/results/step7/dmc-quadruped/review_commit_e.md
scratch/sboughan/se7/dmc-manipulator/stable_envs/__init__.py   <- dmc-manipulator review_phase1.md + profile/*.md
scratch/step7/rocket-landing/* , /measure_tier.py , /nvdiff_identity.py ,
scratch/.../measure_tier.py , scratch/.../oscar_fidelity.sbatch <- rocket-landing review_phase1.md
```
The first three are my own review report quoting the very paths it asked W4 to scrub — the scanner
reads prose in `examples/**/*.md` as citations. Suggest an integration-level fix (exempt
`examples/results/**/review_*.md`, or quote such paths in a non-matching form) and re-baselining the
expected count; W4 should not be asked to carry it.

### (3) INTEG-1 hook — minimal, confined, complete

`git diff -U0 b5c56ea..31bd3e9 -- stable_envs/envs/dmcontrol.py` hunk headers:
`@@ -982,0 +983,13`, `-1005`, `-1295`, `-1314,2`, `-1321`, `-1323`, `-2891,7`, `-2902,53`.
Region map at `b5c56ea`: `_DMCModel` **924–1364**, `DMControlSuite` **1365–2396**,
`DMControlTrackCom` **2491–2836**, `_SWMQuadrupedModel` **2890–**. So every hunk is either in
`_DMCModel` or in the mixin: **W3's `DMControlSuite` region and `DMControlTrackCom` are untouched.**

The base edit is exactly hook + call lines: a new `_load_env(cls, seed)` classmethod (one statement
plus docstring) and three call sites — `_bundle`, `sample_states`, `reference_env` — each a
one-line `_suite().load(...)` → `cls._load_env(...)` swap, plus `reference_env`'s signature
`size: int = 84` → `size: int | None = None` with `int(size or cls.default_size)`. Nothing else in
the base changed. The mixin drops 53 lines of copied `_bundle`/`sample_states`/`reference_env` and
keeps `default_size = 224` + a two-line `_load_env`.

`grep -n "_suite().load(" stable_envs/envs/dmcontrol.py` → **one hit, line 994, inside `_load_env`
itself**. No base loader calls it directly.

The replacement test is stronger than the one it retires: it asserts
`vars(_SWMQuadrupedModel) == {default_size, _load_env}`, that `cls._bundle/​reference_env/​sample_states`
**are** `_DMCModel`'s function objects, that each base loader's source contains `_load_env(` and no
`_suite().load(`, that the hook *is* overridden, and the two sizes. Passes.

Benign behaviour note: `_DMCModel.default_size == 84`, so bare `reference_env()` is unchanged for
every suite/trackcom cell; for the two 224-px cells (`DMCManipulatorBringBall`, `DMCQuadrupedWalk`)
it now returns 224 instead of 84, which is the correct size and matches the old mixin override.
The only bare DMC caller anywhere is `scratch/step_bench.py`; every shipped caller
(`conformance.py`, the gpu/unit tests, `examples/*`) passes `size` explicitly.

### (4) Byte identity vs integration `1d9dccb` — 9/9, 0 px

CPU, `batch=4`, states from `sample_states(4, rng=default_rng(0))` in each tree, at each cell's
registered default size; raw `uint8` buffers compared elementwise.

```
DMCCheetahRun             px_differing=0 max_abs=0 states_equal=True
DMCHopperHop              px_differing=0 max_abs=0 states_equal=True
DMCWalkerWalk             px_differing=0 max_abs=0 states_equal=True
DMCHumanoidStand          px_differing=0 max_abs=0 states_equal=True
DMCQuadrupedWalk  (224)   px_differing=0 max_abs=0 states_equal=True
DMCManipulatorBringBall (224) px_differing=0 max_abs=0 states_equal=True
DMCReacherEasy            px_differing=0 max_abs=0 states_equal=True
DMCCartpoleSwingup        px_differing=0 max_abs=0 states_equal=True
DMCFingerSpin             px_differing=0 max_abs=0 states_equal=True
```
The four trackcom cells, quadruped, W3's manipulator, plus three `DMControlSuite` cells that exercise
the rewritten base loaders. **`states_equal` is also true everywhere**, so `sample_states` through the
hook reproduces the copies bit-for-bit — the hook is neutral on the state distribution as well as on
the pixels.

### (5) `tests/unit` on the branch

```
MUJOCO_GL=glfw OMP_NUM_THREADS=1 pytest tests/unit -q -p no:randomly --deselect tests/unit/test_examples.py
-> 1 failed, 812 passed, 7 skipped, 111 deselected
```
Exactly the author's numbers, and the single failure is `test_every_cited_scratch_path_resolves` —
inherited from integration, zero W4 contribution (item 2). No other failure.

### (6) Hygiene over `76dc5116..31bd3e9`

```
git log --format=%B 76dc5116..31bd3e9 | grep -ciE "co-authored-by|generated with"  -> 0
git log --format='%an <%ae>' 76dc5116..31bd3e9 | sort -u -> sami-bg <samibghanem@gmail.com>
git diff --stat 76dc5116..31bd3e9 -- examples/results/step4 examples/results/step5 \
                                     examples/results/step6                        -> (empty)
```
0 trailers, single author (including every commit that arrived through the integration merge), frozen
Step-4/5/6 result trees untouched.

### Earlier nits — status

| nit | status |
|---|---|
| `interior_exact_frac` 0.205 (median) | fixed → **0.206** (rounded mean) |
| `_QUADRUPED_AT` crediting gpu2104 | fixed → job 6394987 / gpu2107, with the overwrite explained |
| artifacts missing host/job/commit stamps | fixed — `visual_calib_*`, the three identity JSONs, gateA, localize, kind-cost all carry provenance blocks now |
| kind-cost contamination only in findings.md | fixed — in-band `load_note` |
| §P6 untracked inventory incomplete | `sweep_quiet_*.sbatch` committed; the `localize_*/frames/` + `localize_312/sheets/` remain untracked as documented |
| Gate A artifacts predate the shipped declaration | still true by construction; all four measured VF observations clear 4.6 / 0.85 / 0.90, verified earlier |

## Commands run (final pass, abridged)

```
git diff ad842a8..656f9db --stat ; -- tests/unit/test_raycast3d_primitive_bounds.py stable_envs/envs/dmcontrol.py
git diff b5c56ea..31bd3e9 -- stable_envs/envs/dmcontrol.py tests/unit/test_cpu3d.py
git diff -U0 b5c56ea..31bd3e9 -- stable_envs/envs/dmcontrol.py | grep '^@@'
git show b5c56ea:stable_envs/envs/dmcontrol.py | grep -n '^class '      # region map
grep -n "_suite().load(" stable_envs/envs/dmcontrol.py
for c in 656f9db 5adc501 b5c56ea 76dc5116; do git archive $c | tar -x -C <scratch>/t_$c; \
   pytest tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves ; done
git archive 1d9dccb | tar -x -C <scratch>/t_1d9dccb
MUJOCO_GL=glfw OMP_NUM_THREADS=1 pytest tests/unit -q -p no:randomly --deselect tests/unit/test_examples.py
MUJOCO_GL=glfw pytest …::test_the_bounded_kinds_are_the_ones_the_committed_MEASUREMENT_chose \
                     …::test_the_swm_quadruped_mixin_overrides_only_the_loader_hook_and_the_size
MUJOCO_GL=glfw OMP_NUM_THREADS=1 python <scratch>/ident2.py <tree> <out> <9 configs>   # both trees
```
