# Step 9 Wave 1 — Workstream H (render/primitives.py, raycast3d, cpu3d) — independent review

**Reviewed SHA:** `d231a8f332290b5a7d4b971f70da60202aab30b1` (`step9/render-3d` HEAD)
**Code delta reviewed:** `git diff d07ffa8 900065a` (H's own commits) plus the two coordinator
commits after it (`9a43c34` self-compare classification + `d231a8f` artifacts). The merge of
`step9/integration` at `430bb6d` (Workstream A + the Wave-0 harness fix `e1da2f6`) is out of scope
and was reviewed separately (`review_a.md`).
**Step-8 behavioural baseline:** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`
**Reviewer environment:** Apple M5, local CPU only, `/Users/samib/se6-venv` and
`/Users/samib/se8-vm-venv` (torch 2.14.0, mujoco 3.12.0, dm_control), `MUJOCO_GL=glfw`,
`-p no:cacheprovider`. No edits, commits, stashes, resets, pushes; Oscar untouched.

APPROVE

---

## The §17 ten items

1. **Files moved / removed / created** — Confirmed against `git diff d07ffa8 900065a --stat`.
   Created: `stable_envs/render/primitives.py` (182 lines), `tests/unit/test_render_primitives.py`
   (178), `scratch/step9/render-3d/{oracle.py,bench.py,hmg_determinism.py,gpu_oracle.sbatch,
   hmg_determinism.sbatch,oracle_selfcompare.sbatch}`, `examples/results/step9/render-3d/*`.
   Edited: `stable_envs/render/raycast3d.py` (-244/+…, definitions moved out and re-imported),
   `stable_envs/render/cpu3d.py`. Nothing removed from any other file.
   **Scope is exactly the owned set.** `git diff d07ffa8 900065a --stat -- stable_envs/envs tests/gpu`
   is empty (verified, zero output): **zero consumer call-site changes**. `render/primitives.py`
   imports `dataclasses`, `numpy`, `torch` and nothing else — the §8 dependency rule (render never
   imports envs or SWM) holds by construction, and the parametrised
   `test_a_renderer_knows_nothing_about_environments_or_stable_worldmodel` picks the new file up
   automatically (it passes in my own run).

2. **API changes** — Additive only, verified symbol by symbol. `raycast3d` re-exports
   `PLANE/SPHERE/CAPSULE/ELLIPSOID/CYLINDER/BOX`, `INF`, `Primitives`, and binds
   `_dot/_mv/_mtv/_bound_radius` to `primitives.dot3/mv3/mtv3/bound_radius_tensor`, so every private
   name the tests and the five consumer modules reach for resolves to the same object at the same
   name. New: `primitives.*`, `raycast3d.Kinds`/`Cast`/`_Memo`, `cpu3d.WINDOW_EMPTY`. Nothing renamed
   or removed. `cpu3d.MOVER_KINDS = tuple(k for k in PRIMITIVE_KINDS if k != PLANE)` evaluates to
   `(2, 3, 4, 5, 6)` — identical tuple, identical order, to the baseline's literal.

3. **Step-8 IDs affected** — The 44 cells that reach either backend, enumerated in findings §3 and in
   `oracle.py:CELLS` (38 configs × up to 2 devices). Behaviourally **none** changed: see item 6.

4. **Targeted tests** — Re-run by me:
   - `MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit -x
     --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"`
     → `1385 passed, 195 skipped, 112 deselected, 26619 warnings in 108.39s` — the expected count at
     HEAD (findings §4's 1263 is the pre-merge number; the delta is Workstream A's merged tests).
   - `MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python -m pytest -q -p no:cacheprovider
     tests/unit/test_render_primitives.py tests/unit/test_cpu3d.py
     tests/unit/test_variation_maps_dmc_control.py tests/unit/test_raycast3d*.py`
     → `341 passed, 4088 warnings in 33.53s`.
   The 16 new tests are well-aimed at the exact failure modes the extraction could introduce:
   `R3._dot is P.dot3` by **identity** (a duplicated expression would pass a value comparison),
   batch-size invariance of the unrolled product, both bound-radius spellings agreeing per kind
   including ELLIPSOID and actually containing a sampled surface, every memo bounded, and both
   dispatchers refusing `mjGEOM_MESH` with the same message.

5. **Applicable certification checks** — No threshold, declaration, tier or frozen digest is touched
   by the diff (verified: the diff contains no change to any numeric constant other than moving
   `INF = 1e30` between files). The certification harnesses that exercise these paths
   (`tests/gpu/test_conformance.py`, `test_reproducibility.py`, `test_fast_paths.py`) are GPU-only;
   see item 10 and Defect D2.

6. **Default-frame comparison — independently reproduced.** I re-ran the author's harness at branch
   HEAD, in the *other* venv, into a temp dir:
   ```
   cd /Users/samib/stable-envs-step9-render-3d && MUJOCO_GL=glfw \
     /Users/samib/se8-vm-venv/bin/python scratch/step9/render-3d/oracle.py \
     --python /Users/samib/se8-vm-venv/bin/python --out <tmp>/oracle_rev --tag review
   ...
   "identical": true, "cells_ok": 21
   wrote <tmp>/oracle_rev/oracle_review.json
   ORACLE: PASS
   ```
   21 cells ran (the 12 DMC parity cells incl. DMCPointMassEasy and DMCManipulatorBringBall, the
   5 trackcom `cpu3d` cells, and the 4 gym-MuJoCo cells), 64 states each; 17 skipped for absent
   optional packages with byte-identical reasons in both arms. I then diffed **my** record against the
   committed `oracle_cpu_local.json` on `frames_sha256`, `frames_from_rows_sha256`, `states_sha256`,
   `size`, `frames_shape`, `backends` and the switch digests: **zero mismatches on all 21 cells**, and
   the `primitive_bounds_off` / `cover_compaction_off` / `composite_window` digests for
   `DMCReacherHard` and `DMCBallInCupCatch` equal both the committed values and each cell's shipped
   digest — so the bounds-off and compaction-off reference expressions are preserved too.

7. **Variation / state comparison** — Untouched by construction; neither module knows what a
   variation is, and the oracle compares the state digest each arm was handed (identical), so "same
   frames" is not an artefact of different inputs. `tests/unit/test_variation_maps_dmc_control.py`
   passes in the run above.

8. **Performance** — `timings_cpu_local.json`: 8 cells, 5 alternating baseline/head rounds × 5 reps,
   separate interpreter per leg, worst `ratio_head_over_baseline` = **1.0166** (DMCReacherHard, whose
   own same-arm spread is 5.6%), `blocks_merge: false`, bar 1.10.
   `timings_gpu_6464319.json` (gpu1503, RTX A6000, same job, same card): worst **1.0057**
   (DMCReacherHard at batch 256), `blocks_merge: false`. Both are inside each row's own spread and
   far inside §8/§21's ~10% bar. Mechanistically expected: no arithmetic, allocation or kernel launch
   changed, and `_Memo.put` runs only on a cache miss. **No regression.**

9. **Resource lifetime** — Neither module owns a process resource (no EGL/CUDA context, no handle).
   The five module-level memos are now one `_Memo` type, each with a cap (32, 32, 64, 8, 16), and
   `test_every_raycast3d_cache_is_bounded` asserts it. `Primitives._radii_cache` is moved **verbatim**
   (byte-for-byte identical method body, including the `np.isfinite` guard and the
   `(str(device), dtype, float(scale))` key) — confirmed by reading both versions. Nothing outside
   `raycast3d.py` binds or reassigns these caches (only the new test and an unrelated
   `scratch/fpfusion/` copy reference them), so the `dict`-subclass change is invisible to consumers.
   See Defect D1 for the one cap-semantics off-by-one.

10. **Unexplained oracle differences attributable to the patch: ZERO — and the evidence is
    sufficient. I concur with the coordinator's §9d classification.** My reasoning, from the
    artifacts rather than the prose:
    - `oracle_gpu_6464319.json` (gpu1503, A6000): 59 of 76 pairs ran, 58 identical, the sole
      difference `HumanoidmazeGiantNavigate/cuda` (`frames_sha256` + `frames_from_rows_sha256`;
      baseline `929f7c09…` / mean 172.771413583 vs head `6c5385e4…` / 172.771393653).
    - `hmg_determinism_gpu_6464500.json` (gpu1504, A40): 3 fresh processes per tree per device, all
      `6c5385e4…`, `pixels_differing 0/9,633,792`, `frames_touched 0/64`, on the shipped path and on
      `force_torch`, `primitive_bounds_off` and `composite_window`.
    - `oracle_gpu_6464601_{baseline_baseline,head_head,baseline_head}.json`: all three legs on
      **gpu1503**, the very card, in the **original 38-cell process shape**, `"identical": true`,
      27 cuda cells each. **Decisively: in the `baseline,baseline` and `baseline,head` legs the
      BASELINE tree emitted `6c5385e4…` on gpu1503 — a different digest from the `929f7c09…` the same
      unmodified baseline source emitted on the same card in job 6464319.** The same source therefore
      produces both values across processes on that card, which is the definition of a
      non-process-reproducible cell, and the `baseline,head` leg shows baseline and head agreeing
      bit-for-bit on the configuration that produced the anomaly.
    - Independently, the code diff contains no numeric change at all: every expression that produces
      a pixel is character-identical and merely lives in another file (item 11 below). A patch cannot
      move a pixel it does not compute differently.
    Together these close §17 item 10: the one historical difference is a **DOCUMENTED_LIMITATION** of
    `HumanoidmazeGiantNavigate` on cuda (the `_BOUND_SLACK_NOISE` configuration; Step-6 job 6348207
    moved one pixel by 22/255 on this cell alone), not a patch effect. Head has produced exactly one
    digest on every run, on two cards, in both process shapes.
    **Coverage claims are recorded, not claimed:** findings §9 and §11 state explicitly that
    `swm/OGBCube-v0`, `swm/OGBScene-v0` and the four `PointMaze_*-v3` ids (six ids) are *covered
    elsewhere* by coordinator ruling (Workstreams D/F, their own venvs), and that the five DM Control
    trackcom cells' **cuda** arm is Workstream I's MJWarp path — their **cpu** (`cpu3d`) arms did run
    and are identical, which I reproduced. `se-step6` lacks `warp`, `stable_worldmodel` and
    `gymnasium_robotics`, and the oracle records each skip with its reason in both arms. No coverage
    is overstated anywhere I checked.

---

## 11. Exact-output preservation — line-by-line reading of the diff

I read the whole `raycast3d.py` and `cpu3d.py` diff, not a summary. Every numeric expression is moved
verbatim:

- `dot3`/`mv3`/`mtv3` — the nine-multiply/six-add spellings, including the operand order and the
  `torch.stack([...], -1)`, are character-identical to the baseline `_dot`/`_mv`/`_mtv`
  (`primitives.py:66-85` vs `d07ffa8:raycast3d.py:270-293`). No reassociation, no `einsum`, no dtype
  change.
- `bound_radius` (numpy) — same case order, same `np.asarray(size, np.float64)`, same
  `np.zeros(len(kinds))` accumulator, same `np.linalg.norm` for BOX and `s[i].max()` for ELLIPSOID,
  same `INF` fallback. `Primitives.bound_radius` is now a one-line delegation and returns the same
  array.
- `bound_radius_tensor` (torch) — identical branch order and identical expressions, including
  `torch.sqrt(x**2 + y**2 + z**2)` for BOX and the nested `torch.maximum` for ELLIPSOID, and it still
  returns `None` for the plane.
- `cpu3d.Scene.radii` — the throwaway `R3.Primitives(...)`.bound_radius() is replaced by
  `bound_radius(kinds, self.sizes.cpu().numpy())`, same inputs, same dtype/device wrapping.
- `cpu3d._render_chunk` — the only change is deleting the local `INF = R3.INF` rebinding in favour of
  the module-level import of the same `1e30`, and `"empty"` → `WINDOW_EMPTY` (a module constant whose
  value is the string `"empty"`, so `win != WINDOW_EMPTY` is the same comparison).
- `_kind_tensor`'s validation set `(PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX)` →
  `PRIMITIVE_KINDS` — the same six codes in the same order.
- Everything else in the diff is type annotations, docstrings and comments. No loop order, no
  accumulation order, no `dtype`, no tolerance and no switch default changed.

The `INF = 1e30` / `np.isfinite` trap is **recorded, not "fixed"**:
`tests/unit/test_render_primitives.py:88-101` asserts `np.isfinite(r).all()` for a plane and names
`screen_span`'s `torch.isfinite(ndc)` probe as the thing that actually protects the screen-space
caller. That is the right call — fixing it would move pixels.

---

## Defects

**D1 (minor, doc-only recommended) — `_Memo` changes one cache's eviction threshold.**
`stable_envs/render/raycast3d.py:1137` now declares `_KIND_CACHE = _Memo(64)`, and
`_Memo.put` (`raycast3d.py:170-175`) evicts on `if len(self) >= self.cap`. The baseline's kind cache
evicted on `if len(_KIND_CACHE) > 64` (`d07ffa8:stable_envs/render/raycast3d.py:1204`) — strictly
greater — so it held up to 65 entries and now holds up to 64. The other four caches used `>=` in the
baseline (`d07ffa8:raycast3d.py:148,161,1336,1768`) and are unchanged.
Impact: **none on output.** Every cached value is a pure function of its key, eviction is a full
clear and the next call rebuilds it identically; the only effect is one extra rebuild in a process
that holds ≥64 distinct `(device, kinds)` tuples, which no consumer does (a scene has one kind
vector). It is not a behaviour change under any reachable cache pressure, but it *is* a deviation
from "same cap semantics per cache", and findings §10 lists the caps (32, 32, 64, 8, 16) without
recording it. Recommendation: one line in findings §10 (or restore `>` for this one cache). Not
merge-blocking.

**D2 (minor, accuracy / follow-up) — the GPU-test reading is right, but is inference and slightly
over-general.** `gpu_tests_gpu_6464500_summary.txt` reports `66 failed … 462 errors`. I confirm both
readings: the 462 errors are collection/setup `ModuleNotFoundError` for `warp` (252),
`gymnasium_robotics` (168) and `minigrid` (42) — missing optional packages in `se-step6`, nothing to
do with this branch; and the failures are the Wave-0 guard, fixed by `e1da2f6`
(`hasattr(env, "step")` → `dynamics.has_dynamics` in `testing/render_worker.py:74` and
`testing/conformance.py:1073`), which is merged into this branch at `430bb6d` — I verified the fix is
present in the working tree. Two caveats: (a) findings §9c says "every one goes through
`render_worker.py`", but 4 of the 66 are `FAILED tests/gpu/test_conformance.py`, which reach the same
guard through `check_step_matches_reference` — same mechanism, same fix commit, imprecise wording;
(b) the selection has **not** been re-run since the merge, so "fixed" is attribution from the diff
plus the fix, not a measurement. Recommend the integration branch's `tests/gpu` pass confirm the 66
are gone. Not merge-blocking and not this workstream's code.

**D3 (informational, not a defect) — `/oscar` paths.** No committed *result* or source file contains
one (`grep -rn '/oscar'` over `examples/results/step9/render-3d`, `scratch/step9/render-3d`,
`stable_envs/render`, `tests/unit/test_render_primitives.py`): the only hits are the three SLURM
scripts' `#SBATCH --output=` and `SCR=` lines (`scratch/step9/render-3d/gpu_oracle.sbatch:9,30`,
`hmg_determinism.sbatch:9,25`, `oracle_selfcompare.sbatch:10,33`), which are cluster addresses a
batch script cannot avoid and which match the approved precedent in
`scratch/step9/envs-classic-nav/gpu.sbatch:9,19`. The one other hit repo-wide is a pre-existing
comment in `stable_envs/render/mjwarp.py:108`. Findings §9 notes the result artifacts were scrubbed
to `$SCR`, which I confirmed.

**Staleness note (no action):** findings §4's "1263 passed" and the `oracle_cpu_local.json` /
`timings_cpu_local.json` `head_sha` (`6bf885a`) predate the integration merge. I verified
`git diff --stat 900065a HEAD -- stable_envs/render` is empty and that nothing under
`stable_envs/render` or `tests/unit/test_render_primitives.py` changed after `6bf885a`, so the
artifacts remain valid evidence for the code at HEAD — which my own re-run at `d231a8f` confirms.

---

## Commands run (reviewer)

```
git -C /Users/samib/stable-envs-step9-render-3d log -1 --format='%H %s'
  d231a8f332290b5a7d4b971f70da60202aab30b1 step9 render-3d: self-compare artifacts (job 6464601)

git diff d07ffa8 900065a --stat -- stable_envs/envs tests/gpu
  (no output — zero consumer call-site changes)

git diff --stat 900065a HEAD -- stable_envs/render
  (no output — render code frozen since the H delta)

MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit -x \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
  1385 passed, 195 skipped, 112 deselected, 26619 warnings in 108.39s (0:01:48)

MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_render_primitives.py tests/unit/test_cpu3d.py \
  tests/unit/test_variation_maps_dmc_control.py tests/unit/test_raycast3d*.py
  341 passed, 4088 warnings in 33.53s

MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python scratch/step9/render-3d/oracle.py \
  --python /Users/samib/se8-vm-venv/bin/python --out <tmp>/oracle_rev --tag review
  "identical": true, "cells_ok": 21 ... ORACLE: PASS

# digest diff of my record against the committed one, over frames_sha256,
# frames_from_rows_sha256, states_sha256, size, frames_shape, backends and switches:
  cells compared ok: 21   switch cells: 2   MISMATCHES: none
```

No file in any worktree was edited, committed, stashed or pushed by this review except this
report; Oscar was not contacted.
