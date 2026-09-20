# Step 8 — Workstream F findings: reference versions and provenance

**Classification: `PATCH`** — a join and a record. No renderer pixel, no threshold, no frozen
corpus, no dependency pin, and no re-freeze.

| | |
|---|---|
| baseline SHA | `43ac95238f7472137619f3e69e7636e11dd088b1` |
| branch | `step8/reference-provenance` (from integration `4210793`) |
| page | `docs/research/step8_reference_provenance.md` |
| artifacts | `examples/results/step8/provenance/` |
| generators | `scratch/step8/provenance/` |
| SLURM | allocation `6443480`, `gpu2264`, RTX 3090, driver 575.57.08 (three `srun --overlap` steps) |

---

## The merge gate (plan section 18), ten items

| # | item | this workstream |
|---|---|---|
| 1 | explicit workstream scope | Reference versions and provenance only. Files touched: `docs/research/step8_reference_provenance.md` (new), `scratch/step8/provenance/*` (new), `examples/results/step8/provenance/*` (new), `tests/unit/test_step8_provenance.py` (new), one additive `mkdocs.yml` nav entry, and `scratch/step8/report.py` (a reference-arm branch in `_gate_a` plus one `limitations` entry). `stable_envs/` is **untouched** — the declaration the harness reads already existed. `scratch/step4/certify.py` untouched, thresholds untouched, `examples/results/step4/` untouched. |
| 2 | before/after diagnosis | Before: 3 `FAIL` + 1 `UNMEASURED` on the four trackcom rows, from the `mujoco 3.12.0` arm. Cause, named to the line: `certify.py:merge`'s `_richness` tiebreak sorts on the shard tag, `"adhoc" < "dmc"`, so the declared-reference arm lost. After: the arm is selected by the class's `reference_versions` through `provenance.reference_version_mismatch`; all four `PASS`; the 3.12 arm is judged, kept, and reported in the same row. |
| 3 | targeted unit tests | `tests/unit/test_step8_provenance.py`, 18 cases: each cell declares `{"mujoco": "3.11.0"}` and declares nothing of ours; the committed record regenerates byte-for-byte from committed inputs; the authoritative arm satisfies its own declaration and carries host / job / gpu / driver / git_commit / sha256; the drifted arm is present, `blocked_environment` (never `fail`), and keeps the verdict it would have received; the matrix row reads the authoritative arm **and** carries the drift; no committed file holds an absolute cluster path. |
| 4 | affected Gate-A / Gate-F checks | Gate A on `DMCCheetahRun`, `DMCHopperHop`, `DMCWalkerWalk`, `DMCHumanoidStand`, rejudged from **unchanged** measurements with the frozen judge, mandatory **and** family_specific. Gate F: not touched (Workstream D's). |
| 5 | affected frozen-corpus replay | None re-run for item 1: the two arms are committed Step-7 replays of the frozen corpora and are read by sha256. For the 3.13 evaluation, `OGBCube`, `OGBScene`, `DMCCheetahRun` and `FetchReach` were replayed on their committed corpora under three interpreters (see §2 of the page). |
| 6 | pixel-movement classification | **No pixel moves.** Nothing under `stable_envs/` changed. The 3.13 arm's pixels differ from 3.12's, which is the finding, not a change: that interpreter is not adopted. |
| 7 | performance regression check | Not applicable: no hot-path code changed. |
| 8 | independent reviewer verdict | **Pending** — the coordinator's to assign. |
| 9 | semantic follow-up decision | **None.** No renderer pixel moved, so there is nothing for a DINO tripwire or a paired Gate H to detect. |
| 10 | dashboard update | For the coordinator: `gate_a` 40 `PASS` / 2 `UNSUPPORTED_BY_DESIGN`; `pixels_changed: no`; `classification: PATCH`; slurm jobs: allocation `6443480`. |

---

## Buckets, per item

| # | finding | bucket | note |
|---|---|---|---|
| 1 | The four trackcom rows read the non-declared 3.12 arm and reported version drift as `FAIL` / `UNMEASURED` | **`RELEASE_BLOCKING`** — **resolved here** | A false claim about the 42-ID certification, in the shipped support matrix. Fixed in the join; all four now `PASS` against the reference their declaration names. |
| 2 | mujoco ≥ 3.12 re-anchors finite-plane textures and `mujoco_warp` does not, so the reference's floor checker phase-shifts and ours does not | **`DOCUMENTED_LIMITATION`** (containing an **`UPSTREAM_FINDING`**) | Measured, bounded, per-surface localised, and carried in every affected row. Certification needs an interpreter at `mujoco==3.11.0`. |
| 3 | `pyproject.toml`'s `dmcontrol` extra is a floor (`mujoco>=3.11`), not a pin, which is how a 3.12 interpreter came to judge these cells | **`DOCUMENTED_LIMITATION`** → decision for Sami | Turning it into a pin is a dependency-contract change. **Not made here.** |
| 4 | `mujoco_warp` 3.13.0 fixes the UV-less-mesh crash **and** changes ray tie-breaking, vertex normals, the no-light fallback and sample resolution in the same release | **`UPSTREAM_FINDING`** | The fix is real and unadoptable as shipped. This is the item to report upstream. |
| 5 | Under 3.13, `DMCCheetahRun`'s downsample `exact_frac` falls below its frozen floor (0.522233 < 0.528850) | **`RELEASE_BLOCKING` *if the bump were taken*** | It is not taken. Recorded as the reason. |
| 6 | `gymnasium-robotics`' `set_joint_qpos` assertion makes the Fetch **reference** env unbuildable under `mujoco >= 3.12` | **`UPSTREAM_FINDING`** | Predates 3.13; re-measured here on both arms. `se-s3b-fetch311` is the only interpreter where Fetch builds. |
| 7 | `check_resolution_consistency` takes two stable values across four campaigns under nominally identical package versions; the carrier is unidentified | **`STEP9_CLEANUP`** | Nothing is blocked — the declared arm reproduces the frozen bound to 4e-6 — but the number is interpreter-sensitive and nobody has attributed it. |
| 8 | `AntmazeTeleportExplore`'s frozen `digest_ours` does not reproduce at the baseline either; Step 4's plate is a third image of the ≥ 4 the retained `mujoco.Renderer` produces | **`DOCUMENTED_LIMITATION`** | An **oracle hole**. Gate A on the cell is `PASS`; the digest is not a Gate-A check; the cell is not one of the 42 ids. Re-freeze **deferred**: re-freezing a history-dependent digest moves the problem. |
| 9 | The plate bake is a deterministic function of the retained renderer's call history; `check_reference_renderer_is_stable` fires 12/12 on `PointmazeTeleportNavigate` | **`UPSTREAM_FINDING`** (contained) + **`STEP9_CLEANUP`** (the fix) | LRU eviction, merged in Step 6, removes cache pressure as the trigger. A pure bake needs a renderer reset without a new EGL context, against a measured ~96-context budget. |
| 10 | `stable_worldmodel` is recorded as `"present"` with no version by the interpreter that measured the OGB cells | **`DOCUMENTED_LIMITATION`** | `reference_version_mismatch` treats "present" as unknown, so an swm version could never be confirmed if one were declared. Recorded, not fixed. |

---

## Bump recommendation

> **`mujoco` / `mujoco-warp` 3.13.0: DO NOT BUMP.**

Plan section 14's three conditions, answered on hardware in a controlled venv
(`examples/results/step8/provenance/mjwarp313_verdict.json`):

| condition | answer |
|---|---|
| fixes the relevant defect | **yes** — the `texcoord_offset >= 0` guard is in the shipped source and its comment names `mesh_texcoordadr == -1` |
| existing fidelity/regression evidence rechecked and still holding | **no** — `DMCCheetahRun`'s downsample number crosses its frozen floor |
| does not silently move unrelated renderers | **no** — `OGBCube`, `OGBScene` and `DMCCheetahRun` digests all move; four independent render-path changes shipped with the fix |

It also buys nothing that is currently costing anything: `OGBScene` renders today because the cell
bakes the unsamplable material away at a declared 0.139 % residual with `interior_exact` 1.0.
**No `pyproject.toml` pin is changed by this workstream.**

What would change the answer: a release carrying the guard and nothing else in the render path, or a
deliberate re-declaration campaign that re-measures every mjwarp cell under the new pair and
re-freezes with the drift documented. Both are Step-9 scope.

---

## Tests run

```text
MUJOCO_GL=glfw $SE6VENV/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_step8_provenance.py tests/unit/test_step8_gate_record.py tests/unit/test_repo_contracts.py
```

48 passed in the two Step-8 files. `tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves`
fails, and it fails **identically at this branch's parent** (`scratch/step8/semantic/*` and others
are cited by the dashboard and do not exist yet) — pre-existing, not this workstream's.

## SLURM

| step | what | where |
|---|---|---|
| `6443480` (srun --overlap) | build `$SCR/venvs/se8-mjwarp313` | `gpu2264` |
| `6443480` (srun --overlap) | round 1: texcoord source probe, DMC + Fetch replay, both arms | `gpu2264` |
| `6443480` (srun --overlap) | round 2: digest replay, three interpreters (3.11 / 3.12 / 3.13) | `gpu2264` |
| `6443480` (srun --overlap) | round 3: OGB cells under 3.13 | `gpu2264` |

Nothing was submitted to a queue, nothing was cancelled, and no login node computed anything.
