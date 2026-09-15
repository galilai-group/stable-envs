# Step 7 — closure verification (`step7/closure`)

**Classification: PENDING**

Certification of `step7/integration` as a whole. This workstream changed no renderer code and no
test: it ran the existing machinery — `examples/fidelity_step4.py`, `scratch/step4/certify.py`,
`scratch/step7/rocket-landing/refreeze_locomotion.py`, `tests/gpu`, `tests/unit` — against the
frozen Step-4 corpus, against the Step-7 correctness re-freeze, and against the seven cells Step 7
added, on their own committed corpora.

| | |
|---|---|
| parent baseline | `76dc5116f432e61dd47fba8f9e879cf9212c8e46` (Step 6 closed) |
| **integration SHA verified here** | **`275970cc2d9e0d92eebf5297ee175b08944b7d04`** (`step7/integration`, the commit after `fd2b8ef`) |
| my branch | `step7/closure`, branched from that SHA |
| cluster copy | `$SCR/se7/closure`, rsync of the worktree with `.git` excluded and `.git_sha` written first |
| pristine baseline on the cluster | `$SCR/se7/baseline`, `.git_sha` = `76dc5116f432e61dd47fba8f9e879cf9212c8e46` (untouched) |
| SLURM jobs | GPU `6413321` (gpu2107, RTX 3090); CPU `6413322` (Gate A on `cpu`), `6413370` (`tests/unit`); `6413171` superseded, `6413170` cancelled before it measured anything |

## 0. What this sweep is allowed to conclude

Four questions, kept apart, because conflating them is how a coverage claim gets read as a
certification claim:

1. **Did anything that was frozen move?** The 49 frozen Step-4 render-distinct configurations
   replayed on cuda against their frozen digests, each shard in the interpreter it was frozen in.
2. **Did the Step-7 re-freeze hold?** The six Mujoco locomotion nvdiff cells are NOT among the 49 —
   they have no `examples/results/step4/states/<Config>.npz` and appear in no
   `conformance/<shard>.json` — so the artifact they are held to is
   `examples/results/step7/rocket-landing/refreeze/branch.json`, the correctness re-freeze Sami
   authorised.
3. **Do the seven new cells pass Gate A on their own committed corpora**, on every device they
   claim, with the refusals recorded as refusals?
4. **Do the suites still say what they said**, with every remaining failure named in advance?

None of these is a performance question. No tier was measured, re-measured or recomputed anywhere in
this sweep; the coverage matrix copies each tier and prints the file it came from.

## 1. Interpreters — chosen to reproduce, not for convenience

Each frozen shard replays in the interpreter it was frozen in
(`examples/results/step4/conformance/<shard>.json` -> `provenance.reference_packages`); a digest
reproduced in a different dependency set answers a different question. That is Step 6's rule and it
is kept verbatim in `scratch/step7/closure/_env.sh`.

| purpose | interpreter | why this one |
|---|---|---|
| `dmc`, `ogb`, `raster` shards | `$SCR/venvs/se-step3` | dm_control 1.0.45, gym_pusht 0.1.6, ogbench 1.2.1, mujoco + mujoco_warp 3.12.0 — a field-for-field match to those shards' frozen provenance |
| `swm` shard | `$SCR/venvs/se-step4-swm` | craftax 1.6.1 + jax 0.11.1 + stable_worldmodel, no mujoco at all |
| `fetch` shard | `$SCR/venvs/se-s3b-fetch311` | gymnasium_robotics 1.4.2, mujoco + mujoco_warp 3.11.0 |
| the four visual-tier DM Control cells, kept as a shard of their own | `$SCR/venvs/se-vis311` | mujoco 3.11.0 + dm_control 1.0.44 — the pair their `visual_fidelity` is DECLARED against. `certify.py` reads the reference version off the shard, so these rows cannot sit in the 3.12 document |
| `CraftaxClassic`, `CraftaxFull` | `$SCR/venvs/se7-craftax` | jax 0.11.1 with `jax_threefry_partitionable` at its default, which is what the night parity claim names |
| `DMCManipulatorBringBall`, `DMCQuadrupedWalk` | `$SCR/venvs/se-vis311` | the declared 3.11 pair again |
| `OGBCube`, `OGBScene` | `$SCR/venvs/se7-ogb` + the swm checkout on `PYTHONPATH` | ogbench 1.2.1 + mujoco_warp 3.12.0; the reference is swm's own `render()` |
| `PFRocketLanding`, the six locomotion nvdiff cells | `$SCR/venvs/se7-rocket` | nvdiffrast 0.4.0, PyFlyt 0.29.0, pybullet 3.2.7 |
| `tests/gpu`, broad | `$SCR/venvs/se6-verify` | installs nothing: a `--without-pip` venv whose single `.pth` chains se-step6, se-step3, se-step5 and the swm checkout. The only interpreter that reaches the MJWarp families and pytest in one process |

**No venv was created and none was modified.** Every interpreter above already existed.

### One operational defect found and fixed inside this sweep

The first CPU job (`6413171`) set `MUJOCO_GL=osmesa`, reasoning that a batch node has no GPU. That
is wrong on Oscar: `stable_envs.envs.dmcontrol` then fails to import with
`AttributeError: 'NoneType' object has no attribute 'glGetError'`, which **silently removes every DM
Control cell from the registry** — the run reported `NOT_REGISTERED_HERE` for cells that are
registered, and `tests/unit` collapsed into collection errors. `MUJOCO_GL=egl` is what Step 6's own
CPU verification jobs used on these same nodes. Recorded because the failure mode is the dangerous
one: a missing backend and an empty table look identical unless the job prints the import error,
which `gate_a_new_cells.py` now does.

The same reasoning moved the `device="cpu"` refusal check for OGBCube, OGBScene and PFRocketLanding
off the batch node and onto the card. On a batch node `mujoco_warp` and `nvdiffrast` do not import
at all, so a refusal and a missing module are indistinguishable; the question is whether the class
refuses on a machine where the CUDA backend **is** available, and it has to be asked there.

## 2. The corpora the seven new cells were judged on

Copied, never redrawn, by `scratch/step7/closure/stage_corpora.py`, which records the source file and
the sha256 of the array it copied:

| cell | committed corpus | shape | sha256 (first 16) |
|---|---|---|---|
| `CraftaxClassic` | `examples/results/step7/craftax-classic-full/corpus/states.npz` | (64, 149) | `e403a5648d92156e` |
| `CraftaxFull` | `examples/results/step7/craftax-full/corpus/states.npz` | (64, 544) | `b0c38605fc983eca` |
| `DMCManipulatorBringBall` | `examples/results/step7/dmc-manipulator/corpus/states/DMCManipulatorBringBall.npz` | (64, 15) | `ebfc481ad7f23962` |
| `DMCQuadrupedWalk` | `examples/results/step7/dmc-quadruped/phase1/corpus/states/DMCQuadrupedWalk.npz` | (64, 23) | `1f0b38b4dca47b92` |
| `OGBCube` | `examples/results/step7/ogb-manipulation/states/OGBCube.npz` | (64, 24) | `560c1398a8a49a3f` |
| `OGBScene` | `examples/results/step7/ogb-manipulation/states/OGBScene.npz` | (64, 30) | `7a0cdff1849ee221` |
| `PFRocketLanding` | `examples/results/step7/rocket-landing/reference/states.npz` (`state13`, first 64 of 200) | (64, 13) | `f4707e54f3007b2e` |

Two of these needed a copy rather than a symlink: the two Craftax corpora were written without the
`meta` key that `fidelity_step4.corpus_for` reads unconditionally — already recorded as a Step-8
freeze carry-over by W2's reviewer — and PFRocketLanding's reference corpus is a multi-array file
whose states live under `state13`. The copy adds the `meta` key and nothing else; the state bytes
are the committed ones and their sha256 is printed above so that can be checked.

## 6. Two findings this sweep produced that no workstream had seen

### FINDING CLOSURE-1 — `nvdiff.Scene.clip` is batch-size dependent on x86-64 CPU at widths 1 and 2

`tests/unit/test_nvdiff_triangles.py::test_from_triangles_clip_row_is_independent_of_batch_size` —
W6's own test, added in Step 7 — **passes on an Apple M5 and fails on Oscar's x86-64 batch nodes**.
No Step-7 workstream saw it because the Step-7 `tests/unit` runs quoted in the dashboard were taken
on the Mac; this is the first time that suite ran on the certification host at the integration HEAD.

Characterised by `scratch/step7/closure/probe_nvdiff_cpu_batch.py`, job `6413606` on `node1939`
(x86-64, torch 2.14.0+cu126), at three thread counts:

| batch width | `mvp_rows` (W6's float64 product) | `clip` positions | `clip` normals | `clip` world positions |
|---|---|---|---|---|
| 1 | **identical** | **36 of 96 differ**, max 3.05e-05 | identical | identical |
| 2 | **identical** | **60 of 192 differ**, max 3.05e-05 | identical | identical |
| 7, 8, 31, 32, 63 | identical | identical | identical | identical |

- **W6's fix is intact.** `mvp_rows` — the float64 `proj @ view @ M` rounded once, which is the
  product W6 made batch-invariant — is bit-identical at every width on both architectures. The
  residual is one product downstream: `Scene.clip`'s float32 `mvp @ vertices`, whose CPU GEMM picks a
  different summation order for a very small batch, exactly as cuBLAS did for the product W6 fixed.
- **The magnitude is one float32 ulp** at the coordinate scale involved (3.05e-05 at 306 units), on
  clip-space positions only.
- **Not a threading artifact**: identical behaviour at 1, 2 and 4 torch/OMP threads.
- **Architecture-specific**: every width is bit-identical on arm64 (Apple M5, torch 2.14 CPU).

**No shipped renderer is affected, and no frozen or certified artifact depends on it.** Nothing in
the registry draws through `nvdiff` on `device="cpu"`: `PFRocketLanding` raises on cpu by design and
the six Mujoco locomotion nvdiff cells render through nvdiffrast on the card. The CPU path of
`Scene.clip` exists so that the transform can be unit-tested without a GPU, which is what found this.

It is nevertheless a **failing test in a mandatory suite on the certification host**, and the fix
would be inside `stable_envs/render/nvdiff.py` — a shared renderer file — or inside `tests/`, both of
which this workstream may not edit and both of which are frozen for Step 7. **Reported, not fixed.**

### FINDING CLOSURE-2 — the broken-citation set grew from 9 to 10 during Step 7

`tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves` is the known pre-existing
failure. Its **set** is not what it was: at `76dc5116` it names **9** unresolvable paths, at
`275970cc` it names **10**. The tenth is

    scratch/sboughan/se7/swm-src/stable_worldmodel
        cited at examples/results/step7/ogb-manipulation/oscar/provenance_6395175_readback.txt:18

— a `$SCR` path inside a committed Oscar provenance capture, which the scanner reads because it reads
every `.txt` under `examples/`. Measured both ways in the same interpreter on the same tree
(`se6-venv`, Apple M5): baseline 9, integration HEAD 10, same nine plus that one.

Not fixed here, deliberately: the file is W5's committed *evidence capture* of what a job printed, and
editing it would make the provenance record disagree with the job that produced it. The right fixes
are either the exemption W4's reviewer already proposed for committed `review_*.md` (extended to
`oscar/*.txt` provenance captures) or a scrub at capture time — both changes to `tests/` or to a
workstream's artifacts, and both **Step-8 items**.

## 3. The frozen Step-4 replay — PARTIAL (32 of 49 pulled before Oscar SSH was denied)

`--partition=gpu --qos=norm-gpu --gres=gpu:1 --constraint=geforce3090`, job `6413321`, gpu2107,
RTX 3090, driver 575.57.08. Each shard in the interpreter it was frozen in; `compare49_step7.py`
diffs every field of every configuration against `examples/results/step4/conformance/<shard>.json`.

**Of the 32 configurations whose rows had been written when the artifacts were last pulled: 27
bit-identical, 5 expected-known, 0 unexplained, 0 regressions.**

| verdict | configs |
|---|---|
| identical | 27 |
| expected-known | 5 |
| **REGRESSION?** | **0** |

The five are exactly Step 6's known set, with the same mechanisms and no new member:

| config | what moved | classification |
|---|---|---|
| `DMCReacherEasy`, `DMCReacherHard`, `DMCFingerTurnEasy`, `DMCFingerTurnHard` | the DECLARATION only — `style_dim` 17 -> 18 and `style_digest`; `digest_ours` and `digest_ours_default_style` byte-identical | expected-known (reacher-correctness's visibility leaf, Step 6) |
| `AntmazeTeleportExplore` | `digest_ours` and `digest_ours_default_style`; `digest_official` unchanged | expected-known (documented bake nondeterminism; the digest does not reproduce at the BASELINE either) |

`CraftaxClassic` and the five `swm` 2-D cells, plus eleven of the twelve `fetch` cells, had not been
pulled when SSH was denied. The job was still running on the cluster; nothing about it is claimed
here that its artifacts do not show.

### The one NEW delta Step 7 introduces, with its mechanism

**`CraftaxClassic` cannot replay its frozen digest, and must not.** W1 took
`swm/CraftaxClassicPixels-v1` from partial to full coverage: the observation grew from the
`(112,144,3)` world window to the whole `(144,144,3)` SWM frame and the state from 127 slots to 149.
`examples/fidelity_step4.py` therefore raises on the frozen corpus, by design and with the right
message:

```text
ValueError: CraftaxClassic.render: CraftaxClassic expects state_dim 149, got 127
(input shape (64, 127), dtype float32). ... This used to be reshaped to (-1, 149), which regroups
the input into a different number of states that each straddle two of your rows, and renders
without complaint.
```

That raise is the check doing its job; the alternative — a silent reshape — is the failure mode the
message names. The obligation the coordinator set instead is **world-row byte identity at light
1.0**, and `scratch/step7/closure/craftax_world_row_replay.py` measures exactly that against the
frozen artifact:

| | |
|---|---|
| frozen corpus | `examples/results/step4/states/CraftaxClassic.npz`, 127 slots, read-only |
| zero-extended to | 149 slots (empty inventory, awake, day-cycle key zero), `light_level` forced to 1.0 |
| size | 16, the frozen size |
| full frame now | `(64, 144, 144, 3)` |
| world rows taken | `(64, 112, 144, 3)` = the frozen `frames_shape` |
| frozen `digest_ours` | `1aeabbe8bc6a9a2ee1fdb8ffc27a737bd659630dc6e3a1b98f6bae002c894a4d` |
| **world-row digest now** | **`1aeabbe8bc6a9a2ee1fdb8ffc27a737bd659630dc6e3a1b98f6bae002c894a4d`** |
| frozen mean pixel / now | `93.84458414713542` / `93.84458414713542` |
| inventory strip rows | all zero, as a zero inventory must be |

Digested with `conformance.frames_digest`, the function the frozen artifact used — it hashes shape
and dtype before the bytes, so a plain `sha256` of the same pixels is a different number and would
have looked like a failure. Verified on Apple M5 (`arm64`); the cuda arm is in job `6413321`.

## 4. Gate A on the seven cells Step 7 added — `cpu` arm complete

Measured by `scratch/step7/closure/gate_a_new_cells.py` on the committed Step-7 corpora at each
cell's registered default size, through `examples/fidelity_step4.measure`, with each mandatory
check's bound spelled out per check rather than inferred from the evidence's shape.

| cell | device | verdict | mandatory | interpreter | job |
|---|---|---|---|---|---|
| `CraftaxClassic` | cpu | **PASS** | 14/14 | `se7-craftax` | 6413322 |
| `CraftaxFull` | cpu | **PASS** | 14/14 | `se7-craftax` | 6413322 |
| `DMCManipulatorBringBall` | cpu | **PASS** | 14/14 | `se-vis311` (declared 3.11 pair) | 6413322 |
| `DMCQuadrupedWalk` | cpu | **PASS** | 14/14 | `se-vis311` (declared 3.11 pair) | 6413322 |
| `OGBCube`, `OGBScene`, `PFRocketLanding` | cpu | **refuse by design** | n/a | — | asked on the card in 6413321 |
| all seven | cuda | in job `6413321` | | | |

Device coverage recorded honestly: four of the seven have a CPU backend and all four pass every
mandatory check on it; three raise on `cpu` rather than falling back to CUDA, which is the Step-6
policy addendum's rule and is why the refusal is asked on a machine where CUDA *is* available.

## 5. `tests/unit` — the certification host and the Mac, each against its own baseline

Every run is paired with the SAME suite in the SAME interpreter on the SAME machine against the
pristine `76dc5116` tree, because "only the known pre-existing failure" is a claim about a
*difference*.

| host | interpreter | baseline `76dc5116` | integration `275970cc` | new failures |
|---|---|---|---|---|
| Oscar x86-64 | `se-step6` | 2 failed, 807 passed | 3 failed, 1018 passed | **1**: `test_from_triangles_clip_row_is_independent_of_batch_size` |
| Oscar x86-64 | `se7-craftax` | 2 failed, 776 passed | 3 failed, 919 passed | **1**: the same |
| Oscar x86-64 | `se7-rocket` | 11 failed, 780 passed | 12 failed, 974 passed | **1**: the same |
| Apple M5 | `se6-venv` | 2 failed, 822 passed | 2 failed, 1035 passed | **0** |
| Apple M5 | `se7-craftax-venv` | 2 failed, 773 passed | 2 failed, 918 passed | **0** |
| Apple M5 | `se7-rocket-venv` | 11 failed, 767 passed | 11 failed, 913 passed | **0** |

The failure SETS are otherwise identical on both trees:

- `test_repo_contracts.py::test_every_cited_scratch_path_resolves` — the known pre-existing failure,
  **9** broken citations at the baseline and **10** at the integration HEAD (see CLOSURE-2).
- `test_examples.py::test_every_argparse_script_can_print_its_own_help[robocasa_visual/render_mjwarp.py]`
  — pre-existing in every interpreter on both trees; `examples/robocasa_visual/render_mjwarp.py`
  loads its helpers from a hard-coded absolute path,
  `/mnt/vast/home/randall.balestriero/code/stable-envs/stable_envs/render/mjwarp.py`, which exists on
  nobody's machine here. Previous Step-6 and Step-7 runs deselected this file, which is why it does
  not appear in their counts.
- the nine `test_gfxdraw_bounds` / `test_gfxdraw_select` failures in the `rocket` interpreters only —
  same hard-coded path, same pre-existing status on both trees, present at `76dc5116` on both
  machines. Interpreter-specific, not a Step-7 effect.
