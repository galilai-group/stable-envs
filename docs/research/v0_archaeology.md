# v0.1 weight classification (workstream E)

Inventory of the four trees that dominate the `v0-draft` checkout. This wave
**classifies**; it does not delete, `git rm`, retag, or push `main`. Provenance
stays on the immutable Step-8 tag and on the `v0-draft` commit.

Measured in `/Users/samib/stable-envs-v0-archive` at `41ab42a` (branch
`v0/archaeology`, same commit as `v0-draft`). Re-print sizes with
`python scripts/archaeology_sizes.py`.

Binding: [v0_release_surface.md](v0_release_surface.md). Earlier, coarser
Wave-3 labels live in
[`examples/results/step9/cleanup/findings.md`](../../examples/results/step9/cleanup/findings.md).
This file is the size-bearing follow-up those notes asked for.

---

## Checkout vs product

| path | `du -sh` | git blobs (`git ls-tree -r -l`) |
|---|---|---|
| `.` (worktree) | **219 MB** | — |
| `stable_envs/` | **3.2 MB** | product code; out of scope |
| `scratch/` | 82 MB | 83.01 MB / 1614 files |
| `examples/results/` | 79 MB | 77.93 MB / 2102 files |
| `examples/robocasa_visual/` | 22 MB | 22.79 MB / 16 files |
| `docs/assets/` | 28 MB | 29.12 MB / 85 files |
| **four trees** | **~211 MB** | **212.85 MB / 3817 files** |

`du` and git disagree by a megabyte because of directory blocks and how
`ls-tree` reports sizes. The four trees are **96 %** of the checkout;
`stable_envs/` is **1.5 %**.

`.gitignore` already tries to keep *new* scratch renders, meshes, and vendored
C++ out of history. **152 already-tracked files** (70.39 MB) match those
patterns — almost all of them MuJoCo Menagerie meshes under
`scratch/backend3d/scenes/`. They stay tracked until a later wave untracks
them. The ignore rule is the intended future, not the current blob set.

The wheel/sdist does **not** ship any of this (`setuptools` includes
`stable_envs*` only). Weight is a **git / docs-site / CI checkout** problem,
not a PyPI problem.

---

## Tags and what they already pin

| ref | kind | commit | four-tree blobs |
|---|---|---|---|
| `step8-closure` | annotated tag | `21cb306` | 208.33 MB / 3673 files |
| `v0-draft` | **branch** `refs/heads/v0-draft` (no `refs/tags/v0-draft` on `origin` yet) | `41ab42a` | 212.85 MB / 3817 files |
| `step7-closure` | annotated tag | `43ac952` | 203.34 MB / 3456 files |

`v0-draft` is the freeze commit, not yet a tag. Every path in this inventory
exists at `41ab42a`. Everything except **+4.52 MB of Step-9 evidence**
(`examples/results/step9/` 4.15 MB, `scratch/step9/` 0.37 MB) also exists at
`step8-closure`. Moving a pre-Step-9 corpus off `main` later does not need a
new archive: `git checkout step8-closure -- <path>` restores it.

---

## Classes

Exactly one class per major directory or corpus. Not every file.

| class | meaning |
|---|---|
| **KEEP_MAIN** | Ordinary product branch: tests, compact JSON, hashes, MkDocs-cited gallery, probe scripts `test_repo_contracts.py` resolves |
| **KEEP_FIXTURE** | Small golden / state corpora tests `np.load` |
| **RELEASE_ASSET** | Large media or vendored geometry; GitHub release zip, not `main` |
| **ARCHIVE_TAG_ONLY** | Keep via `step8-closure` / `v0-draft`; drop from the slim `main` tree |
| **DELETE_DUPLICATE** | Byte-identical copy. Both paths and sha256 are below. Do not delete until a later reviewed wave |

Intentional snapshot copies (`examples/results/step9/oracle/step8_*.json` ≡
the Step-8 oracles they snapshot) are **KEEP_MAIN**, not DELETE_DUPLICATE.

---

## Rollup: what can leave `main`

Using the slimmer layout in the last section.

| bucket | MB (git blob / file bytes) |
|---|---|
| **Must stay** (KEEP_MAIN + KEEP_FIXTURE) | **~38 MB** |
| **Could leave `main`** | **~175 MB** |
| of which DELETE_DUPLICATE waste | 38.8 MB (≥80 KB groups) + 0.1 MB small STL |
| of which RELEASE_ASSET | ~80 MB |
| of which ARCHIVE_TAG_ONLY | ~56 MB |

38 + 175 = 213 MB, matching the four-tree git total. A later wave can move
weight without rewriting history if it only `git rm`s from `main` and leaves
the tags alone.

CI today (`testing.yml`, `certification.yml`, `docs.yml`) checkouts the whole
repo. Slimming `main` shrinks every job. Docs Pages currently hosts
**all** of `docs/assets/` because MkDocs copies `docs/`; uncited env directories
are still deployed.

---

## 1. `scratch/` — 82 MB (`du -sh`) / 83.01 MB git

`tests/unit/test_repo_contracts.py` extracts 217 distinct `scratch/...`
citations from docs, examples, and `stable_envs/` and fails if a cited probe
vanishes. **872 tracked `*.py` files** are the reproduction path. They are
KEEP_MAIN even when the numbers they produced are ARCHIVE.

| corpus | `du -sh` | class | on `step8-closure` | tests / CI / docs |
|---|---|---|---|---|
| `scratch/backend3d/scenes/flat/*.{obj,stl}` | 34 MB of 34 MB | **DELETE_DUPLICATE** | yes | none; docs cite `scenes/*.xml` only |
| `scratch/backend3d/scenes/*.{obj,stl}` (root) | 34 MB | **RELEASE_ASSET** (or ARCHIVE_TAG_ONLY) | yes | none. `.gitignore` already lists `scratch/**/*.obj` / `*.stl`. Vendored Menagerie UR5e + Robotiq |
| `scratch/backend3d/scenes/*.xml` | ~60 KB | **KEEP_MAIN** | yes | `docs/research/backend3d.md` §Reproducing; `scratch/robocasa/probe_green.py` loads `antmaze_medium.xml` |
| `scratch/backend3d/{*.py,*.sbatch}` | ~0.1 MB | **KEEP_MAIN** | yes | cited from `backend3d.md` |
| `scratch/backend3d/images/` | 584 KB | **ARCHIVE_TAG_ONLY** | yes | probe PNGs; gitignore would exclude new ones |
| `scratch/census/` | 2.4 MB | **KEEP_MAIN** | yes | `union.json` / `assignment.json`: `test_swm_matrix.py`, `tests/gpu/test_conformance.py`, README / `docs/benchmarks.md` |
| `scratch/certification.json` | 20 KB | **KEEP_MAIN** | yes | `tests/certification/test_{certification,step3,step4,step5,step6_report}.py` |
| `scratch/swm_matrix.json` + `swm_matrix.py` | 0.65 MB | **KEEP_MAIN** | yes | `test_certification.py`, `test_envs_fetch.py` |
| `scratch/step{3,4,5,6,7,8}/` | 4.4 MB | **KEEP_MAIN** | yes | certifiers and report generators; `test_repo_contracts`, `test_step6_report`, `test_step8_*` load by path |
| `scratch/step9/` | 444 KB | **KEEP_MAIN** | **no** (Step 9 only) | `.github/workflows/testing.yml` runs `scratch/step9/packaging-ci/test_packaging.py` and `audit_dist.py`; F may move these under `tests/packaging/` + `scripts/ci/` |
| remaining family probe dirs (`ogbench`, `robocasa`, `procgen`, …) | ~8 MB | **KEEP_MAIN** | yes | cited probes; `scratch/ogbench/out` and `scratch/robocasa/out/corpus` are the regenerable exemptions in `test_repo_contracts.REGENERABLE` (3 GB class, **not** in this checkout) |

### 1.1 DELETE_DUPLICATE — Menagerie meshes

28 of 31 paired files under `scenes/` and `scenes/flat/` are byte-identical.
The three XML pairs **differ** (compiler / include layout), and
`pointmaze_medium.xml` exists only in `scenes/`. Both XML trees use
`file="base_0.obj"` relative names, so `flat/` exists so a flattened MJCF can
sit next to a second copy of the meshes. Docs do not cite `flat/`.

Deleting the `flat/` meshes without also dropping or rewriting `flat/*.xml`
would break those three scenes. Recommended later action: drop the whole
`flat/` directory (XML + meshes) and keep one mesh tree next to
`scenes/*.xml`, **or** untrack both mesh trees and fetch Menagerie at probe
time as `.gitignore` already describes.

| file | bytes | sha256 | paths |
|---|---|---|---|
| `upperarm_3.obj` | 5,346,165 | `b257dab441e6fab64ea03b4b87029d8a96554ad1a513cce944d56acba0ad948a` | `scratch/backend3d/scenes/upperarm_3.obj` ≡ `.../flat/upperarm_3.obj` |
| `upperarm_2.obj` | 3,677,923 | `86c05350d60eb0e7e28a81b544267478b7db997baba41cef19088ebbeffecfa1` | `scenes/` ≡ `scenes/flat/` |
| `shoulder_0.obj` | 2,947,794 | `477387a7e7312782a6040484a3c2c2fcee33f40109f419076e31b6f6e1692c68` | same |
| `wrist1_1.obj` | 2,470,967 | `ee6d5c562e6becef94109b0543329080d285d1aef521a01ed98e18a5bf1e8684` | same |
| `shoulder_2.obj` | 2,440,344 | `291714518416bb8f0bafbf123867db011ecbf5146a05245ca5893ee8c57d4e21` | same |
| `wrist2_1.obj` | 2,156,900 | `a721d06dc61d3b3a8ec58c035c9e3358ed30d6991a72e077d8abf3c3c203b633` | same |
| `wrist2_2.obj` | 2,141,392 | `6df4335c8bd5e881fda2c224f770115ffe9ddd0cd321ebe3c0a4a310796f0987` | same |
| `base.stl` | 1,712,484 | `1c7b9f2bd92d705fc4e897c94e905973a3c05f406845e942229433deb7041453` | same |
| `forearm_0.obj` | 1,691,648 | `64c049a55149c2ab355594d52302b53d3e34da3677ceaa1917c427cde7743146` | same |
| `wrist1_2.obj` | 1,657,638 | `88c66dc64f15391287147d50add37729a3f41dadd80b1f859cf2cb6d45daf32a` | same |
| `forearm_3.obj` | 1,095,775 | `fa31af19cfad7a765b62e8a2d5931cb014b88de84195a7850b4e5ee902112e9a` | same |
| `base_mount.stl` | 1,091,784 | `87971dfdd6e5311b5a723e7b73c4701ebf5c2364df671500f30dfc1c53c8e61a` | same |
| `upperarm_1.obj` | 1,090,164 | `272899e927f4a32b7c25e34560ec70654c0346cc8ed1c616c0555f3f53b32d14` | same |
| `forearm_2.obj` | 932,534 | `05b436505fc595aeab26675590c2b56b88a65b2d52d6352096fed9f07cb4c4ca` | same |
| `wrist2_0.obj` | 846,741 | `2ac2e2e39d8f7005bc3b3ab26cd61634be2e14930074615da5318b9ee7c569a8` | same |
| `base_1.obj` | 638,415 | `cf7d54691a2c1eeda280955ebd6bd83c2dbecfc46fb53c10198f05d107511b50` | same |
| `shoulder_1.obj` | 569,523 | `8537bba02804640ddf6e260f79fe66e895b3229ed3d44298a95e8e027eb5d435` | same |
| `base_0.obj` | 556,543 | `9a3b21f1524baef6af8a52d7af90b6e94b79f7e3c2b04a0f6581f74d22291b0b` | same |
| `wrist1_0.obj` | 279,140 | `7ef0c59b5c5d56a9902435c9a058b27d2a38c85fe2bae02d00a9284dd1f4d519` | same |
| `wrist3.obj` | 210,847 | `0e51748f9c93bd66604360a5cf7e2e8a6e754550be64076333ff4fc19ec90c61` | same |
| `upperarm_0.obj` | 155,599 | `14143d4f1175a7dbacc8e2f2f98f1cac5a2b2e238cd46cd03654d31f2610eb53` | same |
| `follower.stl` | 110,484 | `633793332c081641ce200df27a40643cc293b29956e3cb5cb29cb811c33ef1c7` | same |
| `coupler.stl` | 89,084 | `d5ee95e62f8415bf5b6e503c831a958f5fc1990bf9b2865329ec38a28932727c` | same |
| `spring_link.stl` | 85,284 | `4fb74a8a0d76c0e471cf19fd48bc676fd5b19123798e7a39eb6aa56869354283` | same |
| `driver.stl` | 67,084 | `bbd6e868b1778bead60d2516c7ede581d7ad4e431744b1828757d6ea5129c112` | same |
| `pad.stl` | 15,084 | `4af1fa8d9bb285abecbf5dfc88e74cb30f059a7a074c2030dc35e9ee4316019d` | same |
| `silicone_pad.stl` | 15,084 | `ca20c0fa61e6d3ce7b04bed25360e048e31dfd88ace04187aa6d9e8f4ca3fd0f` | same |
| `forearm_1.obj` | 76,202 | `a96fe203015cf532b193f414cd01268c7c36795e51a837dd33afd122334f81f2` | same |

Identical payload (one copy): **34.17 MB**. Duplicate waste: **34.17 MB**.

XML (not duplicates — do not DELETE_DUPLICATE):

| file | `scenes/` sha256 | `flat/` sha256 |
|---|---|---|
| `antmaze_medium.xml` | `a72f43b920fbbd91718da97a232a4df9b846ef48d041aec29f8fc7858963a9a0` | `0bb91dd22397602fac9def230c83a02419516ef66e2415c88cb4f32b4d2a02ca` |
| `cube_single.xml` | `f41e0bd94fe2e9f5267c2314efecd00ebbe66b901f340f603aed6aa1e6f946af` | `1082b799538c93da3d3fe943cf18810311c4803af354458818c65c84430745d7` |
| `humanoidmaze_medium.xml` | `160cea14c2cf2da123a4c36dedc0c51ed781056050289e735f2a2cdce541e238` | `289c1946d50f2364a29050cb5a5cadc1efac1c2afb598258d93aace897d67ceb` |
| `pointmaze_medium.xml` | `49bf25cee9523920bc61c8a861ce80faeea51934cb066205c7a2965ec51ee691` | missing |

---

## 2. `examples/results/` — 79 MB / 77.93 MB git

| corpus | `du -sh` | class | on `step8-closure` | tests / CI / docs |
|---|---|---|---|---|
| `step3/` | 1.2 MB | **KEEP_MAIN** | yes | generated `docs/research/step3_benchmarks.md`; compact JSON |
| `step4/states/` | 416 KB / 49 `*.npz` | **KEEP_FIXTURE** | yes | `test_cpu3d`, `test_envs_fetch`, `test_envs_craftax_rocket`, `test_dmcontrol_tendon`, `tests/gpu/test_{fast_paths,gfxdraw_classic}` |
| `step4/` remainder (certification / merged JSON, comparison PNGs) | 2.1 MB | **KEEP_MAIN** (JSON) + **ARCHIVE_TAG_ONLY** (comparison PNGs) | yes | `scratch/step4/certify.py`, `test_step6_report` reads `step4/certification.json` |
| `step5/` | 2.2 MB | **KEEP_MAIN** | yes | `parity_probe.json` cited by `test_task_object_visibility`, `test_conformance`; `test_step5.py`. `reacher/episodes.jsonl` (993 KB) is ARCHIVE if the JSON stays |
| `step6/cpu-3d/replay/mjwarp_frames_first4.npz` | 50 KB | **KEEP_FIXTURE** | yes | `tests/unit/test_cpu3d.py` |
| `step6/cpu-3d/` remainder | 6.3 MB | **ARCHIVE_TAG_ONLY** | yes | findings cited from research notes; GPU frame dumps not loaded by unit tests |
| `step6/verify/` | 15 MB | **ARCHIVE_TAG_ONLY** | yes | `docs/research/step6_optimization.md` and `scratch/step7/report.py` cite it as the Step-6 control. 12.6 MB JSON shards + copies (see duplicates). No unit test `np.load`s the shards |
| `step6/raycast3d/` | 5.1 MB | **KEEP_MAIN** (JSON / `experiments/`) + **ARCHIVE_TAG_ONLY** (bulk) | yes | `test_raycast3d_primitive_bounds.py` reads `experiments/`; `test_ogbench_scene_cache` cites `bake_probe_branch.json` |
| `step6/lance/` | 2.5 MB | **KEEP_MAIN** | yes | `scratch/step8/perf/write_doc.py` reads `comparison2.json` |
| `step6/texture-fidelity/` | 4.8 MB | **ARCHIVE_TAG_ONLY** | yes | `test_step4.py` cites `findings.md`; sheets are evidence, not fixtures. Several PNGs are duplicated across `localize312/` and `localize312_nolightspec/` |
| `step6/{gfxdraw,raster2d,tilemap,reacher-correctness,fetch-gpu,texture}/` | 4.1 MB | **ARCHIVE_TAG_ONLY** | yes | workstream evidence; report generator walks the tree |
| `step7/rocket-landing/reference/` | (inside 4.3 MB tree) | **KEEP_FIXTURE** (`states.npz`) | yes | `test_rocket_landing.py` |
| `step7/ogb-manipulation/states/` | (inside 3.5 MB tree) | **KEEP_FIXTURE** | yes | `test_ogb_manip.py` |
| `step7/craftax-classic-full/corpus/states.npz` | (inside 336 KB tree) | **KEEP_FIXTURE** | yes | `test_envs_craftax_rocket.py` |
| `step7/craftax-full/` JSON | 516 KB | **KEEP_MAIN** | yes | `test_craftax_full.py` reads certify JSON |
| `step7/closure2/` | 2.4 MB | **KEEP_MAIN** | yes | `test_step8_gate_record.py` (coverage matrix, `gate_a/cuda_rocket.json`); GPU comments in `test_reproducibility.py` |
| `step7/closure/` | 3.4 MB | **ARCHIVE_TAG_ONLY** | yes | superseded by `closure2/` for the live matrix; keep on the tag |
| `step7/{dmc-quadruped,dmc-manipulator,rocket-landing}` large `*.npz` frames | ~12 MB | **ARCHIVE_TAG_ONLY** | yes | research findings cite them; unit tests load the small state corpora above, not the 224 px frame banks |
| `step8/` | 3.0 MB | **KEEP_MAIN** | yes | support matrix, Gate F, variation maps, provenance: `test_foundation_registry`, `test_variation_maps_fetch`, `tests/certification/test_step8_*`, `tests/integration/test_worldmodel.py` |
| `step9/` | 4.2 MB | **KEEP_MAIN** | **no** | Step-9 oracles and reviews; `test_foundation_registry` loads `foundation/registry_baseline.json`; `test_benchmark_surface` loads `benchmarks/` |
| `decode/ALL.json` + `dino/` + `bench_gpu_vs_cpu.json` | ~0.9 MB | **KEEP_MAIN** | yes | `test_swm_matrix.py`; `docs/research/when_does_this_pay_off.md`, `swm_matrix.md` |
| `decode/pre_*` backups | 1.1 MB | **ARCHIVE_TAG_ONLY** | yes | pre-fix copies |
| `storage/lance.json` | 300 KB | **KEEP_MAIN** | yes | `test_benchmarks.py` comments |
| `storage/pre_final/lance.json` | 300 KB | **DELETE_DUPLICATE** | yes | ≡ `storage/lance.json` (`92758344f5c580da8046496348f26476941a7067d9c92860cdf666fbf883213c`) |
| `pusht_rollout.npz` | 460 KB | **KEEP_MAIN** | yes | default corpus for `examples/rollout_with_policy.py`, `hf_roundtrip.py` |
| `domain_randomization_PushT*.png` | 280 KB | **KEEP_MAIN** | yes | `examples/README.md` embeds them |
| `triton_ab/`, `ddp_w*/` | 56 KB | **ARCHIVE_TAG_ONLY** | yes | leftover bench notes |

### 2.1 Step-6 / Step-8 duplicate groups (not the Menagerie set)

| bytes | sha256 | paths | note |
|---|---|---|---|
| 706,751 | `2df6a8cbe70b86017a1454ca99ca7c46edbc1018925d62988f79c2871dffc27c` | `step6/cpu-3d/replay/gpu/frames_cuda_gpu2106.npz` ≡ `.../postmerge/gpu/frames_cuda_gpu2106.npz` | DELETE_DUPLICATE |
| 485,173 | `41b02863b14acee1f7486362b82988b8f14a5c3ab079e1108e272b99f4796f5e` | `step6/verify/replay/certify/certification.json` ≡ `.../final/certification.json` | DELETE_DUPLICATE inside ARCHIVE tree |
| 295,518 | `92758344f5c580da8046496348f26476941a7067d9c92860cdf666fbf883213c` | `storage/lance.json` ≡ `storage/pre_final/lance.json` | DELETE_DUPLICATE |
| 201,388 | `82eac18a4188c4cff292eec6dfb0931f90df1020f4f12cc3fc10b7cf69ebda57` | `step8/performance/tiers.json` ≡ `step9/oracle/step8_tiers.json` | **KEEP_MAIN both** — Step-9 immutable snapshot |
| 159,681 | `ea36716b9ad3168f67a8348f27f3578aa9ac7669ab40ad1186485187a31a50dc` | `step8/support_matrix.json` ≡ `step9/oracle/step8_support_matrix.json` | **KEEP_MAIN both** |
| 152,750 | `e23c90a9dfc6a2d25a7f772fb9f79d62c22ae7e91db3dba377fd0f435097272c` | `step8/swm/gate_f_matrix.json` ≡ `step9/oracle/step8_gate_f_matrix.json` | **KEEP_MAIN both** |

`step6/verify/replay/` also repeats the same `dmc.json` / `fetch.json` /
`ogb.json` shards across `certify/`, `final/`, `final_declared/`,
`final_step312/`, `certify_final1/` (up to 6-way identical fetch JSON). Treat
the whole verify tree as ARCHIVE_TAG_ONLY; collapsing copies is optional
cleanup inside that archive.

---

## 3. `examples/robocasa_visual/` — 22 MB

| corpus | `du -sh` | class | on `step8-closure` | tests / CI / docs |
|---|---|---|---|---|
| `*.py`, `README.md`, `run.sh`, `media/manifest.json` | 68 KB | **KEEP_MAIN** | yes | `tests/unit/test_examples.py` parses / `--help`s the scripts; `docs/research/robocasa.md` §17, `docs/concepts/fidelity-tiers.md` |
| `media/*.mp4` | 7.5 MB | **RELEASE_ASSET** | yes | README and robocasa research note describe them; **MkDocs does not copy `examples/`**, so Pages never serves these. GitHub blob view only |
| `media/*.gif` | 7.4 MB | **RELEASE_ASSET** | yes | GIF fallbacks of the same two clips. Almost the same size as the MP4s (no win) |
| `media/*.png` | 7.8 MB | **RELEASE_ASSET** | yes | contact sheets + kitchen grid + DR batch. Keep one poster PNG on main if the README should still render on GitHub |

`test_examples.py` skips `media/` via `SKIP_DIRS`. No test loads the binaries.

---

## 4. `docs/assets/` — 28 MB

MkDocs copies everything under `docs/` to the site. GitHub Pages therefore
hosts the full 28 MB today even when a page never links the file.

| corpus | `du -sh` | class | on `step8-closure` | tests / CI / docs |
|---|---|---|---|---|
| `brand/` (`mark.svg` ×3) | 96 KB | **KEEP_MAIN** | yes | `mkdocs.yaml` logo + favicon |
| `media/manifest.json` | 95 KB | **KEEP_MAIN** | yes | `docs/gallery.md` |
| Product-page media (linked from `docs/index.md`, `gallery.md`, `concepts/variation-spaces.md`, `docs/testing.md`, `examples/README.md`, `examples/paper/make_figures.py`) | **4.71 MB / 30 files** | **KEEP_MAIN** | yes | PushT mp4/gif/pngs; MiniGrid/BabyAI table (`grid`, `parity`, `mp4`); MiniHack `grid.png`; MiniGridDoorKey parity + mp4 |
| Env dirs **not linked** from those pages: `MetaWorldAssembly/` (12 MB), `DMCWalkerWalk/` (2.3 MB), `PointmazeMediumNavigate/` (1.4 MB), `CarRacing/` (0.42 MB), plus uncited GIFs/states of linked envs | **24.32 MB / 49 files** | **RELEASE_ASSET** | yes | Produced by `examples/gallery.py`. Gallery prose says the page is a snapshot of the registry, but the markdown never embeds these. They still land on Pages while they live under `docs/` |
| `media/*/states.npy` | 1.21 MB | **KEEP_FIXTURE** if gallery regen stays in-tree; else ARCHIVE | yes | Inputs for `examples/gallery.py`, not tests. MiniHack `states.npy` is 796 KB |

`.pre-commit-config.yaml` excludes `docs/assets/media/` from formatters.
`docs.yml` builds MkDocs and uploads `site/`.

### 4.1 GIFs that should become MP4/WebM later (do not convert in this wave)

| clip | GIF | MP4 | GIF/MP4 |
|---|---|---|---|
| `MetaWorldAssembly/sidebyside` | **6.70 MB** | 0.26 MB | **25.7×** |
| `DMCWalkerWalk/sidebyside` | 1.50 MB | 0.46 MB | 3.2× |
| `PushT/sidebyside` | 0.16 MB | 0.06 MB | 2.8× |
| `CarRacing/sidebyside` | 0.16 MB | 0.06 MB | 2.7× |
| MiniGrid family GIFs | 0.19–0.73 MB | smaller except LavaCrossing | 1.4–2.0× |
| `MiniHack/sidebyside` | 0.36 MB | 0.88 MB | GIF already smaller |
| RoboCasa `sidebyside_L52S03` / `L45S33` | 3.81 / 3.63 MB | 3.92 / 3.59 MB | ~1× — transcode will not save space |

The MetaWorld GIF is the single best later transcode. `gallery.md` already
prefers H.264 MP4 and keeps GIF as a `<video>` fallback; only PushT's GIF is
actually referenced.

---

## Proposed slimmer `main` layout

Target: **~38 MB** of these four trees on the ordinary product branch.
Everything else stays reachable at `step8-closure` / `v0-draft`.

```text
docs/assets/brand/                          # KEEP_MAIN
docs/assets/media/manifest.json
docs/assets/media/PushT/{grid,parity,variation_*,sidebyside.mp4,sidebyside.gif,states.npy}
docs/assets/media/MiniGrid*/{grid,parity,sidebyside.mp4}
docs/assets/media/BabyAIGoToLocal/{grid,parity,sidebyside.mp4}
docs/assets/media/MiniHack/grid.png
# optional: one more env (DMCWalker or MetaWorld) as a 3-D poster, MP4 only

examples/robocasa_visual/*.py README.md run.sh media/manifest.json
# media binaries → gh release  v0.1.0-robocasa-visual.zip

examples/results/step3/*.json
examples/results/step4/states/*.npz
examples/results/step4/certification.json
examples/results/step4/merged.json
examples/results/step5/**/*.json            # not episodes.jsonl
examples/results/step6/cpu-3d/replay/mjwarp_frames_first4.npz
examples/results/step6/lance/comparison2.json
examples/results/step6/raycast3d/experiments/   # what tests open
examples/results/step6/**/findings.md       # or keep via tag + docs already generated
examples/results/step7/closure2/            # live Gate-A / coverage matrix
examples/results/step7/rocket-landing/reference/states.npz
examples/results/step7/ogb-manipulation/states/
examples/results/step7/craftax-*/           # compact JSON + corpus npz
examples/results/step8/                     # entire tree (2.8 MB)
examples/results/step9/                     # entire tree (4.2 MB)
examples/results/decode/ALL.json
examples/results/dino/
examples/results/bench_gpu_vs_cpu.json
examples/results/storage/lance.json
examples/results/pusht_rollout.npz
examples/results/domain_randomization*.{png,json}

scratch/**/*.py *.sh *.sbatch *.md
scratch/certification.json scratch/swm_matrix.json scratch/swm_registry*.json
scratch/census/*.json
scratch/backend3d/scenes/*.xml              # no meshes
scratch/step9/                              # until F relocates packaging-ci
```

Release-asset zip (proposed name, not created this wave):

```text
v0.1.0-research-media.zip
  docs/assets/media/MetaWorldAssembly/
  docs/assets/media/DMCWalkerWalk/
  docs/assets/media/{PointmazeMediumNavigate,CarRacing}/
  docs/assets/media/**/sidebyside.gif          # except PushT
  examples/robocasa_visual/media/*.{mp4,gif,png}
  scratch/backend3d/scenes/*.{obj,stl}        # one copy only
```

Archive-only (already on `step8-closure`):

```text
examples/results/step6/verify/
examples/results/step6/cpu-3d/               # minus mjwarp_frames_first4.npz
examples/results/step6/texture-fidelity/
examples/results/step7/closure/              # superseded
examples/results/step7/*/phase*/**/*.npz     # 224 px frame banks
examples/results/decode/pre_*
scratch/backend3d/scenes/flat/
scratch/backend3d/images/
```

Later delete wave (after review; cite hashes above):

1. `scratch/backend3d/scenes/flat/` meshes (and the unused flat XML if probes do not load them).
2. `examples/results/storage/pre_final/lance.json`.
3. `examples/results/step6/cpu-3d/replay/postmerge/gpu/frames_cuda_gpu2106.npz`.
4. Collapse `step6/verify/replay/` shard copies if that tree is still in-tree.

Do **not** delete `examples/results/step9/oracle/step8_*.json`. They are the
Step-9 behavioral oracle, byte-identical **on purpose**.

---

## What this wave did not do

- No `git rm`, no history rewrite, no `main` push, no tag.
- No GIF→MP4 conversion (MetaWorld is the one proof that would pay; keep the GIF if anyone converts).
- No change to `test_repo_contracts.py` exemptions.
- Workstream F still owns moving `scratch/step9/packaging-ci/` into
  `tests/packaging/` + `scripts/ci/`. Until that lands, those scripts stay
  KEEP_MAIN on `main`.

Helper: `python scripts/archaeology_sizes.py` reprints `du -sh` for the four
trees and their children.
