# Step 6C — `raycast3d`: Ticket 1 (profile and diagnose)

**Classification: `FINDING` + `PATCH_CANDIDATE`.** No renderer code was modified. Everything below is
measured on Oscar under SLURM; local (Apple M5) timings are not used for any number.

- baseline SHA: `89fa5ca81341dc4df7186ec325cb691431a2ef17` (`sami/pre-step6-baseline`)
- branch: `step6/raycast3d`, worktree `/Users/samib/stable-envs-step6-raycast3d`
- venv: `$SCR/venvs/se-step6` (python 3.12, torch 2.14.0+cu126, triton 3.8.0, mujoco 3.12.0,
  dm_control 1.0.45, ogbench)
- CPU host: `node2342` (`--partition=batch --qos=normal`, `OMP_NUM_THREADS=MKL_NUM_THREADS=1`,
  `torch.set_num_threads(1)`) — the CPU certification metric
- GPU host: `gpu2101` / `gpu2107`, NVIDIA GeForce RTX 3090, driver 575.57.08
  (`--partition=gpu --qos=norm-gpu --constraint=geforce3090`)
- SLURM jobs, all mine, all named `se6-raycast3d-*`:
  **6343543** backend census on cpu (`node2339`, COMPLETED) ·
  **6343544** CPU phase sweep (`node2342`, COMPLETED, 6 configs) ·
  **6343664** GPU phase sweep + cuda census (`gpu2101`, cancelled by me once its four output files
  were on disk, because its last step had a harness bug — see §6.6) ·
  **6344633** style-is-state bisect (`gpu2107`, COMPLETED, 00:11:08).
  Submitted-then-cancelled by me before they ran, superseded by the above:
  6343478, 6343479, 6343520, 6343545. No job I did not submit was ever touched.

New, versioned tooling (no renderer edits):

| file | what it does |
|---|---|
| `scratch/step6/raycast3d/profile_phases.py` | wraps the renderer's OWN functions (`_fk`, `cast_nearest`, `any_blocking`, `intersect` per primitive kind, `shade`, `composite`, `_reflect`, `_mover_idx`, …) and attributes wall time to phases; reuses `scratch/step6/profile_note.py` for provenance. Also does CUDA-event timing, kernel-launch counts, cProfile, and an `--fk-counterfactual` upper bound. |
| `scratch/step6/raycast3d/backend_census.py` | renders every declared config once with the raycast3d entry points wrapped, to OBSERVE which configs are on the path |
| `scratch/step6/raycast3d/style_bisect.py` | the Gate-A `check_style_is_state` diagnosis |
| `scratch/step6/raycast3d/{phases_cpu,phases_gpu,census_cpu,style_gpu}.sbatch` | the jobs |

Artifacts: `examples/results/step6/raycast3d/{profile_phases_cpu,profile_phases_gpu,profile_phases_gpu_hostonly}/*.json`,
`backend_census_{cpu,cuda}.json`, `style_is_state_bisect.json`.

---

## 1. The live raycast3d set: **21** render-distinct Step-4 configs, not 25

`examples/results/step4/certification.json`'s `backends` field lists 25 configs carrying
`raycast3d`. That field is a **module-level** attribution — `stable_envs/envs/dmcontrol.py` imports
both `mjwarp` and `raycast3d` and serves two different families out of one module — so reading it as
a per-configuration replay list over-counts by four.

Observed (`backend_census_cuda.json`, job 6343664, gpu2101): **21 of the 25 actually call a
raycast3d entry point during `env.render(states)`.** The four that do not are the
`DMControlTrackCom` cells (`dmcontrol.py:1958`, built from `TRACKCOM_DOMAINS` at `dmcontrol.py:2266`),
whose render path is MJWarp; they cannot even be constructed in `se-step6`
(`ModuleNotFoundError: No module named 'warp'`), which is itself the proof.

Every frozen digest in the table was produced on **cuda**, on an **RTX 3090** (`gpu2105`,
`slurm_job 6176998`), at **batch 64**, from the frozen corpus in `examples/results/step4/states/`.
There is **no CPU digest anywhere in the Step-4 oracle** — the CPU path has never been frozen, which
matters for any CPU-side patch (§5).

| config | class | renderer | size / ss | digest_ours (12) | siblings that share it | Gate A |
|---|---|---|---|---|---|---|
| DMCAcrobotSwingup | DMControlSuite | raycast3d | 84 / 1 | `5e37103625bd` | — | PASS |
| DMCBallInCupCatch | DMControlSuite | raycast3d | 84 / 1 | `6976e9f1d114` | — | PASS |
| DMCCartpoleBalance | DMControlSuite | raycast3d | 84 / 1 | `8c12c06d2cc1` | — | PASS |
| DMCCartpoleSwingup | DMControlSuite | raycast3d | 84 / 1 | `fdf1ab60694d` | — | PASS |
| DMCFingerSpin | DMControlSuite | raycast3d | 84 / 1 | `36054a174c4f` | — | PASS |
| DMCFingerTurnEasy | DMControlSuite | raycast3d | 84 / 1 | `11bd4f99e111` | — | PASS |
| DMCFingerTurnHard | DMControlSuite | raycast3d | 84 / 1 | `5dbb921ffed0` | — | PASS |
| DMCPendulumSwingup | DMControlSuite | raycast3d | 84 / 1 | `691524180e13` | — | PASS |
| DMCReacherEasy | DMControlSuite | raycast3d | 84 / 1 | `4d684da4ce6b` | — | PASS |
| DMCReacherHard | DMControlSuite | raycast3d | 84 / 1 | `04e3961cabb2` | — | PASS |
| AntmazeGiantNavigate | OGBenchMaze | raycast3d | 224 / 2 | `9da286404b65` | AntmazeGiantStitch | PASS |
| AntmazeLargeExplore | OGBenchMaze | raycast3d | 224 / 2 | `42bcc3bc0946` | AntmazeLargeNavigate, AntmazeLargeStitch | PASS |
| AntmazeMediumExplore | OGBenchMaze | raycast3d | 224 / 2 | `5229789db19a` | AntmazeMediumNavigate, AntmazeMediumStitch | PASS |
| AntmazeTeleportExplore | OGBenchMaze | raycast3d | 224 / 2 | `06a2e039e1fd` | AntmazeTeleportNavigate, AntmazeTeleportStitch | **FAIL** (`check_style_is_state`) |
| HumanoidmazeGiantNavigate | OGBenchMaze | raycast3d | 224 / 2 | `542e3d8975e6` | HumanoidmazeGiantStitch | PASS |
| HumanoidmazeLargeNavigate | OGBenchMaze | raycast3d | 224 / 2 | `5eba59e2ba7b` | HumanoidmazeLargeStitch | PASS |
| HumanoidmazeMediumNavigate | OGBenchMaze | raycast3d | 224 / 2 | `93e0baf75e8d` | HumanoidmazeMediumStitch | **FAIL** (`check_style_is_state`) |
| PointmazeGiantNavigate | OGBenchMaze | raycast3d | 224 / 2 | `de659acb4f7a` | PointmazeGiantStitch | PASS |
| PointmazeLargeNavigate | OGBenchMaze | raycast3d | 224 / 2 | `e00fd8b6e180` | PointmazeLargeStitch | PASS |
| PointmazeMediumNavigate | OGBenchMaze | raycast3d | 224 / 2 | `7a287e4f02a0` | PointmazeMediumStitch | PASS |
| PointmazeTeleportNavigate | OGBenchMaze | raycast3d | 224 / 2 | `a5ae0b25bef0` | PointmazeTeleportStitch | PASS |

**Declared but NOT on the path** (do not replay for a raycast3d-only change; DO replay if the change
touches `dmcontrol.py` outside `DMControlSuite`): `DMCCheetahRun`, `DMCHopperHop`,
`DMCHumanoidStand`, `DMCWalkerWalk` — all MJWarp, `visual` tier, and three of the four already
FAIL Gate A for the unrelated MuJoCo 3.11→3.12 texture drift recorded in
`docs/research/step6_optimization.md`.

**Replay obligation for any raycast3d patch: those 21 configs, on cuda, at batch 64, against
`digest_ours` above** (35 registered names once siblings are counted). A patch confined to
`DMControlSuite` replays 10; a patch confined to `OGBenchMaze` replays 11; a patch inside
`stable_envs/render/raycast3d.py` replays all 21 **plus** the `maze2d` (`PointMazeLarge/Medium/
Open/UMaze`) and `mujoco_locomotion` (`MujocoReacher/Pusher/InvertedPendulum/
InvertedDoublePendulum`) envs, which call `R3.composite` / `R3.cast_nearest` as well but are NOT in
the frozen 49 — i.e. those have no frozen oracle to replay against. That is an open question for the
coordinator (§6).

---

## 2. CPU profile with attribution (node2342, 1 torch thread)

`render` wall time, median of 5, states from the frozen Step-4 corpora. `isect` is the sum over
`raycast3d.intersect` calls, split by primitive kind. `fk` is `_fk`, the per-environment
`mj_kinematics`/`mj_forward` loop. Columns are INCLUSIVE.

### B = 64

| config | frame | K prims | kinds | render ms | fps | fk ms | cast ms | shadow ms | reflect ms | shade ms | composite ms | **isect ms (%)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PointmazeLargeNavigate | 224 ss2 | 1 | 1 sphere | 89.1 | 717.9 | 0.10 | 17.4 | – | – | 28.5 | 80.9 | 12.6 (**14%**) |
| DMCReacherHard | 84 ss1 | 4 | 2 caps, 2 sph | 980.2 | 65.3 | 0.90 | 446.1 | 351.2 | 307.3 | 42.1 | – | 749.0 (**76%**) |
| DMCFingerSpin | 84 ss1 | 7 | 5 caps, 2 cyl | 1337.9 | 47.8 | 0.79 | 886.5 | 408.4 | 168.8 | 26.7 | – | 1234.8 (**92%**) |
| AntmazeMediumExplore | 224 ss2 | 13 | 12 caps, 1 sph | 929.2 | 68.9 | 0.56 | 863.5 | – | – | 27.2 | 928.1 | 812.2 (**87%**) |
| HumanoidmazeMediumNavigate | 224 ss2 | 19 | 16 caps, 3 sph | 1062.9 | 60.2 | 0.66 | 1024.1 | – | – | 25.0 | 1083.9 | 959.5 (**90%**) |
| DMCBallInCupCatch | 84 ss1 | 36 | 33 caps, 3 sph | 3724.1 | 17.2 | 18.13 | 2395.6 | 1314.8 | 853.6 | 12.1 | – | 3535.4 (**95%**) |

Full sweeps (B = 1 / 16 / 64 / 256, fps):

| config | B=1 | B=16 | B=64 | B=256 |
|---|---|---|---|---|
| PointmazeLargeNavigate | 219.0 | 722.5 | 717.9 | 759.6 |
| DMCReacherHard | 40.7 | 81.9 | 65.3 | 65.7 |
| DMCFingerSpin | 20.1 | 54.3 | 47.8 | 42.4 |
| AntmazeMediumExplore | 37.7 | 74.3 | 68.9 | 54.6 |
| HumanoidmazeMediumNavigate | 29.4 | 61.3 | 60.2 | 44.2 |
| DMCBallInCupCatch | 5.5 | 21.1 | 17.2 | 15.8 |

### The bottleneck, and its mechanism

**`raycast3d.intersect` is 76–95 % of every CPU render that has more than one mover, and the cost is
strictly proportional to the primitive count K.** There is no fused CPU cast: `cast_nearest`
dispatches to Triton only when `_use_triton(o)` holds, and that requires `o.is_cuda`
(`raycast3d.py:827`; `cast_nearest` at `:868`, `cast_nearest_torch` at `:444`, `intersect` at `:405`). On CPU every call falls into `cast_nearest_torch`, which is a **Python
loop over K primitives**; each iteration runs `_capsule`/`_cylinder`/`_box`/`_sphere`, and each of
those is ~150–250 separate full-size elementwise aten kernels over a `(B, N)` grid, every one of
which materialises a `(B, N)` or `(B, N, 3)` temporary.

Normalised, this is a constant: dividing `isect` time by the number of ray-primitive tests actually
issued (from the recorded per-pass ray counts) gives

| config | ray-primitive tests / frame @B=64 | ns per ray-primitive test |
|---|---|---|
| DMCReacherHard | 80,884 | 145 |
| DMCBallInCupCatch | 279,972 | 197 |
| DMCFingerSpin | 93,401 | 207 |
| AntmazeMediumExplore | 53,248 | 238 |

**145–240 ns for one analytic ray/primitive intersection on one x86 core**, against ~60–100 flops of
actual arithmetic. cProfile at B=16 names the terms: for `DMCFingerSpin`, `_capsule` 75.9 ms self,
`torch.stack` 60.1 ms (163 calls — that is `_mv`/`_mtv`/`_dot` rebuilding `(B,N,3)` stacks),
`_mtv` 32.6, `torch.where` 23.4 (245 calls), `_dot` 22.6 (100 calls), `torch.sqrt` 17.4. At B=256
`DMCReacherHard` issues 7,794 aten ops in 3,897 ms — **0.50 ms per aten op**, each touching
1.8–3.2 MB, i.e. ≈10 GB/s: **the path is at one-core memory bandwidth on intermediates that are
dead the moment the nearest-hit test has looked at them.** The answer kept per ray is one float and
one 3-vector.

This is exactly the argument the module docstring makes for the CUDA fused kernel
(`raycast3d.py`, "the fused cast"): *"every quantity between two primitives is dead the moment the
nearest-hit test has looked at it, so it never needs to reach memory at all."* The fused kernel was
built for CUDA only. **The CPU has no equivalent and that is the whole CPU story.**

Two secondary CPU facts, both measured:

- **`_fk` is not the CPU bottleneck anywhere** — 0.06–0.49 % of the render. `--fk-counterfactual`
  on `DMCBallInCupCatch` gives 5.5 → 5.5 fps at B=1 and 17.2 → 17.2 at B=64: removing `_fk`
  entirely changes the CPU number by nothing. (On GPU it changes it by 23x — §3.)
- **Three of the four DM Control ray passes are not bounded to the movers, and two of them cast more
  rays than the mover pass does.** Recorded per-pass ray counts at B=64: `DMCFingerSpin` casts 1,764
  rays/env for the movers and **7,056** (the FULL 84x84 frame) for the reflection pass;
  `DMCReacherHard` 4,489 and 6,720; `DMCBallInCupCatch` 1,764 and 3,136. `_reflect` is 31 % of
  `DMCReacherHard`'s CPU render (307.3 ms of 980.2) and `any_blocking` (the two shadow passes) is
  36 % (351.2 ms). `_fk`'s own docstring already names this — *"the shadow test starts from every
  STATIC pixel and the reflection test from every floor pixel … Bounding them to the movers'
  light-space and screen-space footprints is the next thing worth doing here"* — and the profile
  agrees with it. It belongs to whoever owns the DM Control CPU rows.

- **`PointmazeLargeNavigate` is the exception, and it is the only raycast3d CPU row `plan.json`
  assigns to this workstream.** With K=1 the cast is 14 % and `R3.composite` is 91 % of the render
  (`composite` 80.9 ms incl of 89.1 ms at B=64; at B=256, 308.6 ms of 337.0 ms, of which `shade` is
  115.5 ms and `composite` self ≈ 128 ms). Its bottleneck is the **compositing/output path**, not the
  caster: a full-frame `out.copy_(plate.view(1,H,W,3).expand(B,H,W,3))` stride-0 broadcast into a
  `(B,224,224,3)` uint8 buffer, a float promotion of the plate, `torch.gather`/`scatter_` through
  `(B, tile*tile, 3)` int64 index tensors built per call by `_expand_tile`, and the
  `view(B,tile,ss,tile,ss,3).sum((2,4))` box average.

### Where that leaves the CPU tier

Step 3's frozen 1-thread Oscar rows (`examples/results/step3/certification.json`, `node2349`):
`CPU_FAST := S_cpu >= 5 AND cores_cpu >= 1`. On those rows **S_cpu already passes for five of the
seven raycast3d cells** (Pointmaze 48.0, Cartpole 9.1/10.1, Acrobot 7.05, Pendulum 3.79, Reacher
2.88, FingerTurnHard 1.96, BallInCup 1.62) and **`cores_cpu` fails on every one of them**
(0.003–0.42). The blocking qualifier `BELOW_G4` is `cores_cpu < 1` in all seven cases, i.e. the bar
is one-core JPEG-q90 decode throughput: 2,390.5 fps at 224 px and 7,015-7,187 fps at 84 px on
that node.

| row (node2349, 1 thread) | se_fps | decode fps/core | cores | factor short of cores=1 |
|---|---|---|---|---|
| PointmazeLargeNavigate | 994.3 | 2,390.5 | 0.4159 | **2.40x** |
| DMCCartpoleSwingup | 372.2 | ~7,000 | 0.06 | ~17x |
| DMCAcrobotSwingup | 267.0 | ~7,000 | 0.04 | ~25x |
| DMCReacherHard | 98.6 | 7,014.9 | 0.0141 | **71x** |
| DMCFingerTurnHard | 57.4 | 7,125.3 | 0.0081 | **124x** |
| DMCBallInCupCatch | 20.9 | 7,186.9 | 0.0029 | **344x** |

(The same `PointmazeLargeNavigate` row reaches `CPU_FAST` on the laptop — `cores` 1.0743 at
8,012.8 fps against a laptop decode rate of 7,458.9 — so this row's tier is a statement about the
certification host, not only about the code.)

So: **Pointmaze CPU_FAST is reachable and the DM Control 84 px cells are not, by any single
structural change.** An ideal fused, SIMD CPU cast at one core is worth roughly 2,000 fps scalar /
~16,000 fps at 8-wide on the 4-primitive cells (≈2.4 MFLOP per frame at 84 px) — enough for
`DMCReacherEasy/Hard`, `Cartpole`, `Acrobot`, `Pendulum`, not enough for `DMCBallInCupCatch` at
K=36 (≈1,800 fps ⇒ cores ≈0.27). That is the measured next blocker.

---

## 3. GPU: why `DMCBallInCupCatch` is the outlier — it is not the caster

Both cells in **one job on one card** (job 6343664, gpu2101, RTX 3090), CUDA-event timed with
explicit synchronisation, states from the frozen corpora.

| B | BallInCup render ms | fps | device ms | device/wall | Finger render ms | fps | device ms | device/wall |
|---|---|---|---|---|---|---|---|---|
| 1 | 6.64 | 150.6 | 1.48 | 22 % | 5.51 | 181.6 | 1.01 | 18 % |
| 16 | 12.67 | 1,263.0 | 1.72 | 14 % | 7.75 | 2,065.7 | 1.53 | 20 % |
| 64 | 26.49 | 2,416.1 | 2.29 | 8.6 % | 8.23 | 7,778.6 | 2.15 | 26 % |
| 256 | 97.15 | 2,635.1 | 5.11 | 5.3 % | 11.31 | 22,634.7 | 5.64 | 50 % |
| 1024 | 330.86 | 3,095.0 | 17.09 | 5.2 % | 40.33 | 25,389.9 | 19.23 | 48 % |
| 4096 | 1,313.97 | 3,117.3 | 66.66 | **5.1 %** | 105.47 | 38,834.3 | 72.63 | 69 % |

`set_state` is 0.14–0.29 ms throughout and is not the issue. Kernel launches per render are
**339 for BallInCup and 327 for Finger** — essentially identical, and constant in B. BallInCup's
*device* time at B=4096 (66.66 ms) is **lower** than Finger's (72.63 ms). The GPU is not the
constraint for either cell, and it is emphatically not slower for BallInCup.

### What is different about the scene

| | DMCBallInCupCatch | DMCFingerSpin |
|---|---|---|
| mover primitives K | **36** | 7 |
| kinds | 33 capsule + 3 sphere | 5 capsule + 2 cylinder |
| of which a drawn **spatial tendon** | **28 capsules** | 0 |
| `_fk` kernel | `mj_forward` (needs the constraint set for the tendon's colour) | `mj_kinematics` |
| rays cast per env (mover / reflect pass) | 1,764 / 3,136 | 1,764 / 7,056 |
| reflectance, shadow light | 0.2, light 1 | 0.2, light 1 |

BallInCup casts **fewer** rays than Finger. Its K is 5x larger, which costs it a 5x longer runtime
loop inside one fused Triton kernel — and that shows up as 66 ms of device time at B=4096, which is
5 % of its wall clock.

### The measured cause: a serial per-environment host loop

Phase attribution, and the same run repeated with **no CUDA sync inside the wrappers** so the number
is pure host time:

| | BallInCup B=256 | BallInCup B=4096 | Finger B=256 | Finger B=4096 |
|---|---|---|---|---|
| render (wall) | 97.47 ms | 1,260.72 ms | 10.09 ms | 105.57 ms |
| **`_fk` (host)** | **88.37 ms (90.7 %)** | **1,224.49 ms (97.1 %)** | 3.19 ms (32 %) | 51.03 ms (48 %) |
| `cast_nearest` (host dispatch) | 0.33 ms | 0.82 ms | 0.28 ms | 0.54 ms |
| `shade` | 2.90 ms | 5.94 ms | 2.97 ms | 5.86 ms |
| `_reflect` | 2.50 ms | 22.10 ms | 2.23 ms | 25.99 ms |
| `screen_span` (+ its one sync) | 1.38 ms | 3.26 ms | 1.40 ms | 2.80 ms |
| device total | 5.12 ms | 67.05 ms | 5.64 ms | 73.56 ms |

**Per-environment `_fk` cost: 345 µs at B=256 and 299 µs at B=4096 for BallInCup, against a flat
12.5 µs for Finger — 24-28x.** The asymptotic throughput is its reciprocal: 1/299 µs = 3,344 fps
against a measured 3,117 fps at B=4096 (the residual is the ~67 ms of device work and the
`screen_span` sync). That is the whole of `BATCH_SCALES_FAIL` and `BELOW_G4`: the render saturates
at the speed of one Python thread walking the batch, not at the speed of the card.

The counterfactual confirms it. With `_fk` memoised out of the timed region (`--fk-counterfactual`,
an upper bound, not a proposed implementation):

| B | 1 | 16 | 64 | 256 | 1024 | 4096 |
|---|---|---|---|---|---|---|
| BallInCup fps | 150.6 | 1,263 | 2,416 | 2,635 | 3,095 | **3,117** |
| BallInCup fps, `_fk` frozen | 137.0 | 2,101 | 8,249 | 29,033 | 60,463 | **73,187** (**23.5x**) |
| Finger fps | 181.6 | 2,066 | 7,779 | 22,635 | 25,390 | **38,834** |
| Finger fps, `_fk` frozen | 155.2 | 2,184 | 8,660 | 31,631 | 62,514 | **73,801** |

Both cells converge on the same ceiling (~73 k fps) once `_fk` is out. **`_fk` is the structural
bottleneck of the whole `DMControlSuite` GPU path; BallInCup is simply the cell where it is 24-28x
worse per environment.** Finger at B=4096 is itself 48 % `_fk`, and would nearly double if `_fk`
were batched.

### Inside `_fk` — and the code's own explanation is wrong

cProfile of ONE `DMCBallInCupCatch` render at B=256 (tottime/cumtime in ms; profiler inflates the
total ~30 %, the proportions are the point):

| function | calls | cum ms | share of `_fk` |
|---|---|---|---|
| `_fk` (`dmcontrol.py:1509`) | 1 | 114.08 | 100 % |
| ↳ `_tendon_frame` (`:624`) | 256 | 89.09 | **78 %** |
| ↳↳ `_chain_capsules` (`:541`) | 256 | 52.53 | **46 %** |
| ↳↳↳ `numpy.cross` | 512 | 26.72 | 23 % |
| ↳↳↳ `numpy.linalg.norm` | 768 | 6.86 | 6 % |
| ↳↳ `catenary` (`:485`) | 164 | 31.56 | **28 %** |
| ↳↳↳ `_solve_catenary` (`:467`) | 160 | 3.38 | 3 % |
| ↳↳↳ `_cosh_sinh` (`:454`) | 5,612 | 2.52 | 2 % |
| ↳ `mujoco.mj_forward` | 256 | **3.50** | **3 %** |
| ↳ `torch.tensor` (the 4 pose/size/colour H2D uploads) | 4 | 12.39 | 11 % |

`DMCFingerSpin`'s whole `_fk` is 3.48 ms cum, of which `mj_kinematics` is 0.27 ms.

`_fk`'s docstring says *"A scene with a tendon runs `mj_forward` rather than `mj_kinematics`, and
pays for it … `ball_in_cup` is 4 dof and 6 geoms, so the extra work is microseconds"* and *"a
batched GPU forward kinematics is not [the next thing worth doing]"*. **The first half is right —
`mj_forward` is 3 % — and the conclusion does not follow.** The cost is not MuJoCo at all: it is
pure Python/numpy, run once per environment, building the tendon's 28 capsules —
`_chain_capsules`' four `numpy.cross`/`norm` calls on `(28,3)` arrays (512 numpy dispatches per
render) and `catenary`'s 29-iteration Python point loop with `(3,)` numpy temporaries per iteration.
Both are already array code; neither has a batch axis.

---

## 4. Gate-A `check_style_is_state` on `AntmazeTeleportExplore` and `HumanoidmazeMediumNavigate`

> **SUPERSEDED IN PART — read `RAYCAST-2c and 2c'` below before using this section.** The root cause
> named here (the bake inheriting `qpos` from the last rendered environment) was tested by fixing it
> and it changed nothing; and the fallback reading further down — that the bake is nondeterministic,
> "what a coin flip looks like from one sample" — is also wrong. The eight-bake digest sequences are
> byte-identical across two processes on two different code trees and across two nodes: the bake is a
> DETERMINISTIC function of the renderer's call history. What survives from this section is the
> evidence, not the attribution: the cache-eviction trigger, the per-config measurements, and the
> observation that configurations which PASS Gate A carry the same defect. Kept as written, dated to
> before the correction, because a finding is worth more with its wrong turn visible than tidied away.


### What the check measures

`stable_envs/testing/conformance.py:462`:

```python
st  = env.sample_states(env.B);  sty = env.sample_style(env.B)
a   = env.render(st, style=sty)                       # the env that has already been used
fresh = type(env)(batch=env.B, size=env.size, supersample=env.ss, device=str(env.device))
fresh.set_state(st, style=sty);  b = fresh.render()
return {"max_abs_diff": ...}                          # bound: == 0
```

It is the claim that lets a dataset store states: **`(state, style)` must fully determine the frame**.
Both configs observe `1` — one pixel channel off by 1/255. The other nine OGBench
configurations, and all ten DM Control ones, observe `0`.

### Root cause: `OGBenchMaze._bake` renders the SHARED `MjData` without resetting `qpos`

`stable_envs/envs/ogbench.py:388` `_bake` applies the style to the shared model, hides the agent
(`m.geom_rgba[agent, 3] = 0.0`), and calls `r.update_scene(d, camera=cam); r.render()`. **It never
sets `d.qpos`.** `d` is the single `MjData` held in `_ENV_CACHE` and shared by every scene of that
(loco, maze, size); `_fk` (`ogbench.py:488`) writes `qpos` and runs `mj_kinematics` on it once per
environment per render. So the plate a bake produces is a function of whatever pose the LAST
rendered environment left behind — a hidden (alpha-0) agent still reaches MuJoCo's rasterizer, and
it is enough to move a pixel.

`DMControlSuite._bake` (`dmcontrol.py:1229`) does the opposite and says so:
`d.qpos[:] = m.qpos0; d.qvel[:] = 0; mujoco.mj_forward(m, d)` before the first shot. That is the
control, and it is why the whole DM Control shard passes this check.

The impurity is normally invisible because the plate is memoised in `_SCENE_CACHE`
(`ogbench.py:376`), so the second render of the same appearance gets a cache HIT and never re-bakes.
It becomes visible exactly when the cache evicts — and `_scene` does not evict an entry, it does
`_SCENE_CACHE.clear()` on the whole dict at `_SCENE_CACHE_MAX = 256` (`ogbench.py:382`,
`ogbench.py:116`). A `check_style_is_state` call adds 64 entries (64 sampled appearances, each a
distinct plate key), the cache accumulates across checks AND across configurations in one
certification session, and whichever configuration happens to straddle the wholesale clear fails.

### Measured (job 6344633, gpu2107, RTX 3090, batch 64)

`plate_purity` bakes with the DEFAULT style (one scene), digests the plate, renders frozen states
through the public path, forces the clear the cache forces itself, and re-bakes the SAME key.
`frozen_check` runs the conformance check verbatim and then separates its two axes.

| config | Gate A | plate depends on render history | pixels moved | `reset qpos0` restores plate | frozen check in isolation | same env, same states, after a cache clear |
|---|---|---|---|---|---|---|
| AntmazeTeleportExplore | **FAIL (1)** | **yes** | 2 px, max 1/255, at y 202-207 / x 44-46 | yes | 0 | 0 |
| HumanoidmazeMediumNavigate | **FAIL (1)** | no (these poses) | 0 | yes | 0 | **1** |
| AntmazeTeleportNavigate | PASS (0) | **yes** | **10 px, max 1/255** | **no** | 0 | 0 |
| HumanoidmazeLargeNavigate | PASS (0) | no (these poses) | 0 | yes | 0 | **1** |
| PointmazeLargeNavigate | PASS (0) | no | 0 | yes | 0 | 0 |

`scene_cache_len` goes 1 -> 65 on every row (64 sampled appearances + the default), i.e. **no
eviction happens in an isolated run**, which is why the isolated frozen check returns 0 everywhere.
`plate_t` (`R3.static_depth`, `raycast3d.py:938`) is unchanged on every row.

Three things follow, and the third is the important one.

1. **The plate is not a pure function of (scene, style).** Re-baking the same key after a render
   changes the plate by 1/255 on 2-10 pixels. The analytic depth buffer `plate_t` is unaffected
   (`max |Δ| = 0.0` everywhere) — it is cast by `R3.static_depth` from static geom poses, which do
   not move — so the impurity is entirely in the MuJoCo-rasterized RGB plate. Restoring
   `d.qpos = m.qpos0` and re-baking reproduces the original plate exactly on AntmazeTeleportExplore
   and HumanoidmazeMediumNavigate, which identifies `qpos` as the carrier.
2. **The failure needs a cache eviction to surface.** In isolation the frozen check returns 0 on
   all five configs (`scene_cache_len` goes 1 -> 65, well under 256, so the `fresh` env's 64 keys are
   all HITS and it renders from the identical cached plates). Forcing the clear reproduces
   `max_abs_diff = 1` on `HumanoidmazeMediumNavigate` — i.e. the frozen observation of 1 is
   reproduced deterministically, from the same env, the same states and the same style, with the
   cache eviction as the only difference.
3. **Configurations that PASS Gate A carry the same defect.** `AntmazeTeleportNavigate` (PASS) shows
   it more strongly than `AntmazeTeleportExplore` (FAIL) — 10 pixels rather than 2, and a `qpos0`
   re-bake does not even restore its plate — and `HumanoidmazeLargeNavigate` (PASS) reproduces
   `after a cache clear = 1` exactly as `HumanoidmazeMediumNavigate` (FAIL) does. Every legged cell
   probed exhibits it; the `PointmazeLargeNavigate` control (a single sphere agent) exhibits neither
   probe, which fits — the ant has 13 mover geoms and the humanoid 19, spread across the maze, and
   the pointmaze agent is one sphere sitting on open floor.

   So the fail/pass split across the eleven OGBench configurations is an **ordering artefact of one
   latent defect**, not a property of the two that failed. Which two failed depends on where the
   256-entry wholesale `clear()` landed in the Step-4 session. **`AntmazeTeleportExplore` and
   `HumanoidmazeMediumNavigate` are not two broken configurations; they are the two that happened to
   straddle the clear, and fixing only them would fix nothing.**

### Shape of the fix (NOT applied; report only, per my authorization)

`_bake` should bake from a stated pose rather than an inherited one — `d.qpos[:] = m.qpos0;
d.qvel[:] = 0; mujoco.mj_forward(m, d)` before the first `update_scene`, exactly as
`DMControlSuite._bake` already does. That makes the plate a pure function of (scene, style) and
removes the check's dependence on cache state. It is a **correctness** change and it WILL move the
frozen digests of any OGBench configuration whose cached plate was baked from a non-`qpos0` pose —
which cannot be known without replaying, so all 11 OGBench configs must be re-rendered and diffed,
and a digest change here is a FIX of a latent non-determinism rather than a regression. That
reasoning has to be the coordinator's, not mine; I am reporting, not patching.

A second, independent smell found on the way: `_SCENE_CACHE` and `_RAY_CACHE`/`_PROJ_CACHE`/
`_CORNER_CACHE`/`_KIND_CACHE` (`raycast3d.py`) all `clear()` wholesale when full rather than
evicting one entry. For pure caches that is only a performance cliff; here it is the trigger that
turned a latent purity defect into a Gate A failure on two configurations and not the other nine.



---

## 5. Proposed interventions (Ticket 2 candidates — NOT implemented)

### 5a. GPU / `DMCBallInCupCatch` — batch the tendon geometry over the batch axis

- **Where:** `stable_envs/envs/dmcontrol.py` — `_fk` (`:1509`), `_tendon_frame` (`:624`),
  `_chain_capsules` (`:541`), `catenary` (`:485`). **Not** in `stable_envs/render/raycast3d.py`.
- **Mechanism:** `_chain_capsules` is already vectorised over `nseg`; give it a leading batch axis so
  the whole batch's polylines become one `(B, nseg, 3)` `numpy.cross`/`norm` instead of `2B`/`3B`
  calls on `(28,3)`. Do the same for `catenary`'s point loop (kept scalar per point where `math.exp`
  bit-exactness is load-bearing — see risk). `mj_forward` and the Newton solve stay per environment;
  together they are 6 % of `_fk`.
- **Expected effect:** `_chain_capsules` is 46 % and `catenary`'s point construction ~25 % of `_fk`;
  `_fk` is 90.7 % of the render at B=256. Removing 70 % of `_fk` moves the B=4096 asymptote from
  3,117 fps to ≈9,500 fps. The GPU_FAST bar for this row is `cores >= 1`, i.e.
  **5,965.9 fps** (Step 3, gpu2107: 2,458.7 fps at `cores` 0.4121, `decode_fps_per_core` 5,965.9 — a 2.43x shortfall) with `R = 9.38` already
  above the class-S bar of 4. `BATCH_SCALES` (fps(4096) ≥ 20·fps(1)) also follows: fps(1) is
  launch-bound and barely moves. The `_fk`-frozen upper bound is 73,187 fps (23.5x), so there is
  large headroom for a partial fix.
- **Risk to pixels:** the tendon's capsule poses are float64 geometry that becomes float32 capsule
  primitives; a 1-ulp change flips silhouette pixels (the module docstring measures 1.2e-7 in `dl`
  amplifying to 2.4e-4 in `disc`, and `disc > 0` is a HIT/MISS decision). Batching `numpy.cross` and
  `numpy.linalg.norm` over a leading axis performs the same per-row elementwise multiply/subtract
  and the same 3-element reduction, so it *should* be bit-identical — **that must be asserted on the
  frozen corpus, not assumed.** `catenary`'s `math.exp`/`math.cosh` must NOT be replaced by
  `numpy.exp`: the docstring records a 0-ULP validation against MuJoCo's C `mjv_catenary` over 440
  inputs that depends on calling the same libm.
- **Replay:** the 10 `DMControlSuite` configs (`DMCAcrobotSwingup`, `DMCBallInCupCatch`,
  `DMCCartpoleBalance`, `DMCCartpoleSwingup`, `DMCFingerSpin`, `DMCFingerTurnEasy`,
  `DMCFingerTurnHard`, `DMCPendulumSwingup`, `DMCReacherEasy`, `DMCReacherHard`) on cuda at batch 64.
  Only `DMCBallInCupCatch` has a tendon, so the other nine are a control that must come back
  byte-identical. `DMCPointMassEasy` and `DMCSwimmerSwimmer6` are registered but outside the frozen
  49 and have no oracle.
- **Semantic follow-up:** none if digests are identical (the expected outcome); DINO tripwire only if
  `DMCBallInCupCatch`'s digest moves.

### 5b. CPU — the actionable row is `R3.composite`, which never got `_render_group`'s compactions

The highest-leverage CPU intervention **by mechanism across the whole path** is a **fused CPU cast**:
the exact analogue of `_tri_cast`, keeping the K-primitive loop in registers and writing only
`(t, normal, index)`. It addresses 76–95 % of every multi-mover CPU render and nothing else can. It
is also substantial work (a C++/`torch.compile`/numba kernel that must be bit-identical to
`cast_nearest_torch`, with the same `enable_fp_fusion=False` / `div_rn` / `sqrt_rn` discipline the
Triton kernel needed), it **cannot** reach `CPU_FAST` on the 84 px DM Control cells (§2), and those
rows belong to **`step6/cpu-3d`** in `plan.json`, not to this workstream.

The intervention that is in this workstream's scope AND reaches a tier is different, and the profile
found it:

- **Where:** `stable_envs/render/raycast3d.py:958` `composite` — the function OGBench and `maze2d`
  render through. A file shared with `step6/cpu-3d`; **ownership question for the coordinator.**
- **What the profile says.** `PointmazeLargeNavigate`, B=256, 337.0 ms total:

  | phase | ms | share |
  |---|---|---|
  | `composite` self (plate write, gather/scatter, box average, round/clamp/cast) | 127.46 | **38 %** |
  | `shade` | 115.49 | **34 %** |
  | `cast_nearest` (of which `intersect(sphere)` 48.54) | 64.86 | 19 % |
  | everything else (`screen_span`, `_fk`, `_agent_appearance`, `cached_rays`) | 1.2 | 0.4 % |

- **Mechanism.** `composite` does two things `DMControlSuite._render_group` was already taught not to
  do:
  1. **It shades the whole tile window, not the covered rays.** The window is `TILE=32` output px
     (64×64 supersampled samples per env); the pointmaze agent is a ~13 px sphere, ≈13 % of the
     window. Every one of the ~75 elementwise ops in `shade` runs on 8x more samples than can
     possibly be visible, and the result is then discarded by
     `torch.where(cov.unsqueeze(-1), c, zeros)`. `_render_group` narrows to
     `torch.nonzero(cover_w.any(0))` first (`COVER_COMPACTION`), and argues correctly that
     restricting the SET cannot change a VALUE because each ray's colour is a function of that ray
     alone.
  2. **It rebuilds per call what is constant.** `out.copy_(plate.view(1,H,W,3).expand(B,H,W,3))` is a
     stride-0 broadcast into a `(B,224,224,3)` uint8 buffer; `plate.reshape(1,H*W,3).to(col.dtype)`
     re-promotes the plate to float every call; `_expand_tile` (`raycast3d.py:1119`) rebuilds
     `(B, tile*tile)` int64 index tensors per call and they are then `expand`ed to `(B, N, 3)` for a
     `gather` and a `scatter_` — 6.3 MB of int64 index traffic for 2.4 MB of pixels at B=256.
- **Expected effect.** Cover-compacting `shade` takes 115.5 ms down by the ratio of covered samples to window
  samples. That ratio is estimated, not measured here: `ogbench.py:238` records the pointmaze agent
  as "~13 px across", i.e. ≈133 of the 1,024 output pixels in a 32x32 window (≈13 %), which would
  give ≈17 ms. **Measure it before relying on it.**
  caching the index grid and the promoted plate and writing the frame from a contiguous source
  should take the 127.5 ms of `composite` self down substantially. A conservative estimate of
  ≈122 ms against 337 ms is **2.76x**. Applied to Step 3's certification row
  (`node2349`, best_batch 64, 994.3 fps, decode 2,390.5 fps/core, `cores` 0.4159) that is
  ≈2,744 fps ⇒ **`cores` ≈ 1.15 ⇒ `CPU_FAST`**, with `S_cpu` already at 48.0 against a bar of 5.
  It also carries `PointmazeLargeStitch` (same scene, `cores` 0.4139).
  It does **little** for Antmaze/Humanoidmaze, whose `composite` self is only 151 ms of 4,688 ms —
  those stay cast-bound and need 5b's fused kernel. That is the honest split.
- **Risk to pixels:** none by construction, and the repo already makes the argument twice: the tiled
  and full-frame paths must agree (`tests/gpu/test_fast_paths.py` asserts it), cover compaction is
  the identity outside the covered set, and caching an index grid or a promoted plate changes no
  arithmetic. GPU digests must come back byte-identical — that is the acceptance test, not a
  tolerance.
- **Replay:** all **11** `OGBenchMaze` configs (they all enter `composite`) on cuda at batch 64,
  plus — because `composite` also serves `maze2d` — `PointMazeLarge/Medium/Open/UMaze`, which have
  **no frozen Step-4 oracle** (§6.4). If the change is confined to `composite`, the 10
  `DMControlSuite` configs do not enter it (they use `cast_nearest`/`any_blocking` directly) but
  should be replayed anyway as a cheap control.

### Are 5a and 5b the same intervention?

**No.** They are in different files, have different mechanisms, and the evidence separates them
cleanly: `--fk-counterfactual` changes the GPU number by 23.5x and the CPU number by **0.0x**
(5.5 → 5.5 fps at B=1, 17.2 → 17.2 at B=64 on `DMCBallInCupCatch`); K-proportional `intersect` time
is 76–95 % of the CPU render and ~5 % of the GPU wall clock. A patch that fixed one would not move
the other, and neither of them is in `cast_nearest`'s fused CUDA kernel — which this profile
exonerates on both devices.

---

## 6. Open questions / coordination

1. **Workstream boundary.** `examples/results/step6/plan.json` assigns the `DMCBallInCupCatch` GPU
   row and every DM Control CPU row to **`step6/cpu-3d`** (`raycast3d` gpu shows `rows_below: 0`);
   `STEP_06_AGENT_BRIEFS/03_RAYCAST3D.md` assigns BallInCup to **me**. Both cannot be right, and the
   fix for BallInCup is in `stable_envs/envs/dmcontrol.py`, not in the renderer. Please confirm
   ownership before authorising Ticket 2.
2. **Authorization scope.** My Ticket-1 authorization forbids touching `stable_envs/render/
   raycast3d.py` "or any renderer code". Intervention 5a is in `dmcontrol.py`; 5b is in
   `raycast3d.py`, which AGENT_COMMON explicitly names as shared with `step6/cpu-3d`. Both need
   explicit authorization naming the file.
3. **No frozen CPU oracle.** Every Step-4 digest was taken on cuda. A CPU-side patch has nothing to
   replay against except "cuda digest unchanged"; whether the CPU path is expected to be
   byte-identical to the CUDA path is a contract question (`tests/gpu/test_fast_paths.py` asserts
   the two casts agree, so the answer is probably yes, but it should be stated).
4. **`maze2d` and `mujoco_locomotion` also call `raycast3d`** (`R3.composite`, `R3.cast_nearest`,
   `R3.any_blocking`, `R3.screen_span`) and are NOT among the frozen 49 — a change in
   `raycast3d.py` can move their pixels with no oracle to catch it.
5. **Bug in the shared harness — FIXED on this branch at the coordinator's instruction**
   (commit `6ab122d`): `scratch/step6/profile_note.py:251` read `host = n["host"].split(".")[0]`
   where the dict in scope is `note` (`n` is `render_note`'s parameter). A run measured every batch,
   printed every line, then raised `NameError` before writing either output file, while SLURM
   reported COMPLETED. One-line change, no other edit to the harness. My `profile_phases.py` imports
   only `git_commit`/`versions` from it and never hit this, so none of the Step 6C measurements
   needed re-running.
6. **`--fk-counterfactual` is a diagnostic, not a proposal.** It memoises `_fk` per `(scene, batch)`,
   which is not a legal renderer behaviour (it would break state-dependence). It exists only to put a
   number on the bound.

---
---

# RAYCAST-2 — the three authorised changes

**Classification: `PATCH`** (2a, 2b) + **`FINDING`** (2c: the authorised change is committed and is
correct hygiene, but the measurement shows it does not fix the symptom it was aimed at, and the real
cause is outside the three authorised files).

- baseline SHA: `89fa5ca81341dc4df7186ec325cb691431a2ef17`
- merge of `step6/integration` (`d8bf23b`) into `step6/raycast3d`: **`fa8c3ac`**, no conflicts
- hardware: NVIDIA GeForce RTX 3090 (`gpu2104`, `gpu2110`), driver 575.57.08; CPU `node2337`
  (`--partition=batch --qos=normal`, 1 torch thread, `OMP_NUM_THREADS=MKL_NUM_THREADS=1`)
- venv `$SCR/venvs/se-step6`
- SLURM: **6345115** (2c first pass, gpu2110) · **6346224** (2a, gpu2104) · **6346414**
  (2b + 2c + 2a re-sweep, gpu2104) · **6346429** (2b CPU identity, node2337) · **6346705**
  (final verification on the merged tree, gpu) · **6346706** (CPU certification sweep, node2339)

**Final verification on the merged tree (job 6346705).** All **21** live raycast3d configs replayed
on cuda at batch 64, baseline against merged branch, in one job on one card:
**every frame pixel-identical**, 20 of 21 matching their frozen Step-4 digest on BOTH legs and
`AntmazeTeleportExplore` failing on both (pre-existing — see 2c). `tests/unit/test_dmcontrol_tendon.py`
3 passed; `tests/gpu/test_fast_paths.py` 136 passed, 24 skipped. The BallInCup sweep on the merged
tree, same card: **3,242.9 → 9,906.7 fps** at B=4096 and 3,052.3 → 8,220.9 at B=256.

**A harness defect found and fixed before any of these numbers were trusted.** The Step 6C scripts
did `sys.path.insert(0, ROOT)`, which put this branch ahead of `$PYTHONPATH` — so a before/after job
imported the BRANCH for both legs and the first 2a sweep reported a speed-up of the branch against
itself. They append now, and every note records the `stable_envs.__file__` it actually loaded; every
number below cites a leg whose recorded path is `$SCR/se6/baseline` or `$SCR/se6/raycast3d`.


> **Process note, recorded because it cost real work.** The 2a/2b/2c sections below were written
> once, then silently destroyed: I was pulling artifacts back with
> `rsync -az oscar:.../raycast3d/ examples/results/step6/raycast3d/`, and that directory also holds
> this file, which only ever changes locally. The pull overwrote it with the cluster's older copy and
> the next commit recorded the overwritten version. Pulls now exclude `findings.md`. Anything a
> report says that is not also in a committed artifact or a commit message is one careless `rsync`
> from being gone.

## RAYCAST-2a — `DMCBallInCupCatch` GPU: the tendon is built for the batch, not per environment

`stable_envs/envs/dmcontrol.py`, tendon FK only. Commit `af41d0f` (docstring corrected in `08fb05b`).

**What changed.** `_chain_capsules_batch` takes `(M, K+1, 3)` polylines — every line the
single-polyline version with one more leading axis. `_tendon_polyline` splits the part that must run
per environment (it reads `MjData`, which holds one state at a time) from the part that must not.
`_fk` collects polylines in the loop and builds the chain once per point count afterwards;
`ball_in_cup`'s tendon has exactly three possible counts, so at most three calls replace B.
`catenary`'s 29-point interior loop becomes one numpy expression, with `_cosh_sinh` still scalar and
still `math.exp` (the 0-ULP validation against libmujoco's `mjv_catenary` depends on the same libm).
The per-row pose gathers use `np.take(..., out=)`. The taut branch copies its two wrap points, which
the old code did not need to and the new one does — `np.asarray(d.wrap_xpos, np.float64)` hands back
the MjData buffer itself, and the caller now keeps polylines past the next `mj_forward`.

**Bit-identity** (`tests/unit/test_dmcontrol_tendon.py`, **3 passed**): the pre-change expressions are
written out in the test and the new ones held to them with `==` — all three branches of
`mjv_catenary`, float64 and after the float32 cast the renderer applies, plus `_fk` end to end on the
64 frozen `DMCBallInCupCatch` states.

**Replay** (job 6346224): all **10** `DMControlSuite` configs reproduce their frozen Step-4 digests
exactly at batch 64 on cuda, frames pixel-identical to the baseline's. Nine have no tendon and are
the control.

**The mandatory GPU sweep**, before (`$SCR/se6/baseline`) and after, one job, one RTX 3090
(job 6346414, gpu2104). `fk` and `device util` come from the phase pass, timed separately and with
synchronisation, so they describe the split rather than the wall clock exactly.

| B | before fps | after fps | x | before `_fk` | after `_fk` | before util | after util |
|---|---|---|---|---|---|---|---|
| 1 | 158.7 | 159.3 | 1.00 | 0.61 ms (9.7 %) | 0.57 ms (9.0 %) | 23.6 % | 23.6 % |
| 16 | 1,314.9 | 1,727.1 | 1.31 | 4.77 ms (39.2 %) | 1.99 ms (21.5 %) | 14.1 % | 18.4 % |
| 64 | 2,432.1 | 4,676.0 | 1.92 | 18.90 ms (71.8 %) | 6.23 ms (45.5 %) | 8.7 % | 16.6 % |
| 256 | 3,078.5 | 8,109.4 | 2.63 | 86.67 ms | 22.86 ms (72.4 %) | 6.2 % | 16.1 % |
| 1024 | 2,838.7 | 9,627.2 | 3.39 | 344.37 ms (95.5 %) | 89.84 ms (84.5 %) | 4.7 % | 16.0 % |
| **4096** | **3,271.8** | **9,816.3** | **3.00** | **1,217.24 ms (97.2 %)** | **385.96 ms (92.5 %)** | **5.3 %** | **15.9 %** |

The mechanism moved, not just the number: `_fk` per environment goes **297 µs → 94 µs** and device
utilisation at B=4096 **5.3 % → 15.9 %** on unchanged device time (66.83 → 66.16 ms), which is what a
host-bound render being unblocked looks like.

**Tier, `DMCBallInCupCatch` GPU (class S: `GPU_FAST := R >= 4 AND cores >= 1`)**, against Step 3's
frozen denominators (official CPU renderer 262.2 fps; one-core JPEG-q90 decode 5,965.9 fps):

| | before | after | after, merged tree (job 6346814) |
|---|---|---|---|
| best fps (B=4096) | 3,271.8 | **9,816.3** | **9,956.0** |
| `R` | 12.5 | 37.4 | **38.0** |
| `cores` | 0.55 | 1.65 | **1.67** |
| `BATCH_SCALES` (>= 20x) | 20.6x | 61.6x | **62.7x** |
| `NO_LARGE_BATCH_REGRESSION` | pass | pass | **pass** (9,956.0 >= 7,309.6) |
| **tier** | `GPU_PARITY` + `BELOW_G4` | **`GPU_FAST` + `BATCH_SCALES`** | same |

The whole family gains, because `np.take` costs nothing and every cell paid for the fancy-index
allocations: `DMCFingerSpin` goes 39,152 → **43,759** fps at B=4096 (`_fk` 46.2 → 37.3 ms) with no
tendon at all.

**Pixels changed: no.** Semantic follow-up: **none** — digests identical.

**Headroom left, measured.** `_fk` is still 92.5 % of the render at B=4096; ticket 1's counterfactual
put the renderer's ceiling with `_fk` removed at ~73,000 fps. What remains per environment is
`mj_forward`, the scalar Newton solve, the polyline assembly and four pageable host-to-device
uploads. Not pursued: the row is at tier.


## RAYCAST-2b — `R3.composite` shades the covered rays, not the window

`stable_envs/render/raycast3d.py`, `composite`/`shade` only. Commit `53101dd`.

Cover-compaction under the same `COVER_COMPACTION` name and the same argument
`DMControlSuite._render_group` already uses — restricting the SET cannot change a VALUE, because
every ray's colour is a function of that ray alone and outside the union `cov` is False for every
row — with the expression it replaces kept verbatim behind the switch. Plus a memoised
`_expand_tile` grid: `(y0*ss + a)*W + (x0*ss + b)` is a per-environment scalar plus a grid that
depends only on `(tile, ss, width)`, and it was being rebuilt per call as two `(B,T,T)` int64 expands.

**Identity, asserted on the envs.** A `composite-cover-compaction` case in
`tests/gpu/test_fast_paths.py`, the file that exists for exactly this claim — **136 passed, 24
skipped** on the merged tree — and `test_every_declared_fast_path_is_actually_reachable` now also
asserts the switch still exists, so a rename cannot leave the case comparing the fast path with
itself.

**Replay.** The frozen oracle is CUDA-only, so both devices are covered:

| check | result |
|---|---|
| cuda, 11 OGBench + 10 DMControlSuite, baseline vs branch, one job one card | **pixel-identical**, all digests unchanged |
| cpu, 11 OGBench, baseline vs branch | **pixel-identical** |
| cpu, `COVER_COMPACTION` on vs off, 11 OGBench | **pixel-identical** |
| cpu, out-of-corpus `mujoco_locomotion` (4 envs), seeded draw | **pixel-identical** |
| cpu, out-of-corpus `maze2d` `PointMaze*` (4 envs) | **pixel-identical** — see below |
| `conformance.check_backend_agreement` on these configs | `skipped: env renders without a Scene2D` — the suite has no applicable backend-agreement check for the composite path, which is why the on/off diff stands in for it |

**The 1-thread CPU certification sweep**, before (`$SCR/se6/baseline`) and after, ONE job, ONE batch
node (job **6346706**, `node2339`, `--partition=batch --qos=normal`, `OMP_NUM_THREADS=1`,
`torch.set_num_threads(1)`), merged tree. fps:

| config | B=1 | B=16 | B=64 | B=256 | best x |
|---|---|---|---|---|---|
| PointmazeLargeNavigate **before** | 304.5 | 825.4 | 856.8 | 761.3 | |
| PointmazeLargeNavigate **after** | 300.3 | 1,418.4 | **1,501.1** | 1,256.5 | **1.75x** |
| PointmazeMediumNavigate before | 270.5 | 928.2 | 954.1 | 785.2 | |
| PointmazeMediumNavigate after | 290.7 | 1,053.1 | 1,078.6 | 1,005.5 | 1.28x |
| AntmazeMediumExplore before | 36.3 | 93.4 | 90.0 | 81.2 | |
| AntmazeMediumExplore after | 37.8 | 98.9 | 92.5 | 81.0 | 1.03x |
| HumanoidmazeMediumNavigate before | 28.0 | 71.3 | 69.7 | 58.2 | |
| HumanoidmazeMediumNavigate after | 28.4 | 72.9 | 69.6 | 62.1 | 1.07x |

**Tier, `PointmazeLargeNavigate` CPU (`CPU_FAST := S_cpu >= 5 AND cores_cpu >= 1`).** Two readings,
because they differ and both are honest:

| | before | after |
|---|---|---|
| measured fps @B=64, `node2339` | 856.8 | **1,501.1** |
| `cores_cpu` = fps / 2,390.5 (Step 3's frozen one-core JPEG-q90 decode rate) | 0.358 | **0.628** |
| Step 3's own certified row (`node2349`, best_batch 64) scaled by the measured 1.75x | 994.3 | ~1,740 → `cores` ~0.73 |
| `S_cpu` (bar 5) | 48.0 ✓ | ~84 ✓ |
| tier | `CPU_FASTER` + `BELOW_G4` | **`CPU_FASTER` + `BELOW_G4`** |

Quote the direct number, **`cores` 0.63**: it divides a measurement by a constant. The scaled one is
offered only because the decode rate was frozen on `node2349` while the sweep ran on `node2339` — the
ratio transfers between hosts, the absolute fps does not. Either way **`CPU_FAST` is NOT reached**:
the bar needed 2.40x and the measurement bought 1.75x. That is the stop condition's second branch,
and the residual is measured, not guessed:

| `PointmazeLargeNavigate` @B=64, node2339 | before | after |
|---|---|---|
| render | 74.70 ms | **42.64 ms** |
| `composite` SELF (plate write, index grid, gather/scatter, box average, round/clamp/cast) | 30.95 | **25.31 ms — 59 %** |
| `cast_nearest` (of which `intersect` 9.79) | 12.50 | **13.44 ms — 32 %** |
| `shade` | 21.20 | **4.14 ms — 10 %** |

So 2b did what it claimed and nothing more: **`shade` fell 5.1x and is now the smallest term.** What
is left is `composite`'s own machinery at `ss=2` (59 %) and the cast that decides coverage and so
cannot be compacted (32 %). Both are RAYCAST-3 scope.

**The split across cells is the mechanism, not noise.** Ticket 1 measured `intersect` at 87 % of an
`AntmazeMediumExplore` CPU render (K=13 capsules) and 90 % of `HumanoidmazeMediumNavigate` (K=19)
against 14 % for `PointmazeLargeNavigate` (K=1, one sphere). The row this workstream owns gains
1.75x; the cast-bound rows gain nothing and need the fused CPU cast instead.

**`maze2d`'s `PointMaze*` are now verified** (job **6347137**, `node1829`); they were UNVERIFIED when
2b was committed. `$SCR/venvs/se6-raycast3d-maze2d` is a fresh venv — `python -m venv` from
`se-step6`'s interpreter, `gymnasium-robotics` + `pettingzoo` installed `--no-deps`, and a
`se-step6-inherit.pth` pointing at `se-step6`'s site-packages so every other pin is inherited
read-only. The same job checked that `se-step6` does **not** see `gymnasium_robotics`.

| check, CPU, 64 seeded states | `PointMazeLarge` | `Medium` | `Open` | `UMaze` |
|---|---|---|---|---|
| `COVER_COMPACTION` on vs off | 0 | 0 | 0 | 0 |
| branch vs pristine baseline | 0 | 0 | 0 | 0 |

Two attempts failed first, both worth recording: `python -m venv --system-site-packages` run from a
venv's interpreter inherits the BASE interpreter's site-packages, not the parent venv's (job 6347001
produced a tree with `gymnasium_robotics` and no `numpy`); and `gymnasium_robotics` imports
`pettingzoo` at module scope, which `--no-deps` had skipped (job 6347125). The second exposed a
defect in my own comparator: it printed `ALL_IDENTICAL=True` over two EMPTY dump directories, because
`all([])` is `True`. `compare_dumps` now returns `all_identical: null` with an explicit error and
prints how many configurations it compared. **A verification that goes green having verified nothing
is worse than one that fails**, and this one had already printed a green line.

**The `_expand_tile` memo serves the DM Control path too** — `dmcontrol.py:1898` (`_mover_idx`) and
`raycast3d.composite` both call it — which is why the **21**-config cuda replay, not just the 11
OGBench ones, is the evidence that it moved no pixels. It is integer arithmetic, so the memoised form
is the expression it replaces, and `tests/gpu/test_fast_paths.py` covers the DMC window path in its
own `mover-window-vs-full-frame` case.

**Pixels changed: no**, on either device. Semantic follow-up: **none**.


## RAYCAST-2c and 2c' — the bake is history-dependent and reproducible, not random

`stable_envs/envs/ogbench.py`. Commits `bb4ec44` (bake pose) and `17f28c0` (LRU eviction).

### What changed

**`bb4ec44`**: `_bake` rendered the shared `MjData` without ever setting `qpos`, so the plate was a
function of whichever environment `_fk` had rendered last. It now resets to `qpos0`, exactly as
`DMControlSuite._bake` always has. Real impurity, removed; **pixel-identical on all 21 live configs**.
It does **not** fix `check_style_is_state`, and the measurement said so immediately: with the reset in
place the two failing cells failed identically, byte for byte.

**`17f28c0`**: `_scene` did not evict, it did `_SCENE_CACHE.clear()` on the whole dict at 256 entries,
so crossing the cap threw away the key the caller was actively rendering. Now an `OrderedDict` with
`move_to_end` on a hit and `popitem(last=False)` on overflow.

### The correction: it is not a coin flip

Ticket 1 called the bake "nondeterministic". That was wrong, and the digest sequences say so.
`bake_probe.py` records the plate digest of eight consecutive bakes of one key. The sequences are
**byte-identical between two processes running two different code trees** (`$SCR/se6/baseline` and
this branch), and the teleport key's first bake is the same digest on **two different nodes**
(`5681d6a91dc0` on gpu2107 in ticket 1 and on gpu2104 here):

| config | sequence over 8 bakes | digests |
|---|---|---|
| `AntmazeGiantNavigate` | **A B A B A B A B** (strict alternation) | A `27359fa34b1e`, B `46ea1cfe42eb` |
| `AntmazeTeleportExplore` | **A A A B A A A A** | A `5681d6a91dc0`, B `fb5c8bde6a2b` |
| `HumanoidmazeMediumNavigate` | **A A A A A A A A** | A `f064174642ac` |

So the bake output is a **deterministic function of the call history** — of how many
`update_scene`/`render` cycles that renderer has seen — not of chance. The suspected source is the
renderer's lifetime: `_bake` (`ogbench.py:471-479`) keeps one `mujoco.Renderer` per texture state
forever and drives it `update_scene -> render -> update_scene`, two scene updates and one render per
bake, so successive bakes do not start from the same renderer state. `AntmazeGiantNavigate`'s strict
alternation is what a two-state resource looks like. **Probed separately; see the lifetime probe
below.**

### The host-determinism probe already owns this question — and is not wired to answer it

`conformance.check_reference_renderer_is_stable` (`conformance.py:121`) is, verbatim, *"Is THIS
MACHINE's offscreen rasterizer deterministic at all? Ask before blaming the code."* It renders the
same states twice through one reference env and diffs. `scratch/step4/certify.py:125` states the
routing: it *"is a property of the node, which Step 2 section 4.1 lists as diagnostic-only"*, and
`judges()` declares no bound for it. `check_style_is_state`, by contrast, is a hard `max_abs_diff == 0`
judge.

Run 12 times per config on `gpu2106` (job 6347000):

| config | probe fires | Step 4's recorded `reference_stable` |
|---|---|---|
| `PointmazeTeleportNavigate` | **12/12, max 1** | **1** |
| `AntmazeTeleportExplore` | 0/12 | 0 |
| `HumanoidmazeMediumNavigate` | 0/12 | 0 |
| `AntmazeGiantNavigate` | 0/12 | 0 |
| `PointmazeLargeNavigate` | 0/12 | 0 |

Two things follow. First, the probe is **not** under-powered by sampling — it is deterministic and
**scene-specific**, and it reproduces Step 4's own record exactly, including which single config it
fires on. Second, and this is the routing problem: the same host property reaches the certifier
through two call sites and is judged two different ways. Through `check_reference_renderer_is_stable`
(the OFFICIAL renderer, two renders) it is diagnostic-only. Through `check_style_is_state` (OUR plate
bake, two renders separated by a cache eviction) it is a fidelity FAIL. `AntmazeTeleportExplore` and
`HumanoidmazeMediumNavigate` fail Gate A on a property of `mujoco.Renderer` on this host that the
suite elsewhere declines to blame the code for.

### Proposal (routing only — a contract decision for Sami, not implemented here)

When the host probe demonstrates that the reference rasterizer is not bit-reproducible **for that
configuration's scene at the plate-bake call site**, `check_style_is_state`'s `1/255`-on-N-px outcome
should be reported as an environment property — `blocked_environment` / host-nondeterministic, the
status `scratch/step4/certify.py` already has for a bound judged under the wrong conditions — with
the measured plate spread recorded beside it, rather than as a fidelity FAIL. Concretely:

1. the probe needs a **bake-path arm**. As written it exercises `cls.reference_render`, and that arm
   fires on `PointmazeTeleportNavigate` and not on the two cells that actually fail. A second arm
   that bakes the same key twice and diffs the plates fires on exactly the right ones
   (`bake_probe.py` is that arm, measured: 2 distinct plates on the ant mazes, 1 on humanoid-medium);
2. the routing should be **per configuration**, keyed on that arm's result for that scene, not
   global — nine OGBench cells carry the defect and would still be judged normally on everything else;
3. the record should carry the measured spread (**1/255 over 4-11 pixels**), so "blocked" never means
   "unmeasured";
4. **the bar itself must not move.** This is a proposal to attribute an outcome to the host, not to
   relax `== 0`. A configuration whose bake IS reproducible and still differs is still a FAIL.

I have not implemented any of it: it changes what the certifier concludes, which is above this
workstream.

### The cache-pressure scenario, before and after (2c' item b)

`check_style_is_state` returns 0 in isolation — the second render finds the 64 plates the first one
baked still cached. It disagreed in Step 4 only where the 64 inserts crossed the cap in a long
session. `cache_pressure_probe.py` reproduces that instead of waiting for it: fill the cache to 8
below the cap with keys nobody asks for, then run the frozen check verbatim (job 6347000, gpu2106):

| config | baseline (`dict`, wholesale `clear()`) | branch (`OrderedDict`, LRU) |
|---|---|---|
| `AntmazeTeleportExplore` | 0 | 0 |
| `HumanoidmazeMediumNavigate` | **1** (7 px) | **0** |
| `AntmazeTeleportNavigate` | 0 | 0 |
| `HumanoidmazeLargeNavigate` | **1** (11 px) | **0** |
| `PointmazeLargeNavigate` | 0 | 0 |
| synthetic filler surviving the first render | **0** | **192** |

The filler count is the policy itself: the wholesale clear takes everything; LRU takes exactly the 64
oldest synthetic entries and keeps the 64 the check had just inserted. **Both configurations that
reproduced `observed = 1` now return 0**, and a passing sibling (`HumanoidmazeLargeNavigate`)
reproduced it too — more evidence that the Step-4 fail/pass split was about session order.

**Stated honestly: the nondeterminism still exists in a fresh bake.** A genuinely cold key still
lands on whichever plate its position in the renderer's history selects. What LRU removes is cache
pressure as the trigger — a key in use is never the one dropped — and it is a structural improvement
in its own right, since one appearance rendered a thousand times could previously be re-baked because
an unrelated configuration filled the cache.

Verified: 11 OGBench configs replayed on cuda, baseline vs branch, **pixel-identical, all digests
unchanged**; `tests/unit/test_ogbench_scene_cache.py` 3 policy tests (a hot key survives 300 distinct
inserts past a cap of 256; an overflow costs one entry, not the cache; the common case never evicts),
driven with a stub because proving a one-line eviction rule should not cost 300 MuJoCo bakes.

### The two decisive probes (2c' item 3) — the renderer's lifetime IS the mechanism

Job **6347171**, gpu2105, `bake_lifetime_probe.py`. Report only; nothing here is committed into the
renderer.

**(i) Which plate did Step 4 freeze?** Bake the key repeatedly, render the frozen states against each
bake's plate, digest the frames:

| config | plate sequence | frozen `digest_ours` reproduced? |
|---|---|---|
| `AntmazeGiantNavigate` | A B A B A B A B | **yes, by every ODD bake** (plate `27359fa34b1e` → `9da286404b65dc66`, the frozen value) |
| `HumanoidmazeMediumNavigate` | A A A A A A A A | yes, by all of them |
| `AntmazeTeleportExplore` | A A B A A A A A | **NO — by neither plate.** A → `483968227d628c59`, B → `bf67da3c6fe37298`, frozen is `06a2e039e1fdf3ad` |

So the hypothesis that Step 4 simply froze "the other plate" is **true for Giant and false for
Teleport**. And the reason is visible in probe (ii): driven as shipped, `ant/teleport` produces
**four distinct plates in eight bakes**, not two. Its Step-4 plate is a third image this probe did not
reach. That closes the door on reconciling that digest by re-deriving it and leaves re-freezing as the
only route.

**(ii) Is the renderer's lifetime the mechanism?** Same shot sequence, four ways. `as_shipped` is
`update_scene → render → update_scene` on the Renderer kept forever per texture state
(`ogbench.py:471-479`); `warmup_render` adds a discarded render; `no_trailing_update` drops the
second `update_scene`; `fresh_renderer_per_bake` builds a new `mujoco.Renderer` each time.

| scene | as_shipped | warmup_render | no_trailing_update | **fresh_renderer_per_bake** |
|---|---|---|---|---|
| ant/teleport | `ABCAAADA` (4) | `AAAAAAAB` (2) | `AAAAAAAA` (1) | **`AAAAAA` (1)** |
| ant/giant | `AABABABA` (2) | `AAAAAAAA` (1) | `ABABABAB` (2) | **`AAAAAA` (1)** |
| humanoid/medium | `AAAAAAAA` (1) | `AAAAAAAA` (1) | `AAAAAAAA` (1) | **`AAAAAA` (1)** |

**Confirmed: the retained renderer is the carrier.** And only one variant is constant on every probed
scene. The two cheap ones each fix one scene and break or miss the other — and `no_trailing_update`
does worse than miss: on `ant/giant` its plate is `46ea1cfe42eb`, **the other image**, so it would
move pixels on a configuration that is currently correct.

**The plate a fresh Renderer produces is the one the cache already holds** — `digests[0]` equals the
current first bake on all three scenes. That is the property that makes it a candidate rather than a
rewrite: in the common case (a cold cache, first bake) it changes nothing.

### Proposal for the bake (NOT implemented — needs authorisation, and it has a real cost)

A fresh `mujoco.Renderer` per bake makes the plate a pure function of (scene, style) and preserves
the plate every config currently gets. Its cost is the reason I have not written it:

* **one EGL context per bake, never released.** The module's own note above `OGBenchMaze` records
  that releasing a context corrupts rendering in every other live renderer in the process — that is
  how a batch of maze cells once came back 69–90/255 apart — so they are kept. The measured budget is
  ~96 live contexts, and `check_style_is_state` alone bakes 64 appearances. A certification session
  would exhaust it.
* So the shape of a usable fix is **not** "always build a fresh one": it is a renderer whose state is
  reset between bakes without a new context, or a small pool sized against the context budget with
  the reset property demonstrated. Neither is measured, and the difference between them is exactly
  the kind of thing this probe exists to settle before somebody writes it.
* `warmup_render` is the cheap partial: no extra context, constant on `ant/giant`, plate-preserving
  everywhere, and still flips on `ant/teleport` at the eighth bake. It is not a fix; it is evidence
  that the state being carried is small.

My recommendation: treat this as a separate ticket with the EGL-context budget as its first
constraint, and keep the LRU change (which is committed and needs none of this) as the containment.

### The oracle hole (2c' item c)

**`AntmazeTeleportExplore` does not reproduce its frozen Step-4 digest at the BASELINE.** Measured on
three separate jobs, on two nodes, on both legs of every before/after: `483968227d628c59` against a
frozen `06a2e039e1fdf3ad`. The other 20 live configs reproduce exactly, on both legs.

Root cause: the bake's history dependence above. Step 4 ran the full check suite before taking the
oracle, so its plate was a NON-FIRST bake; a replay that renders immediately after construction takes
the first one. Probe (i) above tested the obvious repair and **refuted it**: neither of the two plates
this config produces in a short session reproduces the frozen digest, and probe (ii) shows the scene
yields **four** distinct plates in eight bakes as shipped. Step 4's plate is a third image. For
contrast, `AntmazeGiantNavigate` alternates between exactly two and its frozen digest IS one of them,
which is why that config replays green.

**Recommended for the Step-8 re-freeze list**, and only after the bake is made deterministic —
re-freezing a digest that is still a function of call history just moves the problem. Nothing under
`examples/results/step4/` has been touched.


## What is still open after RAYCAST-2 / 2c'

1. **`PointmazeLargeNavigate` does not reach `CPU_FAST`** — 1.75x bought against 2.40x needed,
   `cores` 0.63. The residual is measured (`composite` self 59 %, cast 32 %, `shade` 10 %) and is
   RAYCAST-3's scope.
2. **The bake is still history-dependent.** LRU removes cache pressure as the trigger; it does not
   make a cold bake reproducible. The remedies and their costs are in the 2c section; all of them are
   above this workstream.
3. **`AntmazeTeleportExplore`'s frozen digest is unreproducible at the baseline** — recommended for
   the Step-8 re-freeze list, after the bake is fixed.
4. **The routing proposal is a proposal.** Nothing in the certifier or the gate has been changed.
5. **Two pre-existing `tests/unit` failures** on the merged tree, neither mine and neither in my
   authorised files: `test_every_argparse_script_can_print_its_own_help[robocasa_visual/render_mjwarp.py]`
   and `test_every_cited_scratch_path_resolves`. Verified pre-existing by stashing my working tree.
6. **`gymnasium_robotics` is not in `se-step6`**, so the `maze2d` envs are only reachable through
   `$SCR/venvs/se6-raycast3d-maze2d`. Any future CPU verification of `composite` has to use it or
   repeat the 2b gap.


# RAYCAST-3 — a fused CPU cast: design-first, measured, **not implemented**

No renderer edits. The kernel is `scratch/step6/raycast3d/cast_kernel.cpp`, built into a scratch
extension by `cast_microbench.py`; nothing imports it from `stable_envs`. SLURM **6347071**,
**6347142**, **6347404** (`node1828`/`node1829`, Intel Xeon Platinum 8268 @ 2.9 GHz, `batch`
partition, **1 torch thread**, gcc 11.5.0).

## The question, and the bar

Ticket 1 measured the CPU blocker: `raycast3d.intersect` is 76–95 % of every multi-mover CPU render
at **145–240 ns per ray-primitive test**, because `cast_nearest_torch` pays 60–100 flops as ~200
elementwise kernels over materialised `(B,N)` temporaries. `step6/cpu-3d`'s design reproduced the
same constant (220 ns/test) on a different scene shape and computed the bar: `cores_cpu >= 1` at
84 px on this host is one-core JPEG-q90 decode at 6,629 fps — **151 µs per frame, 21.4 ns per pixel**.

## What a compiled kernel costs (measured)

`cast_kernel.cpp` is a scalar C++ transcription of `_sphere` and `_capsule` plus the nearest-hit
reduction — same expressions, same association order, `-O3 -ffp-contract=off -fno-fast-math`. Over
7,056 rays, one thread, median of 9:

| cell | K | torch loop | torch, chunked 4096 | **scalar C++** | speed-up |
|---|---|---|---|---|---|
| DMCReacherHard | 4 | 215.9 ns | 316.7 | **18.0 ns** | **12.0x** |
| DMCFingerSpin | 7 | 228.6 | 337.8 | **18.4** | **12.4x** |
| DMCBallInCupCatch | 36 | 276.3 | 406.4 | **21.6** | **12.8x** |
| AntmazeMediumExplore | 13 | 276.6 | 410.4 | **21.9** | **12.6x** |
| PointmazeLargeNavigate | 1 | 64.6 | 100.5 | **8.5** | 7.6x |

**Chunking the rays in pure torch makes it WORSE, not better** (1.5–4x slower at 1024, 1.5x at 4096):
the shipped call is already one dispatch per op over the whole bundle, and cutting it up multiplies
the 2 µs dispatch floor by the chunk count. So there is no compiler-free win here — the fused CPU
cast has to be compiled or it does not happen.

## The gate is bit-exactness, and it is NOT yet passed

The frozen Step-4 digests were produced by `cast_nearest_torch`, so a CPU kernel is admissible only
if it agrees to the last bit — the same standard the Triton kernel had to meet. Measured on the same
inputs, uint32 bit patterns:

| cell | `t` exact | winner index exact | normal: components differing | rays | max ulp | max abs |
|---|---|---|---|---|---|---|
| DMCReacherHard | yes | yes | 13 / 21,168 | 5 / 7,056 | 3 | 1.19e-07 |
| DMCFingerSpin | **no** (1 ray) | yes | 18 / 21,168 | 6 | **17** | 3.58e-07 |
| DMCBallInCupCatch | yes | yes | 74 / 21,168 | 31 | 16 | 1.79e-07 |
| AntmazeMediumExplore | yes | yes | 25 / 21,168 | 10 | 8 | 1.19e-07 |
| **PointmazeLargeNavigate** (sphere only) | **yes** | **yes** | **0** | **0** | **0** | **0** |

Three facts narrow it a long way:

* **the sphere transcription is exact**, on both toolchains, every ray;
* the capsule NORMAL is exact on Apple clang/arm64 and not on gcc 11.5/x86-64;
* and it is **identical with and without `-march=native`**, so it is not FMA or AVX codegen.

A 3–17 ulp difference on 0.06–0.35 % of components is not a uniform rounding wobble; it is a handful
of rays where the capsule normal is near-degenerate (`nl` almost zero before normalisation — a ray
close to the capsule's own axis), which is exactly where a one-ulp difference upstream is amplified
by the division. **That is the go/no-go item**, and it is a transcription question, not a packaging
one: 17 ulp will not be fixed by pinning a compiler.

## Does the speed reach `CPU_FAST`? Not by itself — the TEST COUNT is the other half

At the measured C++ cost, against each cell's own decode budget and its MEASURED ray-primitive test
count per frame (ticket 1's phase notes):

| cell | tests/frame | budget | cast alone @C++ | cast-only fps ceiling | test-count reduction still needed |
|---|---|---|---|---|---|
| DMCReacherHard | 80,884 | 142.6 µs | 1,453.6 µs | 688 | **10.2x** |
| DMCFingerSpin | 93,401 | 140.3 | 1,716.0 | 583 | **12.2x** |
| DMCBallInCupCatch | 279,972 | 139.1 | 6,033.6 | 166 | **43.4x** |
| AntmazeMediumExplore | 53,248 | 418.3 | 1,166.9 | 857 | **2.8x** |
| PointmazeLargeNavigate | 4,096 | 418.3 | 34.8 | 28,705 | **0.1x** (already fits) |

**So a 12x kernel is necessary and not sufficient for the DM Control parity cells.** `DMCReacherHard`
issues 80,884 ray-primitive tests for a 7,056-pixel frame — **11.5 tests per pixel for a
four-primitive scene** — because four passes each run over a window that is 64–95 % of the frame
(measured: mover 4,489 rays/env, reflect 6,720, the two shadow passes 1,956 and 7,056). The cost is
not the primitives, it is that nothing bounds the rays per primitive.

**Verdict per cell, with a fused kernel and nothing else changed:**

| cell | `CPU_FAST` reachable with the compiled cast alone? |
|---|---|
| PointmazeLargeNavigate / MediumNavigate | the cast already fits the budget; **the blocker is `composite`, not the cast** |
| AntmazeMediumExplore | no — 2.8x of test-count reduction short, plausible with a per-primitive bound |
| DMCReacherHard, DMCFingerSpin | no — 10–12x short; needs both the kernel and a per-primitive screen bound |
| DMCCartpole*, DMCPendulum, DMCAcrobot | not measured here (K=2–3, fewer tests); the arithmetic says they are the closest of the DMC cells and should be measured before any implementation is scoped |
| DMCBallInCupCatch | no — 43x short at K=36; out of reach on one core |

## What it would need, and what to build first

1. **A per-primitive screen bound before the cast.** `screen_span` already projects each primitive's
   bounding sphere; the cast uses only the union. Casting each primitive against the rays inside its
   OWN projected box is the same argument `raster2d`'s tile culling makes and would take Reacher's
   11.5 tests/pixel towards 1–2. This is the larger lever and it needs no compiler.
2. **Then** the compiled kernel, for the residual.
3. **Packaging**, if it goes ahead: a `torch.utils.cpp_extension` build with `-ffp-contract=off
   -fno-fast-math` — note `se-step6` has **no ninja**, so the build path needs the plain-`g++`
   fallback this benchmark already carries, and **no numba** (`ModuleNotFoundError`), so numba was
   not measurable here and would need its own venv to evaluate.

## Scoping note for Pointmaze (review item 6)

For the row this workstream owns the cast is **not** the next target. After RAYCAST-2b,
`PointmazeLargeNavigate` at B=64 is 42.64 ms of which `composite` SELF is **25.31 ms (59 %)** and the
cast 13.44 ms (32 %). That 25 ms is `composite`'s own machinery at `ss=2`: the full-frame stride-0
plate write, the float promotion of the plate, and the `gather`/`scatter_` through `(B, tile*tile, 3)`
int64 indices — 6.3 MB of index traffic for 2.4 MB of pixels at B=256. A micro-benchmark of THAT —
scatter/gather versus a contiguous tile blit, and the box average at `ss=2` — belongs beside the cast
benchmark in any RAYCAST-3 implementation ticket, and for Pointmaze it is the larger half.


---

## Merge record

`step6/integration` was merged into `step6/raycast3d` twice, as the coordinator released it:

| merge commit | integration at | what it brought | conflicts |
|---|---|---|---|
| `fa8c3ac` | `d8bf23b` (texture-fidelity) | declaration fields on the trackcom visual entries, `core/env.py`, `testing/conformance.py`, new `testing/provenance.py` | none |
| `f312c92` | `ca0c577` (reacher-correctness) | `VISIBILITY_KNOB` / `_hideable` / `visibility.*` style leaves, `_apply_variation` alpha zeroing, `sc.statics` recorded at bake and read by `_static_t`, `check_task_object_visibility`, `fidelity_step4.py`, two GPU test files | none |

**Final verification on the TWICE-merged tree** (job **6346814**, gpu2104, RTX 3090, COMPLETED
00:02:46), which is the tree this report describes:

| check | result |
|---|---|
| all 21 live raycast3d configs, cuda batch 64, baseline vs merged branch | **pixel-identical** (`replay/merged_all21_pixel_diff.json`) |
| the same 21 against their frozen Step-4 digests | 20/21 on BOTH legs; `AntmazeTeleportExplore` fails on both (pre-existing, see 2c) |
| `tests/unit/test_dmcontrol_tendon.py` | **3 passed** |
| `tests/gpu/test_fast_paths.py` (composite compaction, mover window, fused cast, reachability) | **136 passed, 24 skipped** |
| `DMCBallInCupCatch` sweep, same card | **3,194.2 -> 9,956.0 fps** at B=4096 (**3.12x**); 3,084.1 -> 8,121.8 at B=256; 155.2 -> 158.7 at B=1 |

which puts the certified tier on the merged tree at `cores` **1.67**, `R` **38.0**, `BATCH_SCALES`
**62.7x**, `NO_LARGE_BATCH_REGRESSION` pass (9,956.0 >= 0.9 x 8,121.8) -- i.e. **`GPU_FAST` +
`BATCH_SCALES`**, unchanged from the pre-merge measurement.

Both merges touch `stable_envs/envs/dmcontrol.py`; this workstream's edits are confined to the tendon FK
(`_fk`, `_tendon_polyline`, `_chain_capsules`, `_chain_capsules_batch`, `catenary`) and git merged
them without a conflict. The movers/statics classification is not touched here — `_fk` reads
`sc.movers` and writes mover rows only, and `sc.statics` is recorded at bake, which 2a does not go
near. `tests/unit` on the merged tree: **527 passed, 7 skipped, 2 failed**, both failures
pre-existing and outside the authorised files (verified by stashing the working tree and re-running):
`test_every_argparse_script_can_print_its_own_help[robocasa_visual/render_mjwarp.py]` and
`test_every_cited_scratch_path_resolves` (9 unresolvable citations, all from
`examples/results/step6/dashboard.yaml` and two pre-existing research docs; none from this
workstream's files after the `$SCR` collapse).
