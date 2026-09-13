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
