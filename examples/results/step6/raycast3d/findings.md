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


# RAYCAST-4 — per-primitive bounds in the CPU cast

`stable_envs/render/raycast3d.py`, the torch cast only. Design first, then the implementation and its
evidence.

## Design

**The lever, restated from RAYCAST-3.** `DMCReacherHard` issues **80,884 ray-primitive tests for a
7,056-pixel frame — 11.5 tests per pixel for a FOUR-primitive scene.** The four ray passes each run
over a window that is 64–95 % of the frame, because the window is a rectangle around *all* the movers
and a reacher arm is long, thin and diagonal. A single capsule covers a small part of it. Nothing
between the window and the intersector bounds a primitive to the rays it can reach.

**The bound: ray/bounding-sphere distance, not a screen box.** A screen-space box is the obvious
answer and is the wrong one here. The cast is handed `o`, `d` and the primitive arrays — no camera,
no pixel indices — and for the reflection pass the origins are per-ray floor points rather than one
pinhole, so "which pixel is this ray" is not a question it can answer without threading a camera
through five call sites. The distance from a primitive's origin to a ray's line needs none of that:

    perpendicular distance^2 <= radius^2   <=>   |oc|^2 |d|^2 - (oc.d)^2 <= r^2 |d|^2 ,   oc = pos - o

multiplied through by `|d|^2` so that a degenerate `d = 0` reads `0 <= 0` and includes the ray
instead of producing a NaN that would silently drop it. A second term, `oc.d + r|d| >= 0`, drops rays
whose whole bound lies behind the origin — where `intersect`'s own `t > 0` would reject them anyway.
About **8 elementwise ops per primitive per ray, against the ~200 an intersector costs**.

**Per primitive, per row.** The radius comes from the `sizes` tensor the caller actually passed, not
from `Primitives.bound_radius`, so a per-environment `size_scale` (which `composite` applies before
calling) is already in it and cannot drift out of sync:

| kind | radius about the primitive's own origin |
|---|---|
| `SPHERE` | `size[0]` |
| `CAPSULE`, `CYLINDER` | `size[0] + size[1]` (conservative; the tight bound is `sqrt(r^2+hh^2)`) |
| `BOX` | `sqrt(sx^2+sy^2+sz^2)` |
| `PLANE`, anything unknown | **unbounded — always tested** |

`pos` is `(B,K,3)`, so the bound is per row for free: the mask is `(B,N)` and costs four broadcasts.

**Grouping without a Python loop over primitives or rows.** There is none. The mask is one vectorised
expression over the whole `(B,N)` bundle; the ray set for primitive `k` is
`nonzero(mask.any(0))` — **the union over the batch of the rays near that primitive in ANY row** —
and the intersector then runs once on `o[:, js]`, `d[:, js]`. The existing Python loop over `K` is
untouched (K is 1–36 and it is the shipped structure); what disappears is the *width* of each
iteration. Restricting the set row-wise instead would need a ragged gather per row, which is the
Python loop over rows the brief rules out, and is unnecessary: the union is already small because
the rows of a chunk hold the same scene at nearby states.

This is the same shape as `raster2d`'s tile binning — a conservative per-primitive bound, evaluated
once, used to skip work that provably cannot contribute — but keyed on ray geometry rather than on a
pixel tile, because that is the information this function has.

**Why it is the identity, not an approximation.** A ray whose line passes further from the primitive's
origin than the primitive's bounding radius cannot intersect it, so `intersect` returns `INF` there,
so `t < best` is False and `best`/`nrm`/`idx` are unchanged; in `any_blocking`, `hit` is False and
`blocked |= hit` is a no-op. The bound is inflated by `1.001x + 1e-6` before it is used to EXCLUDE
anything: the test is a handful of float32 operations on values of order 1–1700, relative error
~1e-6, so three orders of margin, and being over-inclusive costs only speed.

**The invariant it rests on, and why it is checked.** The intersectors assume unit rays — `_sphere`
solves `t^2 - 2bt + c = 0` with the leading `|d|^2` dropped — so with a non-unit `d` the `t` they
return is not the geometric ray parameter and a geometrically exact bound disagrees with them. Every
call site normalises, but the cast now *checks* (`|d|^2` within 1e-4 of 1) and routes a violating
bundle to the unbounded path, the way `_packed` routes an exotic layout to the torch loop. **The
differential fuzz found this rather than my reading it off the source**: feeding `d` scaled by 0–3x
moved 23 of 400 shadow-ray answers before the check existed.

**Scope: the CPU path only.** `_launch`'s fused Triton kernel keeps the whole primitive loop in
registers and has nothing to gain from skipping one; it is untouched, which is why the cuda digests
cannot move and the cuda replay is a control rather than a risk. `nonzero` synchronises on CUDA, so
the torch fallback on a CUDA device pays K syncs per call — that path exists for the tests that diff
the two backends, not for throughput, and the switch makes that explicit.

**Expected effect.** Per-primitive rather than per-window, so tests/pixel should fall from 11.5
towards the number of primitives whose bound actually covers a given pixel — 1–2 for the DM Control
cells, where the movers are small and disjoint, and less for `ball_in_cup`, whose 28 tendon capsules
lie along one thin chain that most rays miss entirely. It does **not** help `PointmazeLargeNavigate`
(K=1: there is nothing to prune against) and it cannot help the passes whose cost is not the cast.
Measured numbers below, not promised ones.


## Implementation and evidence

`PRIMITIVE_BOUNDS` (default on) in `cast_nearest_torch` / `any_blocking_torch`; off restores
`_cast_nearest_torch_unbounded` / `_any_blocking_torch_unbounded`, the current expressions kept
verbatim. Commits `96ef58a` (implementation), `fcbd2b5` (the conservativeness fix below), `f7077a9`.

### Two defects, both found by measurement

**1. The intersectors assume unit rays, and the bound did not.** `_sphere` solves `t^2 - 2bt + c = 0`
with the leading `|d|^2` dropped, so with a non-unit `d` the `t` it returns is not the geometric ray
parameter and a geometrically exact bound disagrees with it. The differential fuzz found this by
feeding `d` scaled by 0-3x: **23 of 400 shadow-ray answers moved.** The cast now CHECKS `|d|^2` within
1e-4 of 1 and routes a violating bundle to the unbounded path, the way `_packed` routes an exotic
layout to the torch loop. Every call site in the repo normalises; the point is that it is checked.

**2. Far-field cancellation — and the larger half of it is in the SHIPPED intersector.** The first
version moved **one pixel of 3,211,264, in one frame of 64, on exactly one of the twenty-one live
configs**: `HumanoidmazeGiantNavigate`, by 22/255. The cuda replay was identical and 176 fast-path
cases passed, so only the CPU `PRIMITIVE_BOUNDS` on-vs-off replay pinned it to this change. The giant
maze puts the camera ~100 m from a ~2 cm capsule, the longest lever arm in the corpus, which is why
that configuration and no other.

* *My test cancelled.* `|oc|^2|d|^2 - (oc.d)^2` is two terms of ~1e4 differing by ~`r^2` ~ 2.5e-3,
  and float32's resolution at 1e4 is 1.2e-3. `|oc x d|^2` is the same geometry with the result of
  order `r|d|` instead, so its absolute error is ~`eps|oc||d|`.
* *The intersector cancels too.* `_capsule` forms `c = olx^2 + oly^2 - r^2` in float32; at
  `|ol| = 100` m a 2 cm capsule's `r^2 = 4e-4` is **below** the 1.2e-3 resolution of the first two
  terms, so the radius is entirely lost and `disc > 0` decides hit-or-miss on rounding noise.
  Measured directly: the shipped cast reports a hit for a ray whose float64 perpendicular distance is
  **0.0277 m from a capsule whose own bounding radius is 0.0221 m — a hit outside the primitive's own
  bound.** A geometrically exact bound is therefore not conservative against this intersector,
  however carefully it is computed.

  So the intersector's own resolution, `|ol| sqrt(eps)` (0.0345 m at 100 m), is added to the radius in
  **quadrature**: invisible wherever `r >> |ol| sqrt(eps)` — every DM Control cell — and dominant only
  where the intersector has already stopped being able to tell. At 100 m the effective radius becomes
  0.14 m, well under a pixel on a giant maze, so the pruning survives.

| `|ol|` | float32 resolution at `|ol|^2` | `r^2` for r = 2 cm | verdict |
|---|---|---|---|
| 2 m | 4.8e-7 | 4.0e-4 | radius survives |
| 30 m | 1.1e-4 | 4.0e-4 | survives, 4x margin |
| **100 m** | **1.2e-3** | **4.0e-4** | **radius is LOST** |
| 300 m | 1.1e-2 | 4.0e-4 | lost by 27x |

**FOR THE RECORD, NOT FIXED HERE.** That second item is a property of the CPU cast *as shipped*, and
the same expressions are in the Triton kernel. It does not make any current frame wrong — the frozen
digests ARE what that arithmetic produces — but far-field hit/miss in these scenes is decided at the
edge of float32, so anything that perturbs it (a different kernel, a different association order, the
compiled cast RAYCAST-3 priced) can move those pixels. Worth a line in whatever ticket next touches
the intersectors.

### The test gap was the actual lesson

Both suites passed while a real configuration moved a pixel:

* the differential fuzz had **no far-field case** — it sampled primitives and rays within a couple of
  metres, which is the regime where everything cancels harmlessly. It now sweeps lever arms of 30,
  100 and 300 m with the rays aimed exactly at the bound, and **those cases fail against the first
  version**;
* the GPU fast-path case built every env at **84 px**, shortening the very lever arm under test. It
  now uses each env's own size, so a giant maze is tested at its certified 224 px.

### Identity, on the device the change targets

| check | result |
|---|---|
| cpu, `PRIMITIVE_BOUNDS` on vs off, all 21 live configs at their native sizes | **identical** (`replay/r4_cpu_bounds_identity.json`) |
| cpu, branch vs pristine baseline, all 21 | **identical** (`replay/r4_cpu_branch_vs_baseline.json`) |
| cuda, branch vs pristine baseline, all 21 | **identical** — the fused kernel is untouched (`replay/r4_cuda_pixel_diff.json`) |
| `tests/unit/test_raycast3d_primitive_bounds.py` | **24 passed** — random scenes, rays aimed at the bound radius, primitives behind the origin, zero radii and half-lengths, non-unit and zero directions, far field at 30/100/300 m, finite and infinite shadow limits, with and without `skip`, all diffed on bit patterns |
| `tests/gpu/test_fast_paths.py` | the `raycast3d-primitive-bounds` case over the whole raycast3d family |
| `tests/unit` overall | 554 passed, 7 skipped, 2 failed (both pre-existing, neither in the authorised files) |

### What it buys, on one thread of the node it was measured on

One job (`6348760`), one node (`node2337`), `OMP_NUM_THREADS=1`, eight rows, both legs, every row
carrying its own `stable_envs.__file__` (`profile_after_r4/`). **The `before` leg here is the
PRISTINE pre-Step-6 baseline**, so these ratios carry RAYCAST-2a and 2b as well as the bound — the
attribution to the bound alone is the table after this one.

| env | px | B | fps before | fps after | fps x | tests/px before | after | tests x |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DMCAcrobotSwingup | 84 | 1 | 42.8 | 49.3 | 1.15x | 2.80 | 0.22 | 12.8x |
| DMCAcrobotSwingup | 84 | 16 | 213.7 | 272.8 | 1.28x | 3.02 | 0.79 | 3.8x |
| DMCAcrobotSwingup | 84 | 64 | 215.7 | 264.2 | 1.22x | 3.23 | 1.12 | 2.9x |
| DMCAcrobotSwingup | 84 | 256 | 204.0 | 252.6 | 1.24x | 3.23 | 1.12 | 2.9x |
| DMCBallInCupCatch | 84 | 1 | 5.5 | 9.5 | 1.72x | 27.39 | 0.13 | 213.3x |
| DMCBallInCupCatch | 84 | 16 | 21.3 | 48.6 | 2.28x | 30.23 | 0.81 | 37.4x |
| DMCBallInCupCatch | 84 | 64 | 16.6 | 51.8 | 3.11x | 39.68 | 1.70 | 23.3x |
| DMCBallInCupCatch | 84 | 256 | 14.5 | 47.1 | 3.23x | 39.68 | 1.70 | 23.3x |
| DMCCartpoleSwingup | 84 | 1 | 64.3 | 62.0 | 0.96x | 1.93 | 0.23 | 8.4x |
| DMCCartpoleSwingup | 84 | 16 | 258.7 | 411.9 | 1.59x | 1.96 | 0.29 | 6.8x |
| DMCCartpoleSwingup | 84 | 64 | 282.2 | 428.9 | 1.52x | 1.96 | 0.29 | 6.7x |
| DMCCartpoleSwingup | 84 | 256 | 224.4 | 404.9 | 1.80x | 1.96 | 0.29 | 6.7x |
| DMCFingerSpin | 84 | 1 | 21.1 | 23.4 | 1.11x | 10.98 | 0.59 | 18.7x |
| DMCFingerSpin | 84 | 16 | 49.1 | 93.0 | 1.90x | 12.39 | 1.81 | 6.9x |
| DMCFingerSpin | 84 | 64 | 45.2 | 81.9 | 1.81x | 13.24 | 2.40 | 5.5x |
| DMCFingerSpin | 84 | 256 | 37.4 | 68.1 | 1.82x | 13.24 | 2.40 | 5.5x |
| DMCPendulumSwingup | 84 | 1 | 47.0 | 44.0 | 0.94x | 6.25 | 0.31 | 20.0x |
| DMCPendulumSwingup | 84 | 16 | 107.3 | 169.2 | 1.58x | 6.43 | 0.84 | 7.7x |
| DMCPendulumSwingup | 84 | 64 | 90.7 | 161.8 | 1.78x | 6.51 | 1.01 | 6.4x |
| DMCPendulumSwingup | 84 | 256 | 80.9 | 142.9 | 1.77x | 6.51 | 1.01 | 6.4x |
| DMCReacherHard | 84 | 1 | 35.9 | 38.5 | 1.07x | 8.87 | 0.24 | 36.8x |
| DMCReacherHard | 84 | 16 | 70.0 | 109.1 | 1.56x | 10.94 | 1.87 | 5.8x |
| DMCReacherHard | 84 | 64 | 53.1 | 81.7 | 1.54x | 11.46 | 3.10 | 3.7x |
| DMCReacherHard | 84 | 256 | 55.2 | 75.2 | 1.36x | 11.46 | 3.10 | 3.7x |
| PointmazeLargeNavigate | 224 | 1 | 214.9 | 235.4 | 1.10x | 0.08 | 0.00 | 21.3x |
| PointmazeLargeNavigate | 224 | 16 | 742.2 | 1019.1 | 1.37x | 0.08 | 0.01 | 6.4x |
| PointmazeLargeNavigate | 224 | 64 | 809.6 | 1110.5 | 1.37x | 0.08 | 0.02 | 5.1x |
| PointmazeLargeNavigate | 224 | 256 | 628.7 | 986.3 | 1.57x | 0.08 | 0.02 | 5.1x |
| PointmazeMediumNavigate | 224 | 1 | 238.8 | 234.2 | 0.98x | 0.08 | 0.01 | 7.6x |
| PointmazeMediumNavigate | 224 | 16 | 751.8 | 698.9 | 0.93x | 0.08 | 0.04 | 1.9x |
| PointmazeMediumNavigate | 224 | 64 | 770.0 | 731.5 | 0.95x | 0.08 | 0.05 | 1.7x |
| PointmazeMediumNavigate | 224 | 256 | 609.2 | 655.8 | 1.08x | 0.08 | 0.05 | 1.7x |

### What the BOUND alone buys — the same branch, switch off and on

Job `6350679`, one node, one thread, both legs on `$SCR/se6/raycast3d` with
`--primitive-bounds off|on`. Added because the sweep above cannot attribute: its `before` leg is
the pre-Step-6 baseline, so every ratio in it also contains 2a's batched tendon FK and 2b's
cover compaction. This is the only leg where the single independent variable is the bound.

| env | B | fps off | fps on | fps x | tests/px off | on | tests x |
|---|---:|---:|---:|---:|---:|---:|---:|
| DMCAcrobotSwingup | 1 | 45.8 | 57.2 | 1.25x | 2.80 | 0.22 | 12.8x |
| DMCAcrobotSwingup | 16 | 233.6 | 309.8 | 1.33x | 3.02 | 0.79 | 3.8x |
| DMCAcrobotSwingup | 64 | 237.6 | 314.6 | 1.32x | 3.23 | 1.12 | 2.9x |
| DMCAcrobotSwingup | 256 | 242.5 | 298.6 | 1.23x | 3.23 | 1.12 | 2.9x |
| DMCBallInCupCatch | 1 | 6.2 | 9.6 | 1.54x | 27.39 | 0.13 | 213.3x |
| DMCBallInCupCatch | 16 | 23.4 | 50.9 | 2.17x | 30.23 | 0.81 | 37.4x |
| DMCBallInCupCatch | 64 | 20.0 | 61.6 | 3.08x | 39.68 | 1.70 | 23.3x |
| DMCBallInCupCatch | 256 | 18.3 | 55.6 | 3.03x | 39.68 | 1.70 | 23.3x |
| DMCCartpoleSwingup | 1 | 73.5 | 69.0 | 0.94x | 1.93 | 0.23 | 8.4x |
| DMCCartpoleSwingup | 16 | 342.2 | 514.6 | 1.50x | 1.96 | 0.29 | 6.8x |
| DMCCartpoleSwingup | 64 | 341.8 | 638.9 | 1.87x | 1.96 | 0.29 | 6.7x |
| DMCCartpoleSwingup | 256 | 324.0 | 610.7 | 1.88x | 1.96 | 0.29 | 6.7x |
| DMCFingerSpin | 1 | 20.0 | 23.8 | 1.19x | 10.98 | 0.59 | 18.7x |
| DMCFingerSpin | 16 | 57.5 | 114.2 | 1.99x | 12.39 | 1.81 | 6.9x |
| DMCFingerSpin | 64 | 51.5 | 107.8 | 2.09x | 13.24 | 2.40 | 5.5x |
| DMCFingerSpin | 256 | 47.3 | 97.9 | 2.07x | 13.24 | 2.40 | 5.5x |
| DMCPendulumSwingup | 1 | 51.6 | 46.9 | 0.91x | 6.25 | 0.31 | 20.0x |
| DMCPendulumSwingup | 16 | 146.0 | 223.3 | 1.53x | 6.43 | 0.84 | 7.7x |
| DMCPendulumSwingup | 64 | 132.2 | 216.4 | 1.64x | 6.51 | 1.01 | 6.4x |
| DMCPendulumSwingup | 256 | 119.7 | 192.0 | 1.60x | 6.51 | 1.01 | 6.4x |
| DMCReacherHard | 1 | 44.0 | 41.2 | 0.94x | 8.87 | 0.24 | 36.8x |
| DMCReacherHard | 16 | 96.4 | 136.9 | 1.42x | 10.94 | 1.87 | 5.8x |
| DMCReacherHard | 64 | 82.1 | 99.9 | 1.22x | 11.46 | 3.10 | 3.7x |
| DMCReacherHard | 256 | 73.8 | 96.0 | 1.30x | 11.46 | 3.10 | 3.7x |
| PointmazeLargeNavigate | 1 | 285.2 | 279.9 | 0.98x | 0.08 | 0.00 | 21.3x |
| PointmazeLargeNavigate | 16 | 1092.5 | 1004.1 | 0.92x | 0.08 | 0.01 | 6.4x |
| PointmazeLargeNavigate | 64 | 1239.7 | 1144.3 | 0.92x | 0.08 | 0.02 | 5.1x |
| PointmazeLargeNavigate | 256 | 1066.2 | 935.3 | 0.88x | 0.08 | 0.02 | 5.1x |
| PointmazeMediumNavigate | 1 | 248.7 | 221.8 | 0.89x | 0.08 | 0.01 | 7.6x |
| PointmazeMediumNavigate | 16 | 894.6 | 747.0 | 0.84x | 0.08 | 0.04 | 1.9x |
| PointmazeMediumNavigate | 64 | 954.9 | 715.9 | 0.75x | 0.08 | 0.05 | 1.7x |
| PointmazeMediumNavigate | 256 | 891.6 | 654.1 | 0.73x | 0.08 | 0.05 | 1.7x |

### Tiers, against `cores_cpu` — and the answer is no

Each frozen Step-3 CPU row on `node2349` scaled by the speed-up measured **at that row's own
certified `best_batch`**, not by the best ratio across batches: `se_fps` in the certification is
that row's number at its own operating point, and scaling it by a ratio measured at a different
batch compares two different things. `S_cpu = se_fps / official_cpu_fps` (bar 5);
`cores_cpu = se_fps / decode_fps_per_core` (bar 1); `CPU_FAST` needs **both**.

| env | Step-3 fps | x at best_batch | projected fps | S_cpu | cores_cpu | short of the cores bar by | tier |
|---|---:|---:|---:|---:|---:|---:|---|
| DMCAcrobotSwingup | 267.0 | 1.22x @B=64 | 327.1 | 7.05 -> 8.64 | 0.0396 -> 0.0485 | **20.6x** | CPU_FASTER+BELOW_G4 |
| DMCBallInCupCatch | 20.9 | 2.28x @B=16 | 47.6 | 1.61 -> 3.68 | 0.0029 -> 0.0066 | **150.8x** | CPU_FASTER+BELOW_G4 |
| DMCCartpoleSwingup | 372.2 | 1.52x @B=64 | 565.7 | 10.06 -> 15.29 | 0.0575 -> 0.0873 | **11.5x** | CPU_FASTER+BELOW_G4 |
| DMCFingerSpin | — | — | — | — | — | — | no frozen Step-3 `node2349` row |
| DMCPendulumSwingup | 145.2 | 1.58x @B=16 | 229.0 | 3.79 -> 5.98 | 0.0187 -> 0.0295 | **33.9x** | CPU_FASTER+BELOW_G4 |
| DMCReacherHard | 98.6 | 1.56x @B=16 | 153.8 | 2.87 -> 4.48 | 0.0141 -> 0.0219 | **45.6x** | CPU_FASTER+BELOW_G4 |
| PointmazeLargeNavigate | 994.3 | 1.37x @B=64 | 1363.8 | 48.01 -> 65.85 | 0.4159 -> 0.5705 | **1.8x** | CPU_FASTER+BELOW_G4 |
| PointmazeMediumNavigate | — | — | — | — | — | — | no frozen Step-3 `node2349` row |

### The cuda path is untouched, and the device agrees

| env | B | device us before | after | x | fps before | after | x |
|---|---:|---:|---:|---:|---:|---:|---:|
| DMCBallInCupCatch | 1 | 1482 | 1475 | 0.995x | 159.0 | 152.1 | 0.96x |
| DMCBallInCupCatch | 64 | 2291 | 2281 | 0.996x | 2409.7 | 4579.4 | 1.90x |
| DMCBallInCupCatch | 256 | 5106 | 5055 | 0.990x | 2662.6 | 8201.8 | 3.08x |
| DMCBallInCupCatch | 4096 | 66635 | 66171 | 0.993x | 3240.4 | 9674.6 | 2.99x |
| DMCFingerSpin | 1 | 1002 | 1001 | 0.999x | 182.6 | 186.8 | 1.02x |
| DMCFingerSpin | 64 | 2150 | 2148 | 0.999x | 7421.5 | 7776.7 | 1.05x |
| DMCFingerSpin | 256 | 5626 | 5631 | 1.001x | 22325.0 | 23606.0 | 1.06x |
| DMCFingerSpin | 4096 | 73803 | 72509 | 0.982x | 21202.8 | 43257.7 | 2.04x |

### The finding the attribution leg produced, which is not the one I expected

**The bound makes both Pointmaze rows SLOWER, and the cumulative table hides it.** Against the
pristine baseline `PointmazeLargeNavigate` reads 1.37x at B=64 and looks like a win; with the switch
as the only variable it reads **0.92x**, and `PointmazeMediumNavigate` reads **0.75x**. The 1.37x is
2b's cover compaction, which is already merged. The bound's own contribution to that row is negative.

| row | K | N (rays/frame) | R4 alone @B=64 | tests/px |
|---|---:|---:|---:|---|
| `DMCBallInCupCatch` | 36 | 7,056 | **3.08x** | 39.68 -> 1.70 |
| `DMCFingerSpin` | 7 | 7,056 | **2.09x** | 13.24 -> 2.40 |
| `DMCCartpoleSwingup` | 2 | 7,056 | **1.87x** | 1.96 -> 0.29 |
| `DMCPendulumSwingup` | 3 | 7,056 | 1.64x | 6.51 -> 1.01 |
| `DMCAcrobotSwingup` | 4 | 7,056 | 1.32x | 3.23 -> 1.12 |
| `DMCReacherHard` | 4 | 7,056 | 1.22x | 11.46 -> 3.10 |
| `PointmazeLargeNavigate` | **1** | **200,704** | **0.92x** | 0.08 -> 0.02 |
| `PointmazeMediumNavigate` | **1** | **200,704** | **0.75x** | 0.08 -> 0.05 |

Two things are worth separating here.

* **Every ray-primitive test the bound removes, it removes.** `tests/px` is bit-for-bit the same in
  the isolation leg as in the cumulative sweep, on all eight rows — 2a and 2b contribute none of that
  reduction. The pruning claim is entirely confirmed.
* **Fewer tests is not less time.** On the two maze rows the bound prunes 1.7x–5.1x of the tests and
  still loses 8–25 % of the frame rate, because what it substitutes is not free: `torch.nonzero`
  plus the advanced-indexing gather of `o[:, js]` / `d[:, js]` and the scatter back into
  `blocked[:, js]`. Those are O(N) on a bundle of **200,704 rays** (224 px at `supersample=2`, 28x
  the DM Control bundles) and they buy the narrowing of **one** intersector call, because these
  scenes have **K = 1** — one mover primitive, the agent sphere. There is no second primitive to
  prune between; there is only the gather.

So the design's own prediction — *"it does not help `PointmazeLargeNavigate` (K=1: there is nothing
to prune against)"* — was directionally right and quantitatively wrong in sign. It does not merely
fail to help. It costs.

**This is a live regression on a certified CPU row and I am not fixing it in this ticket.**
`PointmazeLargeNavigate` has a frozen Step-3 CPU row, so `PRIMITIVE_BOUNDS = True` as it stands ships
a ~8 % CPU slowdown there. The obvious guard is a dispatch predicate — take the bounded path only
when it can pay — but every candidate I can write down is a tuned threshold (`K >= 2`? `|js| < N/2`?
a ray-count ceiling?), the DM Control rows themselves lose 6–9 % at B=1, and picking a constant from
eight rows to make a table look good is the same move that put a far-field bug past two test suites
earlier in this ticket. **It needs its own measurement and the coordinator's call, so it is written
up as RAYCAST-5 below rather than guessed at here.** Until then the switch is the containment: it
exists, its off-path is the verbatim definition, and flipping it restores the baseline exactly.

### The verdict on the tier question, which is "no" either way

* **It prunes.** 3.7x–23x fewer ray-primitive tests per pixel at B=64, 8x–213x at B=1, with no pixel
  moving anywhere in the 21-config corpus.
* **It does not change a tier.** `cores_cpu` binds on all eight rows and is short by **1.8x on
  `PointmazeLargeNavigate`, 11.5x on `DMCCartpoleSwingup`, 151x on `DMCBallInCupCatch`.**
* **One half-threshold moves and does not matter.** `DMCPendulumSwingup` crosses `S_cpu >= 5`
  (3.79 -> 5.98). `CPU_FAST` needs both halves and its `cores_cpu` is 0.0295. Recorded so that nobody
  later reads the 5.98 as a tier.

RAYCAST-4 is a **PATCH, not a tier change**, and on two rows it is currently a negative one. The
honest reading of the tier table is that no cast-level optimisation reaches `CPU_FAST` here:
`cores_cpu` measures a whole analytic render against one core of JPEG decode, and closing 11x — let
alone 151x — is a question about what the renderer does per frame, not about how many primitives it
tests.

### Batch invariance is a claim here, not an identity

The ray set for a primitive is the union over the batch, so **batching changes which rays reach the
intersector.** That row 0's pixels come out the same alone and in company is therefore a consequence
of the bound being conservative, not something readable off the code — and a too-tight bound would
break it asymmetrically, the row alone losing a hit the row in company keeps.
`tests/gpu/test_reproducibility.py` owns this property for the repo but is cuda-only and skips its
whole module without a device, so it has nothing to say about a CPU-only switch.
`test_a_row_is_unchanged_by_the_rows_it_is_batched_WITH` puts row 0 of a B=1 bundle into row 0 of a
B=8 bundle whose other seven rows are a different scene, and compares bit patterns. Job `6351415`,
`node1827`: **27 passed** for that module (24 differential + 3 batch-invariance), and `tests/unit`
overall **570 passed, 9 skipped, 2 failed** — both failures pre-existing and outside the authorised
files: `render_mjwarp.py --help`, and `test_every_cited_scratch_path_resolves`, which after this
commit is down to **two** unresolvable citations (a `carracing` log directory and a `robocasa`
output corpus — deliberately not spelled as paths here, because writing them out would make THIS
file a citation site for them and hand the blame to this workstream), both cited from `docs/research/`
and
`examples/results/step6/texture-fidelity/findings.md` and neither from this workstream. Verified by
re-running that module against the synced tree with this document in place.

### The five jobs this section is made of

| job | node | what |
|---|---|---|
| `6348760` | `node2337` | CPU pixels baseline-vs-branch and `PRIMITIVE_BOUNDS` on-vs-off over all 21, then the 1-thread sweep, both legs |
| `6348761` | `gpu2113` | cuda pixels over all 21, `tests/gpu/test_fast_paths.py` at native sizes, the fuzz, and the cuda no-regression sweep |
| `6350679` | `node2333` | the same branch with the switch off and on, one thread — the only leg that attributes anything to the bound alone |
| `6351415` | `node1827` | `tests/unit`, including the batch-invariance case |
| `6348207` | — | the run that moved one pixel of 3,211,264 and produced the far-field fix |

The sweep and the isolation ran on different nodes (`node2337`, `node2333`), so the two tables are
**not** comparable to each other row by row; each is internally comparable because both of its legs
ran in one job on one node, which is the only comparison either is asked to support.

### The two limits of this design, both measured rather than argued

**1. The union is over the BATCH, and it costs.** The ray set for primitive `k` is the rays near it in
ANY row of the chunk, because a per-row set would need a ragged gather — a Python loop over rows, the
thing this design exists to avoid. So the set grows with `B` while the per-row set does not. Measured
on `DMCReacherHard`: **0.24 ray-primitive tests per pixel at B=1 against 3.10 at B=64 — the union
costs 12.9x.** At B=1 the bound is nearly perfect; at B=64 it still prunes 3.7x but has given most of
it back. That is the honest ceiling of a batch-union bound, and the next lever if this row needs more:
a ragged per-row gather, or binning rows by pose so that the union within a bin is tight.

**2. The mask is evaluated over the FULL bundle, for every primitive.** `_rays_near` is ~14
elementwise ops on `(B,N)` whether it ends up keeping one ray or all of them, so the saving is on the
intersector (~200 ops) and the cost is paid regardless. That is why the fps gain is smaller than the
test-count reduction — on `DMCReacherHard` at B=64, **3.7x fewer tests buys 1.22x of frame rate**
(the bound alone; 1.54x including 2a and 2b) — and the gap is the mask, the `nonzero`/gather/scatter,
and the part of the render that was never the cast. On the two maze rows that gap is larger than the
saving and the net is **negative**; see the attribution table above. Both terms are visible in the
measurements and neither is noise.

Neither limit is a defect of the bound; they are what the bound costs, and they are the two places a
follow-up would look.


# RAYCAST-5 — the bound pays only where the intersector is dearer than the bound

RAYCAST-4 shipped a bound that was on for every primitive kind. It gained 3.08x on `ball_in_cup`
and cost `PointmazeLargeNavigate` 8 % and `PointmazeMediumNavigate` 25 % — a regression on a row
with a frozen Step-3 CPU certification. This is the repair, and the thing it must not be is a
threshold on `K`, on `N`, or on selectivity: those are constants fitted to the eight rows that
happened to be swept.

## The predicate is arithmetic, not a fit

For one primitive over `N` rays:

    unbounded:  cost_intersect(kind) * N
    bounded:    cost_bound * N  +  cost_intersect(kind) * N * p       (p = fraction of rays kept)

so the bound pays iff `cost_bound < cost_intersect(kind) * (1 - p)`. `p` belongs to the scene and
is not knowable inside the cast — but `p >= 0`, so the condition that must hold for **every** scene,
batch and resolution is just `cost_bound < cost_intersect(kind)`. A kind whose intersector is
cheaper than the bound's own mask + `nonzero` + gather + scatter can never pay: there is nothing
left to save even under perfect pruning. That ratio is a property of the intersector's arithmetic.

`scratch/step6/raycast3d/r5_kind_cost.py`, **three independent runs in one job** (`6356311`,
`node1843`, Xeon 8268, 1 thread), B in {1,16,64} x N in {7056, 50176, 200704}. The bound is timed at
its worst case — N−1 rays kept, so the gather and scatter run at essentially full width — in the
**`cast_nearest` shape**, which is the branch that decides the table: it gathers `best` and scatters
`best`/`nrm`/`idx`.

| kind | `ns_intersect / ns_bound`, worst over widths, per run | best | bounded? |
|---|---|---:|---|
| `PLANE` | no finite bounding sphere | — | no |
| `SPHERE` | 0.27 · 0.28 · 0.28 | 0.46 | no — never reaches 1, so it can never pay |
| `BOX` | 1.13 · 1.11 · 1.13 | 2.93 | **yes** |
| `CYLINDER` | 1.24 · 1.23 · 1.24 | 3.11 | **yes** |
| `CAPSULE` | 1.48 · 1.48 · 1.47 | 3.74 | **yes** |

The dearest unbounded kind reaches **0.46** at its best; the cheapest bounded kind never falls below
**1.11**; the line at 1.0 sits inside that **2.4x** gap. `_BOUNDED_KINDS = {CAPSULE, CYLINDER, BOX}`,
and `test_the_bounded_kinds_are_the_ones_the_committed_MEASUREMENT_chose` fails if the module
constant, `experiments/r6_kind_cost.json`, or the rule behind them ever disagree.

### Two corrections this table carries, both found in review

**1. The first version priced the wrong branch.** It timed the `any_blocking`-shaped bound — one bool
scatter — while the branch that decides the table is `cast_nearest`, which additionally gathers
`best` and scatters `best`/`nrm`/`idx`. That understates the bound by about **2x**, so the original
ratios (SPHERE 0.45–0.83, CAPSULE 2.38–3.74) were flattering. Re-priced, capsule's worst case is
1.47, not 2.38.

**2. The margin rule was scaled by the wrong noise, and it was not stable.** `1 + 3*worst_rsd` used
three times the spread of an *individual timing*, which is not the statistic the decision consumes.
Run on a busier node it moved from 1.23 to **1.75** — and the three runs above, whose ratios agree to
within **1–3 %**, carry three *different* predicates in their own `predicate` fields for exactly that
reason: all-False, capsule-only, all-False. **A threshold a node's load average can flip is not a
derived constant.**

What stands instead is the necessary condition itself — `ratio > 1`, at every width, in every run —
which was derived above and needs no constant at all. The error bar that matters is the spread of the
ratio across runs, and it is 1–3 %, nowhere near the boundary. The earlier claim of *"a >2x gap so
line placement doesn't matter"* was about where a **threshold** sat between 0.83 and 1.84; that
framing is retired with the threshold. The separation claim survives in a better form, because 1.0 is
not a tuned number.

**The shipped predicate did not change.** Both corrections make the bound look worse and neither moves
a kind across the line, which is the outcome worth stating plainly: the table was right for
partly-wrong reasons, and it is now right for the right ones.

## What each config actually casts — counted, not assumed

The explanation *"Pointmaze regressed because its mover is a sphere"* is a claim about MuJoCo
models resolved at bake time, so it is measured (`experiments/r5_mover_kinds.json`):

| config | primitives cast per frame | bounded | unbounded |
|---|---|---:|---:|
| `PointmazeLargeNavigate` | SPHERE x1 | 0 | 1 |
| `PointmazeMediumNavigate` | SPHERE x1 | 0 | 1 |
| `DMCBallInCupCatch` | CAPSULE x33, SPHERE x3 | 33 | 3 |
| `DMCCartpoleSwingup` | BOX x1, CAPSULE x1 | 2 | 0 |
| `DMCFingerSpin` | CAPSULE x5, CYLINDER x2 | 7 | 0 |
| `DMCReacherHard` | CAPSULE x2, SPHERE x2 | 2 | 2 |
| `DMCAcrobotSwingup` | CAPSULE x2, CYLINDER x1, SPHERE x1 | 3 | 1 |
| `DMCPendulumSwingup` | CAPSULE x1, CYLINDER x1, SPHERE x1 | 2 | 1 |
| `AntmazeGiantNavigate` | CAPSULE x12, SPHERE x1 | 12 | 1 |
| `HumanoidmazeGiantNavigate` | CAPSULE x16, SPHERE x3 | 16 | 3 |

**All four Pointmaze configs cast exactly one primitive and it is a SPHERE.** Under this table the
bound is not merely cheaper for them, it is *absent* — which is why the regression should return to
exactly 1.00x rather than approximately.

## The call-level short circuit, which is the same table read once

The per-kind table alone still left Pointmaze at 0.92-0.98x — **with its ray-primitive test counts
identical in both legs**, 0.08 per pixel either way. Identical counts mean identical work inside the
`K`-loop, which located the cost precisely: `dd = _dot(d, d)` and the `_unit_rays` reduction run
*before* the loop, unconditionally, over the whole bundle — 200,704 rays at `ss=2`, per cast call,
for a loop that then bounds nothing. `any(_kind_is_bounded(k) for k in kinds)` at the top returns
the verbatim unbounded body instead. No new constant and no second predicate: it is the authorised
per-kind decision applied to the call rather than to each primitive.

| row | B=16 / 64 / 256, per-kind table only | with the short circuit |
|---|---|---|
| `PointmazeLargeNavigate` | 0.98 / 0.95 / 0.91 | **1.06 / 1.01 / 1.05** |
| `PointmazeMediumNavigate` | 0.95 / 0.98 / 0.96 | **1.00 / 1.02 / 0.93** |

## The bar is finer than the harness — measured, not asserted

Four readings still sat under 0.97x, one of them `PointmazeMediumNavigate` at B=256 on a row whose
bounded path is *absent*. A 7 % cost cannot come from an `any()` over a one-element list, so either
the reading was wrong or the harness was. `r5_noise.sbatch` (job `6354150`) runs four **interleaved**
legs — off, on, off, on — same branch, same node, same job, 15 repeats instead of 5. `off2/off1` is
a ratio of two runs of *identical code* and therefore measures nothing but the harness.

| env | B | off1 | off2 | **noise** | on (mean) | **signal** | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `DMCCartpoleSwingup` | 1 | 69.9 | 73.9 | 1.058 | 69.7 | **0.969** | below bar, **B=1 is not a certification batch** |
| `DMCCartpoleSwingup` | 16 | 269.2 | 279.1 | 1.037 | 420.1 | **1.533** | ok |
| `DMCCartpoleSwingup` | 64 | 293.5 | 308.1 | 1.050 | 519.9 | **1.729** | ok |
| `DMCCartpoleSwingup` | 256 | 257.4 | 287.5 | 1.117 | 519.3 | **1.906** | ok |
| `DMCFingerSpin` | 1 | 20.2 | 20.4 | 1.012 | 24.4 | **1.200** | ok |
| `DMCFingerSpin` | 16 | 50.3 | 51.3 | 1.021 | 103.2 | **2.031** | ok |
| `DMCFingerSpin` | 64 | 47.3 | 46.9 | 0.991 | 94.2 | **1.999** | ok |
| `DMCFingerSpin` | 256 | 42.7 | 42.6 | 0.996 | 90.2 | **2.116** | ok |
| `DMCReacherHard` | 1 | 43.3 | 40.4 | 0.934 | 44.1 | **1.053** | ok |
| `DMCReacherHard` | 16 | 81.8 | 77.2 | 0.943 | 115.2 | **1.449** | ok |
| `DMCReacherHard` | 64 | 71.5 | 63.6 | 0.889 | 93.1 | **1.378** | ok |
| `DMCReacherHard` | 256 | 66.8 | 60.0 | 0.898 | 88.1 | **1.389** | ok |
| `PointmazeLargeNavigate` | 1 | 302.2 | 287.3 | 0.951 | 297.1 | **1.008** | ok |
| `PointmazeLargeNavigate` | 16 | 1144.7 | 1126.0 | 0.984 | 1159.5 | **1.021** | ok |
| `PointmazeLargeNavigate` | 64 | 1289.0 | 1283.4 | 0.996 | 1326.0 | **1.031** | ok |
| `PointmazeLargeNavigate` | 256 | 1229.6 | 1214.0 | 0.987 | 1240.7 | **1.015** | ok |
| `PointmazeMediumNavigate` | 1 | 292.1 | 299.7 | 1.026 | 292.9 | **0.990** | ok |
| `PointmazeMediumNavigate` | 16 | 889.6 | 899.8 | 1.012 | 894.9 | **1.000** | ok |
| `PointmazeMediumNavigate` | 64 | 963.1 | 972.0 | 1.009 | 958.2 | **0.990** | ok |
| `PointmazeMediumNavigate` | 256 | 935.6 | 945.1 | 1.010 | 923.2 | **0.982** | ok |

**The worst off-vs-off spread on identical code is 11.7 %, against a bar that asks about
3 %.** One thread on a shared cluster node cannot resolve 3 %. Averaged over both legs each way,
**every row passes at every certification batch**; the single remaining sub-bar reading is
`DMCCartpoleSwingup` at B=1, 0.969 against 0.970, on a row whose own off-vs-off noise is 1.058 —
and B=1 is the batch the frozen spec **forbids** certifying on (`gpu.never_certify_on_batch`,
because the corpus median there is 0.6x the official renderer and it measures launch overhead).

Per the ticket's instruction I reported that and added **no second predicate**, and I did not flip
the default on my own judgment: whether 0.969-inside-5.8 %-noise at an uncertifiable batch clears a
0.97 bar is a release decision. The coordinator took it — acceptance holds — and the default is now
`True`, verified as below.

## The admission window was 400x too loose, and inside it the identity was false

Independent review of `3b9e81c` found the substantive defect of this ticket. `_UNIT_D_TOL` was
`1e-4` on `|d|^2`, and **inside that window the identity this module claims does not hold**: the
cap-sphere test drops the leading `|d|^2`, so at `|d|^2 = 1 + 9e-5` and 100 m the reviewer's probe
lost **113,008 of 126,184** unbounded hits. They are garbage hits either way — an intersector handed
a non-unit ray is not computing a ray parameter — but `_unit_rays` is what decides whether a bundle
is *admitted*, and a bundle it admits must get the same garbage out of both paths.

The window is now sized to the call sites instead of to a round number. `_UNIT_D_TOL = 1e-6`, which
is at or below `_BOUND_SLACK_NOISE` (1.9e-6), the term the bound already carries for the
intersector's own resolution — so nothing can slip between the two.

**Tightening an admission check has its own failure mode**, and it is silent: a real bundle that now
falls *outside* the window takes the unbounded path, the pixels stay identical because that path is
the definition, and the bound quietly stops applying where it was measured to pay. So `_unit_rays`
itself is instrumented across every live config (`scratch/step6/raycast3d/r6_unit_ray_probe.py`,
`experiments/r6_unit_ray_probe_cpu.json`):

| | |
|---|---|
| configs measured | 21 of 21 |
| `_unit_rays` calls | 94 |
| **bundles rejected** | **0** |
| worst `abs(dd - 1)` actually seen | **2.384e-07** |
| headroom to the new tolerance | **4.19x** |

Nothing routes to the unbounded path, and the worst real bundle reproduces the reviewer's measured
2.4e-7 exactly. Three fuzz cases now sit one ulp *inside* the window at `far=100` — the worst bundle
the check still admits — and they fail against the old tolerance.

## The default is on, and the identity is re-taken without pinning anything

`PRIMITIVE_BOUNDS = True` (jobs `6355332` CPU, `6355333` GPU). This is the run that matters now:
**neither leg pins the switch**, so it is the pristine pre-Step-6 baseline against the branch exactly
as a caller gets it. Every earlier identity run had to force the bound on, because with the default
off a replay taking the default would have compared the unbounded path against itself and reported
`ALL_IDENTICAL` for the wrong reason — the job log prints the branch's own
`PRIMITIVE_BOUNDS = True kinds [3, 5, 6]` before it starts, so the leg cannot be silently vacuous.

| check | result |
|---|---|
| cpu, pristine baseline vs branch, all 21 live configs, **defaults on both legs** | **identical** (`replay/r5final_cpu_identity.json`) |
| cuda, pristine baseline vs branch, all 21 | **identical** (`replay/r5final_cuda_identity.json`) |
| `tests/gpu/test_fast_paths.py` at the envs' own sizes | **176 passed**, 33 skipped |
| `tests/unit` | **623 passed**, 9 skipped, 2 failed — the same two pre-existing failures, neither in the authorised files |

Re-run after the review changes (jobs `6356117` CPU, `6356123` GPU), on the tightened tolerance:

| check | result |
|---|---|
| cpu, `PRIMITIVE_BOUNDS` on vs off, all 21, nothing excluded | **identical** (`replay/r6_cpu_bounds_identity.json`) |
| cpu, pristine baseline vs branch, all 21, defaults both legs | **identical** (`replay/r6_cpu_branch_vs_baseline.json`) |
| cuda, pristine baseline vs branch, all 21 | **identical** (`replay/r6_cuda_identity.json`) |
| `tests/gpu/test_fast_paths.py` at the envs' own sizes | **176 passed**, 33 skipped |
| `tests/unit/test_raycast3d_primitive_bounds.py` | **51 passed** (was 45; the three edge-of-window cases x two cast functions) |
| `tests/unit` | **629 passed**, 9 skipped, 2 failed — the same two pre-existing |

## Step-3-shaped shards, so the dashboard can label these rows

`examples/results/step6/raycast3d/sweep/`, produced by `examples/bench_step3.py` itself rather than
back-converted from the profile notes: a shard needs `decode` and `official_cpu`, which are the
**denominators of both tier bars**, and copying those from the frozen `node2349` row onto a
measurement taken elsewhere would bury a cross-host ratio in a file whose whole purpose is to be
classified as if it were Step 3. Three CPU legs because the release decision was open when they were taken; the
`bound on` leg is the shipped build.

The bound-off leg is named `cpu_1thread_branch_bounds_off_control.json`, with **no `before`/`after`
token**, and that is load-bearing rather than tidy: `report.py`'s `leg_of` reads the leg out of the
filename, and under its old name this control was taken for a second *after* leg — which would have
made the page report "CPU after-legs disagree" on `DMCFingerSpin` and `DMCBallInCupCatch`, the two
rows whose tier change is the whole point of the shard. It is a control, not an after.

That rename regressed once before it stuck, and the mechanism is worth recording because it is not
specific to this file: the shard job on the cluster had written the OLD name, so the next
`rsync` pull of the results directory re-created it alongside the new one and it was committed back.
A rename is not complete until the producer is renamed too — the `.sbatch` now writes the control
name — and until the stale copy is removed at the source. This workstream has now been bitten twice
by a pull re-materialising something it had deliberately changed; the first time it was
`findings.md`, which is why the pulls carry `--exclude findings.md`.

CPU: `node2335`, job `6354672`, 1 thread, Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz. Classified here with `scratch/step3/certify.py:classify`.

The **bound on** column is what the branch now ships.

| env | baseline | branch, bound off | branch, bound **on** (shipped) |
|---|---|---|---|
| `DMCBallInCupCatch` | CPU_PARITY · S_cpu 1.40 · cores 0.0026 | CPU_PARITY · S_cpu 1.45 · cores 0.0027 | CPU_FASTER · S_cpu 4.74 · cores 0.0089 |
| `DMCFingerSpin` | CPU_PARITY · S_cpu 1.76 · cores 0.0078 | CPU_PARITY · S_cpu 1.95 · cores 0.0097 | CPU_FASTER · S_cpu 4.01 · cores 0.0194 |
| `DMCReacherHard` | CPU_FASTER · S_cpu 2.72 · cores 0.0131 | CPU_FASTER · S_cpu 2.76 · cores 0.0134 | CPU_FASTER · S_cpu 4.02 · cores 0.0198 |
| `DMCPendulumSwingup` | CPU_FASTER · S_cpu 3.88 · cores 0.0193 | CPU_FASTER · S_cpu 3.50 · cores 0.0184 | CPU_FASTER · S_cpu 6.56 · cores 0.0335 |
| `DMCCartpoleSwingup` | CPU_FASTER · S_cpu 10.41 · cores 0.0582 | CPU_FASTER · S_cpu 10.10 · cores 0.0498 | CPU_FASTER · S_cpu 19.00 · cores 0.0975 |
| `DMCAcrobotSwingup` | CPU_FASTER · S_cpu 7.27 · cores 0.0376 | CPU_FASTER · S_cpu 7.23 · cores 0.0363 | CPU_FASTER · S_cpu 9.25 · cores 0.0452 |
| `PointmazeLargeNavigate` | CPU_FASTER · S_cpu 51.55 · cores 0.3762 | CPU_FASTER · S_cpu 76.89 · cores 0.6171 | CPU_FASTER · S_cpu 78.05 · cores 0.6184 |
| `PointmazeMediumNavigate` | CPU_FASTER · S_cpu 36.72 · cores 0.4150 | CPU_FASTER · S_cpu 44.22 · cores 0.4761 | CPU_FASTER · S_cpu 43.64 · cores 0.4716 |

**Two rows change tier with the bound on**: `DMCFingerSpin` and `DMCBallInCupCatch` go
`CPU_PARITY` -> `CPU_FASTER` (S_cpu 1.95 -> 4.01 and 1.45 -> 4.74). That is a stronger result than
RAYCAST-4 reported, and the difference is method, not luck: R4 projected by scaling the *frozen
`node2349`* rows, while these are a like-for-like pair measured on one node in one job with the
official renderer and the decode rate taken there too. `cores_cpu` is still below 1 on every row,
so `CPU_FAST` remains out of reach and the RAYCAST-4 conclusion on that point stands.

GPU: `gpu2105`, NVIDIA GeForce RTX 3090, job `6354674`.

| env | baseline | branch |
|---|---|---|
| `DMCBallInCupCatch` | GPU_PARITY · R 9.4 · cores 0.44 · BELOW_G4,BATCH_SCALES_FAIL | GPU_FAST · R 28.6 · cores 1.39 · BATCH_SCALES |
| `DMCFingerSpin` | GPU_FAST · R 108.1 · cores 5.94 · BATCH_SCALES | GPU_FAST · R 127.9 · cores 5.82 · BATCH_SCALES |

### Two things these shards do not carry

* **No per-cell spread.** A Step-3 shard records one `fps` per cell and no dispersion, so a tier read
  off one carries no error bar — and this workstream has just measured an **11.7 % off-vs-off spread
  on identical code** at one thread on a shared node. The tier *changes* below are far larger than
  that (S_cpu 1.95 -> 4.01, 1.45 -> 4.74) so they survive it comfortably, but a future row that moved
  a tier by a few per cent on this evidence would not be safe to believe, and the shard format gives
  no way to tell the two situations apart.
* **`Mujoco*` CPU identity is inferred, not replayed.** The 21-config CPU replay covers the DM Control
  and OGBench maze families. The `Mujoco*` family's agreement between the bounded and unbounded paths
  comes from `tests/gpu/test_fast_paths.py`, which builds those envs **on cuda**; no CPU replay of
  them exists. The cast code is shared and the fuzz is device-independent, so the inference is a
  reasonable one — but it is an inference, and it is the obvious gap to close if `Mujoco*` CPU ever
  becomes a certified path.

`DMCBallInCupCatch` goes **`GPU_PARITY` + `BELOW_G4` + `BATCH_SCALES_FAIL` -> `GPU_FAST` +
`BATCH_SCALES`** (R 9.4 -> 28.6, `cores` 0.44 -> 1.39, batch scaling 19.0x -> 55.9x). That is the
dashboard's claim, now carried by a shard the report generator can classify instead of by prose.
The GPU legs are unaffected by `PRIMITIVE_BOUNDS` — the fused kernel has no bound — so the gain
there is RAYCAST-2a's batched tendon FK, already merged.

## Identity, unchanged by any of this

| check | result |
|---|---|
| cpu, `PRIMITIVE_BOUNDS` on vs off, all 21 live configs at native sizes | **identical** (`replay/r5_cpu_bounds_identity.json`) |
| cpu, branch with the bound ON vs pristine baseline, all 21 | **identical** (`replay/r5_cpu_branch_vs_baseline.json`) |
| cuda, branch vs pristine baseline, all 21 | **identical** — the fused kernel has no per-kind table (`replay/r5_cuda_pixel_diff.json`) |
| `tests/gpu/test_fast_paths.py` at the envs' own sizes | **176 passed**, 33 skipped |
| `tests/unit/test_raycast3d_primitive_bounds.py` | **45 passed** (24 differential, 3 batch-invariance, 16 pinned kind mixes, 2 predicate) |
| `tests/unit` overall | **604 passed**, 28 skipped, 2 failed — the same two pre-existing failures |

The identity legs **force** the bound on. With the default at `False` a replay that took the default
would compare the unbounded path against itself and report `ALL_IDENTICAL` for the wrong reason — the
same shape of false green as the empty-directory `compare_dumps` bug this workstream fixed earlier.

## Public API: signatures, defaults and return shapes — not bodies

The CPU-3-D workstream builds on `raycast3d`'s public functions, so this is checked rather than
asserted. For `cast_nearest`, `any_blocking`, `shade`, `composite`, `_render_group`,
`cast_nearest_torch`, `any_blocking_torch`, `intersect` and `static_depth`, the **signatures,
defaults and return shapes are identical** to the pre-Step-6 baseline — the last guaranteed
independently by the 21-config bit-identical replays on both devices.

**Their bodies are not, and an earlier draft of this section said "byte-identical", which was wrong.**
`cast_nearest_torch` and `any_blocking_torch` changed in this ticket (the bound, its per-kind table
and the call-level short circuit) and `composite` changed in RAYCAST-2b (cover compaction). What a
caller can rely on is the interface and the pixels, which is the useful claim; "byte-identical" was a
stronger one that happens to be false and that nobody needs.
The only additions at module scope are switches and derived constants: `COVER_COMPACTION`,
`PRIMITIVE_BOUNDS`, `_BOUNDED_KINDS`, `_BOUND_SLACK_NOISE`, `_TILE_CACHE`, `_TILE_CACHE_MAX`,
`_UNIT_D_TOL`. Nothing was removed.

# Ticket text: OGBENCH-BAKE-1 — make the plate bake a pure function of (scene, style)

Written here at the coordinator's request. **Not authorised and not started**; this is the ticket, not
the work.

## The defect

`OGBenchMaze._bake` keeps one `mujoco.Renderer` per texture state forever and drives it
`update_scene → render → update_scene` (`ogbench.py:471-479`). The plate it produces is a
deterministic function of **how many such cycles that renderer has already seen**, not of the scene
and the style alone. Measured (job 6347171, gpu2105), eight consecutive bakes of one key:

| scene | as shipped | with a discarded warm-up render | without the trailing `update_scene` | with a fresh `Renderer` each bake |
|---|---|---|---|---|
| ant/teleport | `ABCAAADA` (**4 distinct**) | `AAAAAAAB` (2) | `AAAAAAAA` (1) | **`AAAAAA` (1)** |
| ant/giant | `AABABABA` (2) | `AAAAAAAA` (1) | `ABABABAB` (2) | **`AAAAAA` (1)** |
| humanoid/medium | `AAAAAAAA` (1) | `AAAAAAAA` (1) | `AAAAAAAA` (1) | **`AAAAAA` (1)** |

The sequences are byte-identical across two processes on two different code trees, and the teleport
key's first bake is the same digest on two nodes. Consequences already recorded above: two OGBench
cells fail Gate A on `check_style_is_state`, and `AntmazeTeleportExplore`'s frozen Step-4 digest is
not reproducible at the baseline.

## Why it is not a one-line fix

Only `fresh_renderer_per_bake` is constant on every probed scene, **and its plate equals the one the
cache already holds** (`digests[0]` matches the current first bake on all three), so in the common
case it would change nothing. Its cost is the problem:

* a `mujoco.Renderer` owns an EGL context, and the note above `OGBenchMaze` records that RELEASING
  one corrupts rendering in every other live renderer in the process — two fresh reference envs came
  back 44/255 apart, and a pool that evicted was how 25 cells once failed at 69-90/255. So they are
  never closed;
* the measured budget is ~96 live contexts (`stable_envs/testing/resource_guard.py` counts them, and
  the suites print the count);
* `check_style_is_state` alone bakes 64 appearances, and a certification session bakes hundreds.
  "A fresh Renderer per bake" therefore exhausts the budget in one config.

The two cheap variants are not fixes: `warmup_render` leaves ant/teleport flipping at the eighth
bake, and `no_trailing_update` produces the OTHER plate on ant/giant (`46ea1cfe42eb` where the first
bake is `27359fa34b1e`), i.e. it would move pixels on a configuration that is currently correct.

## What the ticket has to do, in order

1. **The EGL context budget is the first constraint, not an afterthought.** Establish what the
   process can hold on the certification hardware and what happens at the limit — `resource_guard`
   already counts, so this is a measurement, not a design.
2. **Find a reset that costs no context.** The candidates are: a `mjvScene` rebuilt per bake rather
   than reused; an explicit `mjr_setBuffer`/clear between bakes; or a renderer pool of fixed size
   whose members are cycled deterministically so that a given key always meets the same renderer
   state. The property to demonstrate is the one the probe measures: **N consecutive bakes of one key
   give one digest**, on ant/teleport and ant/giant, which are the two scenes that flip.
3. **Then check the plate it settles on.** The acceptance bar is the first-bake plate, because that is
   what the cache holds today and what 20 of the 21 live configs' frozen digests were taken from.
   Anything else moves pixels and needs its own authorisation.
4. **Replay all 11 OGBench configs** on cuda against the frozen digests, and re-run the
   `cache_pressure_probe.py` scenario and `check_style_is_state`. Expect `AntmazeTeleportExplore` to
   still fail against its frozen digest — its Step-4 plate is a third image that this session does not
   reach (job 6347171 probe (i)) — and to recommend it for the Step-8 re-freeze once the bake is
   deterministic.
5. **Only then** revisit the `check_style_is_state` routing proposal above. If the bake becomes pure,
   the proposal is moot for these configs and the probe's bake-path arm becomes a regression test
   rather than a router.

## What it must not do

Re-freeze any Step-4 digest as part of the fix; relax `check_style_is_state`'s `== 0`; or release an
EGL context to stay inside the budget.

## Containment already in place

`17f28c0` made `_SCENE_CACHE` evict the least recently used entry instead of clearing wholesale, so a
key in use is never dropped and cache pressure can no longer trigger the flip. The nondeterminism
itself is untouched and a cold bake still lands where its history puts it.


---

# Ticket text: RAYCAST-5 — the bounded cast must not be slower than the unbounded one

**DELIVERED — see the RAYCAST-5 section above for what was measured and built.** Kept verbatim
as written, because the ticket's own reasoning is what the work was judged against and editing
it after the fact would remove the only record of what was asked for. Written originally
because RAYCAST-4's attribution leg (job `6350679`) found the defect and RAYCAST-4 was not
authorised to fix it.

## The defect

`PRIMITIVE_BOUNDS = True` ships a CPU slowdown on the OGBench maze family: **0.92x on
`PointmazeLargeNavigate` and 0.75x on `PointmazeMediumNavigate`** at B=64, same branch, same node,
same job, switch as the only variable. `PointmazeLargeNavigate` carries a frozen Step-3 CPU row
(994.3 fps, `node2349`), so this is a regression on a certified path, not on a curiosity.

It is invisible in the baseline-vs-branch sweep, where 2b's cover compaction more than covers it and
the row reads 1.37x. **Any future "is this faster" question about a switch has to be asked with the
switch as the independent variable; the cumulative leg cannot answer it.**

## Why it happens

These scenes have **K = 1** — one mover primitive — and **N = 200,704 rays** (224 px at
`supersample=2`). The bounded path replaces one full-width intersector call with: a `(B,N)` mask, a
`torch.nonzero`, an advanced-indexing gather of `o[:, js]` and `d[:, js]`, a narrow intersector call,
and a scatter back into `blocked[:, js]`. The gather and scatter are O(N) and the saving they buy is
the narrowing of a single primitive's test. With K = 1 there is nothing to prune *between*; there is
only the gather. The DM Control rows (K = 2–36, N = 7,056) are the opposite regime and gain 1.2x–3.1x.

The same shape shows up weakly at B=1 on DM Control too (0.91x–0.94x on Cartpole, Pendulum and
Reacher), which says the predicate is not simply "K = 1".

## What the ticket has to do, in order

1. **Measure the predicate before choosing it.** Sweep the bounded path's cost against `K`, `N`, `B`
   and the selectivity `|js|/N` — a synthetic bundle is enough and is cheaper and cleaner than eight
   envs. The output is a surface, not a constant.
2. **Then choose a dispatch rule** that the surface supports, in the shape the code already uses:
   `_unit_rays` and the existing `if js.numel() == N: js = None` are both "take the unbounded path
   when the bound cannot pay", and this is a third one. Bit-identity is free for any such rule, since
   both paths are already proven identical — which means the ONLY question is performance, and the
   only risk is picking a constant that fits eight rows and nothing else.
3. **Re-run the isolation leg** (`scratch/step6/raycast3d/r4_isolate_cpu.sbatch`) and require no row
   below 1.00x, rather than requiring any particular row to improve.

## What it must not do

* Do not tune the constant against the eight rows in the table above. That is the move that put a
  far-field precision bug past two test suites earlier in this ticket: a threshold chosen to make
  the measured cases pass is not evidence about the unmeasured ones.
* Do not reach for the gather-free alternative (mask-and-`torch.where` instead of gather-and-scatter)
  without measuring it: it makes the intersector run at full width, which is the entire cost the
  bound exists to avoid, and it would help exactly the rows that are already fast.
* Do not touch the Triton path. It has no bound to switch off.

## Containment already in place

`PRIMITIVE_BOUNDS = False` restores `_cast_nearest_torch_unbounded` / `_any_blocking_torch_unbounded`,
which are the pre-RAYCAST-4 bodies line for line, so the regression is one assignment away from gone
on any deployment that cares more about the mazes than about `ball_in_cup`.


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

# Every SLURM job this workstream ran

Added at the coordinator's request after `6347000` and `6347137` turned out to be cited in prose
but absent from any list — they had written artifacts under this directory that nothing indexed.
The list is `sacct`, not memory, so a job that produced a number cannot be missing from it.
Cancelled and failed runs are kept: a measurement that had to be taken twice is part of how the
number that survived was arrived at.

| job | name | state | elapsed | node | what |
|---|---|---|---:|---|---|
| `6343478` | `se6-raycast3d-cpu` | FAILED | 00:00:50 | `node2342` | Ticket 1 CPU profile — FAILED (EGL/venv), superseded by 6343544 |
| `6343479` | `se6-raycast3d-gpu` | CANCELLED | 00:00:00 | `-` | Ticket 1 GPU profile — cancelled before start |
| `6343520` | `se6-raycast3d-census` | CANCELLED | 00:00:17 | `node2339` | backend census — cancelled |
| `6343543` | `se6-raycast3d-census` | COMPLETED | 00:00:58 | `node2339` | backend census: which 21 configs are live, and on which backend/device |
| `6343544` | `se6-raycast3d-cpu` | COMPLETED | 00:13:57 | `node2342` | Ticket 1 CPU phase profile, 5 envs |
| `6343545` | `se6-raycast3d-gpu` | CANCELLED | 00:00:00 | `-` | Ticket 1 GPU profile — cancelled before start |
| `6343664` | `se6-raycast3d-gpu` | CANCELLED | 00:06:51 | `gpu2101` | Ticket 1 GPU profile — cancelled (wrong card constraint) |
| `6344633` | `se6-raycast3d-style` | COMPLETED | 00:11:08 | `gpu2107` | check_style_is_state bisect, 12 runs per config |
| `6345115` | `se6-raycast3d-2c` | COMPLETED | 00:11:57 | `gpu2110` | RAYCAST-2c: _bake qpos reset, replay + bisect after |
| `6346224` | `se6-raycast3d-2a` | COMPLETED | 00:02:29 | `gpu2104` | RAYCAST-2a: batched tendon FK, GPU replay + BallInCup sweep |
| `6346414` | `se6-raycast3d-2bc` | COMPLETED | 00:03:22 | `gpu2104` | RAYCAST-2b: cover compaction, GPU replay |
| `6346429` | `se6-raycast3d-2bcpu` | COMPLETED | 00:12:18 | `node2337` | RAYCAST-2b: cover compaction, CPU sweep |
| `6346705` | `se6-raycast3d-merged` | COMPLETED | 00:02:50 | `gpu2113` | first merge verification (texture-fidelity) |
| `6346706` | `se6-raycast3d-cpusweep` | COMPLETED | 00:05:22 | `node2339` | RAYCAST-2 CPU sweep on the merged tree |
| `6346814` | `se6-raycast3d-merged` | COMPLETED | 00:02:46 | `gpu2104` | **final RAYCAST-2 verification on the twice-merged tree** |
| `6347000` | `se6-raycast3d-2cp` | COMPLETED | 00:09:30 | `gpu2106` | **RAYCAST-2c' probes (i) and (ii): renderer lifetime as the bake's carrier, cache pressure, host-rasterizer determinism** |
| `6347001` | `se6-raycast3d-maze2d` | COMPLETED | 00:00:20 | `node1829` | maze2d venv attempt 1 — inherited the BASE interpreter, no numpy |
| `6347071` | `se6-raycast3d-mb` | COMPLETED | 00:01:40 | `node1828` | RAYCAST-3 cast micro-benchmark |
| `6347125` | `se6-raycast3d-maze2d` | COMPLETED | 00:00:19 | `node1828` | maze2d venv attempt 2 — gymnasium_robotics needed pettingzoo |
| `6347137` | `se6-raycast3d-maze2d` | COMPLETED | 00:01:03 | `node1829` | **maze2d PointMaze verification in a fresh venv (RAYCAST-2c' item d)** |
| `6347142` | `se6-raycast3d-mb` | COMPLETED | 00:02:03 | `node1829` | RAYCAST-3 micro-benchmark, -march variant |
| `6347171` | `se6-raycast3d-bakelt` | COMPLETED | 00:00:28 | `gpu2105` | bake-lifetime probe |
| `6347404` | `se6-raycast3d-mb2` | COMPLETED | 00:01:20 | `node1829` | RAYCAST-3 micro-benchmark at 224 px |
| `6348207` | `se6-raycast3d-r4cpu` | CANCELLED | 00:10:35 | `node2337` | RAYCAST-4 CPU — **the run that moved one pixel of 3,211,264** and produced the far-field fix |
| `6348293` | `se6-raycast3d-r4gpu` | COMPLETED | 00:03:27 | `gpu2108` | RAYCAST-4 GPU, first pass |
| `6348760` | `se6-raycast3d-r4cpu` | COMPLETED | 00:22:41 | `node2337` | RAYCAST-4 CPU: identity on 21 + the 1-thread sweep, both legs |
| `6348761` | `se6-raycast3d-r4gpu` | COMPLETED | 00:03:16 | `gpu2113` | RAYCAST-4 GPU: cuda identity on 21, fast paths at native sizes, fuzz, no-regression sweep |
| `6350679` | `se6-raycast3d-r4iso` | COMPLETED | 00:13:40 | `node2333` | RAYCAST-4 attribution: the switch against itself — **found that the sweep could not attribute** |
| `6351415` | `se6-raycast3d-r4units` | COMPLETED | 00:01:53 | `node1827` | RAYCAST-4 tests/unit incl. batch invariance |
| `6352389` | `se6-raycast3d-r5cost` | COMPLETED | 00:13:05 | `node2339` | RAYCAST-5 **per-kind cost table** — the measurement the predicate is derived from |
| `6353078` | `se6-raycast3d-r5cpu` | COMPLETED | 00:20:59 | `node1831` | RAYCAST-5 CPU verification, per-kind table only |
| `6353080` | `se6-raycast3d-r5gpu` | COMPLETED | 00:02:07 | `gpu2101` | RAYCAST-5 GPU verification, per-kind table only |
| `6353140` | `se6-raycast3d-r5kinds` | COMPLETED | 00:00:51 | `node2333` | kind census attempt 1 — read the bound METHOD as the scene, reported no prims for all 21 |
| `6353212` | `se6-raycast3d-r5kinds` | COMPLETED | 00:01:05 | `node2335` | **kind census: what each of the 21 configs actually casts** |
| `6353649` | `se6-raycast3d-r5cpu` | COMPLETED | 00:20:22 | `node2333` | RAYCAST-5 CPU verification with the call-level short circuit |
| `6353650` | `se6-raycast3d-r5gpu` | COMPLETED | 00:02:01 | `gpu2264` | RAYCAST-5 GPU verification with the call-level short circuit |
| `6354150` | `se6-raycast3d-r5noise` | COMPLETED | 00:25:05 | `node1831` | RAYCAST-5 **noise floor**: four interleaved legs, off/on/off/on |
| `6354672` | `se6-raycast3d-r5shardcpu` | COMPLETED | 00:27:49 | `node2335` | RAYCAST-5 Step-3-shaped CPU shards, three legs |
| `6354674` | `se6-raycast3d-r5shardgpu` | COMPLETED | 00:01:05 | `gpu2105` | RAYCAST-5 Step-3-shaped GPU shards, two legs |
| `6355332` | `se6-raycast3d-r5finalcpu` | COMPLETED | 00:04:28 | `node2333` | **final identity with the default ON, neither leg pinning the switch**, plus `tests/unit` |
| `6355333` | `se6-raycast3d-r5finalgpu` | COMPLETED | 00:02:09 | `gpu2116` | final cuda identity with the default ON, plus the fast-path suite |
| `6356117` | `se6-raycast3d-r6cpu` | COMPLETED | 00:09:45 | `node2411` | review changes: the `_unit_rays` routing probe, the re-priced cost table, CPU identity on 21 (on-vs-off and vs baseline), `tests/unit` |
| `6356123` | `se6-raycast3d-r6gpu` | COMPLETED | 00:02:04 | `gpu2108` | review changes: cuda identity on 21, fast-path suite, fuzz |
| `6356311` | `se6-raycast3d-r6cost` | COMPLETED | 01:13:49 | `node1843` | **three independent runs of the per-kind cost table in the shape that ships** — the runs that retired the margin rule |


---
---

# RAYCAST-6 — the window `composite` casts is the caller's ceiling, not the bound's answer

`stable_envs/render/raycast3d.py`, `composite` only. Merged `step6/integration` at `5b327af` first,
so everything below is measured on the tree this branch will merge back into; the **before** leg of
every comparison is `$SCR/se6/integration`, NOT the pre-Step-6 baseline.

Commits: `6c954c5` implementation + differential fuzz + fast-path case; `24b3932` design
evidence and the profiler fix; `5f86a75` the re-pointed cover-compaction case; `39e994d` batch
invariance; plus the findings and shard commits listed in `git log`.

## The gap this ticket was given, and what re-measuring the residual changed

RAYCAST-2b left `PointmazeLargeNavigate` at `cores_cpu` 0.618 and `PointmazeMediumNavigate` at
0.472, needing **1.6x** and **2.1x** on the `cores_cpu >= 1` conjunct, and named the residual as
"`composite`'s own gathers/patch merge at ss=2: 25.3 ms of 42.6". That phrase was the ticket's
premise and it did not survive being opened. `composite`'s statements were timed one at a time
through the public path (`env.render(states)`) on a batch node at one thread —
`scratch/step6/raycast3d/r6_composite_profile.py`, job **6365780** (`node2348`, Intel Xeon Platinum
8268), a statement-for-statement copy of the shipped body with each block timed, so the numbers carry
~25 % instrumentation overhead and the PROPORTIONS are what they are for.

`PointmazeLargeNavigate`, B=64, 224 px, ss=2, `tile=32` (composite 53.13 ms of a 56.60 ms render):

| block | ms | share | sized by |
|---|---:|---:|---|
| `cast` | 16.19 | 30 % | **B x N** |
| `box_avg_col` (`view(B,tile,ss,tile,ss,3).sum((2,4))`) | 11.66 | 22 % | **B x N** |
| `shade` | 6.39 | 12 % | B x \|union\| |
| `window_gather` (`d[idx]`, per environment) | 5.47 | 10 % | **B x N** |
| `box_avg_cov` | 3.71 | 7 % | **B x N** |
| `material_gather` | 2.14 | 4 % | B x \|union\| |
| `col_scatter` (`where` + `index_copy_`) | 1.41 | 2.7 % | B x \|union\| |
| `plate_broadcast_write` (`out.copy_(plate.expand)`) | 1.10 | 2.1 % | B x H x W |
| `compact_nonzero` (zeros + `nonzero(cov.any(0))`) | 0.85 | 1.6 % | B x N |
| the **patch merge** (`oidx`, plate promote, gather, blend, round, scatter) | **2.48** | **4.7 %** | B x tile^2 |
| `tile_origin`, `upsample_t`, `cov`, `covf`, `setup`, `sizes` | 1.73 | 3.3 % | mixed |

**The patch merge is 4.7 % of the composite, not the residual.** What the residual actually is:
the terms proportional to `N = (tile*ss)^2` — cast, both box averages, the per-environment ray
gather, the coverage tensors — which are **38.4 ms, 72 %**. That is also the answer to "why ss=2
doubles them": it does not double them, it QUADRUPLES the sample count, because `N` is quadratic in
`ss`; the box averages then fold `ss^2` samples back into one output pixel, so they read `N` and
write `tile^2` and exist only at `ss > 1`.

And the samples are nearly all empty. At the same cell, each row covers **193 of 4,096 samples
(4.7 %)** and 56.5 of 1,024 output pixels.

## Two numbers the design turns on, both measured

**1. The caller's `tile` is a ceiling, and the bound asks for far less.**
`scratch/step6/raycast3d/r6_span_probe.py` intercepts `screen_span` and records the `(w, h)` it
returns, per config, over the frozen corpora (job **6365965**):

| config | span w x h | `tile` asked | N at 32 | N at the span | saving |
|---|---|---:|---:|---:|---:|
| `PointmazeLargeNavigate` | 14 x 13 | 32 | 4,096 | 784 | **5.2x** |
| `PointmazeMediumNavigate` | 21 x 21 | 32 | 4,096 | 1,764 | **2.3x** |
| `PointmazeGiantNavigate` | 11 x 10 | 32 | 4,096 | 484 | 8.5x |
| `PointmazeTeleportNavigate` | 14 x 13 | 32 | 4,096 | 784 | 5.2x |
| `AntmazeGiantNavigate` | 14 x 14 | 32 | 4,096 | 784 | 5.2x |
| `AntmazeLargeExplore` | 18 x 18 | 32 | 4,096 | 1,296 | 3.2x |
| `AntmazeTeleportExplore` | 18 x 18 | 32 | 4,096 | 1,296 | 3.2x |
| `AntmazeMediumExplore` | 28 x 28 | 32 | 4,096 | 3,136 | 1.3x |
| `HumanoidmazeGiantNavigate` | 11 x 9 | 32 | 4,096 | 484 | 8.5x |
| `HumanoidmazeLargeNavigate` | 12 x 12 | 32 | 4,096 | 576 | 7.1x |
| `HumanoidmazeMediumNavigate` | 19 x 20 | 32 | 4,096 | 1,600 | 2.6x |
| `maze2d` `PointMaze{Large,Medium,Open,UMaze}` | 18-30 | 48 | 9,216 | 1,296-3,600 | 2.6-7.1x |

`w_max` is constant across B=16/64/256 on every row, so this is a property of the scene rather than
of the batch. **A power-of-two ladder would have rounded `PointmazeMediumNavigate`'s 21 straight
back to 32**, on the row with the worst gap; the rung is therefore the span itself.

**2. `COVER_COMPACTION`'s union is four times the per-row cover.** The window origin tracks the
movers' projected bound, but the agent still drifts several pixels inside that window across a
batch, and the union is what gets shaded:

| config, B=64 | covered samples per row | union over the batch | ratio |
|---|---:|---:|---:|
| `PointmazeLargeNavigate` | 193 | 785 | **4.06x** |
| `PointmazeMediumNavigate` | 555 | 2,372 | **4.28x** |

## The change

`COMPOSITE_COMPACT_MERGE` (default on). Off restores `_composite_window`, the pre-RAYCAST-6 body
kept verbatim. Three mechanisms, each an identity rather than an approximation:

**(a) The window is sized to the bound.** `_fit_tile` returns `max(w, h)` capped by the caller's
`tile`, or None when the span does not fit — which is exactly `_tile_origin`'s existing
admissibility test, one rung finer. A ray outside the span cannot reach a mover, so `intersect`
returns `INF`, `cov` is False, `col` is `+0.0` and the composite writes back the plate byte
`out.copy_` already put there. **This is the argument the tiled path already makes against the full
frame**, and the repo already asserts it (`tests/gpu/test_fast_paths.py`'s
`mover-window-vs-full-frame`). The clamp is safe at any tile: with `t >= w` the window
`[clamp(x0,0,W-t), +t)` contains `[x0, x0+w) & [0, W)` at both edges.

**(b) The shaded set is per row, not the union.** `nonzero(cov.reshape(-1))` selects the covered
`(row, sample)` pairs — 4.1x fewer than `nonzero(cov.any(0))` — and the `torch.where` that discarded
the union's uncovered members disappears with them, because every selected element is covered by
construction. `shade` is elementwise in every argument (its only reduction is `_dot`, unrolled over
the three components for exactly this reason, and `lamb.sum()` over the lights, which does not
depend on ray shape), so shading a gathered subset returns the values the superset held there.

**(c) The patch merge runs at the covered output pixels.** At an output pixel with `covf == 0` the
blend is `patch * 1.0 + 0.0` on a float that came out of a uint8 — `round` returns it, `clamp(0,255)`
cannot move it — so the scatter would write back the byte the broadcast already wrote. Only
`nonzero((covf > 0).any(0))` needs gathering, blending, rounding and scattering. And the gather takes
the plate as the **uint8 it is** and promotes what it took, instead of promoting all 50,176 pixels of
the frame to float to use 1,024 of them.

Plus one term that does not shrink with the window and therefore had to go: `bt` is now read at the
window's output pixels and blown up, rather than `_upsample_t` building a `(H*ss*W*ss,)` intermediate
per call and being read at the window. Nearest-neighbour upsampling is pure data movement, so
`base_t[(y*ss+a)*Rw + (x*ss+b)] == plate_t[y*W + x]` exactly.

**What was deliberately NOT compacted: the two box averages.** A four-element float sum has an
association order, a compacted reduction is not obliged to reproduce it, and this function's contract
is bit-exactness rather than approximate agreement. They keep their spelling and shrink only because
`N` shrank. That is the one place where a cheaper option was available and refused, and it is worth
naming as such.

## Identity — the acceptance test, not a tolerance

Every leg below is `replay_step4.py` rendering the frozen corpora through the public path and
diffing the raw uint8 dumps, not the digests: a digest says "it moved", a dump says by how much and
where. **The before tree is `$SCR/se6/integration`**, whose `stable_envs/render/raycast3d.py` was
verified byte-identical to this branch's merge commit before the change was applied.

| check | device | configs | result | artifact |
|---|---|---|---|---|
| `COMPOSITE_COMPACT_MERGE` on vs off | cpu | 21 live | **identical** | `replay/r6cm_cpu_switch_identity.json` |
| branch vs integration, **defaults on both legs** | cpu | 21 live | **identical** | `replay/r6cm_cpu_branch_vs_integration.json` |
| branch vs integration, defaults | cuda | 21 live | **identical** | `replay/r6cm_cuda_identity.json` |
| on vs off | cuda | 21 live | **identical** | `replay/r6cm_cuda_switch_identity.json` |
| `maze2d` `PointMaze{Large,Medium,Open,UMaze}`, on vs off | cpu | 4 | **identical** | `replay/r6cm_maze2d_switch_identity.json` |
| `maze2d`, branch vs integration, defaults | cpu | 4 | **identical** | `replay/r6cm_maze2d_branch_vs_integration.json` |
| the four DMControl trackcom cells (`cpu3d.py`'s path) | cpu | 4 | **identical** | `replay/r6cm_trackcom_identity.json` |
| `tests/gpu/test_fast_paths.py` at the envs' own sizes | cuda | — | **223 passed**, 179 skipped | job 6366547 |
| `tests/unit/test_raycast3d_composite_compact.py` | cpu | — | **55 passed** (jobs 6366546, 6366548) -> **63 passed** (job 6366959, after the batch-invariance cases) -> **75 passed** (after the review's `bt`-gather and clamp cases, local) | jobs 6366546 / 6366548 / 6366959 |
| `tests/unit` | cpu | — | **794 passed**, 19 skipped, **2 failed** — the same two pre-existing failures, neither in the authorised file | job 6366959 |
| branch vs integration, defaults, **re-taken on the HEAD that ships** | cpu | 21 live | **identical** | `replay/r6fin_cpu_branch_vs_integration.json` |

The frozen-digest comparison is 20 of 21 on BOTH legs on cuda: `AntmazeTeleportExplore` fails on the
branch and on integration alike, which is RAYCAST-2c's unreproducible frozen digest and not this
change. On CPU it is 0 of 21 on both legs, because **every Step-4 digest was taken on cuda** and the
CPU path has never been frozen — so the CPU evidence here is entirely the dump diff, which is the
point of dumping rather than hashing.

**Neither leg of the default comparison pins the switch**, and the two kinds of leg are evidenced
differently — which an earlier draft of this section got wrong by claiming one mechanism for both.

* The **pinned** legs (`--compact-merge on|off`) print
  `COMPOSITE_COMPACT_MERGE=True/False in .../raycast3d.py` before they start, because
  `replay_step4.py` only reaches that print when a switch is pinned. A vacuous pinned run is
  therefore visible in the job log.
* The **default** legs print nothing, by construction: pinning is what triggers the print, so a leg
  that takes the default cannot announce itself that way. What identifies them is
  `stable_envs_from` in each per-config record of the replay JSON —
  `$SCR/se6/integration/stable_envs/__init__.py` on the before leg and
  `$SCR/se6/raycast3d/stable_envs/__init__.py` on the after leg. That is the field to check, and it
  is the field that proves the two legs imported different trees rather than the same one twice.

### The fuzz, and whether it has teeth

`tests/unit/test_raycast3d_composite_compact.py` — 55 cases: 13 pinned scenes (odd frame sizes at
ss=1 and ss=2, a window that does not fit and falls back to the full frame, `plate_t = INF`, a mover
behind the camera, batch 1 and batch 32, box and cylinder), 24 random ones, an explicit empty-cover
and full-cover pair, a caller-supplied buffer against a fresh one, per-environment colour and scale,
12 cases on `_fit_tile`'s containment property, and the switch-still-exists guard.

A differential test that passes on the first run is a claim about the test as much as about the code,
so three defects were injected and the suite re-run:

| injected defect | caught by |
|---|---|
| window one output pixel too small | `_fit_tile` containment property (11 failures); **NOT by any pixel comparison** |
| window three too small | 20 pixel comparisons |
| window five too small | 26 pixel comparisons |
| the patch merge compacted to `all(0)` instead of `any(0)` — i.e. per-batch instead of per-union | 24 pixel comparisons |
| the per-row material lookup replaced by row 0's | the per-environment-appearance case |

The first row is the useful one: **`screen_span` pads a pixel each way, so a one-pixel error is
invisible in the output and only a property test on the bound catches it.** That is why the
containment test exists beside the pixel diff rather than instead of it.

## The cost model, written down before the measurement and then checked against it

The model is arithmetic on the table above: window-sized terms scale by `N_new/N_old`, union-sized
terms by the per-row-to-union ratio, the patch merge by the covered-pixel fraction, and the rest is
unchanged. `PointmazeLargeNavigate` at B=64 (`composite` 45.67 ms before, `r6_composite_profile.py`
with both bodies instrumented identically, job **6366736**, `node2345`):

| group | before ms | factor | predicted ms | measured ms |
|---|---:|---|---:|---:|
| window-sized (cast, both box averages, ray gather, `cov`/`covf`, the zeros+`nonzero`) | 32.41 | x 784/4096 | 6.20 | 6.45 |
| union-sized (`shade`, material gather, `col` scatter) | 8.86 | / 4.06 | 2.18 | 2.51 |
| patch merge (`oidx`, promote, gather, blend, round, scatter) | 2.32 | x covered pixels | ~0.6 | 0.57 |
| unchanged (frame write, `tile_origin`, `bt`, setup) | 2.08 | x 1 | 2.08 | 1.88 |
| **total** | **45.67** | | **11.06** | **11.41** |

Within 3 %. `PointmazeMediumNavigate` the same way: 62.97 -> predicted 23.07, measured 24.18, within
5 %. The model is not a fit; it is the claim that these terms are proportional to the sample count,
and that claim is what the change acts on.

### Per-op, before and after, at B=64 (instrumented copies of both bodies, one job, one node)

| block | `PointmazeLarge` before -> after | `PointmazeMedium` before -> after |
|---|---|---|
| `cast` | 14.15 -> **2.85** | 14.21 -> **6.53** |
| `box_avg_col` | 9.44 -> **1.86** | 9.44 -> **4.03** |
| `shade` | 5.90 -> **1.57** | 17.64 -> **3.89** |
| `window_gather` | 4.57 -> **0.86** | 4.63 -> **1.83** |
| `box_avg_cov` | 2.98 -> 0.59 | 2.98 -> 1.30 |
| `material_gather` | 1.76 -> 0.81 | 5.03 -> 2.14 |
| `col_scatter` | 1.20 -> 0.13 | 3.41 -> 0.35 |
| `plate_broadcast_write` | 1.03 -> 1.00 | 1.00 -> 1.05 |
| patch merge (6 blocks) | 2.32 -> 0.57 | 2.28 -> 1.13 |
| `plate_promote` | 0.11 -> **0.00** | 0.10 -> **0.00** |
| **composite** | **45.67 -> 11.41** | **62.97 -> 24.18** |

The instrumented render, both bodies, all four envs the ticket names:

| config | B=16 | B=64 | B=256 | `tile` chosen | N |
|---|---|---|---|---|---|
| `PointmazeLargeNavigate` | 13.98 -> 5.35 (2.61x) | 47.22 -> 12.50 (**3.78x**) | 209.29 -> 46.56 (4.50x) | 14 | 784 |
| `PointmazeMediumNavigate` | 19.98 -> 8.67 (2.31x) | 64.95 -> 25.05 (**2.59x**) | 272.79 -> 97.29 (2.80x) | 21 | 1,764 |
| `AntmazeGiantNavigate` | 62.32 -> 30.16 (2.07x) | 280.98 -> 80.44 (**3.49x**) | 1279.16 -> 306.20 (4.18x) | 14 | 784 |
| `HumanoidmazeMediumNavigate` | 99.97 -> 62.70 (1.59x) | 534.45 -> 244.80 (**2.18x**) | 2478.31 -> 1000.89 (2.48x) | 18-20 | 1,296-1,600 |

**These are attribution, not the result.** Both columns carry the instrument's overhead and neither
is the shipped function; the ratio that goes in a report is the interleaved sweep below, which times
the real `composite`.

**The legged mazes were not supposed to gain and they did.** RAYCAST-2b measured `AntmazeMedium` and
`HumanoidmazeMedium` gaining 1.03x and 1.07x from cover compaction and concluded they were
cast-bound and out of this lever's reach. They still are cast-bound — `cast` is 254.8 of 286.0 ms on
`AntmazeGiantNavigate` — but the cast runs over the WINDOW, so narrowing the window narrows the cast
too. That is the part of the design that was not in the ticket's premise: shrinking `N` is not a
compositing optimisation, it is a ray-count optimisation that the compositing path happened to own.

### One thing this change RETIRES, and it had to be said out loud

`COVER_COMPACTION` no longer does anything on the shipped path. RAYCAST-6's composite compacts per
row, which is a strictly tighter set than the union that switch selects, so `composite` never reads
it; it now governs `_composite_window` alone. That is a live hazard rather than a tidy-up, because
`tests/gpu/test_fast_paths.py`'s `composite-cover-compaction` case flips exactly that flag: left
alone it would have flipped a switch the running code ignores, compared the compact path with itself
and gone **green for the wrong reason** — the same shape of false pass as the empty-directory
`compare_dumps` bug this workstream fixed in RAYCAST-2b. The case now pins
`COMPOSITE_COMPACT_MERGE` off for its duration, so RAYCAST-2b's claim keeps being tested on the path
where it still applies, and `test_every_declared_fast_path_is_actually_reachable` asserts both halves
of that arrangement — the pin in the case, and the `COVER_COMPACTION` branch inside
`_composite_window` — so a later edit cannot quietly remove either.

### What the trackcom check could and could not be

The brief asked for the four DMControl trackcom cells through `scratch/step6/cpu-3d`'s cross-backend
comparison script. **That script cannot run: its inputs are gone.** `compare_frozen.py` reads
`localize312/frames/<cfg>.npz` (`frozen_ours`, `frozen_official`, `frozen_seg`) from the
texture-fidelity workstream, and no such directory exists anywhere under `$SCR/se6` any more.

What it would have answered is a FIDELITY question — how the CPU backend compares with MJWarp and
with the official renderer — and the question this ticket owes is narrower: did this change move a
pixel on the path `cpu3d.py` uses. That is answered decisively by rendering the four cells on CPU
under `$SCR/se6/integration` and under this branch and diffing the raw frames, which is what was run:
**identical**, all four. `cpu3d.py` reaches `raycast3d` through `cast_nearest` and `any_blocking` and
does not call `composite` or `shade` at all, so identity there is the expected result and the check
is a control — but a control that had not been run would have been an assumption.

## GPU: no regression, and one reading that had to be taken twice

The fused Triton cast is untouched and `composite`'s arithmetic is unchanged, so the cuda digests
cannot move and do not (21 configs, identical, twice over). Throughput is the separate question, and
the first attempt at it got it wrong. Job **6366547** ran each leg ONCE per cell and returned
`AntmazeMediumExplore` at **0.53x** at B=4096 and `PointmazeMediumNavigate` at **1.39x** at B=1024 —
two numbers that cannot both describe a change that is bit-identical on both devices.

Job **6366608** re-took it INTERLEAVED, int/br/int/br on one card (gpu2104, RTX 3090) in one job, and
`int2/int1` is a ratio of two runs of identical code:

| env | B | noise (int2/int1) | signal (br/int) | reading |
|---|---:|---:|---:|---|
| `PointmazeLargeNavigate` | 64 | 1.00 | 0.993 | unchanged |
| `PointmazeLargeNavigate` | 256 | 1.02 | 0.975 | unchanged (inside noise) |
| `PointmazeLargeNavigate` | 1024 | 1.01 | 1.048 | slight gain |
| `PointmazeLargeNavigate` | 4096 | 1.01 | **1.387** | real gain |
| `PointmazeMediumNavigate` | 64 | 0.99 | 0.986 | unchanged |
| `PointmazeMediumNavigate` | 256 | 0.97 | 0.968 | unchanged (inside noise) |
| `PointmazeMediumNavigate` | 1024 | 0.99 | **1.350** | real gain |
| `PointmazeMediumNavigate` | 4096 | 1.00 | **1.688** | real gain |
| `AntmazeMediumExplore` | 64 | 0.98 | 0.995 | unchanged |
| `AntmazeMediumExplore` | 256 | 0.98 | 1.007 | unchanged |
| `AntmazeMediumExplore` | 1024 | 1.00 | 1.269 | gain |
| `AntmazeMediumExplore` | 4096 | **0.44** | 1.249 | **unreadable** |
| `HumanoidmazeMediumNavigate` | 64 | 0.99 | 0.981 | unchanged |
| `HumanoidmazeMediumNavigate` | 256 | **0.77** | 1.143 | unreadable |
| `HumanoidmazeMediumNavigate` | 1024 | **1.99** | 1.029 | unreadable |
| `HumanoidmazeMediumNavigate` | 4096 | 1.03 | 1.227 | gain |

**The 0.53x was the harness.** The same cell's off-vs-off ratio on identical code is 0.44x, and a
cell whose noise is 2.3x cannot answer a question about 2x. At the batches where the harness IS
stable the answer is: **unchanged at B=64 and B=256 (0.968-1.007), a real gain at the large
batches** where the frame write stops dominating — `PointmazeMediumNavigate` 1.69x at B=4096 against
a 1.00 noise floor. No cell shows a regression outside its own noise, which is the check this ticket
owed.

The single-pair readings from 6366547 are not published: `profile_phases.py` writes a directory, and
`--out` ending in `.json` produced six directories named `*.json` that the Step-6 report scanner
classifies as unaccounted artifacts. They were deleted rather than renamed, because the interleaved
run supersedes them and keeping a superseded reading that the page cannot classify is how a
misleading number survives.

## Public surface: checked, not asserted

`cpu-3d` builds on `raycast3d`'s public functions, so this is compared rather than claimed. Parsing
both trees and diffing the argument lists of `cast_nearest`, `any_blocking`, `shade`, `composite`,
`cast_nearest_torch`, `any_blocking_torch`, `intersect`, `static_depth`, `screen_span`,
`cached_rays`, `_expand_tile`, `_tile_origin` and `_upsample_t`: **every signature and default is
identical**, **no module-level name was removed**, and exactly two were added — `_composite_window`
and `_fit_tile`, both private. Return shapes are guaranteed by the 21-config bit-identical replays on
both devices. `shade` was not touched at all, and `cpu3d.py` reaches this module only through
`cast_nearest`, `any_blocking`, `Primitives.radii` and `INF` — it never enters `composite`.

The diff against `step6/integration` is **+182/-0 lines in one file**: a pure insertion, with not one
line of the old `composite` deleted or edited. That the off-path is verbatim is therefore visible in
`git diff --stat` rather than something to be taken on trust, and it was also checked directly by
extracting both bodies and comparing them character for character.

## The CPU sweep: interleaved on the real function, then a Step-3-shaped pair

Job **6366607**, `node2317`, `--partition=batch --qos=normal`, `OMP_NUM_THREADS=MKL_NUM_THREADS=1`,
`torch.set_num_threads(1)`, Intel Xeon Platinum 8268. Both parts in ONE job on ONE node.

**A second certification pair was taken afterwards on a quieter node** (job 6368126, `node1946`,
`sweep/r6_repeat/`) and it disagrees with this one about `PointmazeMediumNavigate`'s tier. Read the
two together in *The two readings, side by side* below; the numbers in this section are the first
reading and are not the whole answer on their own.

### Part 1 — four interleaved legs, off/on/off/on, on this branch

`off2/off1` is a ratio of two runs of IDENTICAL code, so it measures the harness and nothing else.
RAYCAST-5 put that at 11.7 % on a shared batch node, which is why a single before/after pair is not
evidence at this resolution.

| env | B | off1 | off2 | **noise** | on1 | on2 | **signal** |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PointmazeLargeNavigate` | 1 | 284.6 | 335.5 | 1.18 | 473.1 | 405.7 | 1.417 |
| `PointmazeLargeNavigate` | 16 | 1,082.8 | 1,279.0 | 1.18 | 3,532.8 | 3,465.3 | **2.963** |
| `PointmazeLargeNavigate` | 64 | 1,363.1 | 1,430.3 | 1.05 | 5,425.4 | 5,361.1 | **3.861** |
| `PointmazeLargeNavigate` | 256 | 1,313.8 | 1,279.2 | 0.97 | 5,578.0 | 5,537.0 | **4.287** |
| `PointmazeMediumNavigate` | 1 | 282.2 | 291.8 | 1.03 | 369.7 | 388.6 | 1.321 |
| `PointmazeMediumNavigate` | 16 | 982.0 | 974.5 | 0.99 | 2,116.1 | 2,173.8 | **2.193** |
| `PointmazeMediumNavigate` | 64 | 1,045.3 | 1,031.1 | 0.99 | 2,678.5 | 2,704.5 | **2.592** |
| `PointmazeMediumNavigate` | 256 | 974.8 | 964.5 | 0.99 | 2,588.9 | 2,424.0 | **2.585** |
| `AntmazeGiantNavigate` | 1 | 50.2 | 44.2 | 0.88 | 52.3 | 50.5 | 1.089 |
| `AntmazeGiantNavigate` | 16 | 254.1 | 240.8 | 0.95 | 481.5 | 438.9 | **1.859** |
| `AntmazeGiantNavigate` | 64 | 227.7 | 230.4 | 1.01 | 734.2 | 697.2 | **3.125** |
| `AntmazeGiantNavigate` | 256 | 197.3 | 199.3 | 1.01 | 800.4 | 765.7 | **3.948** |
| `HumanoidmazeMediumNavigate` | 1 | 33.8 | 33.6 | 0.99 | 38.9 | 40.3 | 1.175 |
| `HumanoidmazeMediumNavigate` | 16 | 138.0 | 145.5 | 1.05 | 266.1 | 272.4 | **1.899** |
| `HumanoidmazeMediumNavigate` | 64 | 111.0 | 120.7 | 1.09 | 218.9 | 225.9 | **1.919** |
| `HumanoidmazeMediumNavigate` | 256 | 98.3 | 99.0 | 1.01 | 240.0 | 243.0 | **2.448** |

**Worst off-vs-off spread on identical code, certification batches only: 1.181x.** Every signal at
every certification batch clears it by a wide margin; the narrowest is `HumanoidmazeMediumNavigate`
at B=64, 1.92x against 1.09x. B=1 is shown but does not gate — the frozen spec forbids certifying on
it (`gpu.never_certify_on_batch`).

### Part 2 — the Step-3-shaped pair, with a paired denominator

`examples/results/step6/raycast3d/sweep/r6/{cpu_1thread_before_integration,cpu_1thread_after_branch}.json`.

Produced by `scratch/step6/raycast3d/r6_certify_shard.py`, which runs
`scratch/step6/cpu-3d/certify_shard_cpu3d.py` unmodified and relabels the leg afterwards. That
producer rather than `examples/bench_step3.py` because `bench_step3.py` times the official renderer
ONCE, in a block, after the whole batch sweep, and
`examples/results/step6/gfxdraw/characterize/cpu_shard_attempts.json` is the record of what that does
on a shared node: a single block-timed reading moved **2.3x between the two arms of one job** on
`node2338` and took `S_cpu` far enough with it to relabel four configurations a tier up. The paired
form measures a reference burst between every batch cell and reports the median with its spread.

**The reference spread bound was 1.25x and the worst reading in either leg is 1.17x**
(`DMCReacherHard`, before). Nothing was rejected; had anything exceeded the bound the shard would
have been written and marked rather than published, which is the rule this workstream's own
11.7 %-noise measurement made necessary.

`node2317`, job **6366607**, 1 thread, Intel Xeon Platinum 8268. `S_cpu` = ours / the official
renderer; `cores_cpu` = ours / the one-core JPEG-q90 decode rate, both measured in the same leg on
the same host.

| env | fps before | fps after | x | S_cpu before -> after | **cores_cpu before -> after** | tier before -> after |
|---|---:|---:|---:|---|---|---|
| `PointmazeLargeNavigate` | 1,361 | **6,154** | **4.52** | 70.79 -> **314.45** | **0.663 -> 2.703** | `CPU_FASTER`+`BELOW_G4` -> **`CPU_FAST`** |
| `PointmazeMediumNavigate` | 969 | **2,732** | **2.82** | 37.23 -> **103.11** | **0.362 -> 1.207** | `CPU_FAST` **on this reading only** — the repeat below puts it at 0.925 and back in `CPU_FASTER`+`BELOW_G4` |
| `AntmazeGiantNavigate` | 250 | 804 | 3.21 | 21.18 -> 68.35 | 0.102 -> 0.314 | `CPU_FASTER`+`BELOW_G4`, unchanged |
| `HumanoidmazeMediumNavigate` | 137 | 274 | 2.00 | 8.26 -> 16.65 | 0.057 -> 0.102 | `CPU_FASTER`+`BELOW_G4`, unchanged |
| `DMCReacherHard` | 132 | 121 | 0.92 | 3.62 -> 3.40 | 0.018 -> 0.018 | control |
| `DMCFingerSpin` | 116 | 116 | 1.00 | 3.78 -> 3.70 | 0.017 -> 0.018 | control |
| `DMCCartpoleSwingup` | 562 | 638 | 1.13 | 15.38 -> 17.30 | 0.086 -> 0.099 | control |
| `DMCPendulumSwingup` | 214 | 210 | 0.98 | 5.64 -> 5.60 | 0.028 -> 0.033 | control |
| `DMCAcrobotSwingup` | 309 | 311 | 1.01 | 8.29 -> 8.28 | 0.042 -> 0.047 | control |
| `DMCBallInCupCatch` | 58 | 55 | 0.95 | 3.68 -> 3.61 | 0.009 -> 0.007 | control |

Labels for THIS pair are **`{'CPU_FASTER': 8, 'CPU_FAST': 2}`**, from `scratch/step3/certify.py`'s
`classify`
— the same function `report.py` labels the dashboard with — run over this pair out of tree. Not from
the sweep job: its `certify.py` step was invoked with a `--doc` flag copied from another workstream's
script, `certify.py` accepts no such flag, and the step **exited 2 without producing anything**. The
flag is gone from `r6_sweep.sbatch`, and so is the call: `certify.py` rewrites
`docs/research/step3_benchmarks.md` as a side effect, which is a Step-3 document this workstream must
not touch, so the job now reads the tier with `scratch/step6/raycast3d/r6_tiers.py` and the
certifier's label is taken separately and quoted above.

**The six DM Control rows are a same-code control and they behave like one.** They do not enter
`composite` at all, both legs run the identical build for them, and their before/after ratios are
0.92-1.13x — inside each shard's own reference spread. A shard pair that moved them would have been
measuring the node rather than the change, and this is the row-set that says it did not.

## `PointmazeLargeNavigate` is the first `CPU_FAST` row on this path. `PointmazeMediumNavigate` is not.

**This section retracts half of what an earlier draft of it claimed**, and the retraction is the
result of the one repeat shard the coordinator authorized. The first pair (`node2317`, job 6366607)
put both rows over `cores_cpu >= 1`. The second (`node1946`, job **6368126**) puts
`PointmazeMediumNavigate` under it.

### The two readings, side by side

| | first: `node2317`, job 6366607 | second: `node1946`, job 6368126 |
|---|---|---|
| node load at start | not recorded (the 23xx block was at 31-48 that afternoon) | **4.2 -> 4.8** |
| worst paired reference spread | 1.17x | **1.06x** |
| pre-flight load guard | none | `--max-load 12`, passed |
| `certifiable` | (no guard) | **true** |
| `PointmazeLargeNavigate` | 1,361 -> 6,154 f/s (**4.52x**), cores **2.703** | 1,180 -> 5,535 (**4.69x**), cores **2.152** |
| `PointmazeMediumNavigate` | 969 -> 2,732 f/s (**2.82x**), cores **1.207** | 908 -> 2,479 (**2.73x**), cores **0.925** |
| `certify.classify` | `CPU_FAST`, `CPU_FAST` | **`CPU_FAST`, `CPU_FASTER`+`BELOW_G4`** |

**The speed-up reproduces and the tier does not.** 4.52x against 4.69x and 2.82x against 2.73x is
two independent measurements of the same thing on two nodes. What moved is the DENOMINATOR: the
after-fps for `PointmazeMediumNavigate` differs by 10 % between nodes (2,732 vs 2,479) while the
one-core decode rate measured for it differs by 18 % (2,264 vs 2,680), and the bar is the decode
rate.

### `PointmazeMediumNavigate` straddles the bar

Its two certification-grade after-fps readings against the five one-core decode rates this workstream
has actually measured for that configuration:

| after fps | `n1946` after 2,680 | `n1946` before 2,430 | `n2317` after 2,264 | `n2317` before 2,678 | `n2335` (R5) 2,385 |
|---|---:|---:|---:|---:|---:|
| **2,732** (`node2317`) | 1.019 | 1.124 | **1.207** | 1.020 | 1.145 |
| **2,479** (`node1946`) | **0.925** | 1.020 | 1.095 | 0.926 | 1.039 |

**Ten combinations, range 0.925 to 1.207, and the bar runs through the middle of it.** Four are under
1. The lowest is the one where fps and decode were measured in the same leg of the better-conditioned
job — which is the combination with the strongest claim to being a single coherent measurement, and
it is the one that says no.

**So: `PointmazeMediumNavigate` is NOT established as `CPU_FAST`.** It is `CPU_FASTER` + `BELOW_G4`
with `cores_cpu` somewhere around 0.93-1.21 depending on the host, i.e. **at the bar rather than over
it**, having come from 0.36-0.37. Reporting it as `CPU_FAST` on the strength of the first reading
alone would have been reporting the node, and the repeat exists because the first draft nearly did.

### `PointmazeLargeNavigate` is not close to the bar and crosses on every combination

| after fps | `n1946` after 2,573 | `n1946` before 2,276 | `n2317` after 2,277 | `n2317` before 2,054 | `n2335` (R5) 2,361 |
|---|---:|---:|---:|---:|---:|
| **6,154** (`node2317`) | 2.392 | 2.704 | **2.703** | 2.996 | 2.607 |
| **5,535** (`node1946`) | **2.151** | 2.432 | 2.431 | 2.695 | 2.344 |

Range **2.151 to 2.996**, every combination over the bar by at least 2.1x, from 0.52-0.66 before.
`S_cpu` is 250-314 against a bar of 5. **`PointmazeLargeNavigate` is `CPU_FAST`, and it is the first
`CPU_FAST` row this renderer path has ever had** — Step 3 recorded `cores` 0.4159 for it and ticket 1
put the shortfall at 2.40x; RAYCAST-2b bought 1.75x and stopped at 0.63; RAYCAST-6 lands at 2.15-2.70
depending on the host.

### The two that never came close, and by how much

`AntmazeGiantNavigate` gains 3.21x and reaches `cores` 0.314 — still **3.2x short**.
`HumanoidmazeMediumNavigate` gains 2.00x and reaches 0.102 — **9.8x short**. Both stay `CPU_FASTER` +
`BELOW_G4`. They are cast-bound (`cast` is 90 % of an `AntmazeGiantNavigate` composite after the
change and 95 % of a `HumanoidmazeMediumNavigate` one, on 13 and 19 primitives against Pointmaze's
one), and the remaining lever for them is RAYCAST-3's compiled CPU cast, which was measured at 12x
and found necessary-but-not-sufficient there.

### What this says about the methodology, which is bigger than this ticket

Two certification shards, both taken by the producer with the interleaved denominator, both inside
their own reference-spread bound, on two nodes of the same partition, disagree about a TIER. Not
about the renderer — they agree on the speed-up to within 4 % — but about the bar, because
`cores_cpu` divides by a one-core JPEG decode rate that is itself a measurement on a shared node and
moved 18 % between them. That is a property of `cores_cpu` as a metric, not of this change, and it
will hit any row that lands near 1. **Recommended to the coordinator for Sami's closure packet: a
row within roughly 25 % of `cores_cpu = 1` should not be labelled from a single shard.** This ticket
is the second workstream to arrive at that conclusion from its own evidence, after `gfxdraw`'s
`cpu_shard_attempts.json`.

### Both r6 pairs stay live, and the dashboard is meant to flag them

`sweep/r6_repeat/` sorts after `sweep/r6/`, so `report.py` reads the tier from the repeat — the
better-conditioned measurement — and will mark **CPU after-legs disagree** on
`PointmazeMediumNavigate`. That flag is correct and wanted here. It would be trivial to suppress by
withdrawing one shard, and doing so would publish a tier the evidence does not support; the
withdrawal mechanism in this directory exists for shards describing a build that no longer exists,
not for shards that disagree with the answer one would prefer. `PointmazeLargeNavigate` agrees across
both and raises no flag.

### The RAYCAST-5 CPU shards are superseded, and moved rather than deleted

`sweep/cpu_1thread_{before_baseline,after_branch_bounds_on,branch_bounds_off_control}.json` describe
a build that no longer exists, and leaving them in place would have given `report.py` two after-legs
per row and made it print *CPU after-legs disagree* on exactly the two rows whose tier change is the
result. They are now `sweep/superseded_r5_node2335/`, with the `before`/`after` tokens removed from
their names — `leg_of` reads the leg out of the filename, so a move alone would not have been enough
— and a `README.md` giving the full comparison and the exact steps to reverse it (a
`README.md` rather than a `PROVENANCE_NOTE.md` so the report generator renders the
withdrawal reason rather than listing four unexplained files).
`r5_shards.sbatch`'s header now says it is superseded, because this workstream has already been
bitten once by a re-run re-creating a name it had deliberately changed. **No DM Control row changes
tier between the two shards**, which is what makes the replacement safe; the new pair also covers two
environments the old one did not.

**This is a judgment call and it is reversible.** If the coordinator would rather keep both pairs
live and accept the disagreement flag, the note says how.

## What is still open after RAYCAST-6

1. **`COVER_COMPACTION` is now dead code on the shipped path.** It governs `_composite_window` only.
   Removing it would delete the off-path the fast-path suite diffs against, so it should stay until
   someone decides `_composite_window` itself can go — which is a coordinator call, not mine.
2. **`AntmazeGiantNavigate` and `HumanoidmazeMediumNavigate` remain `BELOW_G4`** by 3.2x and 9.8x.
   They are cast-bound; the lever is RAYCAST-3's compiled CPU cast, still unimplemented and still
   blocked on the capsule-normal bit-exactness question (3-17 ulp on 0.06-0.35 % of components under
   gcc 11.5/x86-64, exact under Apple clang/arm64).
3. **`PointmazeMediumNavigate`'s crossing is 1.02 on the most conservative denominator available.**
   It is a real crossing on every rate measured, but it has no margin, and the decode rate is a
   property of the certification host.
4. **The RAYCAST-5 CPU shard move is a judgment call** (above), reversible, and flagged for the
   coordinator.
5. **`scratch/step6/cpu-3d/compare_frozen.py` cannot run any more** — the texture-fidelity frame
   stacks it reads (`localize312/frames/<cfg>.npz`) are not on the cluster. That is a gap in a
   different workstream's evidence chain, noticed here because this ticket was asked to use it.
6. Unchanged from before: the bake is still history-dependent (OGBENCH-BAKE-1, unauthorised);
   `AntmazeTeleportExplore`'s frozen digest is still unreproducible at the baseline and fails on both
   legs of every cuda replay; the two pre-existing `tests/unit` failures are still there and still
   not in any file this workstream owns.

## Every SLURM job RAYCAST-6 ran

All mine, all named `se6-raycast3d-*`. No job I did not submit was touched; one GPU job at a time.

| job | name | state | elapsed | node | what |
|---|---|---|---:|---|---|
| `6365780` | `se6-raycast3d-r6prof` | COMPLETED | 00:00:30 | `node2348` | the per-op attribution that replaced the ticket's premise |
| `6365965` | `se6-raycast3d-r6span` | COMPLETED | 00:01:21 | `node1828` | the window each caller actually needs, 11 OGBench + 4 maze2d |
| `6366546` | `se6-raycast3d-r6vcpu` | COMPLETED | 00:06:24 | `node2326` | CPU identity: switch on/off, branch vs integration, trackcom, tests |
| `6366547` | `se6-raycast3d-r6vgpu` | COMPLETED | 00:02:59 | `gpu2109` | cuda identity on 21 (both comparisons), fast-path suite, first GPU sweep |
| `6366548` | `se6-raycast3d-r6m2d` | COMPLETED | 00:00:52 | `node2326` | maze2d `PointMaze*` identity in the fresh venv |
| `6366607` | `se6-raycast3d-r6sweep` | COMPLETED | 00:43:43 | `node2317` | **the interleaved sweep and the Step-3-shaped pair** |
| `6366608` | `se6-raycast3d-r6gnr` | COMPLETED | 00:02:37 | `gpu2104` | **the interleaved GPU no-regression check** that retired the 0.53x reading |
| `6366640` | `se6-raycast3d-r6design` | COMPLETED | 00:02:30 | `node2345` | before/after attribution — **discarded**: the profiler's nested timers mis-attributed a block |
| `6366736` | `se6-raycast3d-r6design` | COMPLETED | 00:02:30 | `node2345` | the same, with the recorder fixed; this is the one quoted |
| `6366853` | `se6-raycast3d-r6gt` | COMPLETED | 00:02:41 | `gpu2116` | the fast-path suite after the cover-compaction case was re-pointed |
| `6366959` | `se6-raycast3d-r6fin` | COMPLETED | 00:04:28 | `node2349` | identity re-taken on the HEAD that ships, plus `tests/unit` |
| `6368126` | `se6-raycast3d-r6rpt` | COMPLETED | 00:01:49 | `node1946` | **the authorized repeat shard**, load-guarded, which retracted `PointmazeMediumNavigate`'s tier |

## Report

**Classification: `PATCH`.**

- baseline SHA `89fa5ca81341dc4df7186ec325cb691431a2ef17`; merged `step6/integration` at `5b327af`
- affected configs: the **11** `OGBenchMaze` configurations (35 registered names with siblings) and
  the **4** `maze2d` `PointMaze*`, which are the only callers of `R3.composite`. The 10
  `DMControlSuite` configurations and the 4 trackcom cells were replayed as controls and are
  unchanged.
- pixels changed: **no**, on either device, on any of the 21 live configs, the 4 maze2d envs or the
  4 trackcom cells.
- tiers: **`PointmazeLargeNavigate` `CPU_FASTER` + `BELOW_G4` -> `CPU_FAST`** — the first `CPU_FAST`
  row on this renderer path, and it crosses on all ten (fps, decode) combinations measured, range
  2.15-3.00. **`PointmazeMediumNavigate` is NOT claimed**: its two certification shards straddle the
  bar (`cores_cpu` 1.207 on `node2317`, 0.925 on the quieter `node1946`), so it stays `CPU_FASTER` +
  `BELOW_G4` at roughly 0.93-1.21, up from 0.36-0.37. `AntmazeGiantNavigate` 0.102 -> 0.314 and
  `HumanoidmazeMediumNavigate` 0.057 -> 0.102, both still `BELOW_G4`. No GPU tier moves.
- semantic follow-up recommendation: **none.** The frames are bit-identical, so no downstream metric
  can move.
