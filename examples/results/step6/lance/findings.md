# Step 6 / lance — the 8-worker JPEG-in-Lance baseline, re-run on Oscar

**Classification: FINDING.** Measurement and reporting only. No renderer code, no threshold, no
declaration and no frozen artifact was changed. Everything new lives under
`scratch/step6/lance/` and `examples/results/step6/lance/` on branch `step6/lance-baseline`.

- Baseline SHA: `89fa5ca81341dc4df7186ec325cb691431a2ef17` (verified ancestor of this branch)
- Branch parent: `step6/integration` @ `c975c8c`
- Branch HEAD: the tip of `step6/lance-baseline`; all artifacts and tables landed in `60d6430`,
  and this line was committed on top of it.
- Affected configs: the **36 `role == registered_default`** rows of
  `examples/results/step4/certification.json` — enumerated in §4.

---

## 0. What was actually run

| phase | what | where | job |
|---|---|---|---|
| 1 | re-dump 512 q95 JPEG frames per config at each env's **own certified size**, rendered by our GPU renderer | RTX 3090 | 6366652, 6366696, 6366739, 6366839 |
| 2a | `examples/bench_lance.py`, 8 workers, chunk 256, 8 reps, 3 storage arms + a 1/4/8 worker sweep | `node2317` Xeon 8268, 8 cores | 6366909 |
| 2b | the same, plus the nvJPEG (`local_gpu`) arm | `gpu2109` RTX 3090 | 6366910 |
| 2c | our CPU renderer at 1 thread and at 8 threads, in the same job as 2a | `node2317` | 6366909 |
| 2d | one-core batched decode fps vs JPEG quality (q95 / q90 / q75) on identical frames | `node2349` Xeon 8268 | 6366960 |
| 2e | post-Step-6 GPU renderer-only sweep, **all 36**, because 23 of the 36 GPU cells in the committed corpus were Step-3 numbers | `gpu2104` RTX 3090 | 6366975 |
| 3a | **instrumented** storage arms (cores actually used, read MB/s) at 1 and 8 workers, **plus** our CPU renderer at 1 thread / 8 threads / 8 single-thread processes — all in ONE job so the 8-worker Lance arm and the 8-way renderer arm share a node, a core allocation and a load | `node1956` Xeon 8268, 8 cores | 6367705 |
| 3b | instrumented storage arms + nvJPEG on a 3090; end-to-end `bench_step3_pipeline` for all 36 | `gpu2104` RTX 3090 | 6367697 |
| 4 | **stored JPEG → frames on the device** (8-worker DataLoader + pinned H2D, nvJPEG variant, H2D alone) **and** `bench_step3_pipeline` for all 36 — in ONE job, so both halves of `gpu_pipeline_ratio` come off the same card | `gpu2104` RTX 3090 | 6368790 |

---

## 1. Where the benchmark reads from on Oscar — and the one thing that invalidates the storage column

Recorded from **inside** the jobs (`stat -f`, `df -T`, `findmnt`, `/proc/mounts`,
`scontrol show node`), stored verbatim under `arms` in each result JSON.

| arm | path | filesystem as seen in the job | mount facts | 64 MB fsync write |
|---|---|---|---|---|
| `local` | `/tmp/se_lance_$SLURM_JOB_ID` | **`tmpfs`**, not a local disk | `rootfs[/slurm_tmpfs/<jobid>/.<jobid>/_tmp] /tmp tmpfs rw,relatime,mode=755,inode64` | 2,540 MB/s (batch) · 2,368 MB/s (GPU node) |
| `shared` | `/oscar/scratch/sboughan/...` | `nfs` | `hpcnfs.ccv.brown.edu:/oscar`, `vers=3, proto=rdma, nconnect=8, rsize=wsize=1048576, hard, port=20049` | 621 MB/s (batch) · 735 MB/s (GPU node) |
| `jobtmp` | `/sss/jobtmp/$USER/$SLURM_JOB_ID/se_lance` | `gpfs` | `jobtmp /sss/jobtmp gpfs rw,relatime` | 1,138 MB/s (batch) · 2,095 MB/s (GPU node) |

**Finding S1 — Oscar has no node-local disk arm.** `scontrol show node` reports `TmpDisk=0`, and the
reason is not that SLURM is misconfigured: `/tmp` inside a job is a **per-job `tmpfs`**, bind-mounted
from `/slurm_tmpfs/<jobid>/`. It is RAM. The brief's expectation of "local xfs on `/dev/md2`" does not
hold from inside a job — whatever the node's block devices are, the job cannot see them at `/tmp`.
So the `local_*` arm on Oscar is a **memory** arm, and it is the strongest possible reading of the
stored baseline, not a disk reading. The two real filesystems available to a job are NFS v3 over RDMA
and job-local GPFS; both are reported as their own labelled arms.

**Finding S2 — `bench_lance.py` cannot separate filesystems, by construction.** `build_table` writes
the Lance table from an in-memory blob list microseconds before `time_path` reads it, in the same
process. Every arm is therefore page-cache warm on its first timed read, and the arms differ by a
fraction of a percent (§2). The committed Randall run has the same property — its `local_1` vs
`shared_1` gap of 2,752 vs 2,681 f/s is 2.6%, which is noise of the same order. The honest statement
is that **after the first pass this benchmark measures the JPEG decoder, not storage**, which is what
its own `read_frac` column says (1–2% on the committed run). Any claim about cold NFS or cold GPFS
needs a table larger than node RAM or a cache drop, and neither is available to an unprivileged job.
The per-arm `write_64MB_MBps` probe above is the only real filesystem signal in these runs, and it is
deliberately outside every timed region.

---

## 2. The stored baseline on Oscar — results

Run: `examples/bench_lance.py`'s own `build_table` / `time_path` / `time_parallel`, imported verbatim,
chunk **256**, **8 reps**, **q95**, 36 configs at **their own certified sizes**, on
`node2317` (Xeon Platinum 8268 @2.9 GHz, 8 of 48 cores, affinity `36-43`, node load 34.7 → 40.5),
job **6366909**.

### 2.1 The three storage arms are indistinguishable

| arm | 1 process, median f/s | 8 workers, median f/s | ratio to tmpfs (1 proc / 8 w) | `read_frac` median |
|---|---|---|---|---|
| `local` — per-job **tmpfs** | **2,285** (1,093–13,042) | **13,953** (5,552–103,579) | 1.000 / 1.000 | 3.80% (1.88–12.41%) |
| `shared` — **NFS v3 / RDMA** | 2,276 (1,092–11,091) | 14,152 (5,328–101,459) | 0.986 (0.784–1.017) / 1.000 (0.876–1.161) | 4.28% (2.15–15.70%) |
| `jobtmp` — **GPFS** | 2,282 (1,095–12,721) | 14,310 (5,232–104,287) | 0.998 (0.828–1.023) / 1.004 (0.901–1.174) | 3.65% (1.95–15.09%) |

The one place the filesystem *is* visible is the read term taken on its own. The instrumented run
records `read_fps`, `read_frac`, `read_MB/s`, `decode_MB/s` and the host-side output rate for every
arm (e.g. `DMCReacherHard`, one process: tmpfs 222,629 read f/s = 535 MB/s of JPEG; GPFS 216,994 =
521 MB/s; **NFS 141,037 = 339 MB/s**, a genuine 1.6× penalty). It just does not matter: the read is
5.4% of the chunk on tmpfs and 8.3% on NFS, so a 1.6× slower read moves the end-to-end number by
under 3%.

Within-arm spread across envs is three orders of magnitude; between-arm spread is under 2% at the
median. The read term is 2–16% of the chunk and everything else is `decode_jpeg`. Combined with
Finding S2 (§1) the conclusion is unambiguous: **on Oscar, at this working-set size, the storage
column of this benchmark is a decoder column.** Reporting the arms separately is still worth doing —
it is what shows they do not differ — but no row here should be quoted as "NFS costs X".

### 2.2 The baseline scales almost linearly to 8 cores

Median over the 36 configs, on the local arm: **1 → 4 workers = 3.88×** (3.56–4.34), **1 → 8 workers
= 7.72×** (6.64–8.40). The instrumented re-run (§2.5) confirms this with CPU accounting rather than
inference: the eight workers burn **7.4–7.9 cores** of CPU time over the slowest worker's elapsed
time. "8 workers" really is eight cores of work, and the aggregate really is ~7.7× one worker.

### 2.3 `local_1` and `local_n1` are not the same number

`local_1` (`time_path`, best-of-8) reads ~25% above `local_n1` (one pool worker, mean-of-8) at the
median — 2,285 vs 1,827. Same work, different reduction (asymmetry A6). Where a one-process figure is
quoted below, both are given.

### 2.4 The q95 → q90 gap is worth ~4%, not ~22%

Step 2's frozen `cores` metric and the `decode` block of every committed shard use **q90**; Lance
and `stable_worldmodel` write **q95**. Measured directly on identical frames, one thread, one
process (job **6366960**): q90 decodes **1.040×** faster at the median (range 0.809–1.126) while
being **0.777×** the bytes (0.755–0.939); q75 is 1.095× (0.849–1.326) at 0.513× the bytes. So the
byte column badly overstates the quality asymmetry: as a *decode-throughput* term it is worth about
4%.

### 2.5 What the eight workers actually consume

The instrumented re-run (job **6367705**, `scratch/step6/lance/lance_arms2.py`, which samples
`getrusage(RUSAGE_SELF)` immediately outside each worker's clock and leaves the timed body
byte-identical to `bench_lance._worker`) answers the "is this really 8 cores" question directly:
`sum(worker CPU seconds) / slowest-worker elapsed` is **7.86 cores** at the median, range
**7.41–7.94** over the 36 configs, i.e. **93–99% of the 8 cores the job holds**. The 8-worker
aggregate is a genuine 8-core number, and `local_n / 8` is a fair per-core share.

---

## 3. Equivalence: is the state→render path measured the same way as the stored path?

### 3.1 What `bench_lance.py` times

Read off the code (`examples/bench_lance.py`, imported **verbatim** by
`scratch/step6/lance/run_lance_oscar.py` — none of its timing functions were edited):

`time_path` (the `*_1` arms and `local_gpu`), per 256-row chunk:

1. `tbl.to_lance().take(range(s, s+chunk), columns=["pixels"])` → `.to_pylist()` — a Lance point
   lookup of 256 contiguous row ids on one `pa.binary` column, materialised as 256 Python `bytes`;
2. `[torch.frombuffer(b, uint8) …]` → **one** `torchvision.io.decode_jpeg(ts, mode=RGB)` →
   `torch.stack` — libjpeg-turbo, SIMD, GIL released;
3. `torch.cuda.synchronize()` only on the nvJPEG arm.

The reported value is the **minimum** per-frame time over `reps` starting offsets — a best case.
Output: a freshly allocated `(256, 3, H, W)` uint8 tensor, **NCHW**, in **host** memory.

`time_parallel` (the `*_n` arms): 8 `spawn` processes, each `torch.set_num_threads(1)`, each opening
its own `lancedb` handle, each doing **one untimed warm pass** before timing `reps` chunks. Aggregate
is `total frames / max(worker elapsed)`. Process spawn and the eight `import torch` are outside the
timed region. The per-worker value is a **mean** over that worker's chunks.

I verified against `stable_worldmodel` on Oscar (`$SCR/swm-repo/stable_worldmodel/data/formats/lance.py`)
that this is the real hot path: `_DEFAULT_JPEG_QUALITY = 95`, `_encode_frame` is PIL `save(format='JPEG',
quality=…)`, `_decode_images` is exactly `[torch.frombuffer…] → decode_jpeg(…, mode=RGB) → torch.stack`.
The benchmark is a faithful reproduction, not an approximation.

### 3.2 What the renderer certification rows time

`docs/research/certification.md` §6.1, renderer microbenchmark: *batched states already in device
memory → renderer → frames*; no DataLoader, no `default_collate`, no host transfer inside the timed
loop. `examples/bench_gpu_vs_cpu.py::gpu_throughput` implements exactly that — states are
`torch.as_tensor(…, device=DEVICE)` before the loop, the output buffer
`torch.empty((B,H,W,3), uint8, device=DEVICE)` is **preallocated and reused** via `out=`, five warm-up
renders precede timing, and the value is the **median** of `reps` synchronised calls. Output: uint8
**NHWC**, on device.

CPU rows (§5.1, §5.3): same loop with `DEVICE="cpu"`, `torch.set_num_threads(1)` and
`OMP_NUM_THREADS=MKL_NUM_THREADS=1`; host states in, host frames out. §5.3 fixes **1 thread** as the
certification budget and the all-core figure as a scaling result reported beside it, never substituted.

The end-to-end pipeline question has its own table (`examples/bench_step3_pipeline.py`,
`examples/results/step3/pipeline/3090_pipeline.json`). On the 3090, over PushT / DMCReacherHard /
DMCHumanoidStand at B ∈ {256, 1024, 4096}, the two batched-indexing arms (`batch_sampler`,
`collate_batched`) cost `loader_frac` median **2.2%** (0.6–14.8%) and `transfer_frac` median **0.44%**
(0.04–1.33%) — the "1–7%" of the brief, up to ~15% at the worst cell. `default_collate` costs
5.5–63.6% and is the idiom the repo documents as the one not to use. **That measured overhead is for
moving *states*, which are tens of bytes per row. It is not the cost of moving frames.**

### 3.3 Verdict

**They are not equivalent, and §6.1 of the certification doc explicitly forbids quoting a row from one
table as if it came from the other.** The comparison is still the right one to make — it is the repo's
own G4 economics test (`docs/research/when_does_this_pay_off.md` §G4: *"regeneration faster than
decode, using a renderer that meets the family's parity bar"*, and that document's own note that
"G4 is stated here against one core and in the paper against eight workers, and the two bars disagree
about several families"). But every ratio in §4 must be read with the following list attached.

### 3.4 The asymmetries, enumerated and sized

| # | asymmetry | which way it biases the comparison | measured size on these rows |
|---|---|---|---|
| A1 | **Frame size.** The committed Randall run requested 256 px; the Step-6 rows are certified at each env's own size. | either way, and it is the single largest term | DMC 256² vs certified **84²** (9.3× fewer pixels); PushT 256² vs **96²** (7.1×); Fetch 256² vs **480²** (3.5× *more*); PointmazeLargeNavigate 256² vs **224²**. Only the 8 fixed-size configs coincide. **This re-dump removes the asymmetry**; §5 quantifies what it was worth. |
| A2 | **JPEG quality.** `cores` in Step 2/3 and the `decode` column of every sweep are one core of **q90**; Lance and `stable_worldmodel` write **q95**. | quoting the q90 bar flatters the stored path | measured directly (§2.4): q90 decodes **1.04× median** faster than q95 (range 0.81–1.13) while being **0.78× the bytes**. So the quality term is worth ~4% in fps, not the 20%+ the byte column suggests. |
| A3 | **Core budget.** `local_n` is 8 cores; the certified CPU number is **1 thread** (§5.3). | 8× in the stored path's favour unless divided out | the report gives both `local_1` and `local_n/8`, and an 8-thread supplemental renderer row taken in the same job. |
| A4 | **Output location.** Lance frames land in **host** memory (NCHW) and still need an H2D copy to train on a GPU; our GPU frames land **on device** (NHWC) in a reused buffer. | in the stored path's favour | not charged to Lance anywhere. A 256×3×84×84 uint8 batch is 5.4 MB; at 480² it is 177 MB per 256-frame batch. The Step-3 pipeline table's 0.04–1.33% transfer term does **not** cover this — it is the transfer of states. |
| A5 | **Allocation.** `decode_jpeg` allocates a new stacked tensor per chunk; `gpu_throughput` renders into a preallocated `out=`. | in our renderer's favour | unmeasured here. |
| A6 | **Reduction statistic.** `time_path` reports the **min** over reps; `gpu_throughput` reports the **median**; `time_parallel` reports a **mean bounded by the slowest worker**. | `local_1` flattered vs `local_n`, and both vs our median | visible as `local_n1 ≉ local_1` in §2.3. |
| A7 | **Cache state.** Every Lance arm is page-cache warm because the table is written in-process moments before (§1, S2); the renderer has no analogous warm/cold axis. | in the stored path's favour | `read_frac` 1–2%; arms within a fraction of a percent of each other. |
| A8 | **Hardware.** Randall: Xeon Platinum 8558 (32 CPUs allocated) + H200. Oscar batch: Xeon Platinum 8268 @2.9 GHz, 8 cores of 48, node load ~35–43. Oscar GPU node: AMD EPYC 7532 @2.4 GHz + RTX 3090. | either way | §5 isolates it on the 8 configs whose frames are identical in both runs. |
| A9 | **Node contention.** Our 8 cores are cgroup-pinned (`affinity 36-43`) on a node running at load ~35 with 44/48 CPUs allocated to other jobs; L3 and memory bandwidth are shared. The Randall run held 32 CPUs at load 11.5. | against Oscar | recorded per run as `loadavg` / `loadavg_end`. |
| A10 | **Spawn cost.** `time_parallel` excludes 8 process spawns and 8 `import torch`; a renderer pays nothing analogous. | in the stored path's favour, deliberately | the benchmark's own docstring defends this as once-per-epoch. Accepted. |
| A11 | **States are free in both.** Neither side charges for producing the state to render, nor for the episode index. | neutral | — |

---

## 4. The comparison table

Full per-row tables: **`examples/results/step6/lance/comparison2.md`** (T1 storage arms, T2 CPU,
T3 GPU, T4 the frozen one-core column, T5 per-cell provenance, T6 aggregates, T7 win/lose lists),
machine-readable in `comparison2.json`. Regenerate with `scratch/step6/lance/run_compare.sh gpu2104_6368790`.

Two break-even columns are carried side by side throughout and **neither is substituted for the
other**:

1. **frozen, conservative, in-memory, per-core** — `cores_cpu` / `cores` exactly as
   `certification.md` §5.1/§6.3 define them: our fps over ONE core of **q90** in-memory decode of
   this env's own frame, read out of the committed shards' own `decode` block. Reproduced unchanged
   (T4).
2. **realistic, stored-dataset, training-pipeline** — our fps over the **8-worker JPEG-in-Lance
   pipeline** measured here at q95 and at the certified resolution, with the GPU side compared
   end-to-end so **both sides finish with frames on the device** (T2, T3).

### 4.1 Headline aggregates over the 36 rows

| quantity | median | min | max | n |
|---|---|---|---|---|
| Lance 1 worker (tmpfs) | 1,687 | 683 | 12,169 | 36 |
| Lance 8 workers (tmpfs) | **13,177** | 5,039 | 92,201 | 36 |
| Lance 8 workers, cores actually burned | **7.86** | 7.41 | 7.94 | 36 |
| Lance 8 workers (NFS) | 14,152 | 5,328 | 101,459 | 36 |
| Lance 8 workers (GPFS) | 14,310 | 5,232 | 104,287 | 36 |
| stored JPEG → **device**, 8-worker loader + pinned H2D | **8,100** | 3,578 | 54,839 | 36 |
| stored JPEG → **device**, nvJPEG | 5,830 | 2,653 | 14,920 | 36 |
| H2D alone, pinned, uint8 CHW batch of 256 | 98,481 | 34,850 | 1,086,276 | 36 |
| ours CPU, 1 thread (certified budget) | 498 | 53 | 103,130 | 24 |
| ours CPU, best 8-way | 2,611 | 387 | 203,349 | 24 |
| ours CPU, 8-way scaling efficiency | 0.91 | 0.30 | 1.18 | 24 |
| ours GPU, renderer-only | 40,606 | 5,124 | 1,345,537 | 36 |
| ours GPU, end-to-end states → device | 40,032 | 5,126 | 840,567 | 36 |
| **`cpu_pipeline_ratio`** (best 8-way / Lance 8 workers) | **0.11** | 0.01 | 6.96 | 24 |
| **`gpu_pipeline_ratio`** (ours end-to-end / JPEG→device 8 workers) | **1.44** | 0.18 | 46.21 | 36 |
| `gpu_pipeline_ratio` vs nvJPEG | 3.42 | 0.72 | 124.30 | 36 |
| **`cores_cpu`** (frozen, one core of q90) | **0.20** | 0.01 | 18.89 | 24 |
| **`cores`** (frozen, GPU vs one core of q90) | **6.44** | 1.26 | 246.46 | 36 |

### 4.2 Who wins against the production 8-worker pipeline

**CPU — 1 of 24 measurable rows wins.** Only `CraftaxClassic` (`cpu_pipeline_ratio` **6.96**) beats
the 8-worker stored pipeline. The other 23 lose, by a median factor of ~9 (`cpu_pipeline_ratio`
median 0.11). The same single row is the only one that clears the frozen one-core bar
(`cores_cpu ≥ 1`), and the only one that beats a single Lance worker. The 12 Fetch configs have
**no CPU backend at all** (`mjwarp` only) and are reported as n/a, not as losses.

**GPU — 30 of 36 win end-to-end.** `gpu_pipeline_ratio` median **1.44**.

- **Large-frame 2-D (gfxdraw + raster2d + tilemap + maze):** 33.7–46.2× (`Pendulum` 46.2,
  `CartPole` 44.2, `Acrobot` 40.8, `MountainCar` 37.0, `MountainCarContinuous` 35.6,
  `CraftaxClassic` 33.7), then `PointmazeLargeNavigate` 14.4, `PushT` 13.8, `SimplePointMaze` 10.8,
  `Piecewise` 3.6, `PiecewiseZones` 3.2, `TwoRoom` 2.9, `TwoRoomTarget` 1.7.
- **Fetch (480², mjwarp):** 1.31–1.47× on all twelve. It wins, but narrowly.
- **DMC at 84² (mjwarp visual tier):** `DMCHopperHop` 2.11, `DMCCheetahRun` 2.08, `DMCWalkerWalk`
  1.91, `DMCHumanoidStand` 1.65 — the four mjwarp rows win.
- **DMC at 84² (raycast3d parity tier): six rows LOSE** — `DMCCartpoleBalance` 1.02 and
  `DMCCartpoleSwingup` 0.97 are a coin flip, and `DMCPendulumSwingup` 0.84, `DMCAcrobotSwingup`
  0.68, `DMCFingerTurnHard` 0.56, `DMCReacherHard` 0.47, `DMCBallInCupCatch` 0.18 lose outright.

**Against nvJPEG, 35 of 36 win** (only `DMCBallInCupCatch`, at 0.72, does not). On a 3090 with 8
CPU cores, nvJPEG is *slower* than the 8-worker CPU loader on 25 of 36 rows (`loader8/nvjpeg`
median 1.42): nvJPEG's per-image launch overhead dominates at these frame sizes, and it also has to
move the compressed bytes H2D first.

### 4.3 The three facts that decide the whole table

1. **Frame size, not hardware.** The committed Randall run requested 256 px. Eleven configs refused
   it and fell back to native, so their frames are the same picture in both runs — and for eight of
   those the dump reproduces Randall's `jpeg_bytes_total` **to the byte** (Acrobot 5,443,034;
   CartPole 4,145,024; CraftaxClassic 5,525,459; Piecewise 1,641,925; PiecewiseZones 2,510,318;
   SimplePointMaze 6,219,968; TwoRoom 2,247,734; TwoRoomTarget 2,473,292). On those eleven rows
   Oscar/Randall `local_n` is median **1.05×** (0.92–1.32). **That is the entire hardware term.**
   Everything beyond it is the picture: the DMC rows are **6.1× faster to decode at 84² than at
   256²** (median 6.15 over the ten raycast3d/mjwarp DMC rows), which is why the committed table's
   "DMC beats the stored path on GPU" reverses at the certified size. Fetch moves the other way:
   480² instead of 256² makes the stored baseline **~3× slower**, which is most of why Fetch now
   wins.
2. **Our CPU backends do not scale by threads; they scale by processes.** `procs8` beat `threads8`
   on **24/24** configs, median **1.53×**, up to **5.41×** (DMCCheetahRun: 1 thread 196 → 8 threads
   212 → 8 processes 1,146). Intra-op parallelism is nearly useless in these renderers; process
   parallelism gives median 0.91 scaling efficiency against 8× one thread. This is a structural
   finding about the CPU backends and it is reported, per Step 2 §5.3, strictly as a supplemental
   all-core number.
3. **Storage is not the variable.** §1–§2: all three arms agree within ~2% at the median, and
   `/tmp` inside an Oscar job is a per-job `tmpfs`, i.e. RAM.

### 4.4 Cells that are quoted, not re-measured

Everything in T2/T3 was measured for this report except the frozen `decode_q90_fps_per_core` column
of T4, which comes from the committed shards. In particular the **GPU renderer-only column was
re-measured post-Step-6 for all 36 rows** (job 6366975, one 3090, one commit) because 23 of the 36
GPU cells in the committed corpus were Step-3 numbers; the difference is material —
`PointmazeLargeNavigate` 95,962 → **196,982**, `DMCCheetahRun` 76,589 → **94,570**,
`CraftaxClassic` 1,206,822 → **1,345,537**.

---

## 5. Oscar vs the committed Randall-cluster run

| | Randall (`examples/results/storage/lance.json`) | Oscar (this work) |
|---|---|---|
| node | `gb-west-a1-gpu-114`, Xeon Platinum 8558, 32 CPUs, load 11.5 | `node2317`/`node1956`, Xeon Platinum 8268 @2.9 GHz, **8** of 48 cores, load 33–47 |
| GPU | H200 | RTX 3090 (gpu2104/gpu2109), 8 CPUs of an EPYC 7532, load 23–55 |
| frames | `--size 256` requested; native only where refused | **each env's own certified size**, always |
| configs | 195 | the 36 `registered_default` rows |
| `local_1` median | 2,752 | 2,285 (36-row subset; see below for the like-for-like rows) |
| `local_n` median | 16,984 | 13,177 |

**Like-for-like (same picture in both runs), 11 configs:** Oscar/Randall `local_n` median **1.05×**,
range 0.92–1.32. Eight of those eleven produce **byte-identical** JPEG totals across the two
clusters, two interpreters and two renderer generations — which is also an independent check that
the Step-6 gfxdraw / raster2d / tilemap work did not move those pixels.
`MountainCar`/`MountainCarContinuous` match in size but not in bytes (13,068 vs 12,882 B/frame,
+1.4%): their `sample_states` draw differs because this dump seeds `np.random.default_rng(0)`
explicitly while the original took the env's own generator. **Not** evidence of a pixel change, but
it is not proof of the absence of one either — the clean way to settle it is a paired dump from
`$SCR/se6/baseline`, which is left as a follow-up (the attempt in job 6366910 is unusable: a stale
`/tmp/inspect.py` on the login node shadowed the stdlib for the checker, and the in-job dump was not
re-attempted).

**Not like-for-like, 25 configs:** ratio median 1.11, range 0.29–6.88, and every unit of that spread
is frame size, not machine. Ten DMC rows: 84² vs 256² → Oscar 5.8–6.9× faster. Twelve Fetch rows:
480² vs 256² → Oscar **0.29–0.34×**, i.e. the stored baseline is ~3× *slower* on the real frame.
PushT: 96² vs 256² → 3.28×. PointmazeLargeNavigate: 224² vs 256² → 1.11×.

**So: hardware is worth 1.05×; the 256 px request was worth up to 6.9× in one direction and 3.4× in
the other.** Any conclusion drawn from the committed table about a config whose `size_note` is
`null` (i.e. one that accepted 256) is about a frame that config does not render.

---

## 6. Provenance — SLURM jobs, nodes, interpreters

| job | what | node | interpreter | outcome |
|---|---|---|---|---|
| 6366652 | frame dump, first attempt | gpu2105 (RTX 3090) | `$SCR/venvs/se-step5` | **failed** after 1 env: `np.savez` appends `.npz` to the staging base name, so the atomic rename looked for a file that did not exist. Fixed in `dump_frames_step6.py`. |
| 6366696 | frame dump, 24 non-Fetch configs | gpu2109 (RTX 3090) | `se-step5` | 20/24 dumped; MountainCar / MountainCarContinuous / Pendulum skipped (`No module named 'pygame'`), PointmazeLargeNavigate skipped (`No module named 'ogbench'`) — `se-step5` carries neither. |
| 6366739 | retry of those 4 in `se-step5` | gpu2104 | `se-step5` | reproduced the same four skips; this is what identified the interpreter gap. |
| 6366839 | the remaining 16 | gpu2116 (RTX 3090) | `se6-verify` (4 configs) + `se-s3b-fetch311` (12 Fetch) | **all 36 configs now dumped.** `se6-verify` is the one 3.12 interpreter carrying pygame + ogbench + mujoco_warp + dm_control + lancedb simultaneously. |
| 6366909 | Lance bench + our CPU renderer at 1 and 8 threads, one job | node2317 — Xeon Platinum 8268 @2.9 GHz, 48 cores, 8 allocated (affinity 36–43), node load ≈35 | `se6-verify` | see §2, §4 |
| 6366910 | Lance bench + nvJPEG arm | gpu2109 — AMD EPYC 7532 @2.4 GHz + RTX 3090 (driver 575.57.08), 8 CPUs, node load ≈23 | `se6-verify` | see §2, §4 |
| 6366960 | JPEG-quality probe (q95/q90/q75 on identical frames, 1 thread) | node2349 — Xeon Platinum 8268, node load ≈42 | `se6-verify` | `q90/q95` fps ratio median **1.040** (0.809–1.126); `q90/q95` bytes ratio median **0.777** |
| 6366975 | post-Step-6 GPU renderer-only sweep, all 36 | gpu2104 (RTX 3090), `STABLE_ENVS_GIT_COMMIT=93c81d7` | `se6-verify` + `se-s3b-fetch311` | 24 + 12 envs, 0 blocked |
| 6367705 | instrumented storage arms (1 and 8 workers, CPU accounting, read MB/s) **+** our CPU renderer at 1 thread / 8 threads / 8 processes, one job | node1956 — Xeon Platinum 8268, 8 of 48 cores, node load 33 | `se6-verify` | the CPU half of §4; `cores used (8w)` 7.41–7.94 |
| 6367697 | instrumented storage arms + nvJPEG; `bench_step3_pipeline` for all 36 | gpu2104 (RTX 3090), EPYC 7532, 8 CPUs, node load 55 | `se6-verify` + `se-s3b-fetch311` | pipeline complete; its `bench_jpeg_to_gpu` stage failed on the DataLoader fetcher bug (§7.3) |
| 6367822 | re-run of the JPEG→device arm | — | — | **cancelled** and folded into 6368790 so both halves of `gpu_pipeline_ratio` share a job |
| 6368790 | **stored JPEG → frames on device** (8-worker DataLoader + pinned H2D, unpinned variant, nvJPEG, H2D alone) **and** `bench_step3_pipeline` for all 36 — one job, one card | gpu2104 (RTX 3090) | `se6-verify` + `se-s3b-fetch311` | the GPU half of §4 |

Frame-dump provenance is written into every `.npz` (`meta.provenance`) and into
`_provenance_<node>_<jobid>.json` next to them: renderer, git commit, interpreter path, torch/mujoco/
mujoco_warp versions, GPU name, SLURM job id, seed, size policy, render chunk.

---

## 7. Blockers, defects found, and limits of this measurement

**Nothing blocked the task.** Four things cost time and are worth recording:

1. `np.savez` appends `.npz` to the base name it is given, so the first dump's atomic rename looked
   for a file that did not exist (job 6366652). Fixed.
2. `$SCR/venvs/se-step5` has **no `pygame` and no `ogbench`**, so MountainCar / MountainCarContinuous
   / Pendulum / PointmazeLargeNavigate cannot be rendered there (jobs 6366696, 6366739).
   **`$SCR/venvs/se6-verify` is the only 3.12 interpreter on Oscar that carries pygame + ogbench +
   pymunk + dm_control + mujoco_warp + lancedb + torchvision together**; it should be the default
   for anything that has to touch the whole non-Fetch registry.
3. With `batch_size=None`, torch's `_MapDatasetFetcher` hands the **whole index list to
   `__getitem__`**, not to `__getitems__` (job 6367697). Fixed in `bench_jpeg_to_gpu.py`.
4. A 512-frame table holds exactly two 256-row batches, so a naive DataLoader epoch would have left
   six of eight workers idle and produced a "2-worker" number labelled 8. The loader is given
   `bench_lance`'s own schedule instead (8 × 8 chunks = 16,384 frames), so both sides of
   `gpu_pipeline_ratio` cover the same frames.

**Limits of what was measured — read these before quoting anything above.**

- **No cold storage anywhere.** §1 S2: the table is written in-process immediately before it is
  read, on every arm, in both the committed run and this one. Nothing here measures a cold NFS or
  cold GPFS read, and an unprivileged job cannot drop caches. The per-arm `write_64MB_MBps` probe is
  the only genuine filesystem signal.
- **512 frames per config.** A 16 MB table is not a corpus. Lance's own indexing, fragmentation and
  multi-file behaviour at TB scale is untested here, and so is any cache-miss regime.
- **The GPU-side denominator is CPU-bound on a weaker CPU.** `loader8_h2d` was measured on gpu2104 —
  8 cores of an AMD EPYC 7532 @2.4 GHz at node load 54 — while the CPU-side Lance arms were measured
  on a Xeon 8268 @2.9 GHz. The stored path is decoder-bound, so `gpu_pipeline_ratio` is *flattering
  to our renderer* by roughly the ratio of those two CPUs. A GPU node with 8 Xeon cores would narrow
  it; the six losing DMC rows would lose by more.
- **Node contention.** Every Oscar job here held 8 cores of a 48- or 64-core node at load 23–55.
  Recorded per run as `loadavg` / `loadavg_end`; not controlled for.
- **The 8-way CPU number is supplemental, per Step 2 §5.3**, and is not a certification value.
- **`DMCCheetahRun` / `DMCHopperHop` / `DMCWalkerWalk` are visual-tier cells** (and carry the Step-4
  Gate-A `FAIL` that `texture-fidelity` diagnosed as MuJoCo 3.11→3.12 reference drift, not a
  defect). Their wins above are wins at the visual tier, not parity claims.
- **The 3.12 dump arm.** Everything except Fetch was rendered under mujoco 3.12 / mujoco_warp 3.12;
  the declared DMC reference pair is 3.11. That affects the *pictures'* fidelity claim, not the byte
  counts that drive decode cost, but the frames are 3.12 frames and are labelled as such in
  `meta.provenance`.

---

## 8. Recommendations to the coordinator

1. **Stop quoting `examples/results/storage/lance.json` for any config whose `size_note` is `null`.**
   That run measured a 256 px frame for 25 of these 36 configs, and 25 of 36 do not render at
   256 px. `examples/results/step6/lance/` supersedes it for the registered/default set. A note in
   the committed file pointing at this directory would prevent the mistake being repeated.
2. **The GPU story is sound and now has an honest denominator: 30/36 win end-to-end, median 1.44×,
   35/36 against nvJPEG.** The claim that should be made is the end-to-end one (`gpu_pipeline_ratio`),
   not renderer-only-vs-host-memory-Lance, which is 19/36 and compares different end states.
3. **The six losing raycast3d DMC rows are the actionable Step-6 finding.** `DMCBallInCupCatch`
   (0.18), `DMCReacherHard` (0.47), `DMCFingerTurnHard` (0.56), `DMCAcrobotSwingup` (0.68),
   `DMCPendulumSwingup` (0.84), `DMCCartpoleSwingup` (0.97) do not pay for themselves against a
   stored corpus on a GPU at 84². They are the same envs whose CPU rows are the slowest in the set.
   Whatever is left of the raycast3d per-chunk Python loop is where the remaining headroom is.
4. **Recommend a raycast3d/cpu-3d workstream item: process-parallel, not thread-parallel.**
   `procs8` beats `threads8` on 24/24 configs, median 1.53× and up to 5.41×. Whatever these backends
   do under `torch.set_num_threads(8)`, they do not use the cores. If an all-core CPU story is ever
   wanted, it is a multiprocess story today.
5. **CPU regeneration does not beat a stored corpus for 23 of 24 measurable rows**, on either bar —
   the frozen one-core q90 bar (`cores_cpu` median 0.20) and the realistic 8-worker bar
   (`cpu_pipeline_ratio` median 0.11) agree. Only `CraftaxClassic` clears both. This is consistent
   with `when_does_this_pay_off.md` §G4's own note that the one-core and eight-worker bars disagree
   for several families; here they agree, which makes it a firmer statement than usual.
6. **Fetch has no CPU backend**, so its twelve rows are n/a on CPU, not failures. On GPU they win
   end-to-end by 1.31–1.47× — the narrowest win in the table, and the row most likely to flip if the
   stored side were measured on a Xeon GPU node.
7. **Follow-up worth one small job:** a paired dump from `$SCR/se6/baseline` to date the
   MountainCar/MountainCarContinuous byte difference (§5), and a `--frames 4096` run of the storage
   arms to check whether any of this survives a working set that does not fit in page cache.

---

## 9. Report metadata

- Classification: **FINDING**
- Baseline SHA: `89fa5ca81341dc4df7186ec325cb691431a2ef17` (verified ancestor)
- Branch: `step6/lance-baseline` (artifact commit `60d6430`), forked from `step6/integration` @ `c975c8c`
- Renderer code changed: **none**. Thresholds changed: **none**. Frozen artifacts touched: **none**.
- Step-4 replay: **not run** — no renderer change was made, so there is nothing to replay. Frame
  provenance is instead checked by byte-identity against the committed Randall dump on the eleven
  like-for-like configs (§4.3, §5).
- Tests run: none (measurement-only workstream).
- Pixels changed: **no**.
- Semantic follow-up recommended: **none**.
- SLURM jobs: 6366652, 6366696, 6366739, 6366839, 6366909, 6366910, 6366960, 6366975, 6367697,
  6367705, 6367822 (cancelled), 6368790.
- Artifacts: `examples/results/step6/lance/` — `comparison2.{md,json}` (the deliverable),
  `comparison.{md,json}` (the first-round view), `lance_arms2_*`, `lance_cpu_*`, `lance_gpu_*`,
  `jpeg_to_gpu_*`, `pipeline_gpu_r4_*`, `renderer_cpu_*`, `renderer_gpu_after_*`, `quality_probe_*`,
  `renderer_rows.json`, `findings.md`.
- Harnesses: `scratch/step6/lance/` — `dump_frames_step6.py`, `run_lance_oscar.py`,
  `lance_arms2.py`, `bench_cpu_parallel.py`, `bench_jpeg_to_gpu.py`, `quality_probe.py`,
  `harvest_renderer_rows.py`, `compare.py`, `compare2.py`, `run_compare.sh`, and the `.sbatch`
  files that ran them.
- Frames on Oscar: `$SCR/se6/lance_frames/` (36 × 512 q95 JPEG frames + provenance JSON).

---

## 10. The 36 affected configs, enumerated

| config | backends | Step-4 tier | certified frame | Randall's frame | kB/frame (q95) | `cpu_pipeline_ratio` | `gpu_pipeline_ratio` |
|---|---|---|---|---|---|---|---|
| Acrobot | gfxdraw | parity | 500x500 | 500x500 | 10.6 | 0.34 | 40.84 |
| CartPole | gfxdraw | parity | 400x600 | 400x600 | 8.1 | 0.34 | 44.17 |
| CraftaxClassic | (env-specific) | parity | 112x144 | 112x144 | 10.8 | 6.96 | 33.68 |
| DMCAcrobotSwingup | mjwarp,raycast3d | parity | 84x84 | 256x256 | 3.1 | 0.03 | 0.68 |
| DMCBallInCupCatch | mjwarp,raycast3d | parity | 84x84 | 256x256 | 3.0 | 0.01 | 0.18 |
| DMCCartpoleBalance | mjwarp,raycast3d | parity | 84x84 | 256x256 | 2.9 | 0.05 | 1.02 |
| DMCCartpoleSwingup | mjwarp,raycast3d | parity | 84x84 | 256x256 | 3.0 | 0.04 | 0.97 |
| DMCCheetahRun | mjwarp,raycast3d | visual | 84x84 | 256x256 | 4.6 | 0.02 | 2.08 |
| DMCFingerTurnHard | mjwarp,raycast3d | parity | 84x84 | 256x256 | 3.7 | 0.01 | 0.56 |
| DMCHopperHop | mjwarp,raycast3d | visual | 84x84 | 256x256 | 4.4 | 0.02 | 2.11 |
| DMCHumanoidStand | mjwarp,raycast3d | visual | 84x84 | 256x256 | 3.8 | 0.01 | 1.65 |
| DMCPendulumSwingup | mjwarp,raycast3d | parity | 84x84 | 256x256 | 2.5 | 0.02 | 0.84 |
| DMCReacherHard | mjwarp,raycast3d | parity | 84x84 | 256x256 | 2.4 | 0.01 | 0.47 |
| DMCWalkerWalk | mjwarp,raycast3d | visual | 84x84 | 256x256 | 4.3 | 0.01 | 1.91 |
| FetchPickAndPlace | mjwarp | parity | 480x480 | 256x256 | 25.9 | n/a | 1.34 |
| FetchPickAndPlaceDense | mjwarp | parity | 480x480 | 256x256 | 25.9 | n/a | 1.31 |
| FetchPickAndPlaceDict | mjwarp | parity | 480x480 | 256x256 | 25.9 | n/a | 1.38 |
| FetchPush | mjwarp | parity | 480x480 | 256x256 | 25.9 | n/a | 1.41 |
| FetchPushDense | mjwarp | parity | 480x480 | 256x256 | 25.9 | n/a | 1.43 |
| FetchPushDict | mjwarp | parity | 480x480 | 256x256 | 25.9 | n/a | 1.39 |
| FetchReach | mjwarp | parity | 480x480 | 256x256 | 25.5 | n/a | 1.40 |
| FetchReachDense | mjwarp | parity | 480x480 | 256x256 | 25.5 | n/a | 1.39 |
| FetchReachDict | mjwarp | parity | 480x480 | 256x256 | 25.5 | n/a | 1.37 |
| FetchSlide | mjwarp | parity | 480x480 | 256x256 | 26.7 | n/a | 1.39 |
| FetchSlideDense | mjwarp | parity | 480x480 | 256x256 | 26.7 | n/a | 1.42 |
| FetchSlideDict | mjwarp | parity | 480x480 | 256x256 | 26.7 | n/a | 1.44 |
| MountainCar | gfxdraw | parity | 400x600 | 400x600 | 13.1 | 0.44 | 37.00 |
| MountainCarContinuous | gfxdraw | parity | 400x600 | 400x600 | 13.1 | 0.41 | 35.57 |
| Pendulum | gfxdraw | parity | 500x500 | 500x500 | 8.6 | 0.45 | 46.21 |
| Piecewise | (env-specific) | parity | 224x224 | 224x224 | 3.2 | 0.28 | 3.59 |
| PiecewiseZones | (env-specific) | parity | 224x224 | 224x224 | 4.9 | 0.18 | 3.15 |
| PointmazeLargeNavigate | raycast3d | parity | 224x224 | 256x256 | 12.3 | 0.47 | 14.42 |
| PushT | scene | parity | 96x96 | 256x256 | 2.7 | 0.08 | 13.81 |
| SimplePointMaze | scene | parity | 500x500 | 500x500 | 12.1 | 0.29 | 10.78 |
| TwoRoom | (env-specific) | parity | 224x224 | 224x224 | 4.4 | 0.32 | 2.87 |
| TwoRoomTarget | (env-specific) | parity | 224x224 | 224x224 | 4.8 | 0.14 | 1.73 |

`cpu_pipeline_ratio` is `n/a` on the twelve Fetch rows because Fetch has **no CPU backend** (mjwarp only) — that is an absence of a renderer, not a slow one.

