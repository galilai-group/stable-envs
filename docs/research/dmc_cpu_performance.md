# DMC CPU rendering investigation, 2026-09-29

The four selected regressions run the actual CPU analytic renderer against source
software OpenGL, with saved numeric State loading, source restoration and public
render-State projection excluded from both timers. The driver fixes all CPU thread
counts to one and records the allocated machine. No GPU substitution is involved.

## Measured work

Warmed batch-64 `cProfile` diagnostics from Oscar array **6805886**:

| Registration | Ours total, ms | Primitive bound tests, ms | Pose/FK preparation, ms |
|---|---:|---:|---:|
| swm/BallInCupDMControl-v0 | 5715 | 4185 | 10.62 |
| swm/FingerDMControl-v0 | 3973 | 2065 | 1.71 |
| swm/HumanoidDMControl-v0 | 8375 | 1987 | 1.65 |

These are instrumented cumulative wall times; nested entries must not be added.
Source's largest costs are `mjr_render` and `mjr_readPixels`. Our largest costs
are eager Torch cross products, dot products, primitive intersections and bounds.
Thus pose preparation is less than 0.2% of these renders. The bottleneck is CPU
ray work, not rolling out or uploading new States.

The fixed-camera renderer's `RAY_BUDGET = 2**24` was tuned for CUDA and also
controls its CPU render groups. Primitive bounds and shading use a union over
the group, so larger groups can expand sparse ray work and increase memory
traffic. The separate Humanoid CPU renderer uses groups of 16.

Array **6806007** tests smaller CPU work groups in isolated diagnostics, with no
production edit: fixed-camera budgets `2**18`, `2**19`, `2**20`, `2**21` versus
`2**24`, and Humanoid group sizes 1/2/4/8 versus 16. It compares complete batch-64
outputs against the original implementation and measures three warmed samples
for each setting. All complete outputs were byte-identical.

| Registration | Original B64 ms | Best tested B64 ms | Best setting | Improvement |
|---|---:|---:|---|---:|
| BallInCup | 5171 | 5171 | Original budget | 1.00x |
| Finger | 3620 | 3361 | `2**19` rays | 1.08x |
| Manipulator | 4033 | 3462 | `2**18` rays | 1.16x |
| Humanoid | 8423 | 5846 | 4 worlds | 1.44x |

Smaller chunks are not a universal fix: BallInCup at `2**18` rays regressed to
6702 ms. Humanoid sizes 1/2/8 took 7520/6412/6361 ms, respectively. Full samples
and standard deviations are retained in each `chunk_sweep.json`.

The bounded Humanoid candidate adds an optional internal `cpu3d.render`
`chunk_size` argument (default 16) and selects 4 only for Humanoid Walk. Other CPU
registrations and all CUDA paths keep their existing behavior. Array **6806273**
tested the production candidate against source and baseline on one CPU allocation
at B64/128/256. The terminal gate **passed**: same host/job and States, identical
baseline/candidate full-output hashes, and unchanged full State.

| Batch | Baseline FPS | Candidate FPS | Source CPU FPS | Candidate / source | Improvement |
|---|---:|---:|---:|---:|---:|
| 64 | 7.516 | 10.763 | 10.764 | 0.9999x | 1.432x |
| 128 | 7.559 | 10.966 | 10.742 | 1.0209x | 1.451x |
| 256 | 7.603 | 10.764 | 10.811 | 0.9956x | 1.416x |

The result is a practical tie with source CPU, not a clear source-relative win.
Normalized MSE is unchanged at **0.0004328633 +/- 0.0000732948** across the 16
independent frames. Per-batch timing samples and SD are in the raw results. Local
ruff/compile checks and the CPU3d import check passed. Oscar test **6806724_66**
passed all three current Humanoid lifecycle/projection/render-boundary tests and
invalid work-budget checks. The legacy CPU3d live tests require the removed
`DMCHumanoidStand` registration and would skip; they were not used as evidence.

Array **6806175** independently experiments with Inductor CPU fusion of the
unchanged `_rays_near` function. It disables unsafe floating point and contraction
per callable, compares every predicate during warmup and whole output bytes,
records compilation time and graph counts, and excludes compilation from timings.
It alternates baseline/candidate/source renders. All 12 B64/128/256 diagnostic
comparisons passed exact bound-predicate and whole-image checks, with one compiled
graph for Humanoid and three for each other registration, and no compilation
during timed samples. B256 improvements were
1.27x BallInCup, 1.36x Finger, 1.42x Manipulator, and 1.03x Humanoid, but source
ratios remained 0.916x, 0.851x, 0.812x, and 0.949x respectively. Fusion remains an
isolated harness experiment: these gains do not justify adding compiler/platform
and cold-start complexity in this checkpoint. No shared raycaster production
change has been made. Cold differential warmup costs and all attempted settings
are retained in the evidence instead of being included in steady timings.
The first fusion differential warmup took 46.7–49.8 seconds; this includes both
compilation and reference/candidate predicate checking, not just compilation.
SLURM recorded 4.07–5.14 GiB peak RSS for the complete fusion jobs (including
source contexts and compilation), versus 3.45 GiB for the complete Humanoid
baseline/candidate gate. These are whole-job peaks, not isolated kernel memory.
All jobs are terminal; the initial Manipulator strict-source-repeat failure is
the only failed attempt and its accepted-policy retry passed. No background
collectors remain.

## Evidence and limitations

Raw results, full profiles, source hashes, exact workload, and harness copies are
under `semantic_recovery_evidence/dmc_cpu_profile_20260929/`. The immutable
production baseline is Oscar `render-parity-df80c9c5-20260928/source-v12`.
`candidate_sha256.json` records the exact three local production files tested by
the Humanoid gate; local files were checked against those uploaded hashes after
the gate. `humanoid_chunk_gate/collection_status.json` records the terminal pass,
and its `comparison.csv` links each batch's baseline/candidate/source measurements.

The original Manipulator diagnostic failed the existing strict single-byte source
repeatability check. Its retry uses the already accepted `cpuretry2` protocol:
source repeat MSE at most 0.0001, measured repeated paired MSE outside timers, and
unchanged full-State capture hashes. Candidate render repeatability remains
strict. The failed attempt is retained and is not counted as a successful result.

The corpus contains 16 distinct saved poses repeated into the measured batches.
This is a bounded performance investigation, not 1,000 independent States of
new visual coverage. Existing accepted evidence is preserved.
