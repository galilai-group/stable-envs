# Randall’s matched-State renderer table

Measured on Brown Oscar, 27 September 2026. This is a descriptive measurement campaign at `636bc5863af9a7cb599cce14ece57a605cc5a254` on `repair/environment-semantics`, worktree `/private/tmp/stable-envs-semantic-recovery`. The matrix remains **39 VERIFIED / 3 INCOMPLETE**. No renderer, simulator, shared contract, threshold, implementation map, or accepted evidence was changed. No push occurred.

The table has exactly 42 registration rows and 1488 matched source/candidate trajectory checkpoints collected on CPU. `VERIFIED` is the existing semantic status; these measurements make no acceptance decision. The two PushT rows have no legitimate matched-State corpus. The two symbolic Craftax registrations have no RGB public renderer. All other rows retain actual paired image measurements. GPU quality uses only checkpoints whose numeric State matches the CPU oracle; its exact count is retained in each metric aggregate.

Latency columns are **median warm milliseconds, batch size 1**. MSE columns are **mean / p95 normalized RGB MSE across matched frames**. CPU and GPU MSE stay separate because some renderers produce different frames. Exact CPU/GPU image equality is recorded per row where the whole corpus supports it. `UNSUPPORTED` means the public backend explicitly rejects rendering; `NONCOMPARABLE` means the required State/configuration comparison is unavailable; `NOT IMPLEMENTED` identifies the symbolic registrations’ explicit no-RGB API. None is a synthetic zero.

## Complete table

| Registration | Status | Matched frames | Resolution H×W | CPU MSE vs original CPU (mean / p95) | GPU MSE vs original CPU (mean / p95) | Ours CPU ms | Ours GPU ms | Original CPU ms | Notes |
|---|---|---|---|---|---|---|---|---|---|
| swm/AcrobotControl-v1 | VERIFIED | 24 | 500×500 | 2.18743e-05 / 9.581147e-05 | 2.18743e-05 / 9.581147e-05 | 18.613 | 11.379 | 2.0034 | CPU/GPU RGB exactly equal |
| swm/AcrobotDMControl-v0 | VERIFIED | 24 | 224×224 | 2.899241e-05 / 3.828124e-05 | 3.163895e-05 / 4.101122e-05 | 33.159 | 9.3806 | 38.318 | CPU/GPU RGB differ; MSE kept separate |
| swm/BallInCupDMControl-v0 | VERIFIED | 24 | 224×224 | 0.005639608 / 0.03181913 | 0.005529164 / 0.03215059 | 258.76 | 8.9493 | 66.276 | CPU/GPU RGB differ; MSE kept separate |
| swm/CartPoleControl-v1 | VERIFIED | 22 | 400×600 | 6.380929e-06 / 3.429318e-05 | 6.380929e-06 / 3.429318e-05 | 17.442 | 10.9 | 2.2963 | CPU/GPU RGB exactly equal |
| swm/CartpoleDMControl-v0 | VERIFIED | 24 | 224×224 | 3.395507e-05 / 5.283943e-05 | 8.123011e-05 / 0.0001007562 | 26.756 | 10.392 | 41.321 | CPU/GPU RGB differ; MSE kept separate |
| swm/CheetahDMControl-v0 | VERIFIED | 24 | 224×224 | 0.000483237 / 0.0005037641 | 0.0004830079 / 0.0005035059 | 58.392 | 2.5357 | 46.541 | CPU/GPU RGB differ; MSE kept separate |
| swm/CraftaxClassicPixels-v1 | VERIFIED | 26 | 63×63 | 0 / 0 | 0 / 0 | 19.385 | 31.435 | 0.010145 | GPU matched 22/26 CPU checkpoints; native platform signatures differ; no signature or engine field edits |
| swm/CraftaxClassicSymbolic-v1 | VERIFIED | 0 | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | render is only supported for Pixels variants; symbolic envs do not produce a frame. |
| swm/CraftaxPixels-v1 | VERIFIED | 27 | 130×110 | 0 / 0 | NONCOMPARABLE | 27.179 | NONCOMPARABLE | 0.02282 | GPU matched 0/27 CPU checkpoints; native platform signatures differ; no signature or engine field edits; gpu: CPU/GPU public replay has unequal numeric State leaves: engine/light_map. The public compatibility signature also prohibits cross-device State restoration; no State fields were patched.; CPU/GPU public replay has unequal numeric State leaves: engine/light_map, engine/light_level. The public compatibility signature also prohibits cross-device State restoration; no State fields were patched. |
| swm/CraftaxSymbolic-v1 | VERIFIED | 0 | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | render is only supported for Pixels variants; symbolic envs do not produce a frame. |
| swm/FetchPickAndPlace-v3 | VERIFIED | 18 | 224×224 | UNSUPPORTED | 0.02096089 / 0.02949772 | UNSUPPORTED | 2.882 | 15.52 | RESET_ONLY: Upstream PickAndPlace render calls mj_forward and changes the continuation-relevant gripper position/quaternion caches after steps. Dynamic frames are NONCOMPARABLE under the required render nonmutation rule. No restore, cache repair, or pose teleport is used.; cpu: CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp |
| swm/FetchPickAndPlaceDense-v3 | VERIFIED | 18 | 224×224 | UNSUPPORTED | 0.02096089 / 0.02949772 | UNSUPPORTED | 2.8575 | 15.451 | RESET_ONLY: Upstream PickAndPlace render calls mj_forward and changes the continuation-relevant gripper position/quaternion caches after steps. Dynamic frames are NONCOMPARABLE under the required render nonmutation rule. No restore, cache repair, or pose teleport is used.; cpu: CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp |
| swm/FetchPickAndPlaceDict-v3 | VERIFIED | 18 | 224×224 | UNSUPPORTED | 0.02096089 / 0.02949772 | UNSUPPORTED | 2.8267 | 15.003 | RESET_ONLY: Upstream PickAndPlace render calls mj_forward and changes the continuation-relevant gripper position/quaternion caches after steps. Dynamic frames are NONCOMPARABLE under the required render nonmutation rule. No restore, cache repair, or pose teleport is used.; cpu: CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp |
| swm/FetchPush-v3 | VERIFIED | 42 | 224×224 | UNSUPPORTED | 0.004299026 / 0.02714999 | UNSUPPORTED | 2.9097 | 16.011 | cpu: CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp |
| swm/FetchPushDense-v3 | VERIFIED | 42 | 224×224 | UNSUPPORTED | 0.004299026 / 0.02714999 | UNSUPPORTED | 2.8588 | 16.041 | cpu: CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp |
| swm/FetchPushDict-v3 | VERIFIED | 42 | 224×224 | UNSUPPORTED | 0.004299026 / 0.02714999 | UNSUPPORTED | 2.8345 | 16.161 | cpu: CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp |
| swm/FetchReach-v3 | VERIFIED | 42 | 224×224 | UNSUPPORTED | 0.004337554 / 0.02774732 | UNSUPPORTED | 2.8116 | 15.434 | cpu: CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp |
| swm/FetchReachDense-v3 | VERIFIED | 42 | 224×224 | UNSUPPORTED | 0.004337554 / 0.02774732 | UNSUPPORTED | 2.7834 | 15.525 | cpu: CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp |
| swm/FetchReachDict-v3 | VERIFIED | 42 | 224×224 | UNSUPPORTED | 0.004337554 / 0.02774732 | UNSUPPORTED | 2.79 | 15.462 | cpu: CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp |
| swm/FetchSlide-v3 | VERIFIED | 42 | 224×224 | UNSUPPORTED | 0.008403954 / 0.055868 | UNSUPPORTED | 2.8356 | 16.018 | cpu: CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp |
| swm/FetchSlideDense-v3 | VERIFIED | 42 | 224×224 | UNSUPPORTED | 0.008403954 / 0.055868 | UNSUPPORTED | 2.8338 | 16.047 | cpu: CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp |
| swm/FetchSlideDict-v3 | VERIFIED | 42 | 224×224 | UNSUPPORTED | 0.008403954 / 0.055868 | UNSUPPORTED | 2.8421 | 15.844 | cpu: CPU_BACKEND_MISSING: Fetch full mesh rendering requires CUDA/MJWarp |
| swm/FingerDMControl-v0 | VERIFIED | 24 | 224×224 | 0.009146044 / 0.04172477 | 0.007116541 / 0.03351904 | 85.358 | 8.5246 | 39.539 | CPU/GPU RGB differ; MSE kept separate |
| swm/HopperDMControl-v0 | VERIFIED | 24 | 224×224 | 0.0004573268 / 0.0005370306 | 0.0004570381 / 0.0005366237 | 38.281 | 2.4847 | 41.618 | CPU/GPU RGB differ; MSE kept separate |
| swm/HumanoidDMControl-v0 | VERIFIED | 24 | 224×224 | 0.00043824 / 0.000525648 | 0.0004382271 / 0.000525327 | 122.05 | 2.9609 | 90.213 | CPU/GPU RGB differ; MSE kept separate |
| swm/ManipulatorDMControl-v0 | VERIFIED | 24 | 224×224 | 0.005680322 / 0.02807258 | 0.003718053 / 0.02119746 | 76.963 | 9.0008 | 43.752 | CPU/GPU RGB differ; MSE kept separate |
| swm/MountainCarContinuousControl-v0 | VERIFIED | 42 | 400×600 | 1.422292e-05 / 0.0001002144 | 1.422292e-05 / 0.0001002144 | 16.831 | 11.515 | 1.9966 | CPU/GPU RGB exactly equal |
| swm/MountainCarControl-v0 | VERIFIED | 24 | 400×600 | 2.509052e-05 / 0.0001006064 | 2.509052e-05 / 0.0001006064 | 17.052 | 11.753 | 1.9272 | CPU/GPU RGB exactly equal |
| swm/OGBCube-v0 | VERIFIED | 57 | 224×224 | UNSUPPORTED | 0.001847513 / 0.002862775 | UNSUPPORTED | 25.453 | 190.97 | cpu: Cube/Scene CPU mesh rendering is unavailable; renderer/GPU execution requires CUDA |
| swm/OGBMaze-v0 | VERIFIED | 42 | 224×224 | 0.1017934 / 0.1030644 | 0.1011924 / 0.102435 | 1.8742 | 3.9654 | 163.9 | CPU/GPU RGB differ; MSE kept separate |
| swm/OGBScene-v0 | VERIFIED | 132 | 200×200 | UNSUPPORTED | 0.005975834 / 0.007047075 | UNSUPPORTED | 26.164 | 202.79 | cpu: Cube/Scene CPU mesh rendering is unavailable; renderer/GPU execution requires CUDA |
| swm/PFRocketLanding-v0 | INCOMPLETE | 204 | 480×480 | UNSUPPORTED | 0.01339864 / 0.03403506 | UNSUPPORTED | 21.931 | 45.988 | Measurement does not imply semantic acceptance; Includes retained high-ceiling horizon-600 renderer failure; cpu: Rocket has no CPU textured-triangle renderer |
| swm/PendulumControl-v1 | VERIFIED | 42 | 500×500 | 9.631167e-06 / 6.212872e-05 | 9.631167e-06 / 6.212872e-05 | 10.902 | 11.034 | 3.5576 | CPU/GPU RGB exactly equal |
| swm/PendulumDMControl-v0 | VERIFIED | 24 | 224×224 | 5.632881e-05 / 7.640841e-05 | 6.874135e-05 / 8.898214e-05 | 41.045 | 10.507 | 39.504 | CPU/GPU RGB differ; MSE kept separate |
| swm/Piecewise-v0 | VERIFIED | 42 | 224×224 | 4.402827e-10 / 3.064951e-10 | 4.402827e-10 / 3.064951e-10 | 0.68329 | 1.002 | 1.0402 | CPU/GPU RGB exactly equal |
| swm/PushT-Discrete-v1 | INCOMPLETE | 0 | NONCOMPARABLE | NONCOMPARABLE | NONCOMPARABLE | NONCOMPARABLE | NONCOMPARABLE | NONCOMPARABLE | The current candidate maps to gym_pusht-style absolute-target PushT, not the pinned SWM relative-action reset/step path. All three identical-seed default resets differ numerically before any action. No accepted exhaustive SWM State-equivalence witness exists; installing source poses would not establish matching hidden contact/RNG/task continuation. SWM additionally uses a 16x16 PolarQuantizer MultiDiscrete action API; the current candidate has only continuous absolute targets. |
| swm/PushT-v1 | INCOMPLETE | 0 | NONCOMPARABLE | NONCOMPARABLE | NONCOMPARABLE | NONCOMPARABLE | NONCOMPARABLE | NONCOMPARABLE | The current candidate maps to gym_pusht-style absolute-target PushT, not the pinned SWM relative-action reset/step path. All three identical-seed default resets differ numerically before any action. No accepted exhaustive SWM State-equivalence witness exists; installing source poses would not establish matching hidden contact/RNG/task continuation. |
| swm/QuadrupedDMControl-v0 | VERIFIED | 24 | 224×224 | 0.0004338558 / 0.0005281947 | 0.0004338327 / 0.0005280394 | 110.8 | 3.6812 | 78.803 | CPU/GPU RGB differ; MSE kept separate |
| swm/ReacherDMControl-v0 | VERIFIED | 24 | 224×224 | 3.639446e-05 / 4.822951e-05 | 0.0002166873 / 0.0002291196 | 45.372 | 10.683 | 35.844 | CPU/GPU RGB differ; MSE kept separate |
| swm/SimplePointMaze-v0 | VERIFIED | 42 | 500×500 | 2.153488e-05 / 2.396148e-05 | 2.153488e-05 / 2.396148e-05 | 7.4777 | 4.562 | 12.854 | CPU/GPU RGB exactly equal |
| swm/TwoRoom-v1 | VERIFIED | 42 | 224×224 | 4.913652e-10 / 3.064951e-10 | 4.913652e-10 / 3.064951e-10 | 0.72101 | 1.5465 | 1.3445 | CPU/GPU RGB exactly equal |
| swm/WalkerDMControl-v0 | VERIFIED | 24 | 224×224 | 0.001255864 / 0.001503995 | 0.001256163 / 0.00150377 | 72.941 | 2.6059 | 44.261 | CPU/GPU RGB differ; MSE kept separate |

## How the measurements were made

Every numeric value comes from retained Oscar measurements. Local Mac execution was limited to editing, static inspection, artifact verification and report assembly. Candidate and original measurements within every measured row used the same Oscar host; isolated subprocesses prevented GPU EGL settings from leaking into the source CPU process. Each record contains the actual host, CPU/GPU model, package versions and paths, thread settings and environment. The campaign manifest binds those records to SLURM terminal evidence, NVIDIA driver output, source archives, job recipes and harness overlays.

CPU MuJoCo rendering used Xvfb/GLX with `LIBGL_ALWAYS_SOFTWARE=1`, `GALLIUM_DRIVER=llvmpipe` and a retained live `GL_RENDERER` proof. This is software OpenGL on the CPU. Rocket’s original uses PyBullet TinyRenderer. Native 2-D originals use their software rasterizers. Craftax originals run JAX with the CPU backend: the actual pixel-wrapper `render()` converts its cached observation to RGB. Consequently Craftax’s original render latency does **not** include the upstream observation-generation work performed during reset/step. Our public renderer recomputes pixels from State. Both are the requested actual public calls; the latency difference should be interpreted with that cache boundary in mind.

At each checkpoint the candidate and pinned authoritative source received the same seed, explicit options and identical valid action prefix. The harness either compared every exposed numeric State leaf exactly or invoked the already accepted independent family witness. The latter covers Fetch controller/cache/lifecycle/RNG fields, Cube/Scene’s engine/model/entropy ownership, Craftax engine/PRNG leaves, and Rocket’s continuation capture. Per-frame records retain the witness, State hash, action dtype and prefix hash, exact render kwargs, camera, shape and RGB hashes. Saved numeric States and full action arrays allow every prefix to be checked independently. Source rendering and repeated candidate rendering must preserve supplied/live State, continuation caches, lifecycle and RNG. Replay and State capture are outside timing.

Frames use seeds 0, 17 and 42, early/middle/late or terminal checkpoints, and registered appearance cases. Native continuous tasks, Fetch trajectories, OGBench and Rocket cover float32 and float64 commands; discrete tasks use integer commands. DMC uses its accepted canonical float64 action boundary. The sampling recipes give the complete checkpoint lists and variations. Cube/Scene additionally replay accepted contact controllers (cube, button, drawer, window), camera/light/size/color and lock cases; two reset applications reproduce the source’s appearance lifecycle. Craftax trajectories stop at terminal States. No image is cropped, aligned, recolored, shifted or rescaled by the harness. Fetch’s authoritative wrapper is configured with both its output resolution and underlying raster dimensions at 224×224, so its built-in resize has identical input/output dimensions.

For every supported registration/backend combination there are at least 10 untimed warmups and 100 recorded calls. Warm samples are pooled across that backend’s matched checkpoints. Each timed candidate call includes the public State→RenderState projection and renderer, including real public transfers. No reset, step, get_state, set_state, metrics or PNG encoding is inside the timer. GPU timing synchronizes before and after each call and requires an actual CUDA output tensor. Its primary latency retains native GPU output. The supplemental measurement times the complete same public call plus CPU materialization; it is not a transfer-only subtraction. First-call timing is recorded separately at every checkpoint, with the first checkpoint capturing initial renderer setup/compilation. It is never mixed into the warm distribution.

Metrics use float64 over the complete H×W×3 RGB output. Primary MSE is `mean((ours/255 - original/255)**2)`. Raw RGB-scale MSE, RGB MAE, mean per-pixel maximum-channel absolute error, exact-pixel fraction, within-8 fraction, SSIM and maximum absolute channel error are retained per frame. SSIM uses float64 luminance, a 7×7 uniform window with nearest boundary handling, and constants `(0.01*255)^2` and `(0.03*255)^2`. Fractions require all three channels to satisfy the condition. Each metric includes count, mean, median, p95 and maximum; timings additionally retain every sample, population standard deviation and minimum. Raw RGB arrays were independently rehashed and all metrics recomputed on a CPU compute node before compact export.

## Scope limits and fidelity findings

**PickAndPlace:** all three registrations have independent reset-only measurements: three seeds × six default/object/table-color configurations. Their source renderer calls `mj_forward` after action steps and changes continuation-relevant gripper-position/quaternion caches. Independent float32/float64 probes cover all three registrations, three seeds, zero/random actions and six checkpoints. Dynamic frames therefore fail the required render nonmutation rule and are excluded; no cache repair or State restoration is hidden around rendering. The reset-only measurements have empty action prefixes and do not claim post-action dtype coverage.

**Craftax:** the public State signature includes CPU versus GPU platform. CPU States cannot be transplanted into a GPU environment. The harness instead independently replays the recorded public reset/actions on GPU and compares every numeric leaf except the platform signature. Some checkpoints differ in numeric engine fields, such as `light_level`; these receive `NONCOMPARABLE` records and no GPU image or timing. Only exact semantic matches are rendered/timed. Both full States, all leaf hashes, differing fields and magnitudes are retained. No signature or engine field is edited, and no numerical tolerance admits a mismatch. CPU quality covers the entire CPU corpus; GPU quality covers its explicitly counted subset. Both symbolic registrations independently exercise their source CPU, candidate CPU and candidate GPU no-RGB paths.

**PushT:** current continuous and discrete implementations were independently probed under the source-aligned pymunk 7.2.0 profile. Reset poses differ for seeds 0/17/42 before any action; the current absolute-action implementation also does not match SWM’s relative-action interface, and the discrete quantizer/replay witness is absent. Neither row claims matched frames or renderer latency. No pose teleport or new replay implementation was introduced.

**Appearance fidelity:** semantic State equality does not imply identical renderer appearance. OGBMaze’s historical registered source defaults to state observations while the current renderer was calibrated for pixel appearance; its large image discrepancy remains visible. Fetch uses identical explicit supported reset options and its actual public style defaults/overrides, without sampling unmatched source-only visual knobs. Existing source material-unbinding behavior and renderer approximations remain part of the measured error. These are findings, not repairs or changes to accepted thresholds.

**Rocket:** all 204 canonical checkpoints are included, covering both action dtypes, three seeds, fall/mixed/success/horizon cases, reset/appearance cases and the high-ceiling horizon-600 failure. Its `INCOMPLETE` status is unchanged. The earlier frozen 204-frame artifact was independently rehashed and recomputed separately; it was not overwritten or used to substitute timings.

The fresh seed-0 float32 horizon-600 witness (frame 19) has normalized RGB MSE **0.03403505874**, raw RGB MSE **2213.129695**, and mean per-pixel maximum-channel absolute difference **10.60954861**. The last value is the retained acceptance metric; it is not MSE. Its pair and unamplified absolute-difference image are committed.

## Largest image errors

| Registration | Backend | Frames | MSE mean | MSE p95 | MSE maximum |
|---|---|---:|---:|---:|---:|
| swm/OGBMaze-v0 | cpu | 42 | 0.1017934 | 0.1030644 | 0.1031543 |
| swm/OGBMaze-v0 | gpu | 42 | 0.1011924 | 0.102435 | 0.102525 |
| swm/FetchPickAndPlace-v3 | gpu | 18 | 0.02096089 | 0.02949772 | 0.0295029 |
| swm/FetchPickAndPlaceDense-v3 | gpu | 18 | 0.02096089 | 0.02949772 | 0.0295029 |
| swm/FetchPickAndPlaceDict-v3 | gpu | 18 | 0.02096089 | 0.02949772 | 0.0295029 |
| swm/PFRocketLanding-v0 | gpu | 204 | 0.01339864 | 0.03403506 | 0.0348176 |
| swm/FingerDMControl-v0 | cpu | 24 | 0.009146044 | 0.04172477 | 0.04201596 |
| swm/FetchSlide-v3 | gpu | 42 | 0.008403954 | 0.055868 | 0.05615767 |
| swm/FetchSlideDense-v3 | gpu | 42 | 0.008403954 | 0.055868 | 0.05615767 |
| swm/FetchSlideDict-v3 | gpu | 42 | 0.008403954 | 0.055868 | 0.05615767 |

## Major latency outliers

These are absolute B=1 public-call costs on each row’s recorded hardware, not a cross-hardware speed ranking.

| Registration | Backend | Median ms | p95 ms | Calls |
|---|---|---:|---:|---:|
| swm/BallInCupDMControl-v0 | cpu | 258.76 | 275.21 | 120 |
| swm/OGBScene-v0 | original | 202.79 | 224.6 | 132 |
| swm/OGBCube-v0 | original | 190.97 | 212.34 | 114 |
| swm/OGBMaze-v0 | original | 163.9 | 173.67 | 126 |
| swm/HumanoidDMControl-v0 | cpu | 122.05 | 136.13 | 120 |
| swm/QuadrupedDMControl-v0 | cpu | 110.8 | 116.48 | 120 |
| swm/HumanoidDMControl-v0 | original | 90.213 | 93.683 | 120 |
| swm/FingerDMControl-v0 | cpu | 85.358 | 90.12 | 120 |
| swm/QuadrupedDMControl-v0 | original | 78.803 | 80.288 | 120 |
| swm/ManipulatorDMControl-v0 | cpu | 76.963 | 91.076 | 120 |

## Detailed warm and supplemental latency

Cells show **median / p95 milliseconds (calls)**. Full statistics, cold/first-call samples, backend matched counts and every warm sample are in [summary.json](summary.json).

| Registration | Ours CPU | Ours GPU | Original CPU | Public GPU→CPU output |
|---|---|---|---|---|
| swm/AcrobotControl-v1 | 18.613 / 18.766 (120) | 11.379 / 11.749 (120) | 2.0034 / 41.669 (120) | 11.616 / 11.985 (120) |
| swm/AcrobotDMControl-v0 | 33.159 / 35.075 (120) | 9.3806 / 10.489 (120) | 38.318 / 39.256 (120) | 9.4726 / 10.651 (120) |
| swm/BallInCupDMControl-v0 | 258.76 / 275.21 (120) | 8.9493 / 9.713 (120) | 66.276 / 84.143 (120) | 9.0022 / 9.7877 (120) |
| swm/CartPoleControl-v1 | 17.442 / 17.627 (110) | 10.9 / 11.268 (110) | 2.2963 / 40.03 (110) | 11.109 / 11.61 (110) |
| swm/CartpoleDMControl-v0 | 26.756 / 27.008 (120) | 10.392 / 10.465 (120) | 41.321 / 42.073 (120) | 10.453 / 10.526 (120) |
| swm/CheetahDMControl-v0 | 58.392 / 59.101 (120) | 2.5357 / 2.7892 (120) | 46.541 / 46.91 (120) | 2.5989 / 2.67 (120) |
| swm/CraftaxClassicPixels-v1 | 19.385 / 20.44 (260) | 31.435 / 34.325 (2200) | 0.010145 / 0.020438 (260) | 31.646 / 36.071 (2200) |
| swm/CraftaxClassicSymbolic-v1 | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED |
| swm/CraftaxPixels-v1 | 27.179 / 30.145 (270) | NONCOMPARABLE | 0.02282 / 0.042288 (270) | NONCOMPARABLE |
| swm/CraftaxSymbolic-v1 | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED | NOT IMPLEMENTED |
| swm/FetchPickAndPlace-v3 | UNSUPPORTED | 2.882 / 3.3767 (108) | 15.52 / 16.213 (108) | 2.9104 / 3.4808 (108) |
| swm/FetchPickAndPlaceDense-v3 | UNSUPPORTED | 2.8575 / 3.3002 (108) | 15.451 / 15.976 (108) | 2.894 / 3.2695 (108) |
| swm/FetchPickAndPlaceDict-v3 | UNSUPPORTED | 2.8267 / 2.8596 (108) | 15.003 / 15.55 (108) | 2.8716 / 2.9395 (108) |
| swm/FetchPush-v3 | UNSUPPORTED | 2.9097 / 3.3681 (126) | 16.011 / 16.807 (126) | 2.9146 / 3.2963 (126) |
| swm/FetchPushDense-v3 | UNSUPPORTED | 2.8588 / 3.2481 (126) | 16.041 / 16.993 (126) | 2.8956 / 3.3798 (126) |
| swm/FetchPushDict-v3 | UNSUPPORTED | 2.8345 / 3.2926 (126) | 16.161 / 16.909 (126) | 2.8735 / 3.2816 (126) |
| swm/FetchReach-v3 | UNSUPPORTED | 2.8116 / 3.1681 (126) | 15.434 / 15.896 (126) | 2.8522 / 3.3379 (126) |
| swm/FetchReachDense-v3 | UNSUPPORTED | 2.7834 / 2.8218 (126) | 15.525 / 16.419 (126) | 2.8225 / 2.9893 (126) |
| swm/FetchReachDict-v3 | UNSUPPORTED | 2.79 / 3.1988 (126) | 15.462 / 16.133 (126) | 2.8303 / 3.0513 (126) |
| swm/FetchSlide-v3 | UNSUPPORTED | 2.8356 / 3.2245 (126) | 16.018 / 17.026 (126) | 2.8713 / 3.2074 (126) |
| swm/FetchSlideDense-v3 | UNSUPPORTED | 2.8338 / 3.2695 (126) | 16.047 / 16.82 (126) | 2.8826 / 3.3714 (126) |
| swm/FetchSlideDict-v3 | UNSUPPORTED | 2.8421 / 3.3429 (126) | 15.844 / 16.795 (126) | 2.8867 / 3.3037 (126) |
| swm/FingerDMControl-v0 | 85.358 / 90.12 (120) | 8.5246 / 9.4067 (120) | 39.539 / 41.217 (120) | 8.58 / 9.4565 (120) |
| swm/HopperDMControl-v0 | 38.281 / 43.173 (120) | 2.4847 / 2.7251 (120) | 41.618 / 43.189 (120) | 2.5421 / 2.7939 (120) |
| swm/HumanoidDMControl-v0 | 122.05 / 136.13 (120) | 2.9609 / 2.9789 (120) | 90.213 / 93.683 (120) | 3.018 / 3.0435 (120) |
| swm/ManipulatorDMControl-v0 | 76.963 / 91.076 (120) | 9.0008 / 9.8309 (120) | 43.752 / 47.455 (120) | 9.1093 / 9.9752 (120) |
| swm/MountainCarContinuousControl-v0 | 16.831 / 16.988 (126) | 11.515 / 11.765 (126) | 1.9966 / 22.338 (126) | 11.672 / 12.086 (126) |
| swm/MountainCarControl-v0 | 17.052 / 17.219 (120) | 11.753 / 11.991 (120) | 1.9272 / 26.653 (120) | 11.892 / 12.273 (120) |
| swm/OGBCube-v0 | UNSUPPORTED | 25.453 / 26.75 (114) | 190.97 / 212.34 (114) | 25.193 / 26.007 (114) |
| swm/OGBMaze-v0 | 1.8742 / 1.9578 (126) | 3.9654 / 4.1388 (126) | 163.9 / 173.67 (126) | 3.9985 / 4.0598 (126) |
| swm/OGBScene-v0 | UNSUPPORTED | 26.164 / 26.717 (132) | 202.79 / 224.6 (132) | 26.305 / 27.073 (132) |
| swm/PFRocketLanding-v0 | UNSUPPORTED | 21.931 / 40.843 (204) | 45.988 / 98.39 (204) | 22.074 / 41.123 (204) |
| swm/PendulumControl-v1 | 10.902 / 11.084 (126) | 11.034 / 11.339 (126) | 3.5576 / 36.266 (126) | 11.254 / 11.656 (126) |
| swm/PendulumDMControl-v0 | 41.045 / 43.351 (120) | 10.507 / 10.577 (120) | 39.504 / 40.254 (120) | 10.686 / 10.729 (120) |
| swm/Piecewise-v0 | 0.68329 / 0.69915 (126) | 1.002 / 1.122 (126) | 1.0402 / 1.1377 (126) | 1.0403 / 1.056 (126) |
| swm/PushT-Discrete-v1 | NONCOMPARABLE | NONCOMPARABLE | NONCOMPARABLE | NONCOMPARABLE |
| swm/PushT-v1 | NONCOMPARABLE | NONCOMPARABLE | NONCOMPARABLE | NONCOMPARABLE |
| swm/QuadrupedDMControl-v0 | 110.8 / 116.48 (120) | 3.6812 / 3.8291 (120) | 78.803 / 80.288 (120) | 3.7424 / 3.883 (120) |
| swm/ReacherDMControl-v0 | 45.372 / 45.642 (120) | 10.683 / 10.729 (120) | 35.844 / 36.809 (120) | 10.79 / 10.893 (120) |
| swm/SimplePointMaze-v0 | 7.4777 / 7.6021 (126) | 4.562 / 4.763 (126) | 12.854 / 13.099 (126) | 4.6496 / 5.0978 (126) |
| swm/TwoRoom-v1 | 0.72101 / 0.73581 (126) | 1.5465 / 1.5967 (126) | 1.3445 / 1.3833 (126) | 1.5872 / 1.6347 (126) |
| swm/WalkerDMControl-v0 | 72.941 / 77.057 (120) | 2.6059 / 2.6216 (120) | 44.261 / 49.926 (120) | 2.664 / 2.6912 (120) |

## Retained witness images

These are unmodified renderer outputs and unamplified absolute RGB differences. No visualization affects scoring.

| Witness | Original | Candidate | Difference |
|---|---|---|---|
| Rocket retained horizon-600 | [RGB](../randall_renderer_measurements/swm_PFRocketLanding-v0/retained-horizon600-original.png) | [RGB](../randall_renderer_measurements/swm_PFRocketLanding-v0/retained-horizon600-candidate.png) | [difference](../randall_renderer_measurements/swm_PFRocketLanding-v0/retained-horizon600-absolute-difference.png) |
| OGBMaze worst | [RGB](../randall_renderer_measurements/swm_OGBMaze-v0/worst-original.png) | [RGB](../randall_renderer_measurements/swm_OGBMaze-v0/worst-candidate.png) | [difference](../randall_renderer_measurements/swm_OGBMaze-v0/worst-absolute-difference.png) |
| FetchPush worst | [RGB](../randall_renderer_measurements/swm_FetchPush-v3/worst-original.png) | [RGB](../randall_renderer_measurements/swm_FetchPush-v3/worst-candidate.png) | [difference](../randall_renderer_measurements/swm_FetchPush-v3/worst-absolute-difference.png) |
| Finger worst | [RGB](../randall_renderer_measurements/swm_FingerDMControl-v0/worst-original.png) | [RGB](../randall_renderer_measurements/swm_FingerDMControl-v0/worst-candidate.png) | [difference](../randall_renderer_measurements/swm_FingerDMControl-v0/worst-absolute-difference.png) |
| Scene worst | [RGB](../randall_renderer_measurements/swm_OGBScene-v0/worst-original.png) | [RGB](../randall_renderer_measurements/swm_OGBScene-v0/worst-candidate.png) | [difference](../randall_renderer_measurements/swm_OGBScene-v0/worst-absolute-difference.png) |

## Artifacts and reproduction

- [CSV: exactly 42 rows](summary.csv) and [JSON: exactly 42 registration objects](summary.json), including per-frame quality/hashes, exact raw timings, profiles and recipes.
- [Verification result](verification.json), [campaign manifest](campaign-manifest.json), [compact evidence hashes](../randall_renderer_measurements/manifest.json) and [explicit source selection](../randall_renderer_measurements/remote-selection.json).
- [Starting checkpoint](../randall_renderer_start/start.json), [frozen Rocket independent recomputation](../randall_renderer_start/frozen-rocket-independent-metrics.json), [Fetch float32 diagnostic](../randall_renderer_start/fetch-original-render-mutation-float32.json), [Fetch float64 diagnostic](../randall_renderer_start/fetch-original-render-mutation-float64.json), and [submitted recipes/immutable harness overlays](../randall_renderer_start/submission/).

From the repository root, use a CPU Python environment with NumPy, SciPy, Torch, TensorDict and Pillow:

```bash
python -m scripts.benchmarks.renderer_table.verify \
  docs/research/semantic_recovery_evidence/randall_renderer_results/summary.json \
  --evidence docs/research/semantic_recovery_evidence/randall_renderer_measurements \
  --output /tmp/randall-verification.json
python -m pytest scripts/benchmarks/renderer_table/test_verify.py -q
```

The verifier rejects missing/duplicate registrations, incorrect State/frame/camera/resolution pairing, missing GPU synchronization, GPU-backed CPU labels, missing definitions/counts, invalid metric aggregation, nonfinite numbers and unexplained unavailable cells. It verifies retained State/action-prefix hashes and the independently exported artifact hashes. Strict Craftax replay checks reject any semantic leaf difference in an admitted GPU frame. Production/matrix/profile hashes are checked against the starting checkpoint.

To reproduce measurements, use the selected job recipes in the campaign manifest: reconstruct the exact starting Git tree plus its recorded harness overlay, use the pinned SWM oracle archive and specified isolated profile, then run on Oscar. Every attempt has a distinct output directory; failed or superseded attempts remain identified and are excluded from the numeric table. Full raw RGB NPY files remain in the hashed Oscar scratch paths. Compact committed evidence contains numeric States/actions, metrics/timings, a small set of representative worst-case pairs and unamplified differences. The containing Git commit is discoverable with `git log -1 --format=%H -- <this report>`; no self-referential commit hash is embedded in its own bytes.
