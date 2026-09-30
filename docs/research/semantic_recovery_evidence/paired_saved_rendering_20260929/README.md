# Paired saved-State renderer comparison — 2026-09-29

This campaign compares stable-envs with pinned ground truth on the same saved numeric States, hardware allocation, image dimensions, batch size, and output destination. It measures actual rendering throughput. Disk reads, State restoration, simulator allocation, initial input upload, projection, batching, JIT compilation and warmup are outside the timer. Neither side rolls out trajectories inside the timer. Source images are recomputed for every lane; no saved or cached RGB image is substituted.

- `paired_rendering_42_b256.csv`: 42-row coverage table; empty values are unmeasured, not zero. CPU and GPU statuses are independent.
- `paired_rendering_all_batches.csv`: raw comparisons for every measured batch and destination, with mean, sample SD, throughput, sample count, and evidence path.
- `pilot_comparison.csv`: completed preliminary Fetch measurements (four poses, batches 1/64), distinct from the full campaign.
- `raw/`: full-campaign result records. `pilot/`: initial pilot records, including retained failures.

A speedup above one means ours is faster than source for that row's exact workload. GPU runs report both a contiguous CPU uint8 tensor batch and a contiguous CUDA uint8 tensor batch. Any readback/upload needed for the chosen destination is timed. CPU runs return contiguous CPU tensors. Source render device is explicit: a CPU-only source renderer is never labelled GPU merely because its outputs are uploaded. NVIDIA and software OpenGL are checked separately. No ratio compares timings from different nodes or mismatched output destinations.

Batch sizes are 1, 64, 128 and 256. GPU jobs also measure the selected batch from the earlier RTX 3090 saturation experiment when that completed result exists. Up to 16 distinct saved poses are repeated to fill the batch; these are not 1,000 independent images. Batch 1 uses the first selected pose. Mean/SD describe repeated batch calls, divided by batch size; this is amortized per-image cost rather than single-image latency. Timings alternate candidate/source order; at least three samples per side, up to twelve, with the full campaign stopping after at least one second accumulated on the slower side. Raw samples are retained.

The classic-control, navigation, MuJoCo and PyFlyt source paths use repeated upstream calls over pre-restored instances. This is the actual existing serial source interface, not a claim that multiprocessing or a future vectorized source implementation could not perform better. Craftax instead uses JIT/vmap of the real upstream `get_obs`, including uint8 conversion, checked against scalar upstream output; its public wrapper's cached `render()` is not a recomputation baseline. Compilation is excluded. Source State recapture is checked, and candidate live State and prepared inputs are checked for nonmutation. Pixel MSE is measured between the newly rendered candidate/source images for the selected poses.

PushT's two registrations remain UNESTABLISHED; no accepted matched-State corpus or speedup is invented. The two symbolic Craftax registrations have no RGB renderer. Fetch PickAndPlace reset-only corpus restrictions remain in the scope column. Craftax CPU/GPU saved-State profiles are distinct: compare each candidate to the source on its own profile, not CPU-source timing against a different GPU-State workload.

Production checkout is immutable Oscar `render-parity-df80c9c5-20260928/source-v12`, based on `df80c9c54590afa9fc0482a4dcb241d21f6b63a3` plus the existing renderer overlay. The job driver verifies production file hashes before running. Harness and source helper hashes, package/device profiles, exact input State hashes and raw samples accompany each result. No production implementation, semantic acceptance status, or remote Git branch is changed by this experiment.

Initial pilot array: 6795198. Fetch completed both CPU and GPU checks; Scene failed during source-only variation lookup setup before timing. That failure is preserved. A read-only path accessor was added to the test helper so it survives upstream reset replacing the variation object. The full array 6795386 remained pending on the condo GPU quota and was cancelled without measurements; replacement norm-gpu array 6795405 requests RTX 3090, 4 CPU cores, 32 GB, and 45 minutes per registration. This submission was replaced before starting; see the current scheduling note below. Additional CPU-only setup pilots (6795452) validate source reconstruction while GPUs queue; those are separate evidence and do not replace same-host full results. The older unpaired sweep was held to prioritize this comparison; existing results are preserved.

Current scheduling: CPU work runs independently in array 6796449 (12 concurrent CPU jobs). GPU work is split into disjoint arrays 6796221 (indices 0–31, up to 32 condo RTX 3090s) and 6796222 (indices 32–41, two norm-gpu slots); GPU requests are 20 minutes and 32 GB. Each candidate/source pair shares its hardware allocation. CPU-pair and GPU-pair allocations can differ, so no cross-allocation CPU-versus-GPU speedup is claimed. CPU raw results live in `cpufull/`; GPU results in `raw/`.

Cube/Scene setup retries 6795505 passed exact source-State recapture and exact CPU pixels on four poses after fixing the source-only model-settings capture dtype. This bookkeeping does not copy candidate geometry into the source. Rocket's baseline invokes official `PyFlyt.RocketBaseEnv.render` on a restored upstream physics instance and requires bit-exact agreement with the corpus's saved upstream images before accepting timings. The numeric replay wrapper only restores the saved State outside timing.

The main user-facing file is `paired_rendering_summary.csv`: fixed batch 256, CPU-resident uint8 output for both sides, MSE mean/SD, ground-truth CPU/GPU FPS, our CPU/GPU FPS, and named CPU-vs-CPU / GPU-vs-GPU multipliers. Where source rendering is CPU-only, source GPU FPS and GPU-vs-GPU multiplier remain blank; the GPU-versus-CPU-only-source comparison has its own columns and uses the CPU on that GPU allocation. Timing mean/SD and per-comparison hardware remain in the table. Incomplete cases are explicit. A bounded local collector refreshes results every minute while jobs run.

Manipulator CPU source-repeat diagnostic: unchanged full-State hashes accompany a 27/255 maximum difference on 780 floor-edge pixels, normalized MSE 0.0000496854. The immediate repeated image can be identical, but rotating through the prepared source contexts changes the edge rasterization again. Warming twice did not remove the effect. The original failure and both source images are retained in `manipdiag/` and the failed warmup-only retry in `cpuretry1/`. Its next retry (`cpuretry2/`, job 6797472) explicitly measures MSE after every timed candidate/source pair, outside the timing interval, and records source-repeat variation. Only this evidence-backed family permits source repeat MSE up to 0.0001 (one tenth of the requested 0.001 target), while retaining exact numeric State checks; other source families retain the 1/255 channel check. Candidate batch consistency remains checked at 1/255. This is a measurement qualification, not a rendering-parity or semantic promotion. Successful retry results supersede the failed row in exports without overwriting either raw record.

Comparison qualification correction: a measured CPU source path does not prove that source GPU rendering is unavailable. `paired_rendering_summary.csv` now explicitly marks all eight such rows `CPU_REFERENCE_GPU_AVAILABILITY_UNVERIFIED`. Their measured ratios are retained as `ours_gpu_vs_measured_cpu_source_speedup`; they do not count as qualified GPU performance wins. There are 25 established same-device GPU wins and four same-device GPU losses. The other rows comprise eight provisional CPU-reference comparisons, three failed/missing comparisons, and two RGB-inapplicable registrations. Source GPU availability must be established before accepting a CPU fallback.

## RTX 3090 Cube cache gate overlay (2026-09-29)

The generated tables now use `../cube_context_budget_20260929/candidate/28/result.json` for Cube GPU only, after job 6802707 passed its exact-image, same-host/States, and warmed-cache gate. Both ours and source timings come from that candidate run; CPU rows and other GPU rows retain their original evidence. Original Cube raw measurements remain untouched. `result-sha256.json` and the all-batches CSV identify the overlay evidence.

Regenerate with:

```sh
python3 scripts/benchmarks/renderer_table/export_paired_saved.py docs/research/semantic_recovery_evidence/paired_saved_rendering_20260929 --cube-gate docs/research/semantic_recovery_evidence/cube_context_budget_20260929
```

Current same-device GPU tally at batch 256 with CPU image output: 26 mean speedups above 1x, 3 below 1x, out of 29 comparisons. The eight CPU-reference rows remain unqualified until source GPU availability is established.

## Source-GPU audit and reproducible latest selection

`export_inputs.json` explicitly selects the accepted measurements for each updated row; run `python3 regenerate.py` from this directory to regenerate the CSVs without rerunning benchmarks. The six CPU fallback rows now have source-code evidence that the pinned RGB renderer uses CPU pygame or Matplotlib. Piecewise and TwoRoom instead use completed paired RTX3090 measurements of the unmodified source Torch renderer configured for CUDA. Public source CPU NumPy output (and its transfer) remains timed.

Scene timings compare public rendering against source instances whose State/derived geometry was installed before timing. Our public renderer receives RenderState and performs its internal geometry update inside timing. Disk loading, rollouts, State projection, compilation and setup remain excluded. These are renderer throughput results, not a comparison of total full-State installation plus rendering.

## Earlier completed performance follow-up

All new acceptance jobs are terminal and passed. The selected Scene V2 GPU and CPU results, Humanoid CPU working-set fix, Cube CPU regression result, and both navigation source-GPU comparisons are in the regenerated tables. The exact adopted Scene file matches the staged/tested hash. Validation:69 local tests passed,2 optional skips;3 current Humanoid boundary tests passed on Oscar. No push or semantic acceptance change.

The latest table has38 completed CPU rows and37 completed GPU rows. Of31 same-device GPU comparisons,28 have mean speedup above1; Scene is approximately tied (0.9895x), while both Craftax pixel rows remain slower and deferred. Six additional rows use documented pinned-source CPU fallbacks. All completed mean MSEs remain <=0.001. CPU mean wins are29/38; near-ties must not be treated as substantial gains. Meaningful remaining CPU regressions outside deferred Craftax are Finger,Manipulator,and BallInCup.

Scene B256:1,355.79 versus source1,370.20 GPU FPS (CPU image output), and2.87754 versus2.87605 CPU FPS, MSE0. Humanoid B256 CPU:10.7637 versus source10.8108FPS, MSE0.0004328633±0.0000732948. Raw timing SD and all batches remain available in the detailed CSV and result JSON.

## Latest CPU bounds and Scene scaling follow-up

The accepted CPU component-layout change preserves exact images at B64/128/256
for BallInCup, Finger, and Manipulator. All numerical, cross-consumer, and RTX3090
regression gates passed. Their B256 results now explicitly replace the previous
CPU rows through `export_inputs.json`:

| Environment | Ours CPU FPS | Source CPU FPS | Source-relative speedup | Improvement over paired previous implementation |
|---|---:|---:|---:|---:|
| BallInCup | 15.005 | 14.533 | 1.032× | 1.492× |
| Finger | 12.716 | 21.530 | 0.591× | 1.105× |
| Manipulator | 16.524 | 22.803 | 0.725× | 1.212× |

BallInCup is near source; Finger and Manipulator still need CPU work. Hardware
differs between some prior campaigns, so code-improvement ratios above come
from baseline/candidate measurements on the same new allocation. The Humanoid
B256 row remains unchanged; its fresh residual profile and B64 bounds-regression
timing are separate diagnostics.

Scene's completed RTX3090 sweep covers B256 through B16,384 with seven paired
timings each. Its fresh B256 CPU-output result is 1,361.4 versus 1,379.9 FPS
(0.9866×); that row replaces the prior Scene GPU selection. There is no CPU-output
crossover in the sweep. CUDA output yields approximately 1.004–1.035× source,
with nearly flat candidate FPS; the full sweep is linked from
`../scene_gpu_scaling_20260929/README.md`. The main table stays fixed at B256.

Latest coverage remains 38 completed CPU and 37 completed GPU rows, with all
completed mean MSE <=0.001. CPU mean wins are 30/38, including near-ties; the GPU
tally remains 28/31 same-device mean wins plus six documented CPU fallbacks.
No semantic status promotion or push occurred. Original raw evidence is retained.
