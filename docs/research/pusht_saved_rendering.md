# PushT saved-pose renderer comparison

This checkpoint measures `swm/PushT-v1` and `swm/PushT-Discrete-v1` independently against stable-worldmodel commit `176d54532b3c0a251a4f509b74f2fdacc143a63d`. It does not accept either registration's dynamics or full State implementation. The semantic matrix remains **39 VERIFIED / 3 INCOMPLETE**. The production renderer is unchanged.

## Results

Batch 256, CPU-returned RGB images; RTX 3090 for candidate GPU rendering. Each registration has 1,024 distinct saved source poses and seven paired timing repeats. MSE below is normalized, mean ± sample SD across frames.

| Registration | CPU MSE | GPU MSE | Source CPU FPS | Ours CPU FPS | Ours GPU FPS | CPU speedup | GPU/CPU-source speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `swm/PushT-v1` | 0.000690 ± 0.000060 | 0.000745 ± 0.000064 | 435.5 | 830.7 | 32,584.6 | 1.91× | 74.44× |
| `swm/PushT-Discrete-v1` | 0.000686 ± 0.000056 | 0.000741 ± 0.000061 | 437.2 | 882.5 | 28,288.7 | 2.02× | 62.56× |

All 1,024 frames in each of the four comparisons pass MSE ≤ 0.001. The worst CPU/GPU frames are 0.000849/0.000909 for continuous PushT and 0.000855/0.000915 for discrete PushT. Neither registration meets the stricter 0.0005 mean-MSE target. These are threshold matches, not exact pixel equality.

Source GPU rendering is unavailable. GPU speedups use the separately paired CPU source measurement on the GPU host (437.7 and 452.2 FPS), rather than claiming a GPU-to-GPU comparison. The CSV includes all SDs and raw evidence links. For CPU-returned output, the measured best GPU batches are 256 for continuous (32,585 FPS) and 2,048 for discrete (34,035 FPS). With GPU-resident output, batch 2,048 reaches 228,842 and 244,145 FPS respectively; these are a different output contract from the main table.

The other 40 rows in the main 42-registration summary are byte-for-byte equivalent as parsed CSV records. Measurement totals are 40 completed CPU rows and 39 completed GPU rows, plus the two non-RGB symbolic registrations and Rocket GPU error. No semantic promotions or push occurred.

## Protocol

Each registration has its own on-disk corpus of 1,024 source frames, collected from seeded random-action episodes of at most 20 frames. The discrete registration uses its real discrete actions. Defaults and 224×224 RGB output are retained, including upstream's 512×512 Pygame rasterization followed by OpenCV resizing. Ours uses its existing default supersampling factor of one.

CPU and GPU jobs load the same saved poses and reference images. The candidate receives float32 compact render poses. Candidate scene construction is timed; full-State projection, file reads, initial input upload, rollouts, source reconstruction and warmup/compilation are excluded. This matches the existing table's prepared-input renderer scope, and is not an online environment benchmark.

The source draws collision-debug marks as well as geometry. Pose installation alone failed a byte-exact image check on two pixels in the initial attempt. For the completed protocol, each source owner is reconstructed by seeded replay **before timing**, then its pose and rendered pixels must exactly match the corpus. This preserves source contact marks instead of deleting them to improve MSE. Source rendering runs afresh for every timed frame; no cached image is returned.

For each batch size and output destination, seven pairs alternate candidate/source timing order. Every repeat renders all 1,024 saved frames (2,048 outputs at batch 2,048, cycling the same corpus twice). Both sides return materialized batches, and required output transfers are timed. CUDA calls synchronize before and after timing. All timed outputs are checked outside timers against the saved reference; candidate batch results are also checked against the initial quality pass.

CPU batches: 64, 128, 256. GPU batches: 64, 128, 256, 512, 1,024, 2,048, with both CPU and CUDA output. One Torch/BLAS thread is configured. Source RGB rendering is CPU-only (`pygame.Surface`, Pymunk debug draw, `surfarray`, `cv2.resize`); the discrete class inherits it. A CUDA output tensor does not make this a source GPU renderer. GPU/CPU speedups are explicitly CPU-source fallbacks.

MSE is mean squared RGB error after division by 255. MSE SD describes variation across the 1,024 frames. Timing SD describes the seven repeat-average per-frame times, not the distribution of isolated frame latencies. FPS is total frames divided by mean elapsed time; FPS SD is the SD of seven repeat throughputs.

## Evidence and reproduction

- [Benchmark script](https://github.com/galilai-group/stable-envs/blob/cb000ff100fa6ae90e4ff1da7de87f6489a507ab/scripts/benchmarks/renderer_table/pusht_saved.py)
- [Submission script](semantic_recovery_evidence/pusht_saved_rendering_20260930/run-v2.sbatch)
- [Collection and validation script](semantic_recovery_evidence/pusht_saved_rendering_20260930/collect.py)
- [Worst-frame image gallery](semantic_recovery_evidence/pusht_saved_rendering_20260930/images.html)
- [Two-row comparison](semantic_recovery_evidence/pusht_saved_rendering_20260930/pusht_comparison.csv)
- [42-registration summary](semantic_recovery_evidence/paired_saved_rendering_20260929/paired_rendering_summary.csv)

The evidence directory retains corpus archives, per-frame MSE, raw timing samples, worst/median image pairs, exact source and candidate hashes, hardware/dependency provenance, scheduler evidence, and the failed first attempt. The pre-update CSVs are retained under `previous-csv/`.

The discrete GPU retry uses bounded 32-image chunks for post-timing validation. Every pixel is still checked. Regression tests verify identical per-frame MSE reductions, corpus cycling, and detection of a final-pixel corruption. The timed candidate/source rendering function ASTs are identical between harness v2 and v3. The previous GPU sweep was retained as partial evidence and cancelled to release allocation quota; its completed CPU result is retained.

The lightweight source loader executes the pinned environment/registration modules while bypassing unrelated SWM data and expert-policy imports. It exports the original concrete PushT classes without modifying their rendering or dynamics methods.

Example (run GPU commands only on an allocated Oscar GPU):

```sh
python -m scripts.benchmarks.renderer_table.pusht_saved capture swm/PushT-v1 corpus --oracle /path/to/pinned/swm
python -m scripts.benchmarks.renderer_table.pusht_saved benchmark swm/PushT-v1 cpu-results --oracle /path/to/pinned/swm --corpus corpus --device cpu --batches 64 128 256
python -m scripts.benchmarks.renderer_table.pusht_saved benchmark swm/PushT-v1 gpu-results --oracle /path/to/pinned/swm --corpus corpus --device cuda --batches 64 128 256 512 1024 2048
```

Repeat independently for `swm/PushT-Discrete-v1`. Output directories must not already exist, protecting old evidence from overwrite.
