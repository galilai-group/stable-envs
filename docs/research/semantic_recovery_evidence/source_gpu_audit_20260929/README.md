# Pinned source GPU availability audit

This audit resolves eight registrations previously measured only against source CPU. It concerns the exact pinned source implementations and saved default configurations, not every version or possible backend.

Five classic-control registrations draw CPU pygame/gfxdraw surfaces. SimplePointMaze draws Matplotlib patches through the CPU Agg rasterizer. These six may use explicitly labeled source-CPU fallback comparisons because their pinned RGB call paths have no supported GPU rasterizer.

TwoRoom and Piecewise have Torch raster math. They expose no constructor device argument, but their existing math can be configured for CUDA by placing their owned tensors there and allocating under `torch.device("cuda")`. This is **CUDA-configurable source**, not a native constructor GPU API. No source files or raster equations are changed. Their public `render()` still returns NumPy and its CUDA-to-host copy remains in the timer. The CUDA-output variant necessarily adds host-to-device transfer afterward because that is the source public API.

Piecewise's optional bias-field visualization uses Matplotlib and is outside the saved default configuration. This audit does not claim that every Piecewise visual variation runs wholly on GPU.

`audit.json` provides machine-readable support classifications and exact call-path citations. `source_hashes.json` hashes the preserved source files. Every preserved stable-worldmodel Python file matches git commit `176d54532b3c0a251a4f509b74f2fdacc143a63d` byte-for-byte. Gymnasium 1.3.0 and pygame 2.6.1 identities come from the completed paired benchmark's installed-package record; the Gymnasium files were copied from that exact Oscar environment.

The two-runtime gate uses 16 distinct saved States, repeated into batches 64, 128, and 256. It checks full-State recapture/nonmutation, repeatability, source CUDA intermediate placement, and per-batch pixels. `source_anchor_mse` reports source CUDA versus saved source CPU image error. Timings alternate ours/source order on the same RTX 3090 allocation. No loading, State placement, compilation or rollout is timed. There is no source image cache.

GPU array: `6805885_[34,40]`, initially condo, moved to `gpu`/`norm-gpu` while pending to use available allocations. Both RTX3090 tasks completed and passed; raw results and comparison.csv are collected.

The pinned navigation source exposes a scalar public render API, with no batch argument. The paired harness calls it once per lane and stacks every freshly rendered image, while ours uses its existing batch API. Both produce the same batch size and destination; source images are not reused from a pixel cache.

Supplemental runtime gate **PASS**: job 6806134 on NVIDIA RTX A5500 verified both unchanged source Torch renderers execute CUDA, recapture identical State, and pass B64/128/256 output and nonmutation checks. Public CPU-output B256: Piecewise 21,294 ours / 504 source FPS (42.24x), TwoRoom 19,358 / 414 FPS (46.70x). Mean ours-vs-source GPU normalized MSE is 8.08e-9 / 8.47e-9; source GPU-vs-saved-CPU mean MSE is also about 8e-9. These timings remain supplemental and are not mixed into the RTX3090 table. RTX3090 job 6805885 also passed. Its B256 CPU-output results are Piecewise 52,539.84 / 806.79 FPS (65.12×), and TwoRoom 25,444.33 / 653.20 FPS (38.95×), ours/source respectively. The main table uses these same-device RTX3090 results.
