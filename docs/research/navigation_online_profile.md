# Native Navigation online profile

The three accepted registrations already use native Torch batch dynamics:
`TwoRoom`, `Piecewise`, and `SimplePointMaze`. This checkpoint changes no
Navigation production code and adds no process pool. It measures private task
steps, public steps with/without State history, export, native-resolution render
and the final-frame pipeline. Rates count actual valid transitions; pipeline
reports both steps/s and selected frames/s, so T8 is not mislabeled as rendering
eight frames per lane.

The [profiling script](../../scripts/benchmarks/online_stepping/navigation_profile.py)
keeps restore/warmup outside timing, synchronizes CUDA, records raw samples and
separate cProfile diagnostics, and refuses to overwrite reports. It is not a
source comparison or fresh image-fidelity gate. Local evidence and frozen
Oscar launchers are under
[`navigation_online_20261002`](semantic_recovery_evidence/navigation_online_20261002/).

## Local CPU findings

TwoRoom and Piecewise completed B1/64/256/1024/4096 State-only profiles, with
native-resolution rendering at B1/64. At B64/T1:

| Registration | Raw task steps/s | Public steps/s, all States | Render-only frames/s | Combined final-frame frames/s |
| --- | ---: | ---: | ---: | ---: |
| TwoRoom | 272,200 | 194,613 | 63,485 | 27,680 |
| Piecewise | 381,247 | 240,702 | 60,414 | 28,584 |

At B4096/T1, public State-retaining rates were approximately 830,324 and 860,619
steps/s, respectively. These CPU diagnostics support retaining their tensor
batching. Their combined flow still has copying/capture overhead; individual
stage rates cannot simply be added or assumed independent.

SimplePointMaze's partial profile reveals a different bottleneck. Source
semantics require a **500×500×3 uint8 cached goal image in every step's
`info.goal`**, even when `frames=None`. At B64 the private step measured about
49,610 steps/s but the public State-retaining step only 5,551 steps/s. cProfile
locates roughly 9 ms of an 11 ms public call in `_masked_outputs`, including
cloning and masked assignment over that image. The profile uses all-active
lanes, so this also exposes avoidable empty-mask work. The source-required image
must not be dropped to improve the benchmark.

Its `set_state` also rebuilds each cached goal image through Matplotlib. That
restore cost is excluded from the timing but makes repeated benchmark setup
slow. The local large-batch run was deliberately interrupted during B1024
restore after this was identified; its partial report remains marked FAIL with
`KeyboardInterrupt()`. Completed cells remain useful diagnostics, not a completed
registration benchmark. A B4096/T8 goal-image history alone needs 24.576 GB
decimal before temporary copies, other outputs or renderer storage.

## Oscar findings and next work

Oscar array **`6964157`**, indices 0/1, completed the full TwoRoom/Piecewise
profiles on RTX A6000 nodes. All 144 cells and their tested source hashes were
verified. The [Oscar CSV](semantic_recovery_evidence/navigation_online_20261002/oscar/comparison.csv)
preserves the actual host, device and mean/SD. At B64/T1:

| Registration / device | Public steps/s | Render-only frames/s | Combined frames/s |
| --- | ---: | ---: | ---: |
| TwoRoom / CPU | 25,210 | 10,418 | 4,717 |
| TwoRoom / CUDA | 15,154 | 26,932 | 9,559 |
| Piecewise / CPU | 32,307 | 11,005 | 5,148 |
| Piecewise / CUDA | 21,189 | 44,668 | 14,081 |

At B4096/T1, public steps/s were 97,170 CPU / 95,841 CUDA for TwoRoom and
98,131 CPU / 99,790 CUDA for Piecewise. Physics/task work is native to the chosen
device. GPU rendering helps the composed flow at B64 despite lower standalone
step throughput there. These are general-GPU diagnostics, not RTX 3090 evidence.

Pending SimplePointMaze index 2 was canceled before execution and replaced by
`6964210`, B1/16/64, three repetitions, on the same frozen source. That job failed
because the selected environment lacked Matplotlib. Job `6964294` installed and
import-checked pinned Matplotlib 3.10.9 and its missing dependencies in isolated
scratch, preserving shared environments. Retry **`6964296`** depends on that
successful check and writes a new output directory. Its GPU profile is pending.

Next, measure a semantics-preserving fast path for all-active output masks and
audit whether cached goal-image restoration can reuse images when all of their
State/Variation dependencies are unchanged. Both require owned-output and
mixed terminal-lane regression gates; do not remove the source info payload or
change the user-facing selectors. Use the resulting evidence to decide whether
further native chunk compilation is worthwhile. Process pools are not the
appropriate first change for these already-batched dynamics.
