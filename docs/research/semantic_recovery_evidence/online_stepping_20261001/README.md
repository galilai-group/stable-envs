# Batched stepping and Fetch execution checkpoint

Baseline: `d20190261bec77f2ad68dd94ed4403f86841344b`, branch
`repair/render-config-fixes`. This checkpoint replaces public `transition`,
`unroll`, and `Dynamics` with the [step/capture API](../../../concepts/dynamics.md).
The semantic matrix stays **39 VERIFIED / 3 INCOMPLETE**. Earlier accepted
evidence is unchanged.

`step` accepts `[B,A]` or `[B,T,A]`. Sequence lanes stop after their first reported
termination/truncation, with an explicit validity mask. Capture can retain States,
the final frame, or all frames. Fetch additionally exports State by field and
automatically uses up to four persistent CPU workers for batches of at least 64.
Other families retain their existing execution strategies.

## Measurements

The [CSV](fetch_execution.csv) contains means and sample standard deviations over
five repeats. The measured task is FetchPush, horizon 8, with known actions;
reset, restoration, and warmup are outside the timed regions. Rendering uses the
existing OpenGL backend at 64×64. Runs allocate eight CPU cores and an RTX 3090;
Torch/BLAS threading is limited to one. Fetch physics always runs on CPU.

| Output device | Batch | Original serial steps + State/s | New four-worker steps + State/s | Equally parallel upstream task steps/s |
| --- | ---: | ---: | ---: | ---: |
| CPU | 64 | 642.6 ± 0.6 | 1395.1 ± 5.4 | 1434.0 ± 3.1 |
| CPU | 256 | 622.1 ± 15.4 | 1442.7 ± 14.4 | 1481.1 ± 3.9 |
| CUDA | 64 | 519.4 ± 1.4 | 1391.3 ± 3.7 | 1448.3 ± 4.8 |
| CUDA | 256 | 527.1 ± 8.2 | 1431.4 ± 19.3 | 1486.9 ± 12.4 |

The upstream task-only column excludes full State export and rendering. It shows
the remaining adapter cost at the same worker count; it is not a renderer
speedup comparison. CPU State export improves about 13–14×; CUDA-destined export
improves about 27–35×. Whole-step gains are smaller because physics still costs
time. Eight workers provide little improvement over four in this sweep.

At batch 256 with CUDA output and four workers, capturing every frame reaches
1018.3 ± 7.0 transitions/images per second. Capturing only the final frame reaches
1409.7 ± 14.4 transitions/s, but emits only one image per eight transitions.
Render-only throughput from prepared State is 3516.9 ± 67.2 frames/s. These rates
must not be conflated. No simulation/render overlap is implemented here.

## Acceptance and provenance

- Local targeted API/State suite: 215 passed, four CUDA-only skips subsequently
  passed on Oscar. Separate oracle/API suite: 50 passed, including 12 DMC/Reacher
  oracle cases. Semantic reference harness: two passed.
- Final Oscar suite: 51 passed, one optional Matplotlib skip. That optional
  Navigation coverage passes locally with its dependencies installed.
- All 12 Fetch registrations independently pass source State/task equality,
  worker-count invariance, and distinct per-lane TimeLimit checks at batch 64 on
  CPU and CUDA output paths. Final automatic worker selection is exercised.
- Fetch sequence/all-frame rendering exactly matches repeated single-step
  rendering (MSE 0); final-frame capture exports one full snapshot. CartPole,
  Pendulum, TwoRoom, and Piecewise also pass native CUDA capture checks.
- A fresh wheel builds, excludes the removed public module, and passes an
  unpacked-wheel API smoke test with existing dependencies.

The frame equality check validates capture behavior against the existing
renderer. It is **not a new upstream image-parity measurement** or an update to
the 42-row renderer benchmark.

Oscar jobs 6911735 (timing), 6911954 (native GPU capture), and 6912594 (final
automatic-worker acceptance) completed with exit code 0. Initial staging jobs
6911109 and 6911355 failed because the first archive omitted a fixture and the
second suite required unavailable Matplotlib. Corrected runs use fresh stage
directories. The v3 timing kernel and v4 production kernel are identical; v4
adds automatic selection of the measured four-worker strategy.

`runs/` retains manifests, raw results, scripts and terminal logs. `collect.py`
regenerates this checkpoint's CSV/summary and verifies current production hashes
against the final v4 GPU stage. `sha256.json` inventories this evidence directory.

Broad historical renderer-State tests are not all green: representative Crafter,
Highway, and CraftaxClassic failures also reproduce on the starting commit.
The baseline reproduction log is retained under `local/`; these are not claimed
as new passing acceptance gates.

This completes the shared API and first bounded Fetch execution optimization.
Process-based executors for other families, bounded VRAM buffering,
simulation/render overlap, and closed-loop visual-policy measurements remain
future work.
