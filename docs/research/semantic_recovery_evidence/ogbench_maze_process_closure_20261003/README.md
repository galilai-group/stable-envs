# Maze process CPU/CUDA acceptance and bounded throughput

Job 6964833 completed on gpu2111 with RTX 3090 and eight physical CPUs. All eleven
selected cluster tests pass (four unrelated Cube/Scene tests deselected). Both
CPU/software-GL and CUDA/EGL public gates independently pass all 22 cases and
verify rendering while the next transition is submitted. Full State/task/frame,
RNG continuation, early stopping, variations and retained-output ownership are exact.

The 16-row paired CSV compares local/P4/P8 at B64/256, T1/8, six samples each.
P8/local ranges 4.94–7.10x. B256/T8: 12,733 ± 118 valid steps/s CPU output versus
1,794 ± 9 local; 12,237 ± 61 CUDA output versus 1,769 ± 8 local. These are State-
retaining public step rates with frames=None, not renderer FPS or source speedups.

The submitted source is reconstructed from ../ogbench_maze_process_20261003/oscar/
source_manifest.json and source_overrides.json at commit b14e60b9. In particular,
this gate used the previous all-active masking implementation. Its exact report
source hashes were checked against the frozen bundle. Subsequent sparse masking
has separate native/Fetch acceptance and still needs integration/default gates
before automatic Maze process selection is accepted. The public default remains local.

Run summarize_ogbench_maze_process.py results NEW_OUTPUT --source FROZEN_CHECKOUT.
Full component, larger batch, source and combined pipeline comparisons remain.
