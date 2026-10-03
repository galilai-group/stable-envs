# Native Navigation profiling evidence

Parent `6ce4e82931a629b68218bbe5331296bdc0a946d2` plus the frozen files in
`source_manifest.json`. Production Navigation and the shared step dispatcher are
unchanged. See [the report](../../navigation_online_profile.md).

`local/TwoRoom.json` and `local/Piecewise.json` are complete CPU diagnostics.
`local/SimplePointMaze.json` deliberately remains FAIL/KeyboardInterrupt:
its source-required 500×500 RGB goal image makes large chunk histories expensive
and its repeated State restore rebuilds the cached image via Matplotlib.
The process was interrupted during B1024 restore. Its 27 completed cells are
partial evidence, not a successful complete sweep.

Oscar array `6964157` completed indices 0/1 for TwoRoom/Piecewise. Their CPU/CUDA
reports, logs and 144-cell comparison are preserved under `oscar/`. Index 2 was
canceled while pending. `run-bounded.sh` submits the replacement SimplePointMaze
job `6964210`, on the same benchmark/library source, explicitly selecting
B1/16/64 and three samples. It failed because Matplotlib was absent; preserve
that failure under `oscar/bounded`. `setup-matplotlib.sh` installed pinned 3.10.9
and missing dependencies in task-local scratch, with a successful import check
in job `6964294`. `run-bounded-v2.sh` is dependency-gated retry `6964296` and writes
`bounded-v2`. No output fields or source goal image are dropped.

The manifest hashes the uploaded source archive and all files. Reconstruction
uses `source_overrides.json` when a path is present, otherwise this checkpoint's
committed source. The classic exporter present in this bundle is independent of
the Navigation path. Launch scripts and actual job handles are preserved.
SimplePointMaze measurements remain pending; later reports belong in a new closure folder.

Units matter: State-only and pipeline rates count actual valid transitions;
render-only rates count frames. Pipeline cells additionally report actual
selected-frame FPS. The T8/final-frame pipeline renders one frame per lane,
not eight. Restore/warmup and cProfile are outside the timed intervals. These
diagnostics make no upstream speedup or source-image parity claim.
