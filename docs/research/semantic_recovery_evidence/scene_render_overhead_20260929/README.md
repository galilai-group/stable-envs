# Scene rendering overhead gate

Paired saved-State timing at B64/128/256, same-device source comparisons, no loading/rollouts/projection inside timers. GPU job 6805950 runs unchanged baseline and candidate in separate processes on one RTX 3090 allocation. CPU job 6806083 checks candidate Cube/Scene software GL against source on each allocated host. Candidate production snapshot starts from the completed Cube-cache source and overlays only manipulation_opengl.py.

Gate requires full-State nonmutation, source MSE <=0.001 for every measured State, and GPU baseline/candidate image differences at most 1 uint8 level (the existing repeated-GL tolerance). Raw images, timing SD, source timing and scripts are retained. Passing correctness does not imply a performance win. Sixteen distinct States are repeated to form batches.

CPU checks use a separate frozen snapshot staged by CPU job6806009, so they do not wait for GPU scheduling. Supplemental job6806113 uses an available A5000/A5500 with the same paired workload; these hardware-specific timings do not replace the queued RTX3090 measurements.

All V1 gates completed and passed: RTX3090, supplemental GPU, and Cube/Scene software-GL CPU B64/128/256. CPU images match source exactly on every measured saved State. The separately staged V2 Scene update has its own evidence directory and CPU gate.
